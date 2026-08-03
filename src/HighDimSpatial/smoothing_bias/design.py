"""Finite spatial designs and deterministic smoothing operators.

The routines in this module deliberately return rectangular smoothing
operators.  This makes information loss visible in experiments: the smoothed
observations are ``S @ y`` and their covariance is ``S @ K @ S.T``.
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np


ArrayLike = float | Sequence[float] | np.ndarray


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] not in (1, 2):
        raise ValueError(f"{name} must have shape (n, d), with n > 0 and d in {{1, 2}}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return points


def _per_dimension(value: ArrayLike, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        result = np.repeat(float(array), dimension)
    elif array.ndim == 1 and array.size == dimension:
        result = array.astype(float, copy=True)
    else:
        raise ValueError(f"{name} must be scalar or have one value per spatial dimension")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _stride_per_dimension(stride: int | Sequence[int], dimension: int) -> np.ndarray:
    raw = np.asarray(stride)
    if raw.ndim == 0:
        raw = np.repeat(raw, dimension)
    if raw.ndim != 1 or raw.size != dimension:
        raise ValueError("stride must be scalar or have one value per spatial dimension")
    numeric = raw.astype(float)
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 1) or np.any(numeric % 1 != 0):
        raise ValueError("stride values must be positive integers")
    return numeric.astype(int)


def regular_grid_1d(
    number_of_points: int,
    spacing: float = 1.0,
    origin: float = 0.0,
) -> np.ndarray:
    """Return a one-dimensional regular grid as an ``(n, 1)`` array."""
    if isinstance(number_of_points, bool) or int(number_of_points) != number_of_points:
        raise ValueError("number_of_points must be a positive integer")
    number_of_points = int(number_of_points)
    if number_of_points <= 0 or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("number_of_points and spacing must be positive")
    if not np.isfinite(origin):
        raise ValueError("origin must be finite")
    return (float(origin) + float(spacing) * np.arange(number_of_points, dtype=float))[:, None]


def regular_grid_2d(
    shape: int | tuple[int, int],
    spacing: ArrayLike = 1.0,
    origin: ArrayLike = 0.0,
) -> np.ndarray:
    """Return a two-dimensional Cartesian grid as an ``(n_1 n_2, 2)`` array.

    A scalar ``shape`` requests a square grid.  Scalar spacing and origin are
    applied to both axes; length-two values allow rectangular anisotropic grids.
    """
    if np.asarray(shape).ndim == 0:
        shape_values = (shape, shape)
    else:
        shape_values = tuple(shape)
    if len(shape_values) != 2:
        raise ValueError("shape must be an integer or a pair of integers")
    shape_numeric = np.asarray(shape_values, dtype=float)
    if (
        not np.all(np.isfinite(shape_numeric))
        or np.any(shape_numeric < 1)
        or np.any(shape_numeric % 1 != 0)
    ):
        raise ValueError("shape values must be positive integers")

    spacing_values = _per_dimension(spacing, 2, "spacing")
    if np.any(spacing_values <= 0):
        raise ValueError("spacing values must be positive")
    origin_values = _per_dimension(origin, 2, "origin")
    axes = [
        origin_values[j] + spacing_values[j] * np.arange(int(shape_numeric[j]), dtype=float)
        for j in range(2)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([coordinate.ravel() for coordinate in mesh])


def select_rectangular_centers(
    input_points: np.ndarray,
    stride: int | Sequence[int] = 1,
    boundary_trim: ArrayLike = 0.0,
) -> np.ndarray:
    """Subsample a full Cartesian design and optionally trim its boundary.

    Striding is anchored at the first coordinate on each axis.  The physical
    boundary trim is then applied to the strided coordinates, so the returned
    centers remain a Cartesian product in deterministic lexicographic order.
    """
    points = _points(input_points, "input_points")
    dimension = points.shape[1]
    if np.unique(points, axis=0).shape[0] != points.shape[0]:
        raise ValueError("input_points must not contain duplicate locations")
    axes = [np.unique(points[:, j]) for j in range(dimension)]
    if int(np.prod([axis.size for axis in axes])) != points.shape[0]:
        raise ValueError("input_points must form a full Cartesian grid")

    strides = _stride_per_dimension(stride, dimension)
    trims = _per_dimension(boundary_trim, dimension, "boundary_trim")
    if np.any(trims < 0):
        raise ValueError("boundary_trim must be nonnegative")

    selected_axes: list[np.ndarray] = []
    for axis, current_stride, trim in zip(axes, strides, trims):
        selected = axis[::current_stride]
        coordinate_scale = max(1.0, float(np.max(np.abs(axis))))
        tolerance = 32.0 * np.finfo(float).eps * coordinate_scale
        selected = selected[
            (selected >= axis[0] + trim - tolerance)
            & (selected <= axis[-1] - trim + tolerance)
        ]
        if selected.size == 0:
            raise ValueError("boundary_trim and stride leave no output centers")
        selected_axes.append(selected)

    return np.asarray(list(product(*selected_axes)), dtype=float).reshape(-1, dimension)


def validate_smoothing_matrix(matrix: np.ndarray, *, atol: float = 1e-12) -> None:
    """Raise ``ValueError`` unless a smoothing matrix is finite and row stochastic."""
    smoothing = np.asarray(matrix, dtype=float)
    if smoothing.ndim != 2 or smoothing.shape[0] == 0 or smoothing.shape[1] == 0:
        raise ValueError("smoothing matrix must be nonempty and two-dimensional")
    if not np.all(np.isfinite(smoothing)) or np.any(smoothing < -atol):
        raise ValueError("smoothing matrix must be finite and nonnegative")
    if not np.allclose(smoothing.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise ValueError("each smoothing-matrix row must sum to one")


def epanechnikov_smoothing_matrix(
    input_points: np.ndarray,
    output_centers: np.ndarray,
    bandwidth: ArrayLike,
    *,
    kernel: str = "radial",
    coordinate_tolerance: float = 1e-12,
) -> np.ndarray:
    """Build a row-normalized radial or product Epanechnikov operator.

    ``bandwidth`` may be scalar or anisotropic.  A zero bandwidth in a
    coordinate imposes exact matching in that coordinate; when every bandwidth
    is zero, each row is the exact one-hot selector for its output center.
    Multiplicative normalizing constants are omitted because rows are normalized.
    """
    inputs = _points(input_points, "input_points")
    centers = _points(output_centers, "output_centers")
    if inputs.shape[1] != centers.shape[1]:
        raise ValueError("input_points and output_centers must have the same dimension")
    if kernel not in {"radial", "product"}:
        raise ValueError("kernel must be 'radial' or 'product'")
    if not np.isfinite(coordinate_tolerance) or coordinate_tolerance < 0:
        raise ValueError("coordinate_tolerance must be finite and nonnegative")

    widths = _per_dimension(bandwidth, inputs.shape[1], "bandwidth")
    if np.any(widths < 0):
        raise ValueError("bandwidth must be nonnegative")
    differences = inputs[None, :, :] - centers[:, None, :]
    zero_width = widths == 0
    exact_mask = np.ones(differences.shape[:2], dtype=bool)
    if np.any(zero_width):
        exact_mask = np.all(
            np.abs(differences[:, :, zero_width]) <= coordinate_tolerance,
            axis=2,
        )

    if np.all(zero_width):
        match_count = exact_mask.sum(axis=1)
        if np.any(match_count != 1):
            raise ValueError("zero bandwidth requires each center to match one unique input point")
        smoothing = exact_mask.astype(float)
        validate_smoothing_matrix(smoothing)
        return smoothing

    standardized = differences[:, :, ~zero_width] / widths[~zero_width]
    if kernel == "radial":
        squared_radius = np.sum(standardized**2, axis=2)
        raw_weights = np.maximum(1.0 - squared_radius, 0.0)
    else:
        raw_weights = np.prod(np.maximum(1.0 - standardized**2, 0.0), axis=2)
    raw_weights *= exact_mask

    row_sums = raw_weights.sum(axis=1)
    if np.any(row_sums <= 0):
        failed = np.flatnonzero(row_sums <= 0).tolist()
        raise ValueError(f"output centers have no positive kernel support in rows {failed}")
    smoothing = raw_weights / row_sums[:, None]
    validate_smoothing_matrix(smoothing)
    return smoothing


def jitter_design(
    points: np.ndarray,
    maximum_jitter: ArrayLike,
    rng: np.random.Generator,
    *,
    preserve_bounds: bool = False,
) -> np.ndarray:
    """Return an irregular design using reproducible coordinate-wise jitter.

    Every absolute displacement is bounded by ``maximum_jitter``.  Randomness is
    drawn only from the passed ``Generator``.  With ``preserve_bounds=True``,
    locations are clipped to the original axis-aligned bounding box.
    """
    design = _points(points, "points")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")
    bounds = _per_dimension(maximum_jitter, design.shape[1], "maximum_jitter")
    if np.any(bounds < 0):
        raise ValueError("maximum_jitter must be nonnegative")
    perturbation = rng.uniform(-bounds, bounds, size=design.shape)
    result = design + perturbation
    if preserve_bounds:
        result = np.clip(result, design.min(axis=0), design.max(axis=0))
    return result
