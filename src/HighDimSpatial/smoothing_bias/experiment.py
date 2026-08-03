"""Synthetic finite-design experiments for ignored spatial support.

The experiment fits one decay parameter while variance, smoothness, and nugget
are known.  A dense log-decay profile is evaluated once for every configuration
and all Monte Carlo replicates are solved simultaneously.  A local quadratic
interpolation removes the coarse-grid artifact present in the legacy pilot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from HighDimSpatial.smoothing_bias.design import (
    epanechnikov_smoothing_matrix,
    jitter_design,
    regular_grid_1d,
    regular_grid_2d,
    select_rectangular_centers,
)
from HighDimSpatial.smoothing_bias.kl import (
    exact_smoothed_covariance,
    fit_population_log_decay,
    gaussian_population_criterion,
    matern_covariance,
    naive_point_covariance,
)


@dataclass(frozen=True)
class DecayProfile:
    """Population and replicate optima from a common log-decay grid."""

    population_decay: float
    population_objective: float
    sample_decays: np.ndarray
    sample_objectives: np.ndarray
    sample_at_boundary: np.ndarray
    log_decay_grid: np.ndarray
    population_grid_objective: np.ndarray


@dataclass(frozen=True)
class FiniteDesignResult:
    """Complete result for one finite-design Monte Carlo configuration."""

    records: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _quadratic_grid_minima(
    grid: np.ndarray,
    objectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate columnwise minima on a common equally spaced grid."""
    x = np.asarray(grid, dtype=float)
    values = np.asarray(objectives, dtype=float)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[:, None]
    if x.ndim != 1 or values.ndim != 2 or values.shape[0] != x.size or x.size < 3:
        raise ValueError("objectives must have one row per grid point and at least three rows")
    increments = np.diff(x)
    if not np.allclose(increments, increments[0], rtol=1e-12, atol=1e-14):
        raise ValueError("quadratic interpolation requires an equally spaced grid")

    columns = np.arange(values.shape[1])
    indices = np.argmin(values, axis=0)
    at_boundary = (indices == 0) | (indices == x.size - 1)
    estimates = x[indices].copy()
    minimum_values = values[indices, columns].copy()
    interior_columns = columns[~at_boundary]
    if interior_columns.size:
        center_indices = indices[interior_columns]
        left = values[center_indices - 1, interior_columns]
        center = values[center_indices, interior_columns]
        right = values[center_indices + 1, interior_columns]
        denominator = left - 2.0 * center + right
        valid_curvature = denominator > 0.0
        offsets = np.zeros_like(denominator)
        offsets[valid_curvature] = 0.5 * (
            left[valid_curvature] - right[valid_curvature]
        ) / denominator[valid_curvature]
        offsets = np.clip(offsets, -1.0, 1.0)
        estimates[interior_columns] += offsets * increments[0]
        minimum_values[interior_columns] = center - 0.5 * (
            left - right
        ) * offsets + 0.5 * denominator * offsets**2

    if was_vector:
        return estimates[0], minimum_values[0], at_boundary[0]
    return estimates, minimum_values, at_boundary


def profile_log_decay_grid(
    samples: np.ndarray,
    true_covariance: np.ndarray,
    covariance_builder,
    log_decay_bounds: tuple[float, float],
    grid_size: int = 161,
) -> DecayProfile:
    """Profile population and sample objectives over a dense log-decay grid."""
    values = np.asarray(samples, dtype=float)
    truth = np.asarray(true_covariance, dtype=float)
    if values.ndim != 2 or values.shape[1] != truth.shape[0]:
        raise ValueError("samples must have shape (replicates, covariance dimension)")
    if isinstance(grid_size, bool) or int(grid_size) != grid_size or grid_size < 21:
        raise ValueError("grid_size must be an integer of at least 21")
    lower, upper = (float(value) for value in log_decay_bounds)
    if not np.isfinite(lower + upper) or lower >= upper:
        raise ValueError("log_decay_bounds must be finite and increasing")

    log_grid = np.linspace(lower, upper, int(grid_size))
    population_objectives = np.empty(log_grid.size)
    sample_objectives = np.empty((log_grid.size, values.shape[0]))
    constant = truth.shape[0] * np.log(2.0 * np.pi)
    for index, log_decay in enumerate(log_grid):
        covariance = np.asarray(covariance_builder(float(np.exp(log_decay))), dtype=float)
        factor = cholesky(covariance, lower=True, check_finite=False)
        log_determinant = 2.0 * np.log(np.diag(factor)).sum()
        whitened = solve_triangular(factor, values.T, lower=True, check_finite=False)
        sample_objectives[index] = 0.5 * (
            constant + log_determinant + np.sum(whitened**2, axis=0)
        )
        population_objectives[index] = gaussian_population_criterion(covariance, truth)

    population_fit = fit_population_log_decay(
        truth,
        covariance_builder,
        (lower, upper),
        xatol=1e-10,
    )
    if not population_fit.success:
        raise ValueError(f"population target optimization failed: {population_fit.message}")
    if population_fit.at_lower_bound or population_fit.at_upper_bound:
        raise ValueError("population target lies on the declared decay bounds")
    sample_log_decays, minimum_objectives, sample_at_boundary = _quadratic_grid_minima(
        log_grid, sample_objectives
    )
    return DecayProfile(
        population_decay=population_fit.decay,
        population_objective=population_fit.objective,
        sample_decays=np.exp(sample_log_decays),
        sample_objectives=minimum_objectives,
        sample_at_boundary=sample_at_boundary,
        log_decay_grid=log_grid,
        population_grid_objective=population_objectives,
    )


def _build_design(config: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    dimension = int(config["dimension"])
    spacing = config.get("spacing", 1.0)
    if dimension == 1:
        inputs = regular_grid_1d(int(config["number_of_points"]), float(spacing))
    elif dimension == 2:
        inputs = regular_grid_2d(tuple(config["shape"]), spacing)
    else:
        raise ValueError("dimension must be one or two")
    centers = select_rectangular_centers(
        inputs,
        stride=config.get("output_stride", 1),
        boundary_trim=config.get("boundary_trim", 0.0),
    )
    maximum_jitter = config.get("input_jitter", 0.0)
    if np.any(np.asarray(maximum_jitter, dtype=float) > 0.0):
        inputs = jitter_design(inputs, maximum_jitter, rng, preserve_bounds=True)
    return inputs, centers


def run_finite_design_configuration(
    config: dict[str, Any],
    *,
    seed: int,
) -> FiniteDesignResult:
    """Run one manifest configuration and return auditable replicate records."""
    replicates = int(config["replicates"])
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    inputs, centers = _build_design(config, rng)
    smoothing = epanechnikov_smoothing_matrix(
        inputs,
        centers,
        config["bandwidth"],
        kernel=config.get("kernel", "radial"),
    )
    rank = int(np.linalg.matrix_rank(smoothing))
    if rank != smoothing.shape[0]:
        raise ValueError("smoothing operator is not full row rank")

    variance = float(config["variance"])
    true_decay = float(config["decay"])
    smoothness = float(config["smoothness"])
    nugget = float(config.get("nugget", 0.0))
    raw_covariance = matern_covariance(
        inputs,
        variance=variance,
        decay=true_decay,
        nu=smoothness,
        nugget=nugget,
    )
    raw_factor = cholesky(raw_covariance, lower=True, check_finite=False)
    standard_normals = rng.standard_normal((inputs.shape[0], replicates))
    smoothed_samples = (smoothing @ raw_factor @ standard_normals).T
    true_covariance = smoothing @ raw_covariance @ smoothing.T
    true_covariance = (true_covariance + true_covariance.T) / 2.0

    def corrected_builder(candidate_decay: float) -> np.ndarray:
        return exact_smoothed_covariance(
            inputs,
            smoothing,
            variance,
            candidate_decay,
            smoothness,
            nugget,
        )

    def naive_builder(candidate_decay: float) -> np.ndarray:
        return naive_point_covariance(
            inputs,
            smoothing,
            variance,
            candidate_decay,
            smoothness,
            nugget,
            output_locations=centers,
        )

    bounds = tuple(config.get("log_decay_bounds", [np.log(0.1), np.log(4.0)]))
    grid_size = int(config.get("log_decay_grid_size", 161))
    profiles = {
        "corrected": profile_log_decay_grid(
            smoothed_samples, true_covariance, corrected_builder, bounds, grid_size
        ),
        "naive": profile_log_decay_grid(
            smoothed_samples, true_covariance, naive_builder, bounds, grid_size
        ),
    }

    records: list[dict[str, Any]] = []
    for model, profile in profiles.items():
        for replicate in range(replicates):
            estimate = float(profile.sample_decays[replicate])
            error = estimate - true_decay
            records.append(
                {
                    "model": model,
                    "replicate": replicate,
                    "seed": seed,
                    "decay_estimate": estimate,
                    "population_target": profile.population_decay,
                    "decay_true": true_decay,
                    "signed_error": error,
                    "absolute_error": abs(error),
                    "squared_error": error**2,
                    "objective": float(profile.sample_objectives[replicate]),
                    "at_bound": bool(profile.sample_at_boundary[replicate]),
                }
            )

    corrected_truth_difference = float(
        np.max(np.abs(corrected_builder(true_decay) - true_covariance))
    )
    diagnostics = {
        "number_of_inputs": int(inputs.shape[0]),
        "number_of_outputs": int(centers.shape[0]),
        "smoothing_rank": rank,
        "smoothing_condition": float(np.linalg.cond(smoothing)),
        "smoothing_row_sum_error": float(np.max(np.abs(smoothing.sum(axis=1) - 1.0))),
        "true_covariance_condition": float(np.linalg.cond(true_covariance)),
        "corrected_truth_max_difference": corrected_truth_difference,
        "corrected_population_target": profiles["corrected"].population_decay,
        "naive_population_target": profiles["naive"].population_decay,
        "corrected_boundary_fits": int(np.sum(profiles["corrected"].sample_at_boundary)),
        "naive_boundary_fits": int(np.sum(profiles["naive"].sample_at_boundary)),
    }
    return FiniteDesignResult(records=records, diagnostics=diagnostics)


__all__ = [
    "DecayProfile",
    "FiniteDesignResult",
    "profile_log_decay_grid",
    "run_finite_design_configuration",
]
