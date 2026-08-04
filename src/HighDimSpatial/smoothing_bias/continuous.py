"""Deterministic quadrature for continuously smoothed Matérn fields.

The product Epanechnikov smoother has independent coordinates.  If ``U`` and
``V`` are independent draws from its standardized kernel, then every smoothed
covariance is an expectation over ``D = U - V``.  Tensor Gauss--Legendre
quadrature therefore gives a high-accuracy, simulation-free oracle in one and
two spatial dimensions.  A nonsingular linear kernel transform permits compact
anisotropic supports, and the fitted lag may point in any declared direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import gamma, kve

from HighDimSpatial.smoothing_bias.kl import matern_correlation
from HighDimSpatial.smoothing_bias.theory import epanechnikov_difference_density


@dataclass(frozen=True)
class ContinuousPairTarget:
    """Population pair target after continuous product-kernel smoothing."""

    dimension: int
    smoothness: float
    true_decay: float
    pseudo_decay: float
    bandwidth: float
    lag: float
    variance_factor: float
    covariance_factor: float
    correlation: float
    quadrature_order: int


@dataclass(frozen=True)
class TransitionPairApproximation:
    """Two-term approximation to the pair target near smoothness one."""

    dimension: int
    smoothness: float
    true_decay: float
    pseudo_decay: float
    bandwidth: float
    lag: float
    variance_factor: float
    covariance_factor: float
    correlation: float
    quadrature_order: int


def product_epanechnikov_difference_quadrature(
    dimension: int,
    order: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and probability weights for ``U-V`` in dimension 1 or 2."""
    if dimension not in (1, 2):
        raise ValueError("dimension must be one or two")
    if isinstance(order, bool) or int(order) != order or order < 8 or order % 2:
        raise ValueError("order must be an even integer of at least eight")
    order = int(order)
    # Split at zero because the difference density and several theorem-facing
    # integrands contain absolute values.  A single Gauss rule across the cusp
    # converges unnecessarily slowly.
    positive_nodes, positive_weights = leggauss(order // 2)
    positive_nodes = positive_nodes + 1.0  # map [-1, 1] to [0, 2]
    nodes_1d = np.concatenate((-positive_nodes[::-1], positive_nodes))
    interval_weights = np.concatenate((positive_weights[::-1], positive_weights))
    weights_1d = interval_weights * epanechnikov_difference_density(nodes_1d)
    weights_1d /= np.sum(weights_1d)

    if dimension == 1:
        return nodes_1d[:, None], weights_1d
    first, second = np.meshgrid(nodes_1d, nodes_1d, indexing="ij")
    first_weight, second_weight = np.meshgrid(weights_1d, weights_1d, indexing="ij")
    nodes = np.column_stack((first.ravel(), second.ravel()))
    weights = (first_weight * second_weight).ravel()
    weights /= np.sum(weights)
    return nodes, weights


def epanechnikov_difference_radial_moment(
    power: float,
    dimension: int,
    order: int = 96,
) -> float:
    """Return ``E[||U-V||**power]`` for the product Epanechnikov kernel."""
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be positive and finite")
    nodes, weights = product_epanechnikov_difference_quadrature(dimension, order)
    return float(weights @ np.linalg.norm(nodes, axis=1) ** power)


def _kernel_geometry(
    dimension: int,
    kernel_transform: np.ndarray | None,
    lag_direction: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if dimension not in (1, 2):
        raise ValueError("dimension must be one or two")
    if kernel_transform is None:
        transform = np.eye(dimension)
    else:
        transform = np.asarray(kernel_transform, dtype=float)
        if transform.shape != (dimension, dimension):
            raise ValueError("kernel_transform must have shape (dimension, dimension)")
        if not np.all(np.isfinite(transform)):
            raise ValueError("kernel_transform must contain only finite values")
        if np.linalg.matrix_rank(transform) != dimension:
            raise ValueError("kernel_transform must be nonsingular")
    if lag_direction is None:
        direction = np.zeros(dimension, dtype=float)
        direction[0] = 1.0
    else:
        direction = np.asarray(lag_direction, dtype=float)
        if direction.shape != (dimension,) or not np.all(np.isfinite(direction)):
            raise ValueError("lag_direction must be a finite vector of length dimension")
        norm = float(np.linalg.norm(direction))
        if not np.isclose(norm, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("lag_direction must be a unit vector")
    return transform, direction


def transformed_epanechnikov_difference_radial_moment(
    power: float,
    dimension: int,
    *,
    kernel_transform: np.ndarray,
    order: int = 96,
) -> float:
    """Return ``E[||A(U-V)||**power]`` for a transformed product kernel."""
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be positive and finite")
    transform, _ = _kernel_geometry(dimension, kernel_transform, None)
    nodes, weights = product_epanechnikov_difference_quadrature(dimension, order)
    transformed = nodes @ transform.T
    return float(weights @ np.linalg.norm(transformed, axis=1) ** power)


def product_epanechnikov_decay_shift_coefficient(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    lag: float,
    quadrature_order: int = 96,
    kernel_transform: np.ndarray | None = None,
    lag_direction: np.ndarray | None = None,
) -> float:
    """Return the positive leading coefficient of decay minus pseudo-decay.

    The associated bandwidth scale is h to 2 nu for nu below one,
    h squared times log(1/h) at nu equal to one, and h squared above one.
    The product Epanechnikov kernel has coordinate variance one fifth.
    """
    if dimension not in (1, 2):
        raise ValueError("dimension must be one or two")
    if smoothness <= 0 or decay <= 0 or lag <= 0:
        raise ValueError("smoothness, decay, and lag must be positive")
    transform, direction = _kernel_geometry(
        dimension,
        kernel_transform,
        lag_direction,
    )
    z = decay * lag
    if smoothness < 1.0:
        c_nu = gamma(1.0 - smoothness) / (
            smoothness * 2.0 ** (2.0 * smoothness) * gamma(smoothness)
        )
        if kernel_transform is None:
            radial_moment = epanechnikov_difference_radial_moment(
                2.0 * smoothness,
                dimension,
                quadrature_order,
            )
        else:
            radial_moment = transformed_epanechnikov_difference_radial_moment(
                2.0 * smoothness,
                dimension,
                kernel_transform=transform,
                order=quadrature_order,
            )
        coefficient = (
            c_nu
            * radial_moment
            * decay ** (2.0 * smoothness)
            / lag
            * kve(smoothness, z)
            / kve(smoothness - 1.0, z)
        )
    elif smoothness == 1.0:
        kernel_covariance = transform @ transform.T / 5.0
        m2 = 2.0 * float(np.trace(kernel_covariance))
        coefficient = (
            m2
            * decay**2
            / (2.0 * lag)
            * kve(1.0, z)
            / kve(0.0, z)
        )
    else:
        kernel_covariance = transform @ transform.T / 5.0
        directional_variance = float(direction @ kernel_covariance @ direction)
        total_variance = float(np.trace(kernel_covariance))
        coefficient = (
            decay**2
            * ((2.0 * smoothness - 2.0) * directional_variance + total_variance)
            / (lag * (2.0 * smoothness - 2.0))
            * kve(smoothness - 2.0, z)
            / kve(smoothness - 1.0, z)
        )
    coefficient = float(coefficient)
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("the leading decay-shift coefficient must be positive")
    return coefficient


def product_epanechnikov_direction_contrast_coefficient(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    lag: float,
    kernel_transform: np.ndarray,
    first_direction: np.ndarray,
    second_direction: np.ndarray,
) -> float:
    r"""Return the signed coefficient of the directional decay-shift contrast.

    If ``Delta_e(h) = decay - decay_pseudo(e, h)``, the returned value ``D``
    satisfies

    ``Delta_first(h) - Delta_second(h) = D * h**2 + o(h**2)``.

    This contrast is second order for every positive Matérn smoothness, even
    when the common leading shift is of fractional order or includes a
    logarithmic factor.
    """
    if smoothness <= 0 or decay <= 0 or lag <= 0:
        raise ValueError("smoothness, decay, and lag must be positive")
    transform, first = _kernel_geometry(
        dimension,
        kernel_transform,
        first_direction,
    )
    _, second = _kernel_geometry(
        dimension,
        transform,
        second_direction,
    )
    kernel_covariance = transform @ transform.T / 5.0
    variance_difference = float(
        first @ kernel_covariance @ first - second @ kernel_covariance @ second
    )
    z = decay * lag
    coefficient = (
        decay**2
        * variance_difference
        / lag
        * kve(smoothness - 2.0, z)
        / kve(smoothness - 1.0, z)
    )
    if not np.isfinite(coefficient):
        raise ValueError("the directional contrast coefficient must be finite")
    return float(coefficient)


def transition_aware_matern_pair_approximation(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    bandwidth: float,
    lag: float,
    quadrature_order: int = 96,
    kernel_transform: np.ndarray | None = None,
    lag_direction: np.ndarray | None = None,
) -> TransitionPairApproximation:
    r"""Approximate the pair target with both origin terms around ``nu=1``.

    For noninteger smoothness between zero and two, the Matérn origin series
    contains an analytic quadratic term and a fractional-power term. Each has a
    coefficient that diverges as smoothness approaches one, but their sum has a
    finite logarithmic limit. Retaining both terms removes the slow transition
    layer that makes either one-term phase approximation inaccurate at finite
    bandwidth.

    The fixed-lag covariance retains its quadratic Taylor term. For fixed
    smoothness below one the resulting pseudo-decay error is order
    ``h**(2*nu+2)``; at one it is order ``h**4 * abs(log(h))``; and between one
    and two it is order ``h**4``.
    """
    if not 0.0 < smoothness < 2.0:
        raise ValueError("smoothness must lie strictly between zero and two")
    if decay <= 0 or bandwidth < 0 or lag <= 0:
        raise ValueError("decay and lag must be positive; bandwidth must be nonnegative")
    transform, direction = _kernel_geometry(
        dimension,
        kernel_transform,
        lag_direction,
    )
    nodes, weights = product_epanechnikov_difference_quadrature(
        dimension,
        quadrature_order,
    )
    transformed_nodes = nodes @ transform.T
    radii = np.linalg.norm(transformed_nodes, axis=1)
    kernel_covariance = transform @ transform.T / 5.0
    m2 = 2.0 * float(np.trace(kernel_covariance))

    z = decay * lag
    base_correlation = float(
        matern_correlation(np.asarray(lag), decay, smoothness)
    )
    if bandwidth == 0:
        return TransitionPairApproximation(
            dimension=dimension,
            smoothness=float(smoothness),
            true_decay=float(decay),
            pseudo_decay=float(decay),
            bandwidth=0.0,
            lag=float(lag),
            variance_factor=1.0,
            covariance_factor=base_correlation,
            correlation=base_correlation,
            quadrature_order=int(quadrature_order),
        )

    scaled_bandwidth = decay * bandwidth
    if smoothness == 1.0:
        squared_log_moment = float(
            weights @ (np.square(radii) * np.log(radii))
        )
        variance_shift = 0.5 * scaled_bandwidth**2 * (
            m2 * (np.log(scaled_bandwidth / 2.0) + np.euler_gamma - 0.5)
            + squared_log_moment
        )
    else:
        fractional_moment = float(weights @ radii ** (2.0 * smoothness))
        fractional_coefficient = gamma(-smoothness) / (
            2.0 ** (2.0 * smoothness) * gamma(smoothness)
        )
        variance_shift = (
            m2 * scaled_bandwidth**2 / (4.0 * (1.0 - smoothness))
            + fractional_coefficient
            * fractional_moment
            * scaled_bandwidth ** (2.0 * smoothness)
        )
    variance_factor = 1.0 + float(variance_shift)

    derivative = -base_correlation * (
        kve(smoothness - 1.0, z) / kve(smoothness, z)
    )
    second_derivative = base_correlation + (2.0 * smoothness - 1.0) * derivative / z
    directional_variance = float(direction @ kernel_covariance @ direction)
    total_variance = float(np.trace(kernel_covariance))
    fixed_lag_coefficient = decay**2 * (
        directional_variance * second_derivative
        + (total_variance - directional_variance) * derivative / z
    )
    covariance_factor = base_correlation + bandwidth**2 * fixed_lag_coefficient
    correlation = covariance_factor / variance_factor
    if not 0.0 < correlation < 1.0:
        raise ValueError(
            "the transition-aware correlation must lie strictly between zero and one"
        )

    def root(candidate_decay: float) -> float:
        candidate_correlation = matern_correlation(
            np.asarray(lag),
            candidate_decay,
            smoothness,
        )
        return float(candidate_correlation) - correlation

    lower = np.finfo(float).eps / lag
    upper = max(100.0 / lag, 100.0 * decay)
    pseudo_decay = float(brentq(root, lower, upper, xtol=1e-13, rtol=1e-13))
    return TransitionPairApproximation(
        dimension=dimension,
        smoothness=float(smoothness),
        true_decay=float(decay),
        pseudo_decay=pseudo_decay,
        bandwidth=float(bandwidth),
        lag=float(lag),
        variance_factor=variance_factor,
        covariance_factor=float(covariance_factor),
        correlation=float(correlation),
        quadrature_order=int(quadrature_order),
    )


def continuous_matern_pair_target(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    bandwidth: float,
    lag: float,
    quadrature_order: int = 64,
    kernel_transform: np.ndarray | None = None,
    lag_direction: np.ndarray | None = None,
) -> ContinuousPairTarget:
    """Compute the exact quadrature oracle and its naive point-pair target.

    The pair is separated by ``lag`` along ``lag_direction`` (the first
    coordinate by default).  The naive model has the correct known Matérn
    smoothness but ignores the observation support, fitting a marginal variance
    and decay to the smoothed pair.  ``kernel_transform`` maps the standardized
    product Epanechnikov support into a compact anisotropic support.
    """
    if smoothness <= 0 or decay <= 0 or bandwidth < 0 or lag <= 0:
        raise ValueError(
            "smoothness, decay, and lag must be positive; bandwidth must be nonnegative"
        )
    transform, direction = _kernel_geometry(
        dimension,
        kernel_transform,
        lag_direction,
    )
    nodes, weights = product_epanechnikov_difference_quadrature(
        dimension, quadrature_order
    )
    transformed_nodes = nodes @ transform.T
    if bandwidth == 0:
        variance_factor = 1.0
        covariance_factor = float(matern_correlation(np.asarray(lag), decay, smoothness))
        pseudo_decay = decay
    else:
        variance_distances = bandwidth * np.linalg.norm(transformed_nodes, axis=1)
        variance_factor = float(
            weights @ matern_correlation(variance_distances, decay, smoothness)
        )
        displacement = bandwidth * transformed_nodes
        displacement += lag * direction
        pair_distances = np.linalg.norm(displacement, axis=1)
        covariance_factor = float(
            weights @ matern_correlation(pair_distances, decay, smoothness)
        )
        correlation = covariance_factor / variance_factor
        if not 0.0 < correlation < 1.0:
            raise ValueError("the smoothed pair correlation must lie strictly between zero and one")

        def root(candidate_decay: float) -> float:
            candidate_correlation = matern_correlation(
                np.asarray(lag),
                candidate_decay,
                smoothness,
            )
            return float(candidate_correlation) - correlation

        lower = np.finfo(float).eps / lag
        upper = max(100.0 / lag, 100.0 * decay)
        pseudo_decay = float(brentq(root, lower, upper, xtol=1e-13, rtol=1e-13))

    correlation = covariance_factor / variance_factor
    return ContinuousPairTarget(
        dimension=dimension,
        smoothness=float(smoothness),
        true_decay=float(decay),
        pseudo_decay=float(pseudo_decay),
        bandwidth=float(bandwidth),
        lag=float(lag),
        variance_factor=float(variance_factor),
        covariance_factor=float(covariance_factor),
        correlation=float(correlation),
        quadrature_order=int(quadrature_order),
    )
