"""Deterministic quadrature for continuously smoothed Matérn fields.

The product Epanechnikov smoother has independent coordinates.  If ``U`` and
``V`` are independent draws from its standardized kernel, then every smoothed
covariance is an expectation over ``D = U - V``.  Tensor Gauss--Legendre
quadrature therefore gives a high-accuracy, simulation-free oracle in one to
three spatial dimensions. Product Epanechnikov and product uniform kernels
provide two compact symmetric support families. A nonsingular linear kernel
transform permits compact anisotropic supports, and the fitted lag may point in
any declared direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gamma, kve

from HighDimSpatial.smoothing_bias.kl import (
    gaussian_kl_divergence,
    matern_correlation,
)
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
    kernel_family: str


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
    kernel_family: str


@dataclass(frozen=True)
class PairCompositeAsymptotics:
    """Leading multi-lag pair-composite target and misspecification constants."""

    smoothness: float
    true_decay: float
    lags: tuple[float, ...]
    weights: tuple[float, ...]
    pair_shift_coefficients: tuple[float, ...]
    information_weights: tuple[float, ...]
    decay_shift_coefficient: float
    minimum_kl_coefficient: float


@dataclass(frozen=True)
class ContinuousMultiLagTarget:
    """Exact quadrature target of a Gaussian correlation pair composite."""

    dimension: int
    smoothness: float
    true_decay: float
    pseudo_decay: float
    bandwidth: float
    lags: tuple[float, ...]
    weights: tuple[float, ...]
    pair_correlations: tuple[float, ...]
    pair_pseudo_decays: tuple[float, ...]
    minimum_kl: float
    quadrature_order: int
    kernel_family: str


@dataclass(frozen=True)
class FiniteDesignProjectionAsymptotics:
    """Local full-likelihood projection of the support perturbation."""

    smoothness: float
    true_variance: float
    true_decay: float
    dimension: int
    number_of_locations: int
    log_variance_shift_coefficient: float
    log_decay_shift_coefficient: float
    decay_inflation_coefficient: float
    minimum_kl_coefficient: float
    information_condition_number: float


@dataclass(frozen=True)
class ContinuousFullLikelihoodTarget:
    """Exact finite-design Gaussian KL target under continuous smoothing."""

    smoothness: float
    true_variance: float
    true_decay: float
    pseudo_variance: float
    pseudo_decay: float
    bandwidth: float
    dimension: int
    number_of_locations: int
    minimum_kl: float
    quadrature_order: int
    kernel_family: str


PRODUCT_KERNEL_VARIANCES = {
    "epanechnikov": 1.0 / 5.0,
    "uniform": 1.0 / 3.0,
}


def _validate_product_kernel(kernel_family: str) -> str:
    if kernel_family not in PRODUCT_KERNEL_VARIANCES:
        choices = ", ".join(sorted(PRODUCT_KERNEL_VARIANCES))
        raise ValueError(f"kernel_family must be one of: {choices}")
    return kernel_family


def _product_difference_quadrature(
    dimension: int,
    order: int,
    kernel_family: str,
) -> tuple[np.ndarray, np.ndarray]:
    if dimension not in (1, 2, 3):
        raise ValueError("dimension must be one, two, or three")
    if isinstance(order, bool) or int(order) != order or order < 8 or order % 2:
        raise ValueError("order must be an even integer of at least eight")
    kernel_family = _validate_product_kernel(kernel_family)
    order = int(order)

    # Split at zero because both difference densities and several
    # theorem-facing integrands contain absolute values.
    positive_nodes, positive_weights = leggauss(order // 2)
    positive_nodes = positive_nodes + 1.0  # map [-1, 1] to [0, 2]
    nodes_1d = np.concatenate((-positive_nodes[::-1], positive_nodes))
    interval_weights = np.concatenate((positive_weights[::-1], positive_weights))
    if kernel_family == "epanechnikov":
        density = epanechnikov_difference_density(nodes_1d)
    else:
        density = (2.0 - np.abs(nodes_1d)) / 4.0
    weights_1d = interval_weights * density
    weights_1d /= np.sum(weights_1d)

    if dimension == 1:
        return nodes_1d[:, None], weights_1d
    node_mesh = np.meshgrid(*([nodes_1d] * dimension), indexing="ij")
    nodes = np.column_stack([grid.ravel() for grid in node_mesh])
    weight_grid = np.ones((order,) * dimension)
    for axis in range(dimension):
        shape = [1] * dimension
        shape[axis] = order
        weight_grid *= weights_1d.reshape(shape)
    weights = weight_grid.ravel()
    weights /= np.sum(weights)
    return nodes, weights


def product_epanechnikov_difference_quadrature(
    dimension: int,
    order: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return product-Epanechnikov difference nodes and probability weights."""
    return _product_difference_quadrature(dimension, order, "epanechnikov")


def product_uniform_difference_quadrature(
    dimension: int,
    order: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return product-uniform difference nodes and probability weights."""
    return _product_difference_quadrature(dimension, order, "uniform")


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


def uniform_difference_radial_moment(
    power: float,
    dimension: int,
    order: int = 96,
) -> float:
    """Return the radial moment for a product-uniform difference."""
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be positive and finite")
    nodes, weights = product_uniform_difference_quadrature(dimension, order)
    return float(weights @ np.linalg.norm(nodes, axis=1) ** power)


def _kernel_geometry(
    dimension: int,
    kernel_transform: np.ndarray | None,
    lag_direction: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if dimension not in (1, 2, 3):
        raise ValueError("dimension must be one, two, or three")
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


def _transformed_product_difference_radial_moment(
    power: float,
    dimension: int,
    *,
    kernel_transform: np.ndarray,
    kernel_family: str,
    order: int = 96,
) -> float:
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be positive and finite")
    transform, _ = _kernel_geometry(dimension, kernel_transform, None)
    nodes, weights = _product_difference_quadrature(
        dimension,
        order,
        kernel_family,
    )
    transformed = nodes @ transform.T
    return float(weights @ np.linalg.norm(transformed, axis=1) ** power)


def product_kernel_decay_shift_coefficient(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    lag: float,
    kernel_family: str,
    quadrature_order: int = 96,
    kernel_transform: np.ndarray | None = None,
    lag_direction: np.ndarray | None = None,
) -> float:
    """Return the positive leading coefficient of decay minus pseudo-decay.

    The associated bandwidth scale is h to 2 nu for nu below one,
    h squared times log(1/h) at nu equal to one, and h squared above one.
    The kernel covariance and rough-regime radial moment are evaluated for the
    declared compact product family.
    """
    if smoothness <= 0 or decay <= 0 or lag <= 0:
        raise ValueError("smoothness, decay, and lag must be positive")
    kernel_family = _validate_product_kernel(kernel_family)
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
            nodes, weights = _product_difference_quadrature(
                dimension,
                quadrature_order,
                kernel_family,
            )
            radial_moment = float(
                weights @ np.linalg.norm(nodes, axis=1) ** (2.0 * smoothness)
            )
        else:
            radial_moment = _transformed_product_difference_radial_moment(
                2.0 * smoothness,
                dimension,
                kernel_transform=transform,
                kernel_family=kernel_family,
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
        kernel_covariance = PRODUCT_KERNEL_VARIANCES[kernel_family] * (
            transform @ transform.T
        )
        m2 = 2.0 * float(np.trace(kernel_covariance))
        coefficient = (
            m2
            * decay**2
            / (2.0 * lag)
            * kve(1.0, z)
            / kve(0.0, z)
        )
    else:
        kernel_covariance = PRODUCT_KERNEL_VARIANCES[kernel_family] * (
            transform @ transform.T
        )
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
    """Return the product-Epanechnikov leading decay-shift coefficient."""
    return product_kernel_decay_shift_coefficient(
        dimension=dimension,
        smoothness=smoothness,
        decay=decay,
        lag=lag,
        kernel_family="epanechnikov",
        quadrature_order=quadrature_order,
        kernel_transform=kernel_transform,
        lag_direction=lag_direction,
    )


def _positive_lags_and_weights(
    lags: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    lag_values = np.asarray(lags, dtype=float)
    if (
        lag_values.ndim != 1
        or lag_values.size == 0
        or not np.all(np.isfinite(lag_values))
        or np.any(lag_values <= 0.0)
    ):
        raise ValueError("lags must be a nonempty vector of positive finite values")
    if weights is None:
        weight_values = np.ones(lag_values.size, dtype=float)
    else:
        weight_values = np.asarray(weights, dtype=float)
        if (
            weight_values.shape != lag_values.shape
            or not np.all(np.isfinite(weight_values))
            or np.any(weight_values <= 0.0)
        ):
            raise ValueError("weights must be positive, finite, and match lags")
    weight_values /= np.sum(weight_values)
    return lag_values, weight_values


def gaussian_correlation_kl(
    true_correlation: np.ndarray | float,
    candidate_correlation: np.ndarray | float,
) -> np.ndarray | float:
    """KL divergence between centered unit-variance Gaussian pairs.

    Both arguments are correlations in ``(-1, 1)``. Array inputs broadcast in
    the usual NumPy manner.
    """
    truth = np.asarray(true_correlation, dtype=float)
    candidate = np.asarray(candidate_correlation, dtype=float)
    if (
        not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(candidate))
        or np.any(np.abs(truth) >= 1.0)
        or np.any(np.abs(candidate) >= 1.0)
    ):
        raise ValueError("pair correlations must be finite and strictly between -1 and 1")
    value = 0.5 * (
        np.log1p(-np.square(candidate))
        - np.log1p(-np.square(truth))
        - 2.0
        + 2.0 * (1.0 - candidate * truth) / (1.0 - np.square(candidate))
    )
    value = np.maximum(value, 0.0)
    return float(value) if value.ndim == 0 else value


def product_kernel_multilag_asymptotics(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    lags: np.ndarray,
    weights: np.ndarray | None = None,
    kernel_family: str = "epanechnikov",
    quadrature_order: int = 96,
    kernel_transform: np.ndarray | None = None,
    lag_direction: np.ndarray | None = None,
) -> PairCompositeAsymptotics:
    r"""Return the multi-lag shift and irreducible-KL coefficients.

    Let ``s_nu(h)`` equal ``h**(2*nu)`` below one,
    ``h**2*log(1/h)`` at one, and ``h**2`` above one. For a weighted sum of
    Gaussian KL divergences between smoothed and point-support correlations,
    the population target satisfies

    ``decay - pseudo_decay = C_bar * s_nu(h) + o(s_nu(h))``.

    The minimum composite KL is

    ``V_bar * s_nu(h)**2 + o(s_nu(h)**2)``.

    ``C_bar`` is a positive information-weighted average of the single-lag
    coefficients. ``V_bar`` is half their information-weighted dispersion and
    is positive exactly when the leading single-lag coefficients are not all
    equal.
    """
    if smoothness <= 0.0 or decay <= 0.0:
        raise ValueError("smoothness and decay must be positive")
    lag_values, weight_values = _positive_lags_and_weights(lags, weights)
    base_correlations = np.asarray(
        matern_correlation(lag_values, decay, smoothness),
        dtype=float,
    )
    scaled_lags = decay * lag_values
    decay_derivatives = -lag_values * base_correlations * np.asarray(
        kve(smoothness - 1.0, scaled_lags) / kve(smoothness, scaled_lags),
        dtype=float,
    )
    correlation_information = (1.0 + np.square(base_correlations)) / np.square(
        1.0 - np.square(base_correlations)
    )
    information_weights = weight_values * correlation_information * np.square(
        decay_derivatives
    )
    pair_coefficients = np.asarray(
        [
            product_kernel_decay_shift_coefficient(
                dimension=dimension,
                smoothness=smoothness,
                decay=decay,
                lag=float(lag),
                kernel_family=kernel_family,
                quadrature_order=quadrature_order,
                kernel_transform=kernel_transform,
                lag_direction=lag_direction,
            )
            for lag in lag_values
        ],
        dtype=float,
    )
    total_information = float(np.sum(information_weights))
    if not np.isfinite(total_information) or total_information <= 0.0:
        raise ValueError("the declared lag set has zero or invalid decay information")
    shift_coefficient = float(
        information_weights @ pair_coefficients / total_information
    )
    minimum_kl_coefficient = float(
        0.5
        * np.sum(
            information_weights * np.square(pair_coefficients - shift_coefficient)
        )
    )
    return PairCompositeAsymptotics(
        smoothness=float(smoothness),
        true_decay=float(decay),
        lags=tuple(float(value) for value in lag_values),
        weights=tuple(float(value) for value in weight_values),
        pair_shift_coefficients=tuple(float(value) for value in pair_coefficients),
        information_weights=tuple(float(value) for value in information_weights),
        decay_shift_coefficient=shift_coefficient,
        minimum_kl_coefficient=minimum_kl_coefficient,
    )


def continuous_matern_multilag_target(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    bandwidth: float,
    lags: np.ndarray,
    weights: np.ndarray | None = None,
    kernel_family: str = "epanechnikov",
    quadrature_order: int = 64,
    kernel_transform: np.ndarray | None = None,
    lag_direction: np.ndarray | None = None,
) -> ContinuousMultiLagTarget:
    """Compute the exact Gaussian correlation pair-composite target.

    Unlike the saturated single-lag fit, two or more lags generally cannot all
    be represented by one point-support decay after smoothing. The returned
    minimum KL therefore measures genuine population misspecification.
    """
    if smoothness <= 0.0 or decay <= 0.0 or bandwidth < 0.0:
        raise ValueError(
            "smoothness and decay must be positive; bandwidth must be nonnegative"
        )
    lag_values, weight_values = _positive_lags_and_weights(lags, weights)
    pair_targets = [
        continuous_matern_pair_target(
            dimension=dimension,
            smoothness=smoothness,
            decay=decay,
            bandwidth=bandwidth,
            lag=float(lag),
            kernel_family=kernel_family,
            quadrature_order=quadrature_order,
            kernel_transform=kernel_transform,
            lag_direction=lag_direction,
        )
        for lag in lag_values
    ]
    true_correlations = np.asarray(
        [target.correlation for target in pair_targets],
        dtype=float,
    )

    def objective(log_decay: float) -> float:
        candidate_decay = float(np.exp(log_decay))
        candidate_correlations = np.asarray(
            matern_correlation(lag_values, candidate_decay, smoothness),
            dtype=float,
        )
        divergences = np.asarray(
            gaussian_correlation_kl(true_correlations, candidate_correlations),
            dtype=float,
        )
        return float(weight_values @ divergences)

    log_bounds = (np.log(decay) - np.log(100.0), np.log(decay) + np.log(100.0))
    result = minimize_scalar(
        objective,
        bounds=log_bounds,
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"multi-lag population optimization failed: {result.message}")
    pseudo_decay = float(np.exp(result.x))
    return ContinuousMultiLagTarget(
        dimension=dimension,
        smoothness=float(smoothness),
        true_decay=float(decay),
        pseudo_decay=pseudo_decay,
        bandwidth=float(bandwidth),
        lags=tuple(float(value) for value in lag_values),
        weights=tuple(float(value) for value in weight_values),
        pair_correlations=tuple(float(value) for value in true_correlations),
        pair_pseudo_decays=tuple(target.pseudo_decay for target in pair_targets),
        minimum_kl=objective(float(result.x)),
        quadrature_order=int(quadrature_order),
        kernel_family=kernel_family,
    )


def _finite_locations(locations: np.ndarray) -> np.ndarray:
    values = np.asarray(locations, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] not in (1, 2, 3)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("locations must have shape (p, d), with p >= 2 and d in {1,2,3}")
    if np.unique(values, axis=0).shape[0] != values.shape[0]:
        raise ValueError("locations must be distinct")
    return values


def continuous_matern_covariance_matrix(
    locations: np.ndarray,
    *,
    variance: float,
    decay: float,
    smoothness: float,
    bandwidth: float,
    kernel_family: str = "epanechnikov",
    quadrature_order: int = 64,
    kernel_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Return the continuously smoothed covariance on a finite design."""
    points = _finite_locations(locations)
    if variance <= 0.0 or decay <= 0.0 or smoothness <= 0.0 or bandwidth < 0.0:
        raise ValueError(
            "variance, decay, and smoothness must be positive; bandwidth must be nonnegative"
        )
    kernel_family = _validate_product_kernel(kernel_family)
    transform, _ = _kernel_geometry(points.shape[1], kernel_transform, None)
    differences = points[:, None, :] - points[None, :, :]
    if bandwidth == 0.0:
        distances = np.linalg.norm(differences, axis=2)
        return variance * np.asarray(
            matern_correlation(distances, decay, smoothness), dtype=float
        )
    nodes, weights = _product_difference_quadrature(
        points.shape[1], quadrature_order, kernel_family
    )
    transformed_nodes = nodes @ transform.T
    covariance = np.zeros((points.shape[0], points.shape[0]), dtype=float)
    chunk_size = max(1, min(512, transformed_nodes.shape[0]))
    for start in range(0, transformed_nodes.shape[0], chunk_size):
        stop = min(start + chunk_size, transformed_nodes.shape[0])
        displaced = differences[None, :, :, :] + (
            bandwidth * transformed_nodes[start:stop, None, None, :]
        )
        distances = np.linalg.norm(displaced, axis=3)
        correlations = np.asarray(
            matern_correlation(distances, decay, smoothness), dtype=float
        )
        covariance += np.einsum(
            "q,qij->ij", weights[start:stop], correlations, optimize=True
        )
    covariance *= variance
    return (covariance + covariance.T) / 2.0


def matern_support_covariance_leading_matrix(
    locations: np.ndarray,
    *,
    variance: float,
    decay: float,
    smoothness: float,
    kernel_family: str = "epanechnikov",
    quadrature_order: int = 96,
    kernel_transform: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return ``Gamma`` in ``Sigma_h = Sigma_0 + s_nu(h) Gamma + o(s_nu)``."""
    points = _finite_locations(locations)
    if variance <= 0.0 or decay <= 0.0 or smoothness <= 0.0:
        raise ValueError("variance, decay, and smoothness must be positive")
    kernel_family = _validate_product_kernel(kernel_family)
    transform, _ = _kernel_geometry(points.shape[1], kernel_transform, None)
    kernel_covariance = PRODUCT_KERNEL_VARIANCES[kernel_family] * (
        transform @ transform.T
    )
    m2 = 2.0 * float(np.trace(kernel_covariance))
    perturbation = np.zeros((points.shape[0], points.shape[0]), dtype=float)
    if smoothness < 1.0:
        c_nu = gamma(1.0 - smoothness) / (
            smoothness * 2.0 ** (2.0 * smoothness) * gamma(smoothness)
        )
        nodes, weights = _product_difference_quadrature(
            points.shape[1], quadrature_order, kernel_family
        )
        transformed_nodes = nodes @ transform.T
        radial_moment = float(
            weights @ np.linalg.norm(transformed_nodes, axis=1) ** (2.0 * smoothness)
        )
        diagonal = -c_nu * radial_moment * decay ** (2.0 * smoothness)
    elif smoothness == 1.0:
        diagonal = -0.5 * m2 * decay**2
    else:
        diagonal = -m2 * decay**2 / (4.0 * (smoothness - 1.0))
        differences = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        off_diagonal = ~np.eye(points.shape[0], dtype=bool)
        directions = np.zeros_like(differences)
        directions[off_diagonal] = (
            differences[off_diagonal] / distances[off_diagonal, None]
        )
        z = decay * distances[off_diagonal]
        base = np.asarray(
            matern_correlation(distances[off_diagonal], decay, smoothness),
            dtype=float,
        )
        first_derivative = -base * np.asarray(
            kve(smoothness - 1.0, z) / kve(smoothness, z), dtype=float
        )
        second_derivative = base + (2.0 * smoothness - 1.0) * first_derivative / z
        directional_variances = np.einsum(
            "qi,ij,qj->q",
            directions[off_diagonal],
            kernel_covariance,
            directions[off_diagonal],
            optimize=True,
        )
        total_variance = float(np.trace(kernel_covariance))
        perturbation[off_diagonal] = decay**2 * (
            directional_variances * second_derivative
            + (total_variance - directional_variances) * first_derivative / z
        )
    np.fill_diagonal(perturbation, diagonal)
    perturbation *= variance
    return (perturbation + perturbation.T) / 2.0


def finite_design_full_likelihood_asymptotics(
    locations: np.ndarray,
    *,
    variance: float,
    decay: float,
    smoothness: float,
    kernel_family: str = "epanechnikov",
    quadrature_order: int = 96,
    kernel_transform: np.ndarray | None = None,
) -> FiniteDesignProjectionAsymptotics:
    r"""Project the leading support perturbation onto variance--decay scores.

    Parameters are ``(log variance, log decay)``. The returned vector ``b``
    satisfies ``theta_h - theta_0 = b s_nu(h) + o(s_nu(h))``. The minimum KL
    coefficient is the squared Fisher-metric norm of the component of the
    support perturbation orthogonal to the two-parameter Matérn tangent space.
    """
    points = _finite_locations(locations)
    truth = continuous_matern_covariance_matrix(
        points,
        variance=variance,
        decay=decay,
        smoothness=smoothness,
        bandwidth=0.0,
        kernel_family=kernel_family,
        quadrature_order=quadrature_order,
        kernel_transform=kernel_transform,
    )
    precision = np.linalg.inv(truth)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    base_correlation = np.asarray(
        matern_correlation(distances, decay, smoothness), dtype=float
    )
    decay_derivative = np.zeros_like(base_correlation)
    off_diagonal = distances > 0.0
    z = decay * distances[off_diagonal]
    decay_derivative[off_diagonal] = -variance * z * base_correlation[
        off_diagonal
    ] * np.asarray(kve(smoothness - 1.0, z) / kve(smoothness, z), dtype=float)
    derivatives = (truth, decay_derivative)
    perturbation = matern_support_covariance_leading_matrix(
        points,
        variance=variance,
        decay=decay,
        smoothness=smoothness,
        kernel_family=kernel_family,
        quadrature_order=quadrature_order,
        kernel_transform=kernel_transform,
    )
    information = np.asarray(
        [
            [
                0.5 * np.trace(precision @ first @ precision @ second)
                for second in derivatives
            ]
            for first in derivatives
        ],
        dtype=float,
    )
    forcing = np.asarray(
        [
            0.5 * np.trace(precision @ derivative @ precision @ perturbation)
            for derivative in derivatives
        ],
        dtype=float,
    )
    coefficients = np.linalg.solve(information, forcing)
    residual = perturbation - sum(
        coefficient * derivative
        for coefficient, derivative in zip(coefficients, derivatives)
    )
    minimum_kl_coefficient = float(
        0.25 * np.trace(precision @ residual @ precision @ residual)
    )
    if minimum_kl_coefficient < -1e-10:
        raise RuntimeError("the projected KL coefficient is numerically negative")
    minimum_kl_coefficient = max(minimum_kl_coefficient, 0.0)
    return FiniteDesignProjectionAsymptotics(
        smoothness=float(smoothness),
        true_variance=float(variance),
        true_decay=float(decay),
        dimension=points.shape[1],
        number_of_locations=points.shape[0],
        log_variance_shift_coefficient=float(coefficients[0]),
        log_decay_shift_coefficient=float(coefficients[1]),
        decay_inflation_coefficient=float(-decay * coefficients[1]),
        minimum_kl_coefficient=minimum_kl_coefficient,
        information_condition_number=float(np.linalg.cond(information)),
    )


def continuous_matern_full_likelihood_target(
    locations: np.ndarray,
    *,
    variance: float,
    decay: float,
    smoothness: float,
    bandwidth: float,
    kernel_family: str = "epanechnikov",
    quadrature_order: int = 64,
    kernel_transform: np.ndarray | None = None,
) -> ContinuousFullLikelihoodTarget:
    """Compute the exact finite-design variance--decay Gaussian KL target."""
    points = _finite_locations(locations)
    truth = continuous_matern_covariance_matrix(
        points,
        variance=variance,
        decay=decay,
        smoothness=smoothness,
        bandwidth=bandwidth,
        kernel_family=kernel_family,
        quadrature_order=quadrature_order,
        kernel_transform=kernel_transform,
    )
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    dimension = points.shape[0]

    def profile(log_decay: float) -> tuple[float, float, np.ndarray]:
        candidate_decay = float(np.exp(log_decay))
        base = np.asarray(
            matern_correlation(distances, candidate_decay, smoothness), dtype=float
        )
        inverse_truth_trace = float(np.trace(np.linalg.solve(base, truth)))
        candidate_variance = inverse_truth_trace / dimension
        candidate = candidate_variance * base
        divergence = gaussian_kl_divergence(truth, candidate)
        return float(divergence), float(candidate_variance), candidate

    log_bounds = (np.log(decay) - np.log(100.0), np.log(decay) + np.log(100.0))
    result = minimize_scalar(
        lambda value: profile(float(value))[0],
        bounds=log_bounds,
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"full-likelihood population optimization failed: {result.message}")
    minimum_kl, pseudo_variance, _ = profile(float(result.x))
    return ContinuousFullLikelihoodTarget(
        smoothness=float(smoothness),
        true_variance=float(variance),
        true_decay=float(decay),
        pseudo_variance=pseudo_variance,
        pseudo_decay=float(np.exp(result.x)),
        bandwidth=float(bandwidth),
        dimension=points.shape[1],
        number_of_locations=points.shape[0],
        minimum_kl=minimum_kl,
        quadrature_order=int(quadrature_order),
        kernel_family=kernel_family,
    )


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
    kernel_family: str = "epanechnikov",
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
    kernel_family = _validate_product_kernel(kernel_family)
    transform, direction = _kernel_geometry(
        dimension,
        kernel_transform,
        lag_direction,
    )
    nodes, weights = _product_difference_quadrature(
        dimension,
        quadrature_order,
        kernel_family,
    )
    transformed_nodes = nodes @ transform.T
    radii = np.linalg.norm(transformed_nodes, axis=1)
    kernel_covariance = PRODUCT_KERNEL_VARIANCES[kernel_family] * (
        transform @ transform.T
    )
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
            kernel_family=kernel_family,
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
        kernel_family=kernel_family,
    )


def continuous_matern_pair_target(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    bandwidth: float,
    lag: float,
    kernel_family: str = "epanechnikov",
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
    kernel_family = _validate_product_kernel(kernel_family)
    transform, direction = _kernel_geometry(
        dimension,
        kernel_transform,
        lag_direction,
    )
    nodes, weights = _product_difference_quadrature(
        dimension,
        quadrature_order,
        kernel_family,
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
        kernel_family=kernel_family,
    )
