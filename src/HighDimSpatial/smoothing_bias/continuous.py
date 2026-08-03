"""Deterministic quadrature for continuously smoothed Matérn fields.

The product Epanechnikov smoother has independent coordinates.  If ``U`` and
``V`` are independent draws from its standardized kernel, then every smoothed
covariance is an expectation over ``D = U - V``.  Tensor Gauss--Legendre
quadrature therefore gives a high-accuracy, simulation-free oracle in one and
two spatial dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import gamma, kv

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


def product_epanechnikov_decay_shift_coefficient(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    lag: float,
    quadrature_order: int = 96,
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
    z = decay * lag
    matern_value = float(matern_correlation(np.asarray(lag), decay, smoothness))
    normalizer = 2.0 ** (1.0 - smoothness) / gamma(smoothness)
    matern_derivative = -normalizer * z**smoothness * kv(smoothness - 1.0, z)
    if smoothness < 1.0:
        c_nu = gamma(1.0 - smoothness) / (
            smoothness * 2.0 ** (2.0 * smoothness) * gamma(smoothness)
        )
        radial_moment = epanechnikov_difference_radial_moment(
            2.0 * smoothness,
            dimension,
            quadrature_order,
        )
        signed_coefficient = (
            matern_value
            * c_nu
            * radial_moment
            * decay ** (2.0 * smoothness)
            / (lag * matern_derivative)
        )
    elif smoothness == 1.0:
        m2 = 2.0 * dimension / 5.0
        signed_coefficient = (
            m2
            * matern_value
            * decay**2
            / (2.0 * lag * matern_derivative)
        )
    else:
        kernel_coordinate_variance = 1.0 / 5.0
        g_nu = (
            normalizer
            * z**smoothness
            * kv(smoothness - 2.0, z)
            / (2.0 * smoothness - 2.0)
        )
        a_nu = (
            kernel_coordinate_variance
            * decay**2
            * (2.0 * smoothness + dimension - 2.0)
            * g_nu
        )
        signed_coefficient = a_nu / (lag * matern_derivative)
    coefficient = -float(signed_coefficient)
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("the leading decay-shift coefficient must be positive")
    return coefficient


def continuous_matern_pair_target(
    *,
    dimension: int,
    smoothness: float,
    decay: float,
    bandwidth: float,
    lag: float,
    quadrature_order: int = 64,
) -> ContinuousPairTarget:
    """Compute the exact quadrature oracle and its naive point-pair target.

    The pair is separated by ``lag`` along the first coordinate.  The naive
    model has the correct known Matérn smoothness but ignores the observation
    support, fitting a marginal variance and decay to the smoothed pair.
    """
    if smoothness <= 0 or decay <= 0 or bandwidth < 0 or lag <= 0:
        raise ValueError(
            "smoothness, decay, and lag must be positive; bandwidth must be nonnegative"
        )
    nodes, weights = product_epanechnikov_difference_quadrature(
        dimension, quadrature_order
    )
    if bandwidth == 0:
        variance_factor = 1.0
        covariance_factor = float(matern_correlation(np.asarray(lag), decay, smoothness))
        pseudo_decay = decay
    else:
        variance_distances = bandwidth * np.linalg.norm(nodes, axis=1)
        variance_factor = float(
            weights @ matern_correlation(variance_distances, decay, smoothness)
        )
        displacement = bandwidth * nodes
        displacement[:, 0] += lag
        pair_distances = np.linalg.norm(displacement, axis=1)
        covariance_factor = float(
            weights @ matern_correlation(pair_distances, decay, smoothness)
        )
        correlation = covariance_factor / variance_factor
        if not 0.0 < correlation < 1.0:
            raise ValueError("the smoothed pair correlation must lie strictly between zero and one")

        def root(candidate_decay: float) -> float:
            return float(matern_correlation(np.asarray(lag), candidate_decay, smoothness)) - correlation

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
