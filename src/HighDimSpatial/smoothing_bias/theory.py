"""Closed-form smoothing factors for exponential spatial covariance models.

The exponential covariance ``variance * exp(-decay * |r|)`` is the one-
dimensional Matérn model with smoothness ``nu=1/2`` under the repository's decay-
scale convention.  The formulas here concern continuous Epanechnikov smoothing
with known bandwidth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class ExponentialPseudoTarget:
    """Exact pairwise pseudo-target produced by ignoring smoothing."""

    variance: float
    decay: float
    correlation: float
    variance_factor: float
    far_lag_factor: float


@dataclass(frozen=True)
class SeparableExponentialPseudoTarget:
    """Axis-pair pseudo-target for a separable exponential field."""

    variance: float
    decays: np.ndarray
    correlations: np.ndarray
    variance_factors: np.ndarray
    far_lag_factors: np.ndarray


def epanechnikov_density(value: np.ndarray | float) -> np.ndarray | float:
    """The standard Epanechnikov probability density on ``[-1, 1]``."""
    values = np.asarray(value, dtype=float)
    result = 0.75 * (1.0 - values**2)
    result = np.where(np.abs(values) <= 1.0, result, 0.0)
    return float(result) if result.ndim == 0 else result


def epanechnikov_difference_density(value: np.ndarray | float) -> np.ndarray | float:
    """Density of ``U-V`` for independent standard Epanechnikov variables."""
    values = np.asarray(value, dtype=float)
    absolute = np.abs(values)
    result = 3.0 / 5.0 - 3.0 * absolute**2 / 4.0 + 3.0 * absolute**3 / 8.0
    result -= 3.0 * absolute**5 / 160.0
    result = np.where(absolute <= 2.0, result, 0.0)
    return float(result) if result.ndim == 0 else result


def _as_scalar_if_needed(original: np.ndarray | float, value: np.ndarray) -> np.ndarray | float:
    return float(value) if np.asarray(original).ndim == 0 else value


def epanechnikov_mgf(value: np.ndarray | float) -> np.ndarray | float:
    """Moment-generating function of the standard Epanechnikov density."""
    values = np.asarray(value, dtype=float)
    absolute = np.abs(values)
    result = np.empty_like(values)
    small = absolute < 0.1
    squared = values[small] ** 2
    result[small] = 1.0 + squared / 10.0 + squared**2 / 280.0 + squared**3 / 15120.0
    regular = ~small
    x = values[regular]
    result[regular] = 3.0 * (x * np.cosh(x) - np.sinh(x)) / x**3
    return _as_scalar_if_needed(value, result)


def epanechnikov_far_lag_factor(value: np.ndarray | float) -> np.ndarray | float:
    """Return ``q(x)=M_kappa(x)^2``, the far-lag covariance multiplier."""
    mgf = np.asarray(epanechnikov_mgf(value), dtype=float)
    result = mgf**2
    return _as_scalar_if_needed(value, result)


def epanechnikov_variance_factor(value: np.ndarray | float) -> np.ndarray | float:
    """Return ``s(x)=E[exp(-x|U-V|)]`` with stable small-``x`` evaluation."""
    values = np.asarray(value, dtype=float)
    if np.any(values < 0):
        raise ValueError("the dimensionless smoothing scale must be nonnegative")

    result = np.empty_like(values)
    small = values < 0.2
    x = values[small]
    result[small] = (
        1.0
        - 18.0 * x / 35.0
        + x**2 / 5.0
        - 4.0 * x**3 / 63.0
        + 3.0 * x**4 / 175.0
        - 2.0 * x**5 / 495.0
        + 4.0 * x**6 / 4725.0
    )

    regular = ~small
    x = values[regular]
    exponential = np.exp(-2.0 * x)
    result[regular] = (
        6.0 / (5.0 * x)
        - 3.0 / x**3
        + 4.5 / x**4
        - 4.5 / x**6
        + exponential * (4.5 / x**4 + 9.0 / x**5 + 4.5 / x**6)
    )
    return _as_scalar_if_needed(value, result)


def smoothed_exponential_covariance(
    lag: np.ndarray | float,
    variance: float,
    decay: float,
    bandwidth: float,
) -> np.ndarray | float:
    """Exact covariance after continuous Epanechnikov smoothing.

    The closed far-lag expression is used whenever ``|lag| >= 2*bandwidth``.
    The transition region is evaluated by one-dimensional quadrature against the
    explicit self-convolution density.
    """
    if variance <= 0 or decay <= 0 or bandwidth < 0:
        raise ValueError("variance and decay must be positive; bandwidth must be nonnegative")
    lags = np.abs(np.asarray(lag, dtype=float))
    if bandwidth == 0:
        result = variance * np.exp(-decay * lags)
        return _as_scalar_if_needed(lag, result)

    dimensionless = decay * bandwidth
    result = np.empty_like(lags)
    zero = lags == 0
    result[zero] = variance * float(epanechnikov_variance_factor(dimensionless))
    far = (lags >= 2.0 * bandwidth) & ~zero
    result[far] = (
        variance
        * float(epanechnikov_far_lag_factor(dimensionless))
        * np.exp(-decay * lags[far])
    )

    middle = ~(zero | far)
    for index in np.flatnonzero(middle):
        current_lag = float(lags.flat[index])
        split = -current_lag / bandwidth
        integral, _ = quad(
            lambda difference: epanechnikov_difference_density(difference)
            * np.exp(-decay * abs(current_lag + bandwidth * difference)),
            -2.0,
            2.0,
            points=[split],
            epsabs=1e-12,
            epsrel=1e-12,
        )
        result.flat[index] = variance * integral
    return _as_scalar_if_needed(lag, result)


def naive_exponential_pseudo_target(
    variance: float,
    decay: float,
    bandwidth: float,
    pair_lag: float,
) -> ExponentialPseudoTarget:
    """Exact KL target of a naive single-lag exponential pair likelihood."""
    if variance <= 0 or decay <= 0 or bandwidth < 0 or pair_lag <= 0:
        raise ValueError(
            "variance, decay, and pair_lag must be positive; bandwidth must be nonnegative"
        )
    if bandwidth <= 0:
        return ExponentialPseudoTarget(variance, decay, np.exp(-decay * pair_lag), 1.0, 1.0)
    if pair_lag < 2.0 * bandwidth:
        raise ValueError("pair_lag must be at least twice the smoothing bandwidth")
    dimensionless = decay * bandwidth
    variance_factor = float(epanechnikov_variance_factor(dimensionless))
    far_lag_factor = float(epanechnikov_far_lag_factor(dimensionless))
    correlation = far_lag_factor * np.exp(-decay * pair_lag) / variance_factor
    if not 0.0 < correlation < 1.0:
        raise ValueError("configuration does not define an interior exponential pair target")
    pseudo_decay = -np.log(correlation) / pair_lag
    return ExponentialPseudoTarget(
        variance=variance * variance_factor,
        decay=float(pseudo_decay),
        correlation=float(correlation),
        variance_factor=variance_factor,
        far_lag_factor=far_lag_factor,
    )


def naive_separable_axis_pseudo_target(
    variance: float,
    decays: np.ndarray,
    bandwidths: np.ndarray,
    pair_lags: np.ndarray,
) -> SeparableExponentialPseudoTarget:
    """Exact axis-pair target for a product exponential covariance.

    The latent covariance is

    ``variance * prod_j exp(-decays[j] * abs(r[j]))``

    and the observation kernel is the product of one-dimensional Epanechnikov
    kernels.  Each composite-likelihood pair is separated only along one axis.
    This makes the result genuinely multidimensional while retaining an exact
    target: smoothing in the other coordinates cancels from that axis pair's
    correlation.
    """
    decay_array = np.asarray(decays, dtype=float)
    bandwidth_array = np.asarray(bandwidths, dtype=float)
    lag_array = np.asarray(pair_lags, dtype=float)
    if decay_array.ndim != 1 or decay_array.size == 0:
        raise ValueError("decays must be a nonempty one-dimensional array")
    if bandwidth_array.shape != decay_array.shape or lag_array.shape != decay_array.shape:
        raise ValueError("decays, bandwidths, and pair_lags must have the same shape")
    if variance <= 0 or np.any(decay_array <= 0) or np.any(bandwidth_array < 0):
        raise ValueError("variance and decays must be positive; bandwidths must be nonnegative")
    if np.any(lag_array <= 0) or np.any(lag_array < 2.0 * bandwidth_array):
        raise ValueError("every pair lag must be positive and at least twice its bandwidth")

    dimensionless = decay_array * bandwidth_array
    variance_factors = np.asarray(epanechnikov_variance_factor(dimensionless), dtype=float)
    far_lag_factors = np.asarray(epanechnikov_far_lag_factor(dimensionless), dtype=float)
    correlations = far_lag_factors * np.exp(-decay_array * lag_array) / variance_factors
    if np.any(correlations <= 0) or np.any(correlations >= 1):
        raise ValueError("configuration does not define interior exponential axis-pair targets")
    pseudo_decays = -np.log(correlations) / lag_array
    return SeparableExponentialPseudoTarget(
        variance=float(variance * np.prod(variance_factors)),
        decays=pseudo_decays,
        correlations=correlations,
        variance_factors=variance_factors,
        far_lag_factors=far_lag_factors,
    )
