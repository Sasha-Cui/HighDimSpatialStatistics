"""Moment and pair-composite estimators for smoothed exponential fields."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from HighDimSpatial.smoothing_bias.theory import epanechnikov_variance_factor


@dataclass(frozen=True)
class PairEstimate:
    variance: np.ndarray
    decay: np.ndarray
    correlation: np.ndarray
    valid: np.ndarray
    number_of_pairs: int


@dataclass(frozen=True)
class CorrectedEstimate:
    variance: np.ndarray
    decay: np.ndarray
    covariance_lag_1: np.ndarray
    covariance_lag_2: np.ndarray
    valid: np.ndarray
    number_of_blocks: int


def _as_replicate_matrix(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array[None, :], True
    if array.ndim != 2:
        raise ValueError("values must have shape (n,) or (replicates, n)")
    return array, False


def _base_indices(length: int, maximum_lag: int, block_stride: int) -> np.ndarray:
    if maximum_lag < 1 or block_stride < 1 or maximum_lag >= length:
        raise ValueError("lags and block_stride are incompatible with the sequence length")
    return np.arange(0, length - maximum_lag, block_stride, dtype=int)


def naive_pair_estimate(
    values: np.ndarray,
    lag_steps: int,
    spacing: float,
    block_stride: int = 1,
) -> PairEstimate:
    """Fit the naive exponential covariance to repeated pairs at one lag."""
    matrix, was_vector = _as_replicate_matrix(values)
    bases = _base_indices(matrix.shape[1], lag_steps, block_stride)
    left = matrix[:, bases]
    right = matrix[:, bases + lag_steps]
    moment_left = np.mean(left**2, axis=1)
    moment_right = np.mean(right**2, axis=1)
    cross_moment = np.mean(left * right, axis=1)
    variance = 0.5 * (moment_left + moment_right)
    correlation = cross_moment / variance
    valid = (variance > 0) & (correlation > 0) & (correlation < 1)
    decay = np.full_like(variance, np.nan)
    decay[valid] = -np.log(correlation[valid]) / (lag_steps * spacing)

    if was_vector:
        variance = variance[0]
        decay = decay[0]
        correlation = correlation[0]
        valid = valid[0]
    return PairEstimate(variance, decay, correlation, valid, len(bases))


def corrected_two_lag_estimate(
    values: np.ndarray,
    lag_1_steps: int,
    lag_2_steps: int,
    spacing: float,
    bandwidth: float,
    block_stride: int = 1,
) -> CorrectedEstimate:
    """Recover exponential decay from two far-lag covariances and de-smooth variance."""
    if not 0 < lag_1_steps < lag_2_steps:
        raise ValueError("lags must satisfy 0 < lag_1_steps < lag_2_steps")
    if lag_1_steps * spacing < 2.0 * bandwidth:
        raise ValueError("the first physical lag must be at least twice the bandwidth")

    matrix, was_vector = _as_replicate_matrix(values)
    bases = _base_indices(matrix.shape[1], lag_2_steps, block_stride)
    anchors = matrix[:, bases]
    lag_1_values = matrix[:, bases + lag_1_steps]
    lag_2_values = matrix[:, bases + lag_2_steps]
    covariance_1 = np.mean(anchors * lag_1_values, axis=1)
    covariance_2 = np.mean(anchors * lag_2_values, axis=1)
    smoothed_variance = np.mean(
        np.concatenate((anchors**2, lag_1_values**2, lag_2_values**2), axis=1),
        axis=1,
    )
    ratio = covariance_1 / covariance_2
    valid = (covariance_1 > 0) & (covariance_2 > 0) & (ratio > 1)
    decay = np.full_like(covariance_1, np.nan)
    decay[valid] = np.log(ratio[valid]) / ((lag_2_steps - lag_1_steps) * spacing)
    variance = np.full_like(covariance_1, np.nan)
    factors = np.ones_like(covariance_1)
    factors[valid] = epanechnikov_variance_factor(decay[valid] * bandwidth)
    variance[valid] = smoothed_variance[valid] / factors[valid]

    if was_vector:
        variance = variance[0]
        decay = decay[0]
        covariance_1 = covariance_1[0]
        covariance_2 = covariance_2[0]
        valid = valid[0]
    return CorrectedEstimate(
        variance,
        decay,
        covariance_1,
        covariance_2,
        valid,
        len(bases),
    )
