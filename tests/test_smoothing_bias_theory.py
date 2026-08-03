import numpy as np
import pytest
from scipy.integrate import quad

from HighDimSpatial.smoothing_bias.estimators import (
    corrected_two_lag_estimate,
    naive_pair_estimate,
)
from HighDimSpatial.smoothing_bias.theory import (
    epanechnikov_density,
    epanechnikov_difference_density,
    epanechnikov_far_lag_factor,
    epanechnikov_mgf,
    epanechnikov_variance_factor,
    naive_exponential_pseudo_target,
    naive_separable_axis_pseudo_target,
    smoothed_exponential_covariance,
)


def test_epanechnikov_densities_integrate_to_one() -> None:
    kernel_mass = quad(epanechnikov_density, -1.0, 1.0, epsabs=1e-13)[0]
    difference_mass = quad(epanechnikov_difference_density, -2.0, 2.0, epsabs=1e-13)[0]
    assert kernel_mass == pytest.approx(1.0, abs=1e-12)
    assert difference_mass == pytest.approx(1.0, abs=1e-12)
    grid = np.linspace(-2.1, 2.1, 1001)
    assert np.min(epanechnikov_difference_density(grid)) >= -1e-14


@pytest.mark.parametrize("scale", [0.0, 1e-6, 0.05, 0.2, 1.0, 5.0])
def test_closed_smoothing_factors_match_quadrature(scale: float) -> None:
    variance_oracle = quad(
        lambda value: epanechnikov_difference_density(value)
        * np.exp(-scale * abs(value)),
        -2.0,
        2.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )[0]
    mgf_oracle = quad(
        lambda value: epanechnikov_density(value) * np.exp(scale * value),
        -1.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )[0]
    assert epanechnikov_variance_factor(scale) == pytest.approx(
        variance_oracle, rel=2e-9, abs=2e-11
    )
    assert epanechnikov_mgf(scale) == pytest.approx(mgf_oracle, rel=2e-10, abs=2e-12)
    assert epanechnikov_far_lag_factor(scale) == pytest.approx(mgf_oracle**2, rel=5e-10)


def test_exact_smoothed_covariance_matches_integral_in_all_lag_regions() -> None:
    variance = 1.7
    decay = 0.8
    bandwidth = 0.6
    lags = np.array([0.0, 0.2, 0.9, 1.2, 2.0])
    expected = np.array(
        [
            variance
            * quad(
                lambda value: epanechnikov_difference_density(value)
                * np.exp(-decay * abs(lag + bandwidth * value)),
                -2.0,
                2.0,
                points=[-lag / bandwidth] if lag < 2.0 * bandwidth else None,
                epsabs=1e-12,
                epsrel=1e-12,
            )[0]
            for lag in lags
        ]
    )
    actual = smoothed_exponential_covariance(lags, variance, decay, bandwidth)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


def test_naive_pair_target_has_strict_smoothing_shift() -> None:
    target = naive_exponential_pseudo_target(
        variance=2.0,
        decay=0.75,
        bandwidth=0.4,
        pair_lag=1.2,
    )
    assert 0.0 < target.variance < 2.0
    assert 0.0 < target.decay < 0.75
    assert target.variance_factor < 1.0 < target.far_lag_factor
    assert target.correlation == pytest.approx(np.exp(-target.decay * 1.2))


def test_separable_axis_target_reduces_every_decay_and_variance() -> None:
    decays = np.array([0.5, 1.0, 1.5])
    bandwidths = np.array([0.2, 0.3, 0.0])
    pair_lags = np.array([0.8, 1.0, 0.7])
    target = naive_separable_axis_pseudo_target(
        variance=1.8,
        decays=decays,
        bandwidths=bandwidths,
        pair_lags=pair_lags,
    )
    assert 0.0 < target.variance < 1.8
    assert np.all(target.decays[:2] < decays[:2])
    assert target.decays[2] == pytest.approx(decays[2])
    assert target.variance_factors[2] == pytest.approx(1.0)
    for axis in range(3):
        one_dimensional = naive_exponential_pseudo_target(
            1.0,
            decays[axis],
            bandwidths[axis],
            pair_lags[axis],
        )
        assert target.decays[axis] == pytest.approx(one_dimensional.decay)


def test_pair_estimators_recover_their_distinct_targets() -> None:
    variance = 1.4
    decay = 0.65
    bandwidth = 0.2
    spacing = 1.0
    lag_1 = 1
    lag_2 = 2
    covariance = np.array(
        [
            [smoothed_exponential_covariance(abs(i - j), variance, decay, bandwidth) for j in range(3)]
            for i in range(3)
        ]
    )
    rng = np.random.default_rng(20260802)
    blocks = rng.multivariate_normal(np.zeros(3), covariance, size=40_000)
    values = blocks.reshape(-1)

    naive = naive_pair_estimate(values, lag_steps=lag_1, spacing=spacing, block_stride=3)
    pseudo_target = naive_exponential_pseudo_target(variance, decay, bandwidth, spacing)
    assert naive.valid
    assert naive.variance == pytest.approx(pseudo_target.variance, rel=0.025)
    assert naive.decay == pytest.approx(pseudo_target.decay, rel=0.035)

    corrected = corrected_two_lag_estimate(
        values,
        lag_1_steps=lag_1,
        lag_2_steps=lag_2,
        spacing=spacing,
        bandwidth=bandwidth,
        block_stride=3,
    )
    assert corrected.valid
    assert corrected.variance == pytest.approx(variance, rel=0.025)
    assert corrected.decay == pytest.approx(decay, rel=0.035)


def test_invalid_target_configurations_are_rejected() -> None:
    with pytest.raises(ValueError, match="pair_lag"):
        naive_exponential_pseudo_target(1.0, 1.0, 0.5, 0.9)
    with pytest.raises(ValueError, match="same shape"):
        naive_separable_axis_pseudo_target(
            1.0,
            np.ones(2),
            np.ones(3),
            np.ones(2),
        )
