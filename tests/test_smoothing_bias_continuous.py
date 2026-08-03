import numpy as np
import pytest

from HighDimSpatial.smoothing_bias.continuous import (
    continuous_matern_pair_target,
    epanechnikov_difference_radial_moment,
    product_epanechnikov_decay_shift_coefficient,
    product_epanechnikov_difference_quadrature,
)
from HighDimSpatial.smoothing_bias.theory import naive_exponential_pseudo_target


@pytest.mark.parametrize("dimension", [1, 2])
def test_product_difference_quadrature_is_a_probability_rule(dimension: int) -> None:
    nodes, weights = product_epanechnikov_difference_quadrature(dimension, order=48)
    assert nodes.shape == (48**dimension, dimension)
    assert weights.shape == (48**dimension,)
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=2e-14)


def test_one_dimensional_difference_moments_match_closed_forms() -> None:
    assert epanechnikov_difference_radial_moment(1.0, 1) == pytest.approx(
        18.0 / 35.0, rel=2e-7
    )
    assert epanechnikov_difference_radial_moment(2.0, 1) == pytest.approx(
        2.0 / 5.0, rel=2e-12
    )


def test_continuous_ou_quadrature_matches_exact_closed_target() -> None:
    target = continuous_matern_pair_target(
        dimension=1,
        smoothness=0.5,
        decay=0.8,
        bandwidth=0.3,
        lag=1.0,
        quadrature_order=96,
    )
    exact = naive_exponential_pseudo_target(1.0, 0.8, 0.3, 1.0)
    assert target.variance_factor == pytest.approx(exact.variance_factor, rel=2e-8)
    assert target.correlation == pytest.approx(exact.correlation, rel=2e-8)
    assert target.pseudo_decay == pytest.approx(exact.decay, rel=2e-8)


def test_two_dimensional_rough_matern_has_predicted_bandwidth_order() -> None:
    smoothness = 0.75
    targets = [
        continuous_matern_pair_target(
            dimension=2,
            smoothness=smoothness,
            decay=1.0,
            bandwidth=bandwidth,
            lag=1.0,
            quadrature_order=72,
        )
        for bandwidth in (0.02, 0.04)
    ]
    shifts = np.array([1.0 - target.pseudo_decay for target in targets])
    observed_power = np.log(shifts[1] / shifts[0]) / np.log(2.0)
    assert np.all(shifts > 0.0)
    assert observed_power == pytest.approx(2.0 * smoothness, abs=0.08)


@pytest.mark.parametrize("smoothness", [0.5, 1.0, 1.5, 2.5])
def test_leading_decay_shift_coefficient_is_positive_and_predictive(
    smoothness: float,
) -> None:
    bandwidth = 0.002
    target = continuous_matern_pair_target(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        bandwidth=bandwidth,
        lag=1.0,
        quadrature_order=96,
    )
    coefficient = product_epanechnikov_decay_shift_coefficient(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        lag=1.0,
        quadrature_order=96,
    )
    if smoothness < 1.0:
        scale = bandwidth ** (2.0 * smoothness)
    elif smoothness == 1.0:
        scale = bandwidth**2 * np.log(1.0 / bandwidth)
    else:
        scale = bandwidth**2
    ratio = (1.0 - target.pseudo_decay) / (coefficient * scale)
    assert coefficient > 0.0
    assert ratio == pytest.approx(1.0, rel=0.12)


def test_zero_bandwidth_is_an_exact_no_shift_control() -> None:
    target = continuous_matern_pair_target(
        dimension=2,
        smoothness=1.5,
        decay=0.7,
        bandwidth=0.0,
        lag=1.2,
    )
    assert target.variance_factor == 1.0
    assert target.pseudo_decay == 0.7
