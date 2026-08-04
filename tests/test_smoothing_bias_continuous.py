import numpy as np
import pytest

from HighDimSpatial.smoothing_bias.continuous import (
    continuous_matern_pair_target,
    epanechnikov_difference_radial_moment,
    product_epanechnikov_decay_shift_coefficient,
    product_epanechnikov_direction_contrast_coefficient,
    product_epanechnikov_difference_quadrature,
    transition_aware_matern_pair_approximation,
    transformed_epanechnikov_difference_radial_moment,
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


def test_anisotropic_transform_second_moment_matches_covariance_trace() -> None:
    transform = np.array([[1.7, 0.2], [-0.1, 0.6]])
    moment = transformed_epanechnikov_difference_radial_moment(
        2.0,
        2,
        kernel_transform=transform,
        order=96,
    )
    expected = 2.0 * np.trace(transform @ transform.T / 5.0)
    assert moment == pytest.approx(expected, rel=2e-12)


def test_rotating_kernel_and_lag_together_preserves_anisotropic_target() -> None:
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transform = np.diag([1.8, 0.7])
    direction = np.array([1.0, 0.0])
    first = continuous_matern_pair_target(
        dimension=2,
        smoothness=1.5,
        decay=0.9,
        bandwidth=0.08,
        lag=1.0,
        quadrature_order=96,
        kernel_transform=transform,
        lag_direction=direction,
    )
    second = continuous_matern_pair_target(
        dimension=2,
        smoothness=1.5,
        decay=0.9,
        bandwidth=0.08,
        lag=1.0,
        quadrature_order=96,
        kernel_transform=rotation @ transform,
        lag_direction=rotation @ direction,
    )
    assert second.variance_factor == pytest.approx(first.variance_factor, rel=2e-13)
    assert second.pseudo_decay == pytest.approx(first.pseudo_decay, rel=2e-13)


@pytest.mark.parametrize("smoothness", [0.5, 1.0])
def test_rough_anisotropic_leading_coefficient_is_direction_independent(
    smoothness: float,
) -> None:
    transform = np.diag([2.0, 0.5])
    parallel = product_epanechnikov_decay_shift_coefficient(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        lag=1.0,
        kernel_transform=transform,
        lag_direction=np.array([1.0, 0.0]),
    )
    perpendicular = product_epanechnikov_decay_shift_coefficient(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        lag=1.0,
        kernel_transform=transform,
        lag_direction=np.array([0.0, 1.0]),
    )
    assert parallel == pytest.approx(perpendicular, rel=2e-13)


@pytest.mark.parametrize("smoothness", [1.5, 2.5])
def test_smooth_anisotropic_coefficient_has_closed_directional_contrast(
    smoothness: float,
) -> None:
    aspect_ratio = 4.0
    scale = np.sqrt(2.0 / (aspect_ratio + 1.0 / aspect_ratio))
    transform = scale * np.diag(
        [np.sqrt(aspect_ratio), 1.0 / np.sqrt(aspect_ratio)]
    )
    parallel = product_epanechnikov_decay_shift_coefficient(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        lag=1.0,
        kernel_transform=transform,
        lag_direction=np.array([1.0, 0.0]),
    )
    perpendicular = product_epanechnikov_decay_shift_coefficient(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        lag=1.0,
        kernel_transform=transform,
        lag_direction=np.array([0.0, 1.0]),
    )
    expected_ratio = (
        (2.0 * smoothness - 1.0) * aspect_ratio**2 + 1.0
    ) / (aspect_ratio**2 + 2.0 * smoothness - 1.0)
    assert parallel > perpendicular > 0.0
    assert parallel / perpendicular == pytest.approx(expected_ratio, rel=2e-13)


@pytest.mark.parametrize("smoothness", [0.5, 1.0, 1.5, 2.5])
def test_direction_contrast_coefficient_matches_swapped_sign(
    smoothness: float,
) -> None:
    transform = np.diag([1.6, 0.7])
    major = np.array([1.0, 0.0])
    minor = np.array([0.0, 1.0])
    forward = product_epanechnikov_direction_contrast_coefficient(
        dimension=2,
        smoothness=smoothness,
        decay=0.9,
        lag=1.2,
        kernel_transform=transform,
        first_direction=major,
        second_direction=minor,
    )
    reverse = product_epanechnikov_direction_contrast_coefficient(
        dimension=2,
        smoothness=smoothness,
        decay=0.9,
        lag=1.2,
        kernel_transform=transform,
        first_direction=minor,
        second_direction=major,
    )
    assert forward > 0.0
    assert reverse == pytest.approx(-forward, rel=2e-13)


@pytest.mark.parametrize("smoothness", [0.5, 1.0, 1.5, 2.5])
def test_direction_contrast_coefficient_matches_small_bandwidth_oracle(
    smoothness: float,
) -> None:
    aspect_ratio = 4.0
    scale = np.sqrt(2.0 / (aspect_ratio + 1.0 / aspect_ratio))
    transform = scale * np.diag(
        [np.sqrt(aspect_ratio), 1.0 / np.sqrt(aspect_ratio)]
    )
    major = np.array([1.0, 0.0])
    minor = np.array([0.0, 1.0])
    coefficient = product_epanechnikov_direction_contrast_coefficient(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        lag=1.0,
        kernel_transform=transform,
        first_direction=major,
        second_direction=minor,
    )
    bandwidth = 0.005
    major_target = continuous_matern_pair_target(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        bandwidth=bandwidth,
        lag=1.0,
        quadrature_order=96,
        kernel_transform=transform,
        lag_direction=major,
    )
    minor_target = continuous_matern_pair_target(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        bandwidth=bandwidth,
        lag=1.0,
        quadrature_order=96,
        kernel_transform=transform,
        lag_direction=minor,
    )
    contrast = minor_target.pseudo_decay - major_target.pseudo_decay
    assert contrast / (coefficient * bandwidth**2) == pytest.approx(1.0, rel=2e-4)


def test_anisotropic_geometry_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="nonsingular"):
        continuous_matern_pair_target(
            dimension=2,
            smoothness=1.5,
            decay=1.0,
            bandwidth=0.1,
            lag=1.0,
            kernel_transform=np.diag([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="unit vector"):
        continuous_matern_pair_target(
            dimension=2,
            smoothness=1.5,
            decay=1.0,
            bandwidth=0.1,
            lag=1.0,
            lag_direction=np.array([2.0, 0.0]),
        )


@pytest.mark.parametrize("smoothness", [0.6, 0.95, 1.0, 1.05, 1.4])
@pytest.mark.parametrize("bandwidth", [0.02, 0.05])
def test_transition_aware_approximation_matches_exact_pair_target(
    smoothness: float,
    bandwidth: float,
) -> None:
    exact = continuous_matern_pair_target(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        bandwidth=bandwidth,
        lag=1.0,
        quadrature_order=96,
    )
    approximation = transition_aware_matern_pair_approximation(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        bandwidth=bandwidth,
        lag=1.0,
        quadrature_order=96,
    )
    exact_shift = 1.0 - exact.pseudo_decay
    approximate_shift = 1.0 - approximation.pseudo_decay
    assert exact_shift > 0.0
    assert approximate_shift > 0.0
    assert exact_shift / approximate_shift == pytest.approx(1.0, rel=0.002)


def test_transition_aware_approximation_has_continuous_threshold_limit() -> None:
    values = [
        transition_aware_matern_pair_approximation(
            dimension=2,
            smoothness=smoothness,
            decay=1.0,
            bandwidth=0.05,
            lag=1.0,
            quadrature_order=96,
        )
        for smoothness in (1.0 - 1e-6, 1.0, 1.0 + 1e-6)
    ]
    assert values[0].variance_factor == pytest.approx(
        values[1].variance_factor,
        abs=2e-8,
    )
    assert values[2].variance_factor == pytest.approx(
        values[1].variance_factor,
        abs=2e-8,
    )
    assert values[0].pseudo_decay == pytest.approx(values[1].pseudo_decay, abs=2e-8)
    assert values[2].pseudo_decay == pytest.approx(values[1].pseudo_decay, abs=2e-8)


@pytest.mark.parametrize("smoothness", [0.0, 2.0, 2.5])
def test_transition_aware_approximation_rejects_out_of_scope_smoothness(
    smoothness: float,
) -> None:
    with pytest.raises(ValueError, match="strictly between zero and two"):
        transition_aware_matern_pair_approximation(
            dimension=2,
            smoothness=smoothness,
            decay=1.0,
            bandwidth=0.05,
            lag=1.0,
        )
