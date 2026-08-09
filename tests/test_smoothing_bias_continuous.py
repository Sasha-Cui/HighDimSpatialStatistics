import numpy as np
import pytest

from HighDimSpatial.smoothing_bias.continuous import (
    continuous_matern_multilag_target,
    continuous_matern_covariance_matrix,
    continuous_matern_full_likelihood_target,
    continuous_matern_pair_target,
    epanechnikov_difference_radial_moment,
    finite_design_full_likelihood_asymptotics,
    gaussian_correlation_kl,
    product_kernel_decay_shift_coefficient,
    product_kernel_multilag_asymptotics,
    product_epanechnikov_decay_shift_coefficient,
    product_epanechnikov_direction_contrast_coefficient,
    product_epanechnikov_difference_quadrature,
    product_uniform_difference_quadrature,
    transition_aware_matern_pair_approximation,
    transformed_epanechnikov_difference_radial_moment,
)
from HighDimSpatial.smoothing_bias.kl import matern_correlation
from HighDimSpatial.smoothing_bias.theory import naive_exponential_pseudo_target


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_product_difference_quadrature_is_a_probability_rule(dimension: int) -> None:
    nodes, weights = product_epanechnikov_difference_quadrature(dimension, order=48)
    assert nodes.shape == (48**dimension, dimension)
    assert weights.shape == (48**dimension,)
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=2e-14)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_product_uniform_difference_quadrature_is_a_probability_rule(
    dimension: int,
) -> None:
    order = 32
    nodes, weights = product_uniform_difference_quadrature(dimension, order=order)
    assert nodes.shape == (order**dimension, dimension)
    assert weights.shape == (order**dimension,)
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=2e-14)


def test_one_dimensional_uniform_difference_second_moment_is_exact() -> None:
    nodes, weights = product_uniform_difference_quadrature(1, order=32)
    second_moment = float(weights @ np.square(nodes[:, 0]))
    assert second_moment == pytest.approx(2.0 / 3.0, rel=2e-15)


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


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("kernel_family", ["epanechnikov", "uniform"])
@pytest.mark.parametrize("smoothness", [0.5, 1.0, 1.5, 2.5])
def test_dimension_kernel_robustness_coefficient_is_predictive(
    dimension: int,
    kernel_family: str,
    smoothness: float,
) -> None:
    bandwidth = 0.002
    target = continuous_matern_pair_target(
        dimension=dimension,
        smoothness=smoothness,
        decay=1.0,
        bandwidth=bandwidth,
        lag=1.0,
        kernel_family=kernel_family,
        quadrature_order=48,
    )
    coefficient = product_kernel_decay_shift_coefficient(
        dimension=dimension,
        smoothness=smoothness,
        decay=1.0,
        lag=1.0,
        kernel_family=kernel_family,
        quadrature_order=48,
    )
    if smoothness < 1.0:
        scale = bandwidth ** (2.0 * smoothness)
    elif smoothness == 1.0:
        scale = bandwidth**2 * np.log(1.0 / bandwidth)
    else:
        scale = bandwidth**2
    ratio = (1.0 - target.pseudo_decay) / (coefficient * scale)
    assert target.kernel_family == kernel_family
    assert coefficient > 0.0
    tolerance = 0.18 if smoothness == 1.0 else 0.12
    assert ratio == pytest.approx(1.0, rel=tolerance)


def test_unknown_product_kernel_is_rejected() -> None:
    with pytest.raises(ValueError, match="kernel_family must be one of"):
        continuous_matern_pair_target(
            dimension=2,
            smoothness=1.5,
            decay=1.0,
            bandwidth=0.02,
            lag=1.0,
            kernel_family="triangular",
        )


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


def test_gaussian_correlation_kl_is_nonnegative_and_zero_only_at_match() -> None:
    assert gaussian_correlation_kl(0.4, 0.4) == pytest.approx(0.0, abs=2e-16)
    assert gaussian_correlation_kl(0.4, 0.2) > 0.0
    vector = gaussian_correlation_kl(
        np.array([0.2, 0.4]),
        np.array([0.2, 0.3]),
    )
    assert vector[0] == pytest.approx(0.0, abs=2e-16)
    assert vector[1] > 0.0
    with pytest.raises(ValueError, match="strictly between"):
        gaussian_correlation_kl(1.0, 0.5)


def test_singleton_composite_target_reduces_to_saturated_pair_target() -> None:
    pair = continuous_matern_pair_target(
        dimension=2,
        smoothness=1.5,
        decay=0.8,
        bandwidth=0.03,
        lag=1.2,
        quadrature_order=72,
    )
    composite = continuous_matern_multilag_target(
        dimension=2,
        smoothness=1.5,
        decay=0.8,
        bandwidth=0.03,
        lags=np.array([1.2]),
        quadrature_order=72,
    )
    asymptotics = product_kernel_multilag_asymptotics(
        dimension=2,
        smoothness=1.5,
        decay=0.8,
        lags=np.array([1.2]),
    )
    assert composite.pseudo_decay == pytest.approx(pair.pseudo_decay, rel=2e-8)
    assert composite.minimum_kl == pytest.approx(0.0, abs=2e-15)
    assert asymptotics.minimum_kl_coefficient == pytest.approx(0.0, abs=2e-15)


@pytest.mark.parametrize("smoothness", [0.5, 1.0, 1.5, 2.5])
def test_multilag_composite_has_predicted_shift_and_irreducible_kl(
    smoothness: float,
) -> None:
    lags = np.array([0.5, 1.0, 1.5, 2.0])
    bandwidth = 0.005
    asymptotics = product_kernel_multilag_asymptotics(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        lags=lags,
        quadrature_order=96,
    )
    target = continuous_matern_multilag_target(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        bandwidth=bandwidth,
        lags=lags,
        quadrature_order=96,
    )
    if smoothness < 1.0:
        scale = bandwidth ** (2.0 * smoothness)
    elif smoothness == 1.0:
        scale = bandwidth**2 * np.log(1.0 / bandwidth)
    else:
        scale = bandwidth**2
    shift_ratio = (1.0 - target.pseudo_decay) / (
        asymptotics.decay_shift_coefficient * scale
    )
    kl_ratio = target.minimum_kl / (
        asymptotics.minimum_kl_coefficient * scale**2
    )
    tolerance = 0.12 if smoothness == 1.0 else 0.035
    assert target.pseudo_decay < 1.0
    assert asymptotics.minimum_kl_coefficient > 0.0
    assert np.ptp(asymptotics.pair_shift_coefficients) > 0.0
    assert shift_ratio == pytest.approx(1.0, rel=tolerance)
    assert kl_ratio == pytest.approx(1.0, rel=tolerance)


def test_multilag_inputs_require_positive_matching_weights() -> None:
    with pytest.raises(ValueError, match="match lags"):
        product_kernel_multilag_asymptotics(
            dimension=2,
            smoothness=1.5,
            decay=1.0,
            lags=np.array([0.5, 1.0]),
            weights=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="positive finite"):
        continuous_matern_multilag_target(
            dimension=2,
            smoothness=1.5,
            decay=1.0,
            bandwidth=0.01,
            lags=np.array([0.0, 1.0]),
        )


def test_continuous_covariance_zero_support_matches_point_matern() -> None:
    locations = np.array([(0.0, 0.0), (0.5, 0.0), (0.5, 1.0)])
    covariance = continuous_matern_covariance_matrix(
        locations,
        variance=1.7,
        decay=0.8,
        smoothness=1.5,
        bandwidth=0.0,
    )
    distances = np.linalg.norm(
        locations[:, None, :] - locations[None, :, :], axis=2
    )
    expected = 1.7 * np.asarray(matern_correlation(distances, 0.8, 1.5))
    np.testing.assert_allclose(covariance, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("smoothness", [0.5, 1.0, 1.5, 2.5])
def test_finite_design_full_likelihood_projection_predicts_target_and_kl(
    smoothness: float,
) -> None:
    locations = np.asarray([(i, j) for i in range(3) for j in range(3)], dtype=float)
    bandwidth = 0.005
    asymptotics = finite_design_full_likelihood_asymptotics(
        locations,
        variance=1.0,
        decay=1.0,
        smoothness=smoothness,
        quadrature_order=64,
    )
    target = continuous_matern_full_likelihood_target(
        locations,
        variance=1.0,
        decay=1.0,
        smoothness=smoothness,
        bandwidth=bandwidth,
        quadrature_order=64,
    )
    if smoothness < 1.0:
        scale = bandwidth ** (2.0 * smoothness)
    elif smoothness == 1.0:
        scale = bandwidth**2 * np.log(1.0 / bandwidth)
    else:
        scale = bandwidth**2
    decay_ratio = (1.0 - target.pseudo_decay) / (
        asymptotics.decay_inflation_coefficient * scale
    )
    variance_ratio = np.log(target.pseudo_variance) / (
        asymptotics.log_variance_shift_coefficient * scale
    )
    kl_ratio = target.minimum_kl / (
        asymptotics.minimum_kl_coefficient * scale**2
    )
    tolerance = 0.15 if smoothness == 1.0 else 0.04
    assert asymptotics.decay_inflation_coefficient > 0.0
    assert asymptotics.minimum_kl_coefficient > 0.0
    assert target.pseudo_decay < 1.0
    assert decay_ratio == pytest.approx(1.0, rel=tolerance)
    assert variance_ratio == pytest.approx(1.0, rel=tolerance)
    assert kl_ratio == pytest.approx(1.0, rel=tolerance)


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
@pytest.mark.parametrize("kernel_family", ["epanechnikov", "uniform"])
def test_transition_aware_approximation_matches_exact_pair_target(
    smoothness: float,
    bandwidth: float,
    kernel_family: str,
) -> None:
    exact = continuous_matern_pair_target(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        bandwidth=bandwidth,
        lag=1.0,
        kernel_family=kernel_family,
        quadrature_order=96,
    )
    approximation = transition_aware_matern_pair_approximation(
        dimension=2,
        smoothness=smoothness,
        decay=1.0,
        bandwidth=bandwidth,
        lag=1.0,
        kernel_family=kernel_family,
        quadrature_order=96,
    )
    exact_shift = 1.0 - exact.pseudo_decay
    approximate_shift = 1.0 - approximation.pseudo_decay
    assert exact_shift > 0.0
    assert approximate_shift > 0.0
    assert exact.kernel_family == approximation.kernel_family == kernel_family
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
