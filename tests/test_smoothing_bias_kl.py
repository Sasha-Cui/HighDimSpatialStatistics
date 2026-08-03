import math

import numpy as np
import pytest
from scipy.special import gamma, kv

from HighDimSpatial.smoothing_bias.kl import (
    exact_smoothed_covariance,
    fit_population_log_decay,
    fit_sample_log_decay,
    gaussian_kl_divergence,
    gaussian_population_criterion,
    gaussian_sample_nll,
    matern_correlation,
    matern_covariance,
    naive_point_covariance,
)


def test_matern_uses_ags_decay_convention_in_one_and_two_dimensions():
    distances = np.array([0.0, 0.4, 2.0])
    assert np.allclose(matern_correlation(distances, decay=0.7, nu=0.5), np.exp(-0.7 * distances))

    distance = 1.3
    decay = 0.8
    nu = 1.2
    scaled = decay * distance
    expected = 2.0 ** (1.0 - nu) * scaled**nu * kv(nu, scaled) / gamma(nu)
    assert math.isclose(matern_correlation(distance, decay, nu), expected, rel_tol=1e-12)

    small_scaled = 1e-9
    small_nu = 0.1
    small_expected = (
        2.0 ** (1.0 - small_nu)
        * small_scaled**small_nu
        * kv(small_nu, small_scaled)
        / gamma(small_nu)
    )
    assert matern_correlation(small_scaled, decay=1.0, nu=small_nu) == pytest.approx(
        small_expected, rel=1e-12
    )

    locations = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    covariance = matern_covariance(locations, variance=1.7, decay=decay, nu=nu, nugget=0.2)
    assert covariance.shape == (3, 3)
    assert np.allclose(covariance, covariance.T)
    assert np.allclose(np.diag(covariance), 1.9)
    assert covariance[0, 1] > covariance[0, 2]


def test_exact_and_naive_covariances_accept_rectangular_operator():
    locations = np.arange(4.0)
    operator = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.75],
        ]
    )
    parameters = dict(variance=1.4, decay=0.6, nu=0.5, nugget=0.3)

    latent = matern_covariance(locations, **parameters)
    exact = exact_smoothed_covariance(locations, operator, **parameters)
    assert exact.shape == (2, 2)
    assert np.allclose(exact, operator @ latent @ operator.T)

    naive = naive_point_covariance(locations, operator, **parameters)
    effective_locations = np.array([0.5, 2.75])
    assert np.allclose(naive, matern_covariance(effective_locations, **parameters))
    assert not np.allclose(naive, exact)

    explicit = naive_point_covariance(
        locations, operator, **parameters, output_locations=np.array([0.4, 2.6])
    )
    assert np.allclose(explicit, matern_covariance(np.array([0.4, 2.6]), **parameters))


def test_population_criterion_kl_and_sample_nll_match_direct_formulas():
    truth = np.array([[1.5, 0.25], [0.25, 0.9]])
    candidate = np.array([[1.8, 0.1], [0.1, 1.2]])
    sign, log_determinant = np.linalg.slogdet(candidate)
    assert sign == 1
    expected_population = 0.5 * (
        2 * np.log(2 * np.pi)
        + log_determinant
        + np.trace(np.linalg.solve(candidate, truth))
    )
    assert math.isclose(
        gaussian_population_criterion(candidate, truth), expected_population, rel_tol=1e-13
    )
    assert gaussian_kl_divergence(truth, truth) == pytest.approx(0.0, abs=1e-13)
    assert gaussian_kl_divergence(truth, candidate) > 0.0

    samples = np.array([[0.4, -0.2], [1.1, 0.8], [-0.3, 0.5]])
    quadratic = sum(sample @ np.linalg.solve(candidate, sample) for sample in samples)
    expected_nll = 0.5 * (
        samples.shape[0] * (2 * np.log(2 * np.pi) + log_determinant) + quadratic
    )
    assert gaussian_sample_nll(samples, candidate) == pytest.approx(expected_nll)
    assert gaussian_sample_nll(samples, candidate, average=True) == pytest.approx(
        expected_nll / samples.shape[0]
    )


def test_population_and_exact_second_moment_sample_fits_recover_decay():
    locations = np.linspace(0.0, 3.5, 8)
    operator = np.zeros((4, 8))
    operator[np.arange(4)[:, None], np.column_stack((2 * np.arange(4), 2 * np.arange(4) + 1))] = 0.5
    variance = 1.3
    true_decay = 0.75
    nu = 0.5
    nugget = 0.08

    def builder(decay):
        return exact_smoothed_covariance(
            locations, operator, variance=variance, decay=decay, nu=nu, nugget=nugget
        )

    truth = builder(true_decay)
    bounds = (np.log(0.15), np.log(2.5))
    population_fit = fit_population_log_decay(truth, builder, bounds)
    assert population_fit.success
    assert population_fit.decay == pytest.approx(true_decay, rel=2e-6)
    assert population_fit.nfev > 0
    assert population_fit.nit > 0
    assert not population_fit.at_lower_bound
    assert not population_fit.at_upper_bound

    # These deterministic rows have empirical second moment exactly equal to
    # truth, so their average NLL has the same optimum as the population target.
    dimension = truth.shape[0]
    exact_second_moment_samples = np.sqrt(dimension) * np.linalg.cholesky(truth).T
    assert np.allclose(
        exact_second_moment_samples.T @ exact_second_moment_samples / dimension, truth
    )
    assert gaussian_sample_nll(exact_second_moment_samples, truth, average=True) == pytest.approx(
        gaussian_population_criterion(truth, truth)
    )
    sample_fit = fit_sample_log_decay(exact_second_moment_samples, builder, bounds)
    assert sample_fit.success
    assert sample_fit.decay == pytest.approx(true_decay, rel=2e-6)


def test_validation_rejects_malformed_or_non_positive_definite_inputs_without_jitter():
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        matern_covariance(np.zeros((3, 3)), variance=1.0, decay=1.0, nu=0.5)
    with pytest.raises(ValueError, match="one column per latent location"):
        exact_smoothed_covariance(
            np.arange(4.0), np.ones((2, 3)), variance=1.0, decay=1.0, nu=0.5
        )
    with pytest.raises(ValueError, match="nonzero sum"):
        naive_point_covariance(
            np.arange(3.0),
            np.array([[1.0, -1.0, 0.0]]),
            variance=1.0,
            decay=1.0,
            nu=0.5,
        )
    with pytest.raises(ValueError, match="no numerical jitter"):
        gaussian_sample_nll(np.ones(2), np.ones((2, 2)))
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_population_log_decay(np.eye(2), lambda decay: np.eye(2), (0.0, 0.0))
