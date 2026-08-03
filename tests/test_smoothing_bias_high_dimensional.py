import numpy as np
import pytest

from HighDimSpatial.smoothing_bias.high_dimensional import (
    gaussian_likelihood_uniform_bound,
    normalized_gaussian_population_objectives,
    normalized_gaussian_sample_objectives,
    relative_precision_matrix,
)


def test_relative_precision_matches_scalar_generalized_eigenvalues() -> None:
    truth = np.diag([1.0, 2.0, 4.0])
    candidate = np.diag([2.0, 1.0, 8.0])
    relative = relative_precision_matrix(truth, candidate)
    np.testing.assert_allclose(np.linalg.eigvalsh(relative), [0.5, 0.5, 2.0])


def test_uniform_bound_has_declared_gaussian_quadratic_constants() -> None:
    dimension = 3
    truth = np.eye(dimension)
    candidates = np.stack([truth, 2.0 * truth])
    result = gaussian_likelihood_uniform_bound(
        truth,
        candidates,
        sample_size=25,
        delta=0.1,
    )
    log_factor = np.log(2.0 * 2.0 / 0.1)
    first = result.candidates[0]
    second = result.candidates[1]
    assert first.frobenius_norm == pytest.approx(np.sqrt(dimension))
    assert first.operator_norm == pytest.approx(1.0)
    assert first.stable_rank == pytest.approx(dimension)
    assert second.frobenius_norm == pytest.approx(0.5 * np.sqrt(dimension))
    assert second.operator_norm == pytest.approx(0.5)
    assert second.stable_rank == pytest.approx(dimension)
    expected_first = np.sqrt(dimension) / dimension * np.sqrt(log_factor / 25.0)
    expected_first += log_factor / (dimension * 25.0)
    assert first.radius == pytest.approx(expected_first)
    assert result.radius == pytest.approx(first.radius)


def test_sample_and_population_objectives_match_when_sample_covariance_is_exact() -> None:
    dimension = 4
    truth = np.eye(dimension)
    candidates = np.stack([truth, 1.5 * truth, np.diag([1.0, 1.2, 1.4, 1.6])])
    samples = np.sqrt(dimension) * np.eye(dimension)
    population = normalized_gaussian_population_objectives(truth, candidates)
    empirical = normalized_gaussian_sample_objectives(samples, candidates)
    np.testing.assert_allclose(empirical, population, rtol=1e-13, atol=1e-13)


def test_seeded_likelihood_grid_satisfies_its_simultaneous_certificate() -> None:
    truth = np.array(
        [
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.35],
            [0.2, 0.35, 1.0],
        ]
    )
    candidates = np.stack(
        [
            truth,
            0.9 * truth + 0.1 * np.eye(3),
            1.1 * truth,
        ]
    )
    rng = np.random.default_rng(20260802)
    samples = rng.multivariate_normal(np.zeros(3), truth, size=400)
    empirical = normalized_gaussian_sample_objectives(samples, candidates)
    population = normalized_gaussian_population_objectives(truth, candidates)
    certificate = gaussian_likelihood_uniform_bound(
        truth,
        candidates,
        sample_size=samples.shape[0],
        delta=0.01,
    )
    assert np.max(np.abs(empirical - population)) <= certificate.radius


@pytest.mark.parametrize(
    ("sample_size", "delta"),
    [(0, 0.05), (True, 0.05), (10, 0.0), (10, 1.0)],
)
def test_uniform_bound_rejects_invalid_probability_inputs(
    sample_size: int,
    delta: float,
) -> None:
    with pytest.raises(ValueError):
        gaussian_likelihood_uniform_bound(
            np.eye(2),
            np.eye(2),
            sample_size=sample_size,
            delta=delta,
        )

