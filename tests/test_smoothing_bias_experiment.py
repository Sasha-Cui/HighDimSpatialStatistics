import numpy as np
import pytest

from HighDimSpatial.smoothing_bias.experiment import (
    profile_log_decay_grid,
    run_finite_design_configuration,
)


def _configuration(bandwidth: float) -> dict:
    return {
        "dimension": 1,
        "number_of_points": 24,
        "spacing": 0.2,
        "output_stride": 2,
        "boundary_trim": 0.4,
        "bandwidth": bandwidth,
        "kernel": "radial",
        "variance": 1.0,
        "decay": 0.8,
        "smoothness": 0.5,
        "nugget": 0.05,
        "replicates": 10,
        "log_decay_bounds": [np.log(0.2), np.log(2.5)],
        "log_decay_grid_size": 61,
    }


def test_profile_grid_recovers_scalar_population_target() -> None:
    truth = np.array([[1.4]])

    def builder(decay: float) -> np.ndarray:
        return np.array([[decay]])

    samples = np.sqrt(1.4) * np.ones((3, 1))
    fit = profile_log_decay_grid(
        samples,
        truth,
        builder,
        (np.log(0.5), np.log(3.0)),
        grid_size=101,
    )
    assert fit.population_decay == pytest.approx(1.4, rel=2e-4)
    np.testing.assert_allclose(fit.sample_decays, 1.4, rtol=2e-4)
    assert not np.any(fit.sample_at_boundary)


def test_zero_bandwidth_is_exact_corrected_naive_control() -> None:
    result = run_finite_design_configuration(_configuration(0.0), seed=713)
    diagnostics = result.diagnostics
    assert diagnostics["corrected_truth_max_difference"] < 1e-12
    assert diagnostics["corrected_population_target"] == pytest.approx(0.8, rel=3e-4)
    assert diagnostics["naive_population_target"] == pytest.approx(0.8, rel=3e-4)
    corrected = [record["decay_estimate"] for record in result.records if record["model"] == "corrected"]
    naive = [record["decay_estimate"] for record in result.records if record["model"] == "naive"]
    np.testing.assert_allclose(corrected, naive, rtol=0.0, atol=1e-12)


def test_smoothing_shifts_naive_target_but_not_corrected_target() -> None:
    result = run_finite_design_configuration(_configuration(0.35), seed=917)
    diagnostics = result.diagnostics
    assert diagnostics["corrected_population_target"] == pytest.approx(0.8, rel=3e-4)
    assert diagnostics["naive_population_target"] < 0.8
    assert diagnostics["smoothing_rank"] == diagnostics["number_of_outputs"]
    assert diagnostics["smoothing_row_sum_error"] < 1e-14


def test_configuration_is_exactly_reproducible() -> None:
    first = run_finite_design_configuration(_configuration(0.35), seed=111)
    second = run_finite_design_configuration(_configuration(0.35), seed=111)
    assert first.records == second.records
    assert first.diagnostics == second.diagnostics
