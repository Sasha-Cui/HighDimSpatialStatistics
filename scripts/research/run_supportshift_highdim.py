"""Run the replicated-field track of the SupportShift synthetic benchmark.

For each spatial design, this script draws independent vectors

    X_i ~ N_p(0, S K_{v0, alpha0} S.T),  i = 1, ..., N,

where coordinates within a vector remain spatially dependent.  It fits a
deterministic finite grid of process variances and inverse ranges under both
the support-aware covariance and a naive point-support covariance.  The
reported likelihood deviations and certificate use exactly the same finite
candidate library.

The default ``full`` preset is intended for batch execution.  Use
``--preset shakedown`` for a quick local numerical audit.  No numerical jitter
is added anywhere in the experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist


FloatArray = NDArray[np.float64]
CovarianceBuilder = Callable[[float], FloatArray]


PRESETS: dict[str, dict[str, Any]] = {
    "full": {
        "grid_sides": [4, 6, 8, 10],
        "sample_sizes": [1, 4, 16, 64],
        "trials": 200,
        "smoothness": [0.5, 1.5],
        "decay_grid_size": 161,
        "variance_grid_size": 101,
        "grid_objective_gap_tolerance": 5e-5,
    },
    "shakedown": {
        "grid_sides": [2],
        "sample_sizes": [1, 3],
        "trials": 4,
        "smoothness": [0.5],
        "decay_grid_size": 5,
        "variance_grid_size": 3,
        "grid_objective_gap_tolerance": 0.1,
    },
}


def add_src_to_path() -> Path:
    """Add this checkout's source tree to ``sys.path`` and return its root."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    return repo_root


def write_csv_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace ``path`` with a nonempty rectangular CSV."""
    if not records:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(records[0])
    if any(list(record) != fieldnames for record in records):
        raise ValueError("every CSV record must have identical ordered fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace ``path`` with sorted, indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_metadata(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def derive_seed(root_seed: int, grid_side: int, smoothness: float) -> int:
    """Derive an order-independent configuration seed."""
    payload = f"{root_seed}:q={grid_side}:nu={float(smoothness).hex()}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def derive_child_seed(parent_seed: int, label: str) -> int:
    """Derive a deterministic substream without advancing the parent generator."""
    digest = hashlib.sha256(f"{parent_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def grid_with_anchor(
    lower: float,
    upper: float,
    size: int,
    anchor: float,
    *,
    name: str,
) -> FloatArray:
    """Return a positive geometric grid containing ``anchor`` exactly."""
    values = np.asarray([lower, upper, anchor], dtype=float)
    if not np.all(np.isfinite(values)) or lower <= 0.0 or not lower < anchor < upper:
        raise ValueError(f"{name} bounds must be positive and strictly contain its anchor")
    if isinstance(size, bool) or int(size) != size or size < 3:
        raise ValueError(f"{name} grid size must be an integer of at least three")
    size = int(size)
    left_count = size // 2
    right_count = size - left_count - 1
    left = np.geomspace(lower, anchor, left_count + 1)[:-1]
    right = np.geomspace(anchor, upper, right_count + 1)[1:]
    grid = np.concatenate([left, np.asarray([anchor]), right])
    if grid.size != size or not np.all(np.diff(grid) > 0.0):
        raise RuntimeError(f"failed to construct the {name} candidate grid")
    return grid


def _positive_int_list(values: Sequence[int], *, name: str) -> list[int]:
    numeric = np.asarray(values, dtype=float)
    if (
        numeric.ndim != 1
        or numeric.size == 0
        or not np.all(np.isfinite(numeric))
        or np.any(numeric <= 0.0)
        or np.any(numeric % 1.0 != 0.0)
    ):
        raise ValueError(f"{name} must contain positive integers")
    result = sorted({int(value) for value in numeric})
    return result


def _positive_float_list(values: Sequence[float], *, name: str) -> list[float]:
    numeric = np.asarray(values, dtype=float)
    if (
        numeric.ndim != 1
        or numeric.size == 0
        or not np.all(np.isfinite(numeric))
        or np.any(numeric <= 0.0)
    ):
        raise ValueError(f"{name} must contain positive finite values")
    return sorted({float(value) for value in numeric})


def _sample_covariances(samples: FloatArray, sample_sizes: Sequence[int]) -> FloatArray:
    """Compute trial-wise empirical covariances at nested sample sizes."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3:
        raise ValueError("samples must have shape (trials, max_N, p)")
    covariances = []
    for sample_size in sample_sizes:
        prefix = values[:, :sample_size, :]
        covariance = np.matmul(prefix.transpose(0, 2, 1), prefix) / sample_size
        covariances.append(covariance)
    return np.stack(covariances)


def _smoothed_correlation(
    latent_correlation: FloatArray,
    smoothing: csr_matrix,
) -> FloatArray:
    """Compute ``S K S.T`` using both sparse sides of the local operator."""
    left = np.asarray(smoothing @ latent_correlation, dtype=float)
    result = np.asarray(smoothing @ left.T, dtype=float)
    return (result + result.T) / 2.0


def _profiled_population_target(
    true_covariance: FloatArray,
    covariance_builder: CovarianceBuilder,
    decay_bounds: tuple[float, float],
    variance_bounds: tuple[float, float],
    *,
    fixed_variance: float | None,
) -> dict[str, Any]:
    """Minimize the population criterion after profiling process variance."""
    dimension = true_covariance.shape[0]

    def evaluate(log_decay: float) -> tuple[float, float]:
        base = covariance_builder(float(np.exp(log_decay)))
        factor = cholesky(base, lower=True, check_finite=False)
        precision_truth = cho_solve((factor, True), true_covariance, check_finite=False)
        trace_term = float(np.trace(precision_truth))
        if fixed_variance is None:
            variance = float(np.clip(trace_term / dimension, *variance_bounds))
        else:
            variance = float(fixed_variance)
        log_determinant = 2.0 * float(np.log(np.diag(factor)).sum())
        objective = 0.5 * (
            dimension * np.log(variance) + log_determinant + trace_term / variance
        ) / dimension
        return float(objective), variance

    log_bounds = tuple(float(np.log(value)) for value in decay_bounds)
    result = minimize_scalar(
        lambda log_decay: evaluate(float(log_decay))[0],
        bounds=log_bounds,
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 300},
    )
    objective, variance = evaluate(float(result.x))
    tolerance = max(1e-7 * (log_bounds[1] - log_bounds[0]), 1e-8)
    log_decay_boundary_margin = float(
        min(result.x - log_bounds[0], log_bounds[1] - result.x)
    )
    log_variance_boundary_margin = (
        None
        if fixed_variance is not None
        else float(
            min(
                np.log(variance / variance_bounds[0]),
                np.log(variance_bounds[1] / variance),
            )
        )
    )
    return {
        "decay": float(np.exp(result.x)),
        "variance": variance,
        "objective": objective,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "log_decay_boundary_margin": log_decay_boundary_margin,
        "log_variance_boundary_margin": log_variance_boundary_margin,
        "at_decay_lower_bound": bool(result.x - log_bounds[0] <= tolerance),
        "at_decay_upper_bound": bool(log_bounds[1] - result.x <= tolerance),
        "at_variance_lower_bound": bool(
            fixed_variance is None
            and np.isclose(variance, variance_bounds[0], rtol=0.0, atol=1e-10)
        ),
        "at_variance_upper_bound": bool(
            fixed_variance is None
            and np.isclose(variance, variance_bounds[1], rtol=0.0, atol=1e-10)
        ),
    }


def _evaluate_candidate_model(
    *,
    model: str,
    true_covariance: FloatArray,
    true_factor: FloatArray,
    samples: FloatArray,
    sample_covariances: FloatArray,
    sample_sizes: Sequence[int],
    decay_grid: FloatArray,
    variance_grid: FloatArray,
    covariance_builder: CovarianceBuilder,
    decay_bounds: tuple[float, float],
    variance_bounds: tuple[float, float],
    true_decay: float,
    true_variance: float,
    fit_mode: str,
    delta: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one product candidate grid with one factorization per decay."""
    dimension = true_covariance.shape[0]
    n_decays = decay_grid.size
    n_variances = variance_grid.size
    candidate_count = n_decays * n_variances
    n_sample_sizes, trials = sample_covariances.shape[:2]
    identity = np.eye(dimension)

    base_covariances: list[FloatArray] = []
    log_determinants = np.empty(n_decays)
    population_traces = np.empty(n_decays)
    sample_quadratics = np.empty((n_decays, n_sample_sizes, trials))
    frobenius_norms = np.empty(n_decays)
    operator_norms = np.empty(n_decays)
    stable_ranks = np.empty(n_decays)

    for decay_index, decay in enumerate(decay_grid):
        base = np.asarray(covariance_builder(float(decay)), dtype=float)
        factor = cholesky(base, lower=True, check_finite=False)
        precision = cho_solve((factor, True), identity, check_finite=False)
        precision = (precision + precision.T) / 2.0
        base_covariances.append(base)
        log_determinants[decay_index] = 2.0 * np.log(np.diag(factor)).sum()
        population_traces[decay_index] = np.einsum(
            "ij,ji->", precision, true_covariance, optimize=True
        )
        sample_quadratics[decay_index] = np.einsum(
            "ij,ntji->nt", precision, sample_covariances, optimize=True
        )
        relative = true_factor.T @ precision @ true_factor
        eigenvalues = np.linalg.eigvalsh((relative + relative.T) / 2.0)
        operator_norms[decay_index] = np.max(np.abs(eigenvalues))
        frobenius_norms[decay_index] = np.linalg.norm(eigenvalues)
        stable_ranks[decay_index] = (
            frobenius_norms[decay_index] / operator_norms[decay_index]
        ) ** 2

    log_variances = np.log(variance_grid)
    population_objectives = 0.5 / dimension * (
        dimension * log_variances[None, :]
        + log_determinants[:, None]
        + population_traces[:, None] / variance_grid[None, :]
    )
    sample_objectives = 0.5 / dimension * (
        dimension * log_variances[None, :, None, None]
        + log_determinants[:, None, None, None]
        + sample_quadratics[:, None, :, :] / variance_grid[None, :, None, None]
    )
    criterion_deviations = 0.5 / dimension * (
        sample_quadratics[:, None, :, :] - population_traces[:, None, None, None]
    ) / variance_grid[None, :, None, None]

    population_flat = population_objectives.reshape(candidate_count)
    sample_flat = sample_objectives.reshape(candidate_count, n_sample_sizes, trials)
    deviation_flat = criterion_deviations.reshape(
        candidate_count, n_sample_sizes, trials
    )
    population_index = int(np.argmin(population_flat))
    population_decay_index = population_index // n_variances
    population_variance_index = population_index % n_variances
    estimate_indices = np.argmin(sample_flat, axis=0)
    estimate_decay_indices = estimate_indices // n_variances
    estimate_variance_indices = estimate_indices % n_variances

    max_deviations = np.max(np.abs(deviation_flat), axis=0)
    sample_at_estimate = np.take_along_axis(
        sample_flat, estimate_indices[None, :, :], axis=0
    )[0]
    population_at_estimate = population_flat[estimate_indices]
    deviation_at_estimate = np.take_along_axis(
        deviation_flat, estimate_indices[None, :, :], axis=0
    )[0]
    population_excess = population_at_estimate - population_flat[population_index]
    sample_at_population_target = sample_flat[population_index]
    deviation_at_population_target = deviation_flat[population_index]

    log_factor = float(np.log(2.0 * candidate_count / delta))
    scaled_frobenius = frobenius_norms[:, None] / variance_grid[None, :]
    scaled_operator = operator_norms[:, None] / variance_grid[None, :]
    bounds = np.empty((n_sample_sizes, n_decays, n_variances))
    worst_indices = np.empty(n_sample_sizes, dtype=int)
    for sample_index, sample_size in enumerate(sample_sizes):
        bounds[sample_index] = (
            scaled_frobenius / dimension * np.sqrt(log_factor / sample_size)
            + scaled_operator / dimension * log_factor / sample_size
        )
        worst_indices[sample_index] = int(np.argmax(bounds[sample_index]))
    candidate_bounds = bounds.reshape(n_sample_sizes, candidate_count).T[:, :, None]
    candidatewise_bound_ratios = np.abs(deviation_flat) / candidate_bounds
    maximum_candidatewise_bound_ratios = np.max(
        candidatewise_bound_ratios,
        axis=0,
    )
    simultaneous_candidatewise_bound_holds = np.all(
        np.abs(deviation_flat) <= candidate_bounds + 1e-12,
        axis=0,
    )

    continuous_target = _profiled_population_target(
        true_covariance,
        covariance_builder,
        decay_bounds,
        variance_bounds,
        fixed_variance=(float(variance_grid[0]) if fit_mode == "fixed" else None),
    )
    if not continuous_target["success"]:
        raise RuntimeError(
            f"{model} continuous population optimization failed: "
            f"{continuous_target['message']}"
        )

    # Audit the optimized product-grid algebra against the public theorem API
    # on three candidates and one actual replicated dataset.
    from HighDimSpatial.smoothing_bias.high_dimensional import (
        normalized_gaussian_population_objectives,
        normalized_gaussian_sample_objectives,
    )

    audit_indices = sorted({0, population_index, candidate_count - 1})
    audit_covariances = np.stack(
        [
            variance_grid[index % n_variances]
            * base_covariances[index // n_variances]
            for index in audit_indices
        ]
    )
    api_population = normalized_gaussian_population_objectives(
        true_covariance, audit_covariances
    )
    api_sample = normalized_gaussian_sample_objectives(
        samples[0, : sample_sizes[-1]], audit_covariances
    )
    optimized_population = population_flat[audit_indices]
    optimized_sample = sample_flat[audit_indices, -1, 0]
    criterion_api_max_difference = float(
        max(
            np.max(np.abs(api_population - optimized_population)),
            np.max(np.abs(api_sample - optimized_sample)),
        )
    )
    criterion_api_scale = float(
        max(1.0, np.max(np.abs(api_population)), np.max(np.abs(api_sample)))
    )
    criterion_api_tolerance = 1e-8 * criterion_api_scale
    if criterion_api_max_difference > criterion_api_tolerance:
        raise RuntimeError(
            "optimized likelihood algebra disagrees with theorem API: "
            f"model={model}, max_difference={criterion_api_max_difference:.3e}, "
            f"tolerance={criterion_api_tolerance:.3e}"
        )

    records: list[dict[str, Any]] = []
    for sample_index, sample_size in enumerate(sample_sizes):
        bound = float(bounds[sample_index].ravel()[worst_indices[sample_index]])
        worst_decay_index = worst_indices[sample_index] // n_variances
        worst_variance_index = worst_indices[sample_index] % n_variances
        for trial in range(trials):
            decay_index = int(estimate_decay_indices[sample_index, trial])
            variance_index = int(estimate_variance_indices[sample_index, trial])
            decay_estimate = float(decay_grid[decay_index])
            variance_estimate = float(variance_grid[variance_index])
            grid_decay_target = float(decay_grid[population_decay_index])
            grid_variance_target = float(variance_grid[population_variance_index])
            maximum_deviation = float(max_deviations[sample_index, trial])
            excess = float(population_excess[sample_index, trial])
            records.append(
                {
                    "model": model,
                    "sample_size": sample_size,
                    "sample_size_times_dimension": sample_size * dimension,
                    "inverse_sqrt_sample_size_times_dimension": float(
                        1.0 / np.sqrt(sample_size * dimension)
                    ),
                    "trial": trial,
                    "decay_estimate": decay_estimate,
                    "variance_estimate": variance_estimate,
                    "range_estimate": float(1.0 / decay_estimate),
                    "population_grid_decay_target": grid_decay_target,
                    "population_grid_variance_target": grid_variance_target,
                    "population_continuous_decay_target": continuous_target["decay"],
                    "population_continuous_variance_target": continuous_target["variance"],
                    "decay_error_to_truth": decay_estimate - true_decay,
                    "variance_error_to_truth": variance_estimate - true_variance,
                    "decay_error_to_grid_target": decay_estimate - grid_decay_target,
                    "variance_error_to_grid_target": variance_estimate
                    - grid_variance_target,
                    "sample_objective_at_estimate": float(
                        sample_at_estimate[sample_index, trial]
                    ),
                    "population_objective_at_estimate": float(
                        population_at_estimate[sample_index, trial]
                    ),
                    "criterion_deviation_at_estimate": float(
                        deviation_at_estimate[sample_index, trial]
                    ),
                    "sample_objective_at_population_target": float(
                        sample_at_population_target[sample_index, trial]
                    ),
                    "population_objective_at_population_target": float(
                        population_flat[population_index]
                    ),
                    "criterion_deviation_at_population_target": float(
                        deviation_at_population_target[sample_index, trial]
                    ),
                    "max_abs_criterion_deviation": maximum_deviation,
                    "population_excess_at_estimate": excess,
                    "erm_deterministic_excess_limit": 2.0 * maximum_deviation,
                    "erm_inequality_holds": bool(excess <= 2.0 * maximum_deviation + 1e-12),
                    "uniform_likelihood_bound": bound,
                    "uniform_excess_bound": 2.0 * bound,
                    "max_candidatewise_deviation_to_bound_ratio": float(
                        maximum_candidatewise_bound_ratios[sample_index, trial]
                    ),
                    "simultaneous_candidatewise_bound_holds": bool(
                        simultaneous_candidatewise_bound_holds[sample_index, trial]
                    ),
                    "uniform_bound_holds": bool(maximum_deviation <= bound + 1e-12),
                    "population_excess_within_uniform_bound": bool(
                        excess <= 2.0 * bound + 1e-12
                    ),
                    "finite_grid_oracle_selected": bool(
                        int(estimate_indices[sample_index, trial]) == population_index
                    ),
                    "worst_bound_decay": float(decay_grid[worst_decay_index]),
                    "worst_bound_variance": float(variance_grid[worst_variance_index]),
                    "worst_bound_frobenius_norm": float(
                        scaled_frobenius[worst_decay_index, worst_variance_index]
                    ),
                    "worst_bound_operator_norm": float(
                        scaled_operator[worst_decay_index, worst_variance_index]
                    ),
                    "worst_bound_stable_rank": float(stable_ranks[worst_decay_index]),
                }
            )

    diagnostics = {
        "model": model,
        "candidate_count": candidate_count,
        "factorizations_per_model": n_decays,
        "population_grid_decay_target": float(decay_grid[population_decay_index]),
        "population_grid_variance_target": float(variance_grid[population_variance_index]),
        "population_grid_objective": float(population_flat[population_index]),
        "population_grid_objective_gap": float(
            population_flat[population_index] - continuous_target["objective"]
        ),
        "population_grid_at_decay_bound": bool(
            population_decay_index in {0, n_decays - 1}
        ),
        "population_grid_at_variance_bound": bool(
            fit_mode == "joint" and population_variance_index in {0, n_variances - 1}
        ),
        "population_grid_decay_index_margin": int(
            min(population_decay_index, n_decays - 1 - population_decay_index)
        ),
        "population_grid_variance_index_margin": (
            None
            if fit_mode == "fixed"
            else int(
                min(
                    population_variance_index,
                    n_variances - 1 - population_variance_index,
                )
            )
        ),
        "population_grid_log_decay_boundary_margin": float(
            min(
                np.log(decay_grid[population_decay_index] / decay_bounds[0]),
                np.log(decay_bounds[1] / decay_grid[population_decay_index]),
            )
        ),
        "population_grid_log_variance_boundary_margin": (
            None
            if fit_mode == "fixed"
            else float(
                min(
                    np.log(
                        variance_grid[population_variance_index] / variance_bounds[0]
                    ),
                    np.log(
                        variance_bounds[1] / variance_grid[population_variance_index]
                    ),
                )
            )
        ),
        "continuous_population_target": continuous_target,
        "criterion_api_max_difference": criterion_api_max_difference,
        "criterion_api_tolerance": criterion_api_tolerance,
        "minimum_relative_stable_rank": float(np.min(stable_ranks)),
        "maximum_relative_stable_rank": float(np.max(stable_ranks)),
        "coverage_by_sample_size": {
            str(sample_size): float(
                np.mean(simultaneous_candidatewise_bound_holds[index])
            )
            for index, sample_size in enumerate(sample_sizes)
        },
        "worst_envelope_coverage_by_sample_size": {
            str(sample_size): float(
                np.mean(max_deviations[index] <= np.max(bounds[index]) + 1e-12)
            )
            for index, sample_size in enumerate(sample_sizes)
        },
        "oracle_selection_by_sample_size": {
            str(sample_size): float(np.mean(estimate_indices[index] == population_index))
            for index, sample_size in enumerate(sample_sizes)
        },
    }
    return records, diagnostics


def parse_arguments(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the high-dimensional replicated-field SupportShift benchmark."
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--grid-sides", type=int, nargs="+")
    parser.add_argument("--sample-sizes", type=int, nargs="+")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--smoothness", type=float, nargs="+")
    parser.add_argument("--fit-mode", choices=["joint", "fixed"], default="joint")
    parser.add_argument("--true-variance", type=float, default=1.0)
    parser.add_argument("--true-decay", type=float, default=1.0)
    parser.add_argument("--decay-minimum", type=float, default=0.15)
    parser.add_argument("--decay-maximum", type=float, default=1.6)
    parser.add_argument("--decay-grid-size", type=int)
    parser.add_argument("--variance-minimum", type=float, default=0.35)
    parser.add_argument("--variance-maximum", type=float, default=2.5)
    parser.add_argument("--variance-grid-size", type=int)
    parser.add_argument("--grid-objective-gap-tolerance", type=float)
    parser.add_argument("--input-spacing", type=float, default=0.25)
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--coverage-floor", type=float, default=0.90)
    parser.add_argument("--root-seed", type=int, default=20260803)
    parser.add_argument("--raw-example-output", type=Path)
    parser.add_argument(
        "--raw-example-grid-side",
        type=int,
        help="Grid side for the raw latent/averaged illustration (default: largest side).",
    )
    parser.add_argument("--raw-example-replicates", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    for name in (
        "grid_sides",
        "sample_sizes",
        "trials",
        "smoothness",
        "decay_grid_size",
        "variance_grid_size",
        "grid_objective_gap_tolerance",
    ):
        if getattr(args, name) is None:
            setattr(args, name, preset[name])
    if args.output is None:
        filename = (
            "supportshift_highdim_v1.csv"
            if args.preset == "full"
            else "supportshift_highdim_shakedown.csv"
        )
        args.output = repo_root / "outputs" / "smoothing_bias" / filename
    return args


def _resolved_settings(args: argparse.Namespace) -> dict[str, Any]:
    grid_sides = _positive_int_list(args.grid_sides, name="grid_sides")
    if min(grid_sides) < 2:
        raise ValueError("grid_sides must be at least two")
    sample_sizes = _positive_int_list(args.sample_sizes, name="sample_sizes")
    smoothness = _positive_float_list(args.smoothness, name="smoothness")
    if isinstance(args.trials, bool) or args.trials <= 0:
        raise ValueError("trials must be a positive integer")
    if not np.isfinite(args.input_spacing) or args.input_spacing <= 0.0:
        raise ValueError("input_spacing must be finite and positive")
    if (
        not np.isfinite(args.bandwidth)
        or args.bandwidth <= args.input_spacing
        or args.bandwidth > 2.0 * args.input_spacing
    ):
        raise ValueError(
            "bandwidth must exceed one input spacing and not exceed the two-spacing pad"
        )
    if not np.isfinite(args.delta) or not 0.0 < args.delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    if not np.isfinite(args.coverage_floor) or not 0.0 <= args.coverage_floor <= 1.0:
        raise ValueError("coverage_floor must lie between zero and one")
    if (
        not np.isfinite(args.grid_objective_gap_tolerance)
        or args.grid_objective_gap_tolerance < 0.0
    ):
        raise ValueError("grid_objective_gap_tolerance must be finite and nonnegative")
    if args.true_variance <= 0.0 or args.true_decay <= 0.0:
        raise ValueError("true variance and decay must be positive")
    if args.raw_example_replicates <= 0:
        raise ValueError("raw_example_replicates must be positive")
    raw_example_grid_side = (
        max(grid_sides)
        if args.raw_example_grid_side is None
        else int(args.raw_example_grid_side)
    )
    if raw_example_grid_side not in grid_sides:
        raise ValueError("raw_example_grid_side must be one of the requested grid_sides")
    variance_grid_size = 1 if args.fit_mode == "fixed" else int(args.variance_grid_size)
    settings = {
        "benchmark": "SupportShift",
        "benchmark_version": "1.1",
        "preset": args.preset,
        "grid_sides": grid_sides,
        "sample_sizes": sample_sizes,
        "trials": int(args.trials),
        "smoothness": smoothness,
        "fit_mode": args.fit_mode,
        "true_variance": float(args.true_variance),
        "true_decay": float(args.true_decay),
        "decay_minimum": float(args.decay_minimum),
        "decay_maximum": float(args.decay_maximum),
        "decay_grid_size": int(args.decay_grid_size),
        "variance_minimum": float(args.variance_minimum),
        "variance_maximum": float(args.variance_maximum),
        "variance_grid_size": variance_grid_size,
        "input_spacing": float(args.input_spacing),
        "bandwidth": float(args.bandwidth),
        "delta": float(args.delta),
        "coverage_floor": float(args.coverage_floor),
        "grid_objective_gap_tolerance": float(args.grid_objective_gap_tolerance),
        "root_seed": int(args.root_seed),
        "raw_example_grid_side": raw_example_grid_side,
    }
    return settings


def _validation_gates(
    records: list[dict[str, Any]],
    configuration_diagnostics: list[dict[str, Any]],
    settings: dict[str, Any],
    raw_details: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate predeclared numerical and statistical benchmark gates."""
    nonfinite: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        for field, value in record.items():
            if isinstance(value, Real) and not np.isfinite(value):
                nonfinite.append({"row": row_index, "field": field, "value": str(value)})
    keys = [
        (
            record["config_id"],
            record["model"],
            record["sample_size"],
            record["trial"],
        )
        for record in records
    ]

    corrected_oracle_failures: list[str] = []
    boundary_failures: list[str] = []
    api_failures: list[str] = []
    grid_approximation_failures: dict[str, float] = {}
    for configuration in configuration_diagnostics:
        for model in configuration["models"]:
            label = f"{configuration['config_id']}:{model['model']}"
            if model["model"] == "corrected" and (
                model["population_grid_decay_target"] != settings["true_decay"]
                or model["population_grid_variance_target"] != settings["true_variance"]
            ):
                corrected_oracle_failures.append(label)
            continuous = model["continuous_population_target"]
            decay_at_boundary = (
                model["population_grid_at_decay_bound"]
                or continuous["at_decay_lower_bound"]
                or continuous["at_decay_upper_bound"]
            )
            variance_at_boundary = settings["fit_mode"] == "joint" and (
                model["population_grid_at_variance_bound"]
                or continuous["at_variance_lower_bound"]
                or continuous["at_variance_upper_bound"]
            )
            if decay_at_boundary or variance_at_boundary:
                boundary_failures.append(label)
            if model["criterion_api_max_difference"] > model["criterion_api_tolerance"]:
                api_failures.append(label)
            objective_gap = model["population_grid_objective_gap"]
            if abs(objective_gap) > settings["grid_objective_gap_tolerance"] + 1e-15:
                grid_approximation_failures[label] = objective_gap

    coverage_cells: dict[tuple[str, str, int], list[bool]] = {}
    for record in records:
        key = (record["config_id"], record["model"], int(record["sample_size"]))
        coverage_cells.setdefault(key, []).append(
            bool(record["simultaneous_candidatewise_bound_holds"])
        )
    coverage_rates = {
        f"{config_id}:{model}:N={sample_size}": float(np.mean(values))
        for (config_id, model, sample_size), values in sorted(coverage_cells.items())
    }
    coverage_failures = {
        cell: coverage
        for cell, coverage in coverage_rates.items()
        if coverage + 1e-15 < settings["coverage_floor"]
    }
    erm_failures = sum(not bool(record["erm_inequality_holds"]) for record in records)
    raw_tolerance = 1e-12
    raw_difference = None if raw_details is None else raw_details["max_reapplication_difference"]
    raw_passed = raw_details is None or raw_difference <= raw_tolerance

    gates: dict[str, Any] = {
        "corrected_grid_oracle_is_exact_truth": {
            "passed": not corrected_oracle_failures,
            "failures": corrected_oracle_failures,
        },
        "population_oracles_are_interior": {
            "passed": not boundary_failures,
            "failures": boundary_failures,
            "margins_recorded_in": "configuration_diagnostics.models",
        },
        "result_rows_are_finite_and_unique": {
            "passed": not nonfinite and len(set(keys)) == len(keys),
            "nonfinite_values": nonfinite[:20],
            "row_count": len(keys),
            "unique_key_count": len(set(keys)),
            "key": ["config_id", "model", "sample_size", "trial"],
        },
        "exact_erm_deterministic_inequality": {
            "passed": erm_failures == 0,
            "failure_count": erm_failures,
            "inequality": "population_excess <= 2 * max_abs_criterion_deviation",
        },
        "empirical_uniform_bound_coverage": {
            "passed": not coverage_failures,
            "predeclared_floor": settings["coverage_floor"],
            "cell_coverage": coverage_rates,
            "failures": coverage_failures,
            "event": "all candidates satisfy their own theorem radius",
        },
        "public_likelihood_api_algebra": {
            "passed": not api_failures,
            "failures": api_failures,
        },
        "finite_grid_approximates_continuous_oracle": {
            "passed": not grid_approximation_failures,
            "maximum_absolute_normalized_nll_gap": max(
                (
                    abs(model["population_grid_objective_gap"])
                    for configuration in configuration_diagnostics
                    for model in configuration["models"]
                ),
                default=0.0,
            ),
            "tolerance": settings["grid_objective_gap_tolerance"],
            "failures": grid_approximation_failures,
        },
        "raw_s_reapplication": {
            "passed": raw_passed,
            "requested": raw_details is not None,
            "max_difference": raw_difference,
            "tolerance": raw_tolerance,
        },
    }
    gates["all_passed"] = all(gate["passed"] for gate in gates.values())
    return gates


def main() -> None:
    repo_root = add_src_to_path()
    from HighDimSpatial.smoothing_bias.design import (
        epanechnikov_smoothing_matrix,
        regular_grid_2d,
        select_rectangular_centers,
    )
    from HighDimSpatial.smoothing_bias.kl import (
        exact_smoothed_covariance,
        matern_correlation,
    )

    args = parse_arguments(repo_root)
    settings = _resolved_settings(args)
    started = datetime.now(timezone.utc)
    commit, dirty = git_metadata(repo_root)

    decay_grid = grid_with_anchor(
        settings["decay_minimum"],
        settings["decay_maximum"],
        settings["decay_grid_size"],
        settings["true_decay"],
        name="decay",
    )
    if settings["fit_mode"] == "joint":
        variance_grid = grid_with_anchor(
            settings["variance_minimum"],
            settings["variance_maximum"],
            settings["variance_grid_size"],
            settings["true_variance"],
            name="variance",
        )
    else:
        variance_grid = np.asarray([settings["true_variance"]], dtype=float)

    all_records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    raw_details: dict[str, Any] | None = None
    configuration_diagnostics: list[dict[str, Any]] = []
    maximum_sample_size = max(settings["sample_sizes"])

    for grid_side in settings["grid_sides"]:
        latent_shape = 2 * grid_side + 3
        inputs = regular_grid_2d(latent_shape, settings["input_spacing"])
        centers = select_rectangular_centers(
            inputs,
            stride=2,
            boundary_trim=2.0 * settings["input_spacing"],
        )
        dimension = grid_side**2
        if centers.shape != (dimension, 2):
            raise RuntimeError("the padded latent grid did not produce the requested p")
        smoothing_dense = epanechnikov_smoothing_matrix(
            inputs,
            centers,
            settings["bandwidth"],
            kernel="radial",
        )
        smoothing = csr_matrix(smoothing_dense)
        if np.linalg.matrix_rank(smoothing_dense) != dimension:
            raise RuntimeError("the smoothing operator must have full row rank")
        latent_distances = cdist(inputs, inputs)
        output_distances = cdist(centers, centers)

        for smoothness in settings["smoothness"]:
            config_id = f"q{grid_side}_nu{smoothness:g}"
            seed = derive_seed(settings["root_seed"], grid_side, smoothness)
            rng = np.random.default_rng(seed)

            def corrected_base(candidate_decay: float) -> FloatArray:
                latent = np.asarray(
                    matern_correlation(latent_distances, candidate_decay, smoothness),
                    dtype=float,
                )
                return _smoothed_correlation(latent, smoothing)

            def naive_base(candidate_decay: float) -> FloatArray:
                correlation = np.asarray(
                    matern_correlation(output_distances, candidate_decay, smoothness),
                    dtype=float,
                )
                return (correlation + correlation.T) / 2.0

            true_base = corrected_base(settings["true_decay"])
            true_covariance = settings["true_variance"] * true_base
            true_factor = cholesky(true_covariance, lower=True, check_finite=False)
            standard_normals = rng.standard_normal(
                (settings["trials"], maximum_sample_size, dimension)
            )
            samples = standard_normals @ true_factor.T
            sample_covariances = _sample_covariances(samples, settings["sample_sizes"])

            api_truth = exact_smoothed_covariance(
                inputs,
                smoothing_dense,
                settings["true_variance"],
                settings["true_decay"],
                smoothness,
                0.0,
            )
            exact_api_max_difference = float(
                np.max(np.abs(api_truth - true_covariance))
            )
            if exact_api_max_difference > 5e-12:
                raise RuntimeError("sparse exact covariance disagrees with the public API")

            if (
                args.raw_example_output is not None
                and grid_side == settings["raw_example_grid_side"]
                and not raw_records
            ):
                raw_replicates = min(args.raw_example_replicates, maximum_sample_size)
                raw_seed = derive_child_seed(seed, "raw-latent-illustration")
                raw_rng = np.random.default_rng(raw_seed)
                latent_true_correlation = np.asarray(
                    matern_correlation(
                        latent_distances, settings["true_decay"], smoothness
                    ),
                    dtype=float,
                )
                latent_true_covariance = (
                    settings["true_variance"] * latent_true_correlation
                )
                latent_true_factor = cholesky(
                    latent_true_covariance, lower=True, check_finite=False
                )
                raw_latent = (
                    raw_rng.standard_normal((raw_replicates, inputs.shape[0]))
                    @ latent_true_factor.T
                )
                raw_averaged = np.asarray(raw_latent @ smoothing_dense.T, dtype=float)
                for replicate in range(raw_replicates):
                    for field_stage, locations, field_values in (
                        ("latent_input", inputs, raw_latent[replicate]),
                        ("averaged_output", centers, raw_averaged[replicate]),
                    ):
                        for location_index, (coordinate, value) in enumerate(
                            zip(locations, field_values, strict=True)
                        ):
                            raw_records.append(
                                {
                                    "benchmark": settings["benchmark"],
                                    "benchmark_version": settings["benchmark_version"],
                                    "config_id": config_id,
                                    "sample_role": "standalone_raw_illustration",
                                    "field_stage": field_stage,
                                    "replicate": replicate,
                                    "location_index": location_index,
                                    "x": float(coordinate[0]),
                                    "y": float(coordinate[1]),
                                    "value": float(value),
                                    "data_generating_model": "latent_matern_then_exact_S",
                                    "fitted_models": "corrected;naive",
                                    "grid_side": grid_side,
                                    "dimension_p": dimension,
                                    "smoothness": smoothness,
                                    "decay_true": settings["true_decay"],
                                    "variance_true": settings["true_variance"],
                                    "bandwidth": settings["bandwidth"],
                                    "seed": raw_seed,
                                }
                            )
                reapplied = raw_latent @ smoothing_dense.T
                raw_details = {
                    "config_id": config_id,
                    "seed": raw_seed,
                    "replicates": raw_replicates,
                    "latent_locations_per_replicate": int(inputs.shape[0]),
                    "averaged_locations_per_replicate": dimension,
                    "max_reapplication_difference": float(
                        np.max(np.abs(raw_averaged - reapplied))
                    ),
                }

            model_diagnostics: list[dict[str, Any]] = []
            for model, builder in (
                ("corrected", corrected_base),
                ("naive", naive_base),
            ):
                model_records, diagnostics = _evaluate_candidate_model(
                    model=model,
                    true_covariance=true_covariance,
                    true_factor=true_factor,
                    samples=samples,
                    sample_covariances=sample_covariances,
                    sample_sizes=settings["sample_sizes"],
                    decay_grid=decay_grid,
                    variance_grid=variance_grid,
                    covariance_builder=builder,
                    decay_bounds=(
                        settings["decay_minimum"],
                        settings["decay_maximum"],
                    ),
                    variance_bounds=(
                        settings["variance_minimum"],
                        settings["variance_maximum"],
                    ),
                    true_decay=settings["true_decay"],
                    true_variance=settings["true_variance"],
                    fit_mode=settings["fit_mode"],
                    delta=settings["delta"],
                )
                common = {
                    "benchmark": settings["benchmark"],
                    "benchmark_version": settings["benchmark_version"],
                    "config_hash": canonical_hash(settings),
                    "config_id": config_id,
                    "fit_mode": settings["fit_mode"],
                    "grid_side": grid_side,
                    "dimension_p": dimension,
                    "smoothness": smoothness,
                    "decay_true": settings["true_decay"],
                    "variance_true": settings["true_variance"],
                    "range_true": 1.0 / settings["true_decay"],
                    "bandwidth": settings["bandwidth"],
                    "input_spacing": settings["input_spacing"],
                    "candidate_count": int(decay_grid.size * variance_grid.size),
                    "decay_candidate_count": int(decay_grid.size),
                    "variance_candidate_count": int(variance_grid.size),
                    "delta": settings["delta"],
                    "seed": seed,
                    "git_commit": commit,
                    "git_dirty": dirty,
                }
                all_records.extend([{**common, **record} for record in model_records])
                model_diagnostics.append(diagnostics)

            configuration_diagnostics.append(
                {
                    "config_id": config_id,
                    "grid_side": grid_side,
                    "dimension_p": dimension,
                    "latent_grid_side": latent_shape,
                    "latent_location_count": int(inputs.shape[0]),
                    "smoothness": smoothness,
                    "derived_seed": seed,
                    "smoothing_nonzeros": int(smoothing.nnz),
                    "smoothing_row_sum_max_error": float(
                        np.max(np.abs(smoothing_dense.sum(axis=1) - 1.0))
                    ),
                    "true_covariance_condition": float(np.linalg.cond(true_covariance)),
                    "exact_covariance_api_max_difference": exact_api_max_difference,
                    "models": model_diagnostics,
                }
            )

    expected_rows = (
        len(settings["grid_sides"])
        * len(settings["smoothness"])
        * 2
        * len(settings["sample_sizes"])
        * settings["trials"]
    )
    if len(all_records) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows but constructed {len(all_records)}")
    write_csv_atomic(args.output, all_records)
    raw_artifact: dict[str, Any] | None = None
    if args.raw_example_output is not None:
        write_csv_atomic(args.raw_example_output, raw_records)
        raw_artifact = {
            "path": str(args.raw_example_output.resolve()),
            "rows": len(raw_records),
            "sha256": sha256_file(args.raw_example_output),
            **(raw_details or {}),
        }

    validation_gates = _validation_gates(
        all_records,
        configuration_diagnostics,
        settings,
        raw_details,
    )

    completed = datetime.now(timezone.utc)
    metadata = {
        "benchmark": settings["benchmark"],
        "benchmark_version": settings["benchmark_version"],
        "created_utc": completed.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "resolved_settings": settings,
        "config_hash": canonical_hash(settings),
        "candidate_grids": {
            "decay": decay_grid.tolist(),
            "variance": variance_grid.tolist(),
            "cartesian_order": "decay-major, variance-minor",
        },
        "statistical_model": {
            "data": "iid N_p(0, S K_{v0,alpha0} S^T) replicated spatial fields",
            "corrected_candidates": "v S R_alpha S^T",
            "naive_candidates": "v R_alpha evaluated at output centers",
            "objective": "average zero-mean Gaussian NLL divided by p, constant omitted",
            "mean": "known and fixed at zero",
            "finite_grid_estimator": (
                "exact Cartesian-grid argmin by enumeration; no interpolation or "
                "continuous sample optimization"
            ),
            "bound_scope": (
                "for each fixed configuration, model, and N separately; simultaneous over "
                "the complete deterministic variance-decay candidate grid"
            ),
            "nugget": 0.0,
        },
        "rows": len(all_records),
        "expected_rows": expected_rows,
        "result_csv": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
        "raw_example": raw_artifact,
        "configuration_diagnostics": configuration_diagnostics,
        "validation_gates": validation_gates,
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "git_commit": commit,
            "git_dirty": dirty,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    write_json_atomic(metadata_path, metadata)
    if not validation_gates["all_passed"]:
        failed = [
            name
            for name, gate in validation_gates.items()
            if name != "all_passed" and not gate["passed"]
        ]
        raise RuntimeError(
            f"SupportShift validation gates failed after metadata write: {failed}"
        )
    print(
        f"Wrote {len(all_records)} benchmark rows to {args.output} in "
        f"{metadata['elapsed_seconds']:.2f} seconds"
    )
    if raw_artifact is not None:
        print(f"Wrote {raw_artifact['rows']} raw example rows to {args.raw_example_output}")


if __name__ == "__main__":
    main()
