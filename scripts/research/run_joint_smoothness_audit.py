"""Joint smoothness--decay SupportShift audit with an intermediate support model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.linalg import cho_solve, cholesky


def add_src_to_path() -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root / "src"))
    return repository_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _candidate_library(
    *,
    latent_locations: np.ndarray,
    output_locations: np.ndarray,
    smoothing: np.ndarray | None,
    smoothness_grid: np.ndarray,
    decay_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    from HighDimSpatial.smoothing_bias.kl import matern_covariance

    dimension = output_locations.shape[0]
    count = smoothness_grid.size * decay_grid.size
    precisions = np.empty((count, dimension, dimension), dtype=float)
    log_determinants = np.empty(count, dtype=float)
    condition_numbers = np.empty(count, dtype=float)
    candidate_smoothness = np.empty(count, dtype=float)
    candidate_decay = np.empty(count, dtype=float)
    index = 0
    for smoothness in smoothness_grid:
        for decay in decay_grid:
            locations = output_locations if smoothing is None else latent_locations
            base = matern_covariance(
                locations, variance=1.0, decay=float(decay), nu=float(smoothness)
            )
            if smoothing is not None:
                base = smoothing @ base @ smoothing.T
            factor = cholesky(base, lower=True, check_finite=False)
            precisions[index] = cho_solve(
                (factor, True), np.eye(dimension), check_finite=False
            )
            log_determinants[index] = 2.0 * np.log(np.diag(factor)).sum()
            condition_numbers[index] = np.linalg.cond(base)
            candidate_smoothness[index] = smoothness
            candidate_decay[index] = decay
            index += 1
    return {
        "precisions": precisions,
        "log_determinants": log_determinants,
        "condition_numbers": condition_numbers,
        "smoothness": candidate_smoothness,
        "decay": candidate_decay,
    }


def _profile_library(
    library: dict[str, np.ndarray],
    second_moments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moments = np.asarray(second_moments, dtype=float)
    if moments.ndim == 2:
        moments = moments[None, :, :]
    dimension = moments.shape[1]
    traces = np.einsum(
        "kij,tji->kt", library["precisions"], moments, optimize=True
    )
    variances = traces / dimension
    objectives = 0.5 / dimension * (
        library["log_determinants"][:, None]
        + dimension * np.log(variances)
        + dimension
    )
    indices = np.argmin(objectives, axis=0)
    selected_variances = variances[indices, np.arange(moments.shape[0])]
    selected_objectives = objectives[indices, np.arange(moments.shape[0])]
    return indices, selected_variances, selected_objectives


def main() -> None:
    repository_root = add_src_to_path()
    from HighDimSpatial.smoothing_bias.design import (
        epanechnikov_smoothing_matrix,
        regular_grid_2d,
        select_rectangular_centers,
    )
    from HighDimSpatial.smoothing_bias.kl import matern_covariance

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-side", type=int, default=11)
    parser.add_argument("--spacing", type=float, default=0.25)
    parser.add_argument("--boundary-trim", type=float, default=0.5)
    parser.add_argument(
        "--true-smoothness", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.5]
    )
    parser.add_argument("--bandwidths", type=float, nargs="+", default=[0.35, 0.5])
    parser.add_argument("--partial-bandwidth-fraction", type=float, default=0.75)
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root
        / "outputs/smoothing_bias/supportshift_joint_smoothness.csv",
    )
    args = parser.parse_args()
    if args.input_side < 7 or args.spacing <= 0.0 or args.boundary_trim < 0.0:
        raise ValueError("the finite design arguments are invalid")
    if args.sample_size <= 0 or args.replicates <= 1:
        raise ValueError("sample size must be positive and replicates must exceed one")
    if not 0.0 < args.partial_bandwidth_fraction < 1.0:
        raise ValueError("partial bandwidth fraction must lie strictly between zero and one")
    true_smoothness = sorted(set(args.true_smoothness))
    bandwidths = sorted(set(args.bandwidths))
    if any(value <= 0.0 for value in true_smoothness + bandwidths):
        raise ValueError("smoothness and bandwidth values must be positive")

    smoothness_grid = np.asarray(
        [
            0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5,
            1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0,
        ],
        dtype=float,
    )
    smoothness_grid = np.unique(np.concatenate((smoothness_grid, true_smoothness)))
    decay_grid = np.unique(np.concatenate((np.geomspace(0.7, 2.7, 51), [1.0])))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository_root, check=True,
        capture_output=True, text=True,
    ).stdout
    dirty = bool(status.strip())
    if dirty and not args.allow_dirty:
        raise RuntimeError("refusing to generate a promoted audit from a dirty worktree")

    latent = regular_grid_2d(args.input_side, spacing=args.spacing)
    outputs = select_rectangular_centers(
        latent, stride=2, boundary_trim=args.boundary_trim
    )
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    maximum_condition_number = 0.0
    for bandwidth in bandwidths:
        true_smoothing = epanechnikov_smoothing_matrix(latent, outputs, bandwidth)
        model_smoothing = {
            "support_aware": true_smoothing,
            "partial_support": epanechnikov_smoothing_matrix(
                latent, outputs, args.partial_bandwidth_fraction * bandwidth
            ),
            "point_support": None,
        }
        libraries = {
            model: _candidate_library(
                latent_locations=latent,
                output_locations=outputs,
                smoothing=smoothing,
                smoothness_grid=smoothness_grid,
                decay_grid=decay_grid,
            )
            for model, smoothing in model_smoothing.items()
        }
        maximum_condition_number = max(
            maximum_condition_number,
            *(
                float(np.max(library["condition_numbers"]))
                for library in libraries.values()
            ),
        )
        for current_smoothness in true_smoothness:
            latent_truth = matern_covariance(
                latent,
                variance=1.0,
                decay=1.0,
                nu=current_smoothness,
            )
            truth = true_smoothing @ latent_truth @ true_smoothing.T
            truth_factor = cholesky(truth, lower=True, check_finite=False)
            standard = rng.standard_normal(
                (args.replicates, args.sample_size, outputs.shape[0])
            )
            samples = standard @ truth_factor.T
            second_moments = np.einsum(
                "tnp,tnq->tpq", samples, samples, optimize=True
            ) / args.sample_size
            for model, library in libraries.items():
                population_indices, population_variances, population_objectives = (
                    _profile_library(library, truth)
                )
                population_index = int(population_indices[0])
                sample_indices, sample_variances, sample_objectives = _profile_library(
                    library, second_moments
                )
                population = {
                    "bandwidth": bandwidth,
                    "smoothness_true": current_smoothness,
                    "decay_true": 1.0,
                    "variance_true": 1.0,
                    "model": model,
                    "assumed_bandwidth": (
                        0.0
                        if model == "point_support"
                        else bandwidth
                        if model == "support_aware"
                        else args.partial_bandwidth_fraction * bandwidth
                    ),
                    "population_smoothness_target": float(
                        library["smoothness"][population_index]
                    ),
                    "population_decay_target": float(
                        library["decay"][population_index]
                    ),
                    "population_variance_target": float(population_variances[0]),
                    "population_objective": float(population_objectives[0]),
                    "population_index": population_index,
                }
                population_rows.append(population)
                for trial, candidate_index in enumerate(sample_indices):
                    candidate_index = int(candidate_index)
                    records.append(
                        {
                            **population,
                            "sample_size": args.sample_size,
                            "trial": trial,
                            "smoothness_estimate": float(
                                library["smoothness"][candidate_index]
                            ),
                            "decay_estimate": float(library["decay"][candidate_index]),
                            "variance_estimate": float(sample_variances[trial]),
                            "sample_objective": float(sample_objectives[trial]),
                            "smoothness_error_to_target": float(
                                library["smoothness"][candidate_index]
                                - population["population_smoothness_target"]
                            ),
                            "decay_error_to_target": float(
                                library["decay"][candidate_index]
                                - population["population_decay_target"]
                            ),
                        }
                    )

    grouped: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            float(record["bandwidth"]),
            float(record["smoothness_true"]),
            str(record["model"]),
        )
        grouped.setdefault(key, []).append(record)
    summary: list[dict[str, Any]] = []
    for _, group in sorted(grouped.items()):
        first = group[0]
        smoothness_estimates = np.asarray(
            [row["smoothness_estimate"] for row in group], dtype=float
        )
        decay_estimates = np.asarray(
            [row["decay_estimate"] for row in group], dtype=float
        )
        summary.append(
            {
                key: first[key]
                for key in (
                    "bandwidth", "smoothness_true", "decay_true", "model",
                    "assumed_bandwidth", "population_smoothness_target",
                    "population_decay_target", "population_variance_target",
                    "population_objective",
                )
            }
            | {
                "replicates": len(group),
                "sample_size": args.sample_size,
                "median_smoothness_estimate": float(np.median(smoothness_estimates)),
                "median_decay_estimate": float(np.median(decay_estimates)),
                "smoothness_target_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                smoothness_estimates
                                - float(first["population_smoothness_target"])
                            )
                        )
                    )
                ),
                "decay_target_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                decay_estimates
                                - float(first["population_decay_target"])
                            )
                        )
                    )
                ),
            }
        )

    population_lookup = {
        (row["bandwidth"], row["smoothness_true"], row["model"]): row
        for row in population_rows
    }
    comparison_cells = [
        (bandwidth, smoothness)
        for bandwidth in bandwidths
        for smoothness in true_smoothness
    ]
    corrected_exact = all(
        np.isclose(
            population_lookup[(bandwidth, smoothness, "support_aware")][
                "population_smoothness_target"
            ],
            smoothness,
        )
        and np.isclose(
            population_lookup[(bandwidth, smoothness, "support_aware")][
                "population_decay_target"
            ],
            1.0,
        )
        for bandwidth, smoothness in comparison_cells
    )
    partial_beats_naive = all(
        population_lookup[(bandwidth, smoothness, "partial_support")][
            "population_objective"
        ]
        <= population_lookup[(bandwidth, smoothness, "point_support")][
            "population_objective"
        ]
        + 1e-12
        for bandwidth, smoothness in comparison_cells
    )
    point_targets = [
        population_lookup[(bandwidth, smoothness, "point_support")]
        for bandwidth, smoothness in comparison_cells
    ]
    gates = {
        "complete_fit_grid": {
            "observed": len(records),
            "required": len(comparison_cells) * 3 * args.replicates,
            "passed": len(records) == len(comparison_cells) * 3 * args.replicates,
        },
        "support_aware_population_target_is_truth": {"passed": corrected_exact},
        "partial_support_improves_population_criterion": {
            "passed": partial_beats_naive
        },
        "point_support_changes_joint_smoothness_target": {
            "changed_cells": sum(
                not np.isclose(row["population_smoothness_target"], row["smoothness_true"])
                for row in point_targets
            ),
            "required": len(point_targets),
            "passed": all(
                not np.isclose(
                    row["population_smoothness_target"], row["smoothness_true"]
                )
                for row in point_targets
            ),
        },
        "joint_fit_can_reverse_fixed_smoothness_decay_direction": {
            "reversed_cells": sum(
                row["population_decay_target"] > 1.0 for row in point_targets
            ),
            "passed": any(row["population_decay_target"] > 1.0 for row in point_targets),
        },
        "candidate_conditioning": {
            "observed_maximum": maximum_condition_number,
            "predeclared_maximum": 1e10,
            "passed": maximum_condition_number <= 1e10,
        },
    }
    gates["all_passed"] = all(
        value["passed"] for value in gates.values() if isinstance(value, dict)
    )
    if not gates["all_passed"]:
        failures = [
            name for name, value in gates.items()
            if isinstance(value, dict) and not value["passed"]
        ]
        raise RuntimeError(f"joint smoothness gates failed: {failures}")

    write_csv_atomic(args.output, records)
    summary_path = args.output.with_name(f"{args.output.stem}.summary.csv")
    write_csv_atomic(summary_path, summary)
    metadata = {
        "benchmark": "SupportShift joint smoothness--decay finite library",
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output": str(args.output)},
        "design": {
            "latent_locations": latent.shape[0],
            "output_locations": outputs.shape[0],
            "smoothness_grid": smoothness_grid.tolist(),
            "decay_grid": decay_grid.tolist(),
            "candidate_count_per_model": int(smoothness_grid.size * decay_grid.size),
        },
        "rows": len(records),
        "summary_rows": len(summary),
        "validation_gates": gates,
        "provenance": {
            "git_commit": commit, "git_dirty": dirty,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "result_csv": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "summary_csv": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
    }
    write_json_atomic(args.output.with_suffix(".metadata.json"), metadata)
    print(
        f"Wrote {len(records)} joint fits and {len(summary)} summary rows to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
