"""Matched-dimension boundary audit for finite-support Matérn observations.

The original boundary stress test changed both the location of the output block
and its dimension.  This audit uses translated 4-by-4 blocks on the same latent
grid, so boundary truncation is the only design difference.  Variance is
profiled analytically and decay is optimized continuously.
"""
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
from typing import Any, Callable

import numpy as np
import scipy
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize_scalar


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


def profiled_population_target(
    truth: np.ndarray,
    unit_covariance: Callable[[float], np.ndarray],
    decay_bounds: tuple[float, float],
) -> dict[str, float | bool]:
    """Profile variance and optimize decay for a Gaussian KL projection."""
    dimension = truth.shape[0]

    def evaluate(log_decay: float) -> tuple[float, float, np.ndarray]:
        base = unit_covariance(float(np.exp(log_decay)))
        factor = cholesky(base, lower=True, check_finite=False)
        trace = float(
            np.trace(cho_solve((factor, True), truth, check_finite=False))
        )
        variance = trace / dimension
        log_determinant = 2.0 * float(np.log(np.diag(factor)).sum())
        objective = 0.5 * (
            log_determinant + dimension * np.log(variance) + dimension
        )
        return objective, variance, base

    log_bounds = tuple(float(np.log(value)) for value in decay_bounds)
    result = minimize_scalar(
        lambda log_decay: evaluate(log_decay)[0],
        bounds=log_bounds,
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    objective, variance, base = evaluate(float(result.x))
    candidate = variance * base
    truth_factor = cholesky(truth, lower=True, check_finite=False)
    candidate_factor = cholesky(candidate, lower=True, check_finite=False)
    trace = float(
        np.trace(cho_solve((candidate_factor, True), truth, check_finite=False))
    )
    logdet_candidate = 2.0 * float(np.log(np.diag(candidate_factor)).sum())
    logdet_truth = 2.0 * float(np.log(np.diag(truth_factor)).sum())
    kl = 0.5 * (trace - dimension + logdet_candidate - logdet_truth)
    if -1e-9 <= kl < 0.0:
        kl = 0.0
    return {
        "decay": float(np.exp(result.x)),
        "variance": float(variance),
        "minimum_kl": float(kl),
        "objective_without_constant": float(objective),
        "optimizer_success": bool(result.success),
        "at_decay_bound": bool(
            min(result.x - log_bounds[0], log_bounds[1] - result.x) <= 1e-6
        ),
        "condition_number": float(np.linalg.cond(candidate)),
    }


def main() -> None:
    repository_root = add_src_to_path()
    from HighDimSpatial.smoothing_bias.design import (
        epanechnikov_smoothing_matrix,
        regular_grid_2d,
    )
    from HighDimSpatial.smoothing_bias.kl import matern_covariance

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-side", type=int, default=19)
    parser.add_argument("--spacing", type=float, default=0.25)
    parser.add_argument("--block-side", type=int, default=4)
    parser.add_argument("--block-spacing", type=float, default=0.5)
    parser.add_argument("--interior-origin", type=float, default=1.5)
    parser.add_argument(
        "--smoothness", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.5]
    )
    parser.add_argument("--bandwidths", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--partial-bandwidth-fraction", type=float, default=0.75)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument(
        "--decay-bounds", type=float, nargs=2, default=[0.08, 3.0]
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root
        / "outputs/smoothing_bias/supportshift_matched_boundary.csv",
    )
    args = parser.parse_args()
    if args.latent_side < 7 or args.spacing <= 0.0 or args.block_side < 2:
        raise ValueError("the latent-grid and block arguments are invalid")
    if args.block_spacing <= 0.0 or args.interior_origin <= 0.0:
        raise ValueError("block spacing and interior origin must be positive")
    if not 0.0 < args.partial_bandwidth_fraction < 1.0:
        raise ValueError("partial bandwidth fraction must lie strictly between zero and one")
    smoothness_values = sorted(set(args.smoothness))
    bandwidth_values = sorted(set(args.bandwidths))
    if any(value <= 0.0 for value in smoothness_values + bandwidth_values):
        raise ValueError("smoothness and bandwidth values must be positive")
    if args.decay <= 0.0 or args.variance <= 0.0:
        raise ValueError("decay and variance must be positive")
    if not 0.0 < args.decay_bounds[0] < args.decay_bounds[1]:
        raise ValueError("decay bounds must be positive and increasing")

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

    latent = regular_grid_2d(args.latent_side, spacing=args.spacing)
    axes = {
        "boundary": np.arange(args.block_side, dtype=float) * args.block_spacing,
        "interior": (
            args.interior_origin
            + np.arange(args.block_side, dtype=float) * args.block_spacing
        ),
    }
    output_blocks = {
        name: np.asarray(np.meshgrid(axis, axis, indexing="ij"))
        .reshape(2, -1)
        .T
        for name, axis in axes.items()
    }
    domain_maximum = float(np.max(latent))
    if any(np.min(block) < 0.0 or np.max(block) > domain_maximum for block in output_blocks.values()):
        raise ValueError("output blocks must lie inside the latent domain")
    output_counts = {name: block.shape[0] for name, block in output_blocks.items()}
    if len(set(output_counts.values())) != 1:
        raise RuntimeError("matched boundary blocks must have identical dimensions")

    records: list[dict[str, Any]] = []
    for bandwidth in bandwidth_values:
        for smoothness in smoothness_values:
            latent_truth = matern_covariance(
                latent, variance=args.variance, decay=args.decay, nu=smoothness
            )
            for region, outputs in output_blocks.items():
                true_smoothing = epanechnikov_smoothing_matrix(
                    latent, outputs, bandwidth
                )
                truth = true_smoothing @ latent_truth @ true_smoothing.T
                smoothing_models = {
                    "support_aware": true_smoothing,
                    "partial_support": epanechnikov_smoothing_matrix(
                        latent,
                        outputs,
                        args.partial_bandwidth_fraction * bandwidth,
                    ),
                    "point_support": None,
                }
                for model, assumed_smoothing in smoothing_models.items():
                    def unit_covariance(
                        decay: float,
                        operator: np.ndarray | None = assumed_smoothing,
                        locations: np.ndarray = outputs,
                    ) -> np.ndarray:
                        if operator is None:
                            return matern_covariance(
                                locations,
                                variance=1.0,
                                decay=decay,
                                nu=smoothness,
                            )
                        latent_candidate = matern_covariance(
                            latent,
                            variance=1.0,
                            decay=decay,
                            nu=smoothness,
                        )
                        return operator @ latent_candidate @ operator.T

                    target = profiled_population_target(
                        truth, unit_covariance, tuple(args.decay_bounds)
                    )
                    records.append(
                        {
                            "region": region,
                            "output_dimension": outputs.shape[0],
                            "block_side": args.block_side,
                            "block_spacing": args.block_spacing,
                            "bandwidth": bandwidth,
                            "smoothness": smoothness,
                            "model": model,
                            "assumed_bandwidth": (
                                0.0
                                if model == "point_support"
                                else bandwidth
                                if model == "support_aware"
                                else args.partial_bandwidth_fraction * bandwidth
                            ),
                            "decay_true": args.decay,
                            "decay_target": target["decay"],
                            "variance_true": args.variance,
                            "variance_target": target["variance"],
                            "minimum_kl": target["minimum_kl"],
                            "optimizer_success": target["optimizer_success"],
                            "at_decay_bound": target["at_decay_bound"],
                            "condition_number": target["condition_number"],
                        }
                    )

    lookup = {
        (row["bandwidth"], row["smoothness"], row["region"], row["model"]): row
        for row in records
    }
    cells = [
        (bandwidth, smoothness, region)
        for bandwidth in bandwidth_values
        for smoothness in smoothness_values
        for region in output_blocks
    ]
    pairs = [
        (bandwidth, smoothness)
        for bandwidth in bandwidth_values
        for smoothness in smoothness_values
    ]
    boundary_differences = [
        abs(
            float(lookup[(bandwidth, smoothness, "boundary", "point_support")]["decay_target"])
            - float(lookup[(bandwidth, smoothness, "interior", "point_support")]["decay_target"])
        )
        for bandwidth, smoothness in pairs
    ]
    gates = {
        "complete_matched_grid": {
            "observed": len(records),
            "required": len(cells) * 3,
            "passed": len(records) == len(cells) * 3,
        },
        "identical_output_dimension": {
            "dimensions": output_counts,
            "passed": len(set(output_counts.values())) == 1,
        },
        "support_aware_recovers_truth": {
            "maximum_decay_error": max(
                abs(float(lookup[cell + ("support_aware",)]["decay_target"]) - args.decay)
                for cell in cells
            ),
            "maximum_kl": max(
                float(lookup[cell + ("support_aware",)]["minimum_kl"])
                for cell in cells
            ),
            "passed": all(
                abs(float(lookup[cell + ("support_aware",)]["decay_target"]) - args.decay)
                <= 2e-5
                and float(lookup[cell + ("support_aware",)]["minimum_kl"]) <= 1e-8
                for cell in cells
            ),
        },
        "partial_support_beats_point_support": {
            "passed": all(
                float(lookup[cell + ("partial_support",)]["minimum_kl"])
                <= float(lookup[cell + ("point_support",)]["minimum_kl"]) + 1e-10
                for cell in cells
            ),
        },
        "boundary_effect_at_matched_dimension": {
            "maximum_target_difference": max(boundary_differences),
            "passed": max(boundary_differences) >= 1e-4,
        },
        "optimizer_and_conditioning": {
            "maximum_condition_number": max(
                float(row["condition_number"]) for row in records
            ),
            "passed": all(
                bool(row["optimizer_success"])
                and not bool(row["at_decay_bound"])
                and float(row["condition_number"]) <= 1e10
                for row in records
            ),
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
        raise RuntimeError(f"matched boundary gates failed: {failures}")

    write_csv_atomic(args.output, records)
    metadata = {
        "benchmark": "SupportShift matched-dimension boundary audit",
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output": str(args.output)},
        "rows": len(records),
        "validation_gates": gates,
        "provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "result_csv": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    write_json_atomic(args.output.with_suffix(".metadata.json"), metadata)
    print(f"Wrote {len(records)} passing matched-boundary rows to {args.output}")


if __name__ == "__main__":
    main()
