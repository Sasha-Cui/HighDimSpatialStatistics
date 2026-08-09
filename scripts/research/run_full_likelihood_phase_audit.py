"""Audit the finite-design full-Gaussian SupportShift projection theorem."""
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


def asymptotic_scale(bandwidth: float, smoothness: float) -> float:
    if smoothness < 1.0:
        return bandwidth ** (2.0 * smoothness)
    if smoothness == 1.0:
        return bandwidth**2 * np.log(1.0 / bandwidth)
    return bandwidth**2


def main() -> None:
    repository_root = add_src_to_path()
    from HighDimSpatial.smoothing_bias.continuous import (
        continuous_matern_full_likelihood_target,
        finite_design_full_likelihood_asymptotics,
    )
    from HighDimSpatial.smoothing_bias.design import regular_grid_2d

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-side", type=int, default=3)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument(
        "--smoothness", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.5]
    )
    parser.add_argument(
        "--bandwidths", type=float, nargs="+", default=[0.005, 0.01, 0.02]
    )
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--coefficient-relative-tolerance", type=float, default=0.16)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root
        / "outputs/smoothing_bias/supportshift_full_likelihood_phase.csv",
    )
    args = parser.parse_args()
    if args.grid_side < 2 or args.spacing <= 0.0:
        raise ValueError("grid side must be at least two and spacing must be positive")
    if args.variance <= 0.0 or args.decay <= 0.0:
        raise ValueError("variance and decay must be positive")
    smoothness_values = sorted(set(args.smoothness))
    bandwidth_values = sorted(set(args.bandwidths))
    if any(value <= 0.0 for value in smoothness_values + bandwidth_values):
        raise ValueError("smoothness and bandwidths must be positive")
    if max(bandwidth_values) >= args.spacing / (4.0 * np.sqrt(2.0)):
        raise ValueError("bandwidths must satisfy the declared Taylor neighborhood")

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

    locations = regular_grid_2d(args.grid_side, spacing=args.spacing)
    records: list[dict[str, Any]] = []
    for smoothness in smoothness_values:
        asymptotics = finite_design_full_likelihood_asymptotics(
            locations,
            variance=args.variance,
            decay=args.decay,
            smoothness=smoothness,
            quadrature_order=args.quadrature_order,
        )
        for bandwidth in bandwidth_values:
            target = continuous_matern_full_likelihood_target(
                locations,
                variance=args.variance,
                decay=args.decay,
                smoothness=smoothness,
                bandwidth=bandwidth,
                quadrature_order=args.quadrature_order,
            )
            scale = asymptotic_scale(bandwidth, smoothness)
            predicted_decay_shift = asymptotics.decay_inflation_coefficient * scale
            predicted_log_variance_shift = (
                asymptotics.log_variance_shift_coefficient * scale
            )
            predicted_minimum_kl = asymptotics.minimum_kl_coefficient * scale**2
            records.append(
                {
                    "grid_side": args.grid_side,
                    "number_of_locations": locations.shape[0],
                    "spacing": args.spacing,
                    "smoothness": smoothness,
                    "bandwidth": bandwidth,
                    "variance_true": args.variance,
                    "decay_true": args.decay,
                    "variance_pseudo": target.pseudo_variance,
                    "decay_pseudo": target.pseudo_decay,
                    "decay_shift": args.decay - target.pseudo_decay,
                    "log_variance_shift": np.log(
                        target.pseudo_variance / args.variance
                    ),
                    "asymptotic_scale": scale,
                    "decay_inflation_coefficient": (
                        asymptotics.decay_inflation_coefficient
                    ),
                    "log_variance_shift_coefficient": (
                        asymptotics.log_variance_shift_coefficient
                    ),
                    "predicted_decay_shift": predicted_decay_shift,
                    "decay_shift_ratio": (
                        (args.decay - target.pseudo_decay) / predicted_decay_shift
                    ),
                    "predicted_log_variance_shift": predicted_log_variance_shift,
                    "log_variance_shift_ratio": (
                        np.log(target.pseudo_variance / args.variance)
                        / predicted_log_variance_shift
                    ),
                    "minimum_kl": target.minimum_kl,
                    "minimum_kl_coefficient": asymptotics.minimum_kl_coefficient,
                    "predicted_minimum_kl": predicted_minimum_kl,
                    "minimum_kl_ratio": target.minimum_kl / predicted_minimum_kl,
                    "information_condition_number": (
                        asymptotics.information_condition_number
                    ),
                    "quadrature_order": args.quadrature_order,
                }
            )

    smallest_bandwidth = min(bandwidth_values)
    smallest = [
        row for row in records if np.isclose(row["bandwidth"], smallest_bandwidth)
    ]
    errors = {
        "decay": max(abs(row["decay_shift_ratio"] - 1.0) for row in smallest),
        "variance": max(
            abs(row["log_variance_shift_ratio"] - 1.0) for row in smallest
        ),
        "minimum_kl": max(
            abs(row["minimum_kl_ratio"] - 1.0) for row in smallest
        ),
    }
    expected_rows = len(smoothness_values) * len(bandwidth_values)
    gates = {
        "complete_factor_grid": {
            "observed": len(records), "required": expected_rows,
            "passed": len(records) == expected_rows,
        },
        "genuine_full_likelihood_misspecification": {
            "observed_minimum_kl": min(row["minimum_kl"] for row in records),
            "passed": all(row["minimum_kl"] > 0.0 for row in records),
        },
        "selected_design_decay_inflation": {
            "observed_minimum_shift": min(row["decay_shift"] for row in records),
            "passed": all(row["decay_shift"] > 0.0 for row in records),
        },
        "smallest_bandwidth_coefficient_error": {
            "observed": errors,
            "predeclared_maximum": args.coefficient_relative_tolerance,
            "passed": max(errors.values()) <= args.coefficient_relative_tolerance,
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
        raise RuntimeError(f"full-likelihood phase gates failed: {failures}")

    write_csv_atomic(args.output, records)
    metadata = {
        "benchmark": "SupportShift finite-design full-likelihood phase",
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output": str(args.output)},
        "rows": len(records),
        "expected_rows": expected_rows,
        "validation_gates": gates,
        "provenance": {
            "git_commit": commit, "git_dirty": dirty,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "result_csv": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    write_json_atomic(args.output.with_suffix(".metadata.json"), metadata)
    print(f"Wrote {len(records)} passing full-likelihood rows to {args.output}")


if __name__ == "__main__":
    main()
