"""Audit the genuinely misspecified multi-lag SupportShift phase law."""
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
        json.dump(value, handle, indent=2, sort_keys=True, default=json_scalar)
        handle.write("\n")
    os.replace(temporary, path)


def json_scalar(value: Any) -> Any:
    """Convert NumPy scalar diagnostics without weakening JSON validation."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__} as JSON")


def asymptotic_scale(bandwidth: float, smoothness: float) -> float:
    if smoothness < 1.0:
        return bandwidth ** (2.0 * smoothness)
    if smoothness == 1.0:
        return bandwidth**2 * np.log(1.0 / bandwidth)
    return bandwidth**2


def main() -> None:
    repository_root = add_src_to_path()
    from HighDimSpatial.smoothing_bias.continuous import (
        continuous_matern_multilag_target,
        product_kernel_multilag_asymptotics,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument(
        "--lags", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0]
    )
    parser.add_argument(
        "--smoothness", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.5]
    )
    parser.add_argument(
        "--bandwidths", type=float, nargs="+", default=[0.005, 0.01, 0.02]
    )
    parser.add_argument("--kernel-family", default="epanechnikov")
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--coefficient-relative-tolerance", type=float, default=0.12)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root
        / "outputs/smoothing_bias/supportshift_multilag_composite.csv",
    )
    args = parser.parse_args()
    lags = np.asarray(sorted(set(args.lags)), dtype=float)
    smoothness_values = sorted(set(args.smoothness))
    bandwidth_values = sorted(set(args.bandwidths))
    if args.decay <= 0.0 or np.any(lags <= 0.0):
        raise ValueError("decay and lags must be positive")
    if any(value <= 0.0 for value in smoothness_values + bandwidth_values):
        raise ValueError("smoothness and bandwidths must be positive")
    if max(bandwidth_values) >= min(lags) / (4.0 * np.sqrt(args.dimension)):
        raise ValueError("bandwidths must satisfy the declared Taylor neighborhood")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = bool(status.strip())
    if dirty and not args.allow_dirty:
        raise RuntimeError("refusing to generate a promoted audit from a dirty worktree")

    records: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, float]] = []
    for smoothness in smoothness_values:
        asymptotics = product_kernel_multilag_asymptotics(
            dimension=args.dimension,
            smoothness=smoothness,
            decay=args.decay,
            lags=lags,
            kernel_family=args.kernel_family,
            quadrature_order=args.quadrature_order,
        )
        pair_coefficients = np.asarray(asymptotics.pair_shift_coefficients)
        information_weights = np.asarray(asymptotics.information_weights)
        for bandwidth in bandwidth_values:
            target = continuous_matern_multilag_target(
                dimension=args.dimension,
                smoothness=smoothness,
                decay=args.decay,
                bandwidth=bandwidth,
                lags=lags,
                kernel_family=args.kernel_family,
                quadrature_order=args.quadrature_order,
            )
            scale = asymptotic_scale(bandwidth, smoothness)
            shift = args.decay - target.pseudo_decay
            predicted_shift = asymptotics.decay_shift_coefficient * scale
            predicted_kl = asymptotics.minimum_kl_coefficient * scale**2
            shift_ratio = shift / predicted_shift
            minimum_kl_ratio = target.minimum_kl / predicted_kl
            cell_summaries.append(
                {
                    "smoothness": smoothness,
                    "bandwidth": bandwidth,
                    "shift_ratio": shift_ratio,
                    "minimum_kl_ratio": minimum_kl_ratio,
                    "minimum_kl": target.minimum_kl,
                }
            )
            for index, lag in enumerate(lags):
                records.append(
                    {
                        "dimension": args.dimension,
                        "kernel_family": args.kernel_family,
                        "smoothness": smoothness,
                        "bandwidth": bandwidth,
                        "lag": float(lag),
                        "decay_true": args.decay,
                        "pair_pseudo_decay": target.pair_pseudo_decays[index],
                        "pair_shift_coefficient": pair_coefficients[index],
                        "information_weight": information_weights[index],
                        "composite_pseudo_decay": target.pseudo_decay,
                        "composite_decay_shift": shift,
                        "composite_shift_coefficient": (
                            asymptotics.decay_shift_coefficient
                        ),
                        "asymptotic_scale": scale,
                        "predicted_composite_shift": predicted_shift,
                        "composite_shift_ratio": shift_ratio,
                        "minimum_composite_kl": target.minimum_kl,
                        "minimum_kl_coefficient": (
                            asymptotics.minimum_kl_coefficient
                        ),
                        "predicted_minimum_kl": predicted_kl,
                        "minimum_kl_ratio": minimum_kl_ratio,
                        "quadrature_order": args.quadrature_order,
                    }
                )

    smallest_bandwidth = min(bandwidth_values)
    smallest_cells = [
        row
        for row in cell_summaries
        if np.isclose(row["bandwidth"], smallest_bandwidth)
    ]
    maximum_shift_error = max(abs(row["shift_ratio"] - 1.0) for row in smallest_cells)
    maximum_kl_error = max(
        abs(row["minimum_kl_ratio"] - 1.0) for row in smallest_cells
    )
    expected_rows = len(smoothness_values) * len(bandwidth_values) * len(lags)
    gates = {
        "complete_factor_grid": {
            "observed": len(records),
            "required": expected_rows,
            "passed": len(records) == expected_rows,
        },
        "all_composite_shifts_positive": {
            "observed_minimum": min(
                float(record["composite_decay_shift"]) for record in records
            ),
            "passed": all(
                float(record["composite_decay_shift"]) > 0.0 for record in records
            ),
        },
        "genuine_multilag_misspecification": {
            "observed_minimum_kl": min(
                float(record["minimum_composite_kl"]) for record in records
            ),
            "passed": all(
                float(record["minimum_composite_kl"]) > 0.0 for record in records
            ),
        },
        "smallest_bandwidth_shift_coefficient_error": {
            "observed_maximum": maximum_shift_error,
            "predeclared_maximum": args.coefficient_relative_tolerance,
            "passed": maximum_shift_error <= args.coefficient_relative_tolerance,
        },
        "smallest_bandwidth_kl_coefficient_error": {
            "observed_maximum": maximum_kl_error,
            "predeclared_maximum": args.coefficient_relative_tolerance,
            "passed": maximum_kl_error <= args.coefficient_relative_tolerance,
        },
    }
    gates["all_passed"] = all(
        value["passed"] for value in gates.values() if isinstance(value, dict)
    )
    if not gates["all_passed"]:
        failures = [
            name
            for name, value in gates.items()
            if isinstance(value, dict) and not value["passed"]
        ]
        raise RuntimeError(f"multi-lag composite gates failed: {failures}")

    write_csv_atomic(args.output, records)
    metadata = {
        "benchmark": "SupportShift multi-lag Gaussian pair composite",
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output": str(args.output)},
        "factor_grid": {
            "lags": lags.tolist(),
            "smoothness": smoothness_values,
            "bandwidths": bandwidth_values,
        },
        "rows": len(records),
        "expected_rows": expected_rows,
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
    print(f"Wrote {len(records)} passing multi-lag audit rows to {args.output}")


if __name__ == "__main__":
    main()
