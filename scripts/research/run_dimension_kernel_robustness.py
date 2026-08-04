"""Audit the SupportShift phase law across dimensions and compact kernels."""
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


SUPPORTED_DIMENSIONS = (1, 2, 3)
SUPPORTED_KERNELS = ("epanechnikov", "uniform")


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


def require_factors(args: argparse.Namespace) -> None:
    if not args.dimensions or any(value not in SUPPORTED_DIMENSIONS for value in args.dimensions):
        raise ValueError(f"dimensions must be drawn from {SUPPORTED_DIMENSIONS}")
    if not args.kernel_families or any(
        value not in SUPPORTED_KERNELS for value in args.kernel_families
    ):
        raise ValueError(f"kernel families must be drawn from {SUPPORTED_KERNELS}")
    if not args.smoothness or any(value <= 0 for value in args.smoothness):
        raise ValueError("smoothness values must be positive")
    if not args.bandwidths or any(value <= 0 for value in args.bandwidths):
        raise ValueError("bandwidths must be positive")
    if args.decay <= 0 or args.lag <= 0:
        raise ValueError("decay and lag must be positive")
    all_orders = [args.quadrature_order, *args.refinement_orders]
    if any(
        isinstance(order, bool) or order < 8 or order % 2
        for order in all_orders
    ):
        raise ValueError("quadrature orders must be even integers of at least eight")
    if not args.refinement_orders or set(args.refinement_orders) == {
        args.quadrature_order
    }:
        raise ValueError("at least one distinct refinement order is required")
    if (
        not np.isfinite(args.coefficient_relative_tolerance)
        or args.coefficient_relative_tolerance <= 0
        or not np.isfinite(args.quadrature_relative_tolerance)
        or args.quadrature_relative_tolerance <= 0
    ):
        raise ValueError("validation tolerances must be positive and finite")
    if max(args.bandwidths) >= args.lag / (4.0 * np.sqrt(max(args.dimensions))):
        raise ValueError("bandwidths must satisfy the declared Taylor neighborhood")


def main() -> None:
    repository_root = add_src_to_path()
    from HighDimSpatial.smoothing_bias.continuous import (
        continuous_matern_pair_target,
        product_kernel_decay_shift_coefficient,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--kernel-families",
        nargs="+",
        default=["epanechnikov", "uniform"],
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.5],
    )
    parser.add_argument(
        "--bandwidths",
        type=float,
        nargs="+",
        default=[0.002, 0.004, 0.008],
    )
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--lag", type=float, default=1.0)
    parser.add_argument("--quadrature-order", type=int, default=48)
    parser.add_argument(
        "--refinement-orders",
        type=int,
        nargs="*",
        default=[32, 64],
    )
    parser.add_argument("--coefficient-relative-tolerance", type=float, default=0.20)
    parser.add_argument("--quadrature-relative-tolerance", type=float, default=2e-4)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root
        / "outputs/smoothing_bias/supportshift_dimension_kernel_robustness.csv",
    )
    args = parser.parse_args()
    require_factors(args)

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
    for dimension in sorted(set(args.dimensions)):
        for kernel_family in sorted(set(args.kernel_families)):
            for smoothness in sorted(set(args.smoothness)):
                coefficient = product_kernel_decay_shift_coefficient(
                    dimension=dimension,
                    smoothness=smoothness,
                    decay=args.decay,
                    lag=args.lag,
                    kernel_family=kernel_family,
                    quadrature_order=args.quadrature_order,
                )
                for bandwidth in sorted(set(args.bandwidths)):
                    target = continuous_matern_pair_target(
                        dimension=dimension,
                        smoothness=smoothness,
                        decay=args.decay,
                        bandwidth=bandwidth,
                        lag=args.lag,
                        kernel_family=kernel_family,
                        quadrature_order=args.quadrature_order,
                    )
                    decay_shift = args.decay - target.pseudo_decay
                    scale = asymptotic_scale(bandwidth, smoothness)
                    leading_shift = coefficient * scale
                    records.append(
                        {
                            "dimension": dimension,
                            "kernel_family": kernel_family,
                            "smoothness": smoothness,
                            "bandwidth": bandwidth,
                            "decay_true": args.decay,
                            "lag": args.lag,
                            "pseudo_decay": target.pseudo_decay,
                            "decay_shift": decay_shift,
                            "variance_factor": target.variance_factor,
                            "covariance_factor": target.covariance_factor,
                            "smoothed_correlation": target.correlation,
                            "asymptotic_scale": scale,
                            "leading_coefficient": coefficient,
                            "leading_shift": leading_shift,
                            "coefficient_ratio": decay_shift / leading_shift,
                            "quadrature_order": args.quadrature_order,
                        }
                    )

    refinement: dict[str, dict[str, float]] = {}
    for order in sorted(set(args.refinement_orders)):
        if order == args.quadrature_order:
            continue
        absolute_differences: list[float] = []
        relative_shift_differences: list[float] = []
        for record in records:
            refined = continuous_matern_pair_target(
                dimension=int(record["dimension"]),
                smoothness=float(record["smoothness"]),
                decay=args.decay,
                bandwidth=float(record["bandwidth"]),
                lag=args.lag,
                kernel_family=str(record["kernel_family"]),
                quadrature_order=order,
            )
            difference = abs(refined.pseudo_decay - float(record["pseudo_decay"]))
            absolute_differences.append(difference)
            relative_shift_differences.append(
                difference / abs(float(record["decay_shift"]))
            )
        refinement[str(order)] = {
            "max_abs_pseudo_decay_difference": max(absolute_differences),
            "max_relative_decay_shift_difference": max(relative_shift_differences),
        }

    minimum_bandwidth = min(args.bandwidths)
    smallest = [
        record
        for record in records
        if np.isclose(float(record["bandwidth"]), minimum_bandwidth)
    ]
    maximum_coefficient_error = max(
        abs(float(record["coefficient_ratio"]) - 1.0) for record in smallest
    )
    maximum_refinement_error = max(
        value["max_relative_decay_shift_difference"] for value in refinement.values()
    )
    expected_rows = (
        len(set(args.dimensions))
        * len(set(args.kernel_families))
        * len(set(args.smoothness))
        * len(set(args.bandwidths))
    )
    gates = {
        "complete_factor_grid": {
            "observed": len(records),
            "required": expected_rows,
            "passed": len(records) == expected_rows,
        },
        "all_decay_shifts_positive": {
            "observed_minimum": min(float(record["decay_shift"]) for record in records),
            "passed": all(float(record["decay_shift"]) > 0 for record in records),
        },
        "smallest_bandwidth_coefficient_error": {
            "observed_maximum": maximum_coefficient_error,
            "predeclared_maximum": args.coefficient_relative_tolerance,
            "passed": maximum_coefficient_error <= args.coefficient_relative_tolerance,
        },
        "quadrature_refinement": {
            "observed_maximum_relative_shift_difference": maximum_refinement_error,
            "predeclared_maximum": args.quadrature_relative_tolerance,
            "passed": maximum_refinement_error <= args.quadrature_relative_tolerance,
        },
    }
    gates["all_passed"] = all(
        bool(value["passed"]) for value in gates.values() if isinstance(value, dict)
    )
    if not gates["all_passed"]:
        failures = [
            name
            for name, value in gates.items()
            if isinstance(value, dict) and not value["passed"]
        ]
        raise RuntimeError(f"dimension-kernel robustness gates failed: {failures}")

    write_csv_atomic(args.output, records)
    metadata = {
        "benchmark": "SupportShift dimension-kernel robustness",
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output": str(args.output)},
        "factor_grid": {
            "dimensions": sorted(set(args.dimensions)),
            "kernel_families": sorted(set(args.kernel_families)),
            "smoothness": sorted(set(args.smoothness)),
            "bandwidths": sorted(set(args.bandwidths)),
        },
        "rows": len(records),
        "expected_rows": expected_rows,
        "quadrature_refinement": refinement,
        "validation_gates": gates,
        "provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "result_csv": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
        },
    }
    write_json_atomic(args.output.with_suffix(".metadata.json"), metadata)
    print(
        "Wrote "
        f"{len(records)} passing dimension-kernel audit rows to {args.output}"
    )


if __name__ == "__main__":
    main()
