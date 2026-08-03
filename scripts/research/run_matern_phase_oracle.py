"""Evaluate the continuous two-dimensional Matérn phase theorem by quadrature."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy


def add_src_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    return repo_root


def write_csv_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: dict) -> None:
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
    repo_root = add_src_to_path()
    from HighDimSpatial.smoothing_bias.continuous import (
        continuous_matern_pair_target,
        product_epanechnikov_decay_shift_coefficient,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2, choices=[1, 2])
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--lag", type=float, default=1.0)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument(
        "--refinement-orders",
        type=int,
        nargs="*",
        default=[64, 128],
        help="Additional quadrature orders used only for a deterministic error audit.",
    )
    parser.add_argument("--minimum-bandwidth", type=float, default=0.003)
    parser.add_argument("--maximum-bandwidth", type=float, default=0.3)
    parser.add_argument("--bandwidth-count", type=int, default=18)
    parser.add_argument(
        "--smoothness",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.5],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs" / "smoothing_bias" / "phase_oracle.csv",
    )
    args = parser.parse_args()
    if args.minimum_bandwidth <= 0 or args.maximum_bandwidth >= args.lag / 2:
        raise ValueError("bandwidths must be positive and smaller than half the fitted lag")
    bandwidths = np.geomspace(
        args.minimum_bandwidth,
        args.maximum_bandwidth,
        args.bandwidth_count,
    )
    records: list[dict] = []
    for smoothness in args.smoothness:
        for bandwidth in bandwidths:
            target = continuous_matern_pair_target(
                dimension=args.dimension,
                smoothness=smoothness,
                decay=args.decay,
                bandwidth=float(bandwidth),
                lag=args.lag,
                quadrature_order=args.quadrature_order,
            )
            decay_shift = args.decay - target.pseudo_decay
            scale = asymptotic_scale(bandwidth, smoothness)
            leading_coefficient = product_epanechnikov_decay_shift_coefficient(
                dimension=args.dimension,
                smoothness=smoothness,
                decay=args.decay,
                lag=args.lag,
                quadrature_order=args.quadrature_order,
            )
            leading_shift = leading_coefficient * scale
            records.append(
                {
                    "dimension": args.dimension,
                    "smoothness": smoothness,
                    "bandwidth": bandwidth,
                    "decay_true": args.decay,
                    "decay_pseudo": target.pseudo_decay,
                    "decay_shift": decay_shift,
                    "relative_decay_shift": decay_shift / args.decay,
                    "variance_factor": target.variance_factor,
                    "variance_loss": 1.0 - target.variance_factor,
                    "covariance_factor": target.covariance_factor,
                    "smoothed_correlation": target.correlation,
                    "asymptotic_scale": scale,
                    "scaled_decay_shift": decay_shift / scale,
                    "leading_coefficient": leading_coefficient,
                    "leading_shift": leading_shift,
                    "coefficient_ratio": decay_shift / leading_shift,
                    "quadrature_order": args.quadrature_order,
                }
            )
    refinement: dict[str, dict[str, float]] = {}
    for order in args.refinement_orders:
        if order == args.quadrature_order:
            continue
        absolute_differences: list[float] = []
        relative_shift_differences: list[float] = []
        for record in records:
            refined = continuous_matern_pair_target(
                dimension=args.dimension,
                smoothness=float(record["smoothness"]),
                decay=args.decay,
                bandwidth=float(record["bandwidth"]),
                lag=args.lag,
                quadrature_order=order,
            )
            difference = abs(refined.pseudo_decay - float(record["decay_pseudo"]))
            absolute_differences.append(difference)
            relative_shift_differences.append(
                difference / abs(float(record["decay_shift"]))
            )
        refinement[str(order)] = {
            "max_abs_pseudo_decay_difference": max(absolute_differences),
            "max_relative_decay_shift_difference": max(relative_shift_differences),
        }
    write_csv_atomic(args.output, records)
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
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output": str(args.output)},
        "rows": len(records),
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "quadrature_refinement": refinement,
    }
    write_json_atomic(args.output.with_suffix(".metadata.json"), metadata)
    print(f"Wrote {len(records)} quadrature rows to {args.output}")


if __name__ == "__main__":
    main()
