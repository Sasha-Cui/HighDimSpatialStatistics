"""Audit the finite-bandwidth transition layer around Matérn smoothness one."""
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

import numpy as np
import scipy


def add_src_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    return repo_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        transition_aware_matern_pair_approximation,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2, choices=[1, 2])
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--lag", type=float, default=1.0)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--refinement-orders", type=int, nargs="*", default=[64, 128])
    parser.add_argument("--smoothness-minimum", type=float, default=0.55)
    parser.add_argument("--smoothness-maximum", type=float, default=1.45)
    parser.add_argument("--smoothness-count", type=int, default=37)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument(
        "--maximum-transition-relative-error",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "outputs"
        / "smoothing_bias"
        / "supportshift_transition_stress.csv",
    )
    args = parser.parse_args()
    if not 0.0 < args.smoothness_minimum < 1.0:
        raise ValueError("smoothness-minimum must lie strictly between zero and one")
    if not 1.0 < args.smoothness_maximum < 2.0:
        raise ValueError("smoothness-maximum must lie strictly between one and two")
    if args.smoothness_count < 3 or args.smoothness_count % 2 == 0:
        raise ValueError("smoothness-count must be an odd integer of at least three")
    if any(value <= 0 or value >= args.lag / 4.0 for value in args.bandwidths):
        raise ValueError("bandwidths must be positive and smaller than one quarter lag")

    smoothness_values = np.linspace(
        args.smoothness_minimum,
        args.smoothness_maximum,
        args.smoothness_count,
    )
    smoothness_values[np.argmin(np.abs(smoothness_values - 1.0))] = 1.0
    records: list[dict] = []
    for smoothness in smoothness_values:
        smoothness = float(smoothness)
        leading_coefficient = product_epanechnikov_decay_shift_coefficient(
            dimension=args.dimension,
            smoothness=smoothness,
            decay=args.decay,
            lag=args.lag,
            quadrature_order=args.quadrature_order,
        )
        for bandwidth in sorted(args.bandwidths):
            exact = continuous_matern_pair_target(
                dimension=args.dimension,
                smoothness=smoothness,
                decay=args.decay,
                bandwidth=bandwidth,
                lag=args.lag,
                quadrature_order=args.quadrature_order,
            )
            transition = transition_aware_matern_pair_approximation(
                dimension=args.dimension,
                smoothness=smoothness,
                decay=args.decay,
                bandwidth=bandwidth,
                lag=args.lag,
                quadrature_order=args.quadrature_order,
            )
            exact_shift = args.decay - exact.pseudo_decay
            leading_shift = leading_coefficient * asymptotic_scale(
                bandwidth,
                smoothness,
            )
            transition_shift = args.decay - transition.pseudo_decay
            exact_variance_loss = 1.0 - exact.variance_factor
            transition_variance_loss = 1.0 - transition.variance_factor
            records.append(
                {
                    "dimension": args.dimension,
                    "smoothness": smoothness,
                    "bandwidth": bandwidth,
                    "decay_true": args.decay,
                    "lag": args.lag,
                    "exact_pseudo_decay": exact.pseudo_decay,
                    "exact_decay_shift": exact_shift,
                    "leading_decay_shift": leading_shift,
                    "transition_pseudo_decay": transition.pseudo_decay,
                    "transition_decay_shift": transition_shift,
                    "exact_to_leading_ratio": exact_shift / leading_shift,
                    "exact_to_transition_ratio": exact_shift / transition_shift,
                    "leading_relative_error": abs(leading_shift - exact_shift)
                    / exact_shift,
                    "transition_relative_error": abs(transition_shift - exact_shift)
                    / exact_shift,
                    "exact_variance_factor": exact.variance_factor,
                    "transition_variance_factor": transition.variance_factor,
                    "transition_variance_loss_relative_error": abs(
                        transition_variance_loss - exact_variance_loss
                    )
                    / exact_variance_loss,
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
            difference = abs(
                refined.pseudo_decay - float(record["exact_pseudo_decay"])
            )
            absolute_differences.append(difference)
            relative_shift_differences.append(
                difference / float(record["exact_decay_shift"])
            )
        refinement[str(order)] = {
            "max_abs_pseudo_decay_difference": max(absolute_differences),
            "max_relative_decay_shift_difference": max(relative_shift_differences),
        }

    maximum_transition_error = max(
        float(record["transition_relative_error"]) for record in records
    )
    maximum_variance_error = max(
        float(record["transition_variance_loss_relative_error"])
        for record in records
    )
    minimum_leading_ratio = min(
        float(record["exact_to_leading_ratio"]) for record in records
    )
    validation_gates = {
        "all_exact_shifts_positive": all(
            float(record["exact_decay_shift"]) > 0 for record in records
        ),
        "all_transition_shifts_positive": all(
            float(record["transition_decay_shift"]) > 0 for record in records
        ),
        "transition_shift_relative_error_below_tolerance": maximum_transition_error
        <= args.maximum_transition_relative_error,
        "transition_variance_relative_error_below_tolerance": maximum_variance_error
        <= args.maximum_transition_relative_error,
    }
    validation_gates["all_passed"] = all(validation_gates.values())
    if not validation_gates["all_passed"]:
        raise RuntimeError(f"transition stress audit failed: {validation_gates}")

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
        "benchmark": "SupportShift transition stress audit",
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output": str(args.output)},
        "rows": len(records),
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "quadrature_refinement": refinement,
        "diagnostics": {
            "maximum_transition_shift_relative_error": maximum_transition_error,
            "maximum_transition_variance_loss_relative_error": maximum_variance_error,
            "minimum_exact_to_leading_ratio": minimum_leading_ratio,
        },
        "validation_gates": validation_gates,
    }
    write_json_atomic(args.output.with_suffix(".metadata.json"), metadata)
    metadata["result_csv"] = {
        "path": str(args.output),
        "sha256": sha256_file(args.output),
    }
    write_json_atomic(args.output.with_suffix(".metadata.json"), metadata)
    print(
        f"Wrote {len(records)} transition-stress rows to {args.output}; "
        f"maximum two-term error {maximum_transition_error:.6g}."
    )


if __name__ == "__main__":
    main()
