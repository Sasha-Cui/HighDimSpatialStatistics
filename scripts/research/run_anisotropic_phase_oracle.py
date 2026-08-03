"""Run the directional SupportShift oracle for anisotropic observation support."""
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


def fixed_trace_transform(aspect_ratio: float) -> np.ndarray:
    """Return a diagonal transform with singular-value ratio rho and fixed trace."""
    if not np.isfinite(aspect_ratio) or aspect_ratio < 1.0:
        raise ValueError("aspect_ratio must be finite and at least one")
    scale = np.sqrt(2.0 / (aspect_ratio + 1.0 / aspect_ratio))
    return scale * np.diag(
        [np.sqrt(aspect_ratio), 1.0 / np.sqrt(aspect_ratio)]
    )


def main() -> None:
    repo_root = add_src_to_path()
    from HighDimSpatial.smoothing_bias.continuous import (
        continuous_matern_pair_target,
        product_epanechnikov_decay_shift_coefficient,
        product_epanechnikov_direction_contrast_coefficient,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--lag", type=float, default=1.0)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--refinement-orders", type=int, nargs="*", default=[64, 128])
    parser.add_argument("--minimum-bandwidth", type=float, default=0.003)
    parser.add_argument("--maximum-bandwidth", type=float, default=0.15)
    parser.add_argument("--bandwidth-count", type=int, default=14)
    parser.add_argument(
        "--smoothness",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.5],
    )
    parser.add_argument(
        "--aspect-ratios",
        type=float,
        nargs="+",
        default=[1.0, 4.0],
    )
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs" / "smoothing_bias" / "anisotropic_phase_v1.csv",
    )
    args = parser.parse_args()
    if args.minimum_bandwidth <= 0.0 or args.maximum_bandwidth <= args.minimum_bandwidth:
        raise ValueError("bandwidth limits must be positive and increasing")
    if args.angle_step <= 0.0 or 90.0 % args.angle_step > 1e-10:
        raise ValueError("angle_step must divide 90 degrees")

    bandwidths = np.geomspace(
        args.minimum_bandwidth,
        args.maximum_bandwidth,
        args.bandwidth_count,
    )
    angles = np.arange(0.0, 90.0 + args.angle_step / 2.0, args.angle_step)
    records: list[dict] = []
    for aspect_ratio in args.aspect_ratios:
        transform = fixed_trace_transform(aspect_ratio)
        kernel_covariance = transform @ transform.T / 5.0
        support_radius = float(np.linalg.norm(transform @ np.ones(2)))
        if args.maximum_bandwidth >= args.lag / (4.0 * support_radius):
            raise ValueError(
                "maximum bandwidth leaves the theorem's fixed-lag Taylor neighborhood"
            )
        for smoothness in args.smoothness:
            for angle_degrees in angles:
                angle = np.deg2rad(angle_degrees)
                direction = np.array([np.cos(angle), np.sin(angle)])
                directional_variance = float(
                    direction @ kernel_covariance @ direction
                )
                coefficient = product_epanechnikov_decay_shift_coefficient(
                    dimension=2,
                    smoothness=smoothness,
                    decay=args.decay,
                    lag=args.lag,
                    quadrature_order=args.quadrature_order,
                    kernel_transform=transform,
                    lag_direction=direction,
                )
                for bandwidth in bandwidths:
                    target = continuous_matern_pair_target(
                        dimension=2,
                        smoothness=smoothness,
                        decay=args.decay,
                        bandwidth=float(bandwidth),
                        lag=args.lag,
                        quadrature_order=args.quadrature_order,
                        kernel_transform=transform,
                        lag_direction=direction,
                    )
                    scale = asymptotic_scale(float(bandwidth), smoothness)
                    shift = args.decay - target.pseudo_decay
                    leading_shift = coefficient * scale
                    records.append(
                        {
                            "aspect_ratio": aspect_ratio,
                            "angle_degrees": angle_degrees,
                            "smoothness": smoothness,
                            "bandwidth": bandwidth,
                            "decay_true": args.decay,
                            "decay_pseudo": target.pseudo_decay,
                            "decay_shift": shift,
                            "implied_range_ratio": args.decay / target.pseudo_decay,
                            "variance_factor": target.variance_factor,
                            "smoothed_correlation": target.correlation,
                            "asymptotic_scale": scale,
                            "leading_coefficient": coefficient,
                            "leading_shift": leading_shift,
                            "coefficient_ratio": shift / leading_shift,
                            "directional_kernel_variance": directional_variance,
                            "kernel_total_variance": float(np.trace(kernel_covariance)),
                            "support_radius": support_radius,
                            "quadrature_order": args.quadrature_order,
                        }
                    )

    refinement: dict[str, dict[str, float]] = {}
    for order in args.refinement_orders:
        if order == args.quadrature_order:
            continue
        absolute_differences: list[float] = []
        relative_differences: list[float] = []
        refined_by_key: dict[tuple[float, float, float, float], float] = {}
        for record in records:
            transform = fixed_trace_transform(float(record["aspect_ratio"]))
            angle = np.deg2rad(float(record["angle_degrees"]))
            direction = np.array([np.cos(angle), np.sin(angle)])
            refined = continuous_matern_pair_target(
                dimension=2,
                smoothness=float(record["smoothness"]),
                decay=args.decay,
                bandwidth=float(record["bandwidth"]),
                lag=args.lag,
                quadrature_order=order,
                kernel_transform=transform,
                lag_direction=direction,
            )
            difference = abs(refined.pseudo_decay - float(record["decay_pseudo"]))
            absolute_differences.append(difference)
            relative_differences.append(difference / abs(float(record["decay_shift"])))
            refined_by_key[
                (
                    float(record["aspect_ratio"]),
                    float(record["smoothness"]),
                    float(record["angle_degrees"]),
                    float(record["bandwidth"]),
                )
            ] = refined.pseudo_decay
        contrast_differences: list[float] = []
        for aspect_ratio in args.aspect_ratios:
            if aspect_ratio == 1.0:
                continue
            for smoothness in args.smoothness:
                for bandwidth in bandwidths:
                    primary = {
                        float(record["angle_degrees"]): float(record["decay_shift"])
                        for record in records
                        if record["aspect_ratio"] == aspect_ratio
                        and record["smoothness"] == smoothness
                        and record["bandwidth"] == bandwidth
                        and record["angle_degrees"] in (0.0, 90.0)
                    }
                    primary_contrast = primary[0.0] - primary[90.0]
                    refined_parallel = args.decay - refined_by_key[
                        (aspect_ratio, smoothness, 0.0, float(bandwidth))
                    ]
                    refined_perpendicular = args.decay - refined_by_key[
                        (aspect_ratio, smoothness, 90.0, float(bandwidth))
                    ]
                    contrast_differences.append(
                        abs(
                            refined_parallel
                            - refined_perpendicular
                            - primary_contrast
                        )
                        / abs(primary_contrast)
                    )
        refinement[str(order)] = {
            "max_abs_pseudo_decay_difference": max(absolute_differences),
            "max_relative_decay_shift_difference": max(relative_differences),
            "max_relative_directional_contrast_difference": max(
                contrast_differences,
                default=0.0,
            ),
        }

    # For every smoothness, the directional h^2 contrast has this closed
    # coefficient.  Record endpoint checks separately from the individual-shift
    # phase ratios because the contrast is lower order when nu <= 1.
    contrast_checks: list[dict] = []
    for aspect_ratio in args.aspect_ratios:
        transform = fixed_trace_transform(aspect_ratio)
        for smoothness in args.smoothness:
            contrast_coefficient = (
                product_epanechnikov_direction_contrast_coefficient(
                    dimension=2,
                    smoothness=smoothness,
                    decay=args.decay,
                    lag=args.lag,
                    kernel_transform=transform,
                    first_direction=np.array([1.0, 0.0]),
                    second_direction=np.array([0.0, 1.0]),
                )
            )
            subset = [
                record
                for record in records
                if record["aspect_ratio"] == aspect_ratio
                and record["smoothness"] == smoothness
                and record["angle_degrees"] in (0.0, 90.0)
            ]
            by_key = {
                (float(record["bandwidth"]), float(record["angle_degrees"])): record
                for record in subset
            }
            ratios = []
            for bandwidth in bandwidths:
                parallel = by_key[(float(bandwidth), 0.0)]["decay_shift"]
                perpendicular = by_key[(float(bandwidth), 90.0)]["decay_shift"]
                if contrast_coefficient > 0.0:
                    ratios.append(
                        (parallel - perpendicular)
                        / (contrast_coefficient * float(bandwidth) ** 2)
                    )
            contrast_checks.append(
                {
                    "aspect_ratio": aspect_ratio,
                    "smoothness": smoothness,
                    "contrast_coefficient": contrast_coefficient,
                    "smallest_bandwidth_ratio": ratios[0] if ratios else None,
                }
            )

    smallest_bandwidth = float(bandwidths[0])
    endpoint_phase_error = max(
        abs(float(record["coefficient_ratio"]) - 1.0)
        for record in records
        if record["bandwidth"] == smallest_bandwidth
    )
    nonzero_contrast_checks = [
        check for check in contrast_checks if check["contrast_coefficient"] > 0.0
    ]
    endpoint_contrast_error = max(
        (
            abs(float(check["smallest_bandwidth_ratio"]) - 1.0)
            for check in nonzero_contrast_checks
        ),
        default=0.0,
    )
    equal_axis_spreads: list[float] = []
    for smoothness in args.smoothness:
        for bandwidth in bandwidths:
            values = [
                float(record["decay_pseudo"])
                for record in records
                if record["aspect_ratio"] == 1.0
                and record["smoothness"] == smoothness
                and record["bandwidth"] == bandwidth
                and record["angle_degrees"] in (0.0, 90.0)
            ]
            if values:
                equal_axis_spreads.append(max(values) - min(values))
    maximum_equal_axis_spread = max(equal_axis_spreads, default=0.0)
    maximum_absolute_refinement = max(
        (
            check["max_abs_pseudo_decay_difference"]
            for check in refinement.values()
        ),
        default=0.0,
    )
    maximum_relative_shift_refinement = max(
        (
            check["max_relative_decay_shift_difference"]
            for check in refinement.values()
        ),
        default=0.0,
    )
    maximum_relative_contrast_refinement = max(
        (
            check["max_relative_directional_contrast_difference"]
            for check in refinement.values()
        ),
        default=0.0,
    )
    validation_gates = {
        "all_decay_shifts_positive": all(
            float(record["decay_shift"]) > 0.0 for record in records
        ),
        "smallest_bandwidth_phase_error_at_most_0.15": endpoint_phase_error
        <= 0.15,
        "smallest_bandwidth_contrast_error_at_most_0.10": endpoint_contrast_error
        <= 0.10,
        "equal_axis_direction_spread_at_most_1e-10": maximum_equal_axis_spread
        <= 1e-10,
        "quadrature_absolute_difference_at_most_1e-7": maximum_absolute_refinement
        <= 1e-7,
        "quadrature_relative_shift_difference_at_most_0.01": maximum_relative_shift_refinement
        <= 0.01,
        "quadrature_relative_contrast_difference_at_most_0.02": maximum_relative_contrast_refinement
        <= 0.02,
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
        "directional_contrast_checks": contrast_checks,
        "validation_diagnostics": {
            "smallest_bandwidth_phase_error": endpoint_phase_error,
            "smallest_bandwidth_contrast_error": endpoint_contrast_error,
            "maximum_equal_axis_direction_spread": maximum_equal_axis_spread,
            "maximum_absolute_quadrature_difference": maximum_absolute_refinement,
            "maximum_relative_shift_quadrature_difference": maximum_relative_shift_refinement,
            "maximum_relative_contrast_quadrature_difference": maximum_relative_contrast_refinement,
        },
        "validation_gates": validation_gates,
    }
    write_json_atomic(args.output.with_suffix(".metadata.json"), metadata)
    failed_gates = [name for name, passed in validation_gates.items() if not passed]
    if failed_gates:
        raise RuntimeError(f"anisotropic oracle validation failed: {failed_gates}")
    print(f"Wrote {len(records)} anisotropic oracle rows to {args.output}")


if __name__ == "__main__":
    main()
