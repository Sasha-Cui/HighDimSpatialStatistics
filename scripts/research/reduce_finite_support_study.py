"""Audit and reduce immutable finite-support simulation shards."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot write an empty aggregate")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(records[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    run_directory = args.output_root / manifest["run_id"]

    audit: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "manifest_hash": canonical_hash(manifest),
        "missing_tasks": [],
        "invalid_tasks": [],
        "valid_tasks": [],
    }
    aggregate: list[dict[str, str]] = []
    for task_index, override in enumerate(manifest["configurations"]):
        config = dict(manifest.get("defaults", {}))
        config.update(override)
        expected_hash = canonical_hash(config)
        expected_rows = 2 * int(config["replicates"])
        shard = run_directory / "shards" / f"task_{task_index:04d}.csv"
        if not shard.exists():
            audit["missing_tasks"].append(task_index)
            continue
        try:
            with shard.open("r", encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
        except (OSError, csv.Error) as error:
            audit["invalid_tasks"].append({"task_index": task_index, "reason": str(error)})
            continue
        reasons: list[str] = []
        if len(records) != expected_rows:
            reasons.append(f"expected {expected_rows} rows, found {len(records)}")
        if any(record.get("config_hash") != expected_hash for record in records):
            reasons.append("config hash mismatch")
        key_pairs = {(record.get("model"), record.get("replicate")) for record in records}
        if len(key_pairs) != len(records):
            reasons.append("duplicate model/replicate key")
        if reasons:
            audit["invalid_tasks"].append(
                {"task_index": task_index, "reason": "; ".join(reasons)}
            )
            continue
        audit["valid_tasks"].append(task_index)
        aggregate.extend(records)

    audit["complete"] = not audit["missing_tasks"] and not audit["invalid_tasks"]
    audit["valid_task_count"] = len(audit["valid_tasks"])
    audit["expected_task_count"] = len(manifest["configurations"])
    audit["aggregate_rows"] = len(aggregate)
    write_json_atomic(run_directory / "audit.json", audit)
    if not audit["complete"]:
        raise RuntimeError(
            f"run is incomplete: missing={audit['missing_tasks']}, invalid={audit['invalid_tasks']}"
        )

    write_csv_atomic(run_directory / "results.csv", aggregate)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for record in aggregate:
        grouped.setdefault((record["config_id"], record["model"]), []).append(record)
    summary: list[dict[str, Any]] = []
    for (config_id, model), records in sorted(grouped.items()):
        estimates = np.asarray([float(record["decay_estimate"]) for record in records])
        truth = float(records[0]["decay_true"])
        population_target = float(records[0]["population_target"])
        boundary_fits = sum(record["at_bound"].lower() == "true" for record in records)
        summary.append(
            {
                "config_id": config_id,
                "model": model,
                "dimension": records[0]["dimension"],
                "bandwidth": records[0]["bandwidth"],
                "smoothness": records[0]["smoothness"],
                "number_of_inputs": records[0]["number_of_inputs"],
                "number_of_outputs": records[0]["number_of_outputs"],
                "replicates": len(records),
                "decay_true": truth,
                "population_target": population_target,
                "mean_estimate": float(np.mean(estimates)),
                "median_estimate": float(np.median(estimates)),
                "standard_deviation": float(np.std(estimates, ddof=1)),
                "bias_from_truth": float(np.mean(estimates) - truth),
                "bias_from_population_target": float(np.mean(estimates) - population_target),
                "rmse_from_truth": float(np.sqrt(np.mean((estimates - truth) ** 2))),
                "monte_carlo_standard_error_mean": float(
                    np.std(estimates, ddof=1) / np.sqrt(len(estimates))
                ),
                "boundary_fits": boundary_fits,
            }
        )
    write_csv_atomic(run_directory / "summary.csv", summary)
    print(
        f"Validated {len(audit['valid_tasks'])} tasks and wrote "
        f"{len(aggregate)} rows to {run_directory}"
    )


if __name__ == "__main__":
    main()
