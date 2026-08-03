"""Run one immutable-manifest task for the finite support-misspecification study."""
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy


def add_src_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    return repo_root


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_seed(root_seed: int, common_random_number_group: str) -> int:
    digest = hashlib.sha256(
        f"{root_seed}:{common_random_number_group}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


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


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot write an empty result shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(records[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def valid_existing_shard(path: Path, config_hash: str, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    return (
        len(records) == expected_rows
        and all(record.get("config_hash") == config_hash for record in records)
    )


def main() -> None:
    repo_root = add_src_to_path()
    from HighDimSpatial.smoothing_bias.experiment import run_finite_design_configuration

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--task-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "outputs" / "smoothing_bias",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    configurations = manifest["configurations"]
    if not 0 <= args.task_index < len(configurations):
        raise IndexError(
            f"task index {args.task_index} outside manifest range 0..{len(configurations) - 1}"
        )
    config = dict(manifest.get("defaults", {}))
    config.update(configurations[args.task_index])
    config_id = str(config["config_id"])
    crn_group = str(config.get("common_random_number_group", config_id))
    run_id = str(manifest["run_id"])
    root_seed = int(manifest["root_seed"])
    seed = derive_seed(root_seed, crn_group)
    config_hash = canonical_hash(config)
    expected_rows = 2 * int(config["replicates"])
    run_directory = args.output_root / run_id
    shard_path = run_directory / "shards" / f"task_{args.task_index:04d}.csv"
    metadata_path = run_directory / "metadata" / f"task_{args.task_index:04d}.json"
    if not args.force and valid_existing_shard(shard_path, config_hash, expected_rows):
        print(f"Validated existing shard {shard_path}; nothing to do")
        return

    started = datetime.now(timezone.utc)
    commit, dirty = git_metadata(repo_root)
    result = run_finite_design_configuration(config, seed=seed)
    completed = datetime.now(timezone.utc)
    common = {
        "run_id": run_id,
        "task_index": args.task_index,
        "config_id": config_id,
        "config_hash": config_hash,
        "common_random_number_group": crn_group,
        "dimension": config["dimension"],
        "bandwidth": config["bandwidth"],
        "smoothness": config["smoothness"],
        "nugget": config.get("nugget", 0.0),
        "boundary_trim": json.dumps(config.get("boundary_trim", 0.0)),
        "input_jitter": json.dumps(config.get("input_jitter", 0.0)),
        "number_of_inputs": result.diagnostics["number_of_inputs"],
        "number_of_outputs": result.diagnostics["number_of_outputs"],
        "git_commit": commit,
        "git_dirty": dirty,
    }
    records = [{**common, **record} for record in result.records]
    write_csv_atomic(shard_path, records)
    metadata = {
        **common,
        "manifest": str(args.manifest.resolve()),
        "manifest_hash": canonical_hash(manifest),
        "config": config,
        "root_seed": root_seed,
        "derived_seed": seed,
        "diagnostics": result.diagnostics,
        "expected_rows": expected_rows,
        "written_rows": len(records),
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    write_json_atomic(metadata_path, metadata)
    if not valid_existing_shard(shard_path, config_hash, expected_rows):
        raise RuntimeError("atomic shard failed post-write validation")
    print(
        f"Wrote {len(records)} rows for {config_id} to {shard_path} "
        f"in {metadata['elapsed_seconds']:.2f} seconds"
    )


if __name__ == "__main__":
    main()
