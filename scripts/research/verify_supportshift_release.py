"""Verify a SupportShift run and its paper-artifact hash contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_recorded_path(path_text: str, repository_root: Path) -> Path:
    recorded = Path(path_text)
    candidates = [recorded] if recorded.is_absolute() else [repository_root / recorded]
    candidates.append(repository_root / "outputs" / "smoothing_bias" / recorded.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def verify_run(
    metadata_path: Path,
    repository_root: Path,
    *,
    allow_dirty: bool,
    require_full: bool,
) -> tuple[dict[str, Any], list[str]]:
    metadata = load_json(metadata_path)
    failures: list[str] = []
    require(metadata.get("benchmark") == "SupportShift", "wrong benchmark name", failures)
    require(metadata.get("benchmark_version") == "1.1", "wrong benchmark version", failures)
    require(metadata.get("rows") == metadata.get("expected_rows"), "row count mismatch", failures)
    require(
        bool(metadata.get("validation_gates", {}).get("all_passed")),
        "one or more statistical validation gates failed",
        failures,
    )
    provenance = metadata.get("provenance", {})
    if not allow_dirty:
        require(provenance.get("git_dirty") is False, "run provenance is dirty", failures)
    coverage_gate = metadata.get("validation_gates", {}).get(
        "empirical_uniform_bound_coverage", {}
    )
    coverage = coverage_gate.get("cell_coverage", {})
    floor = float(coverage_gate.get("predeclared_floor", 0.0))
    require(bool(coverage), "candidatewise coverage cells are missing", failures)
    require(
        bool(coverage) and min(map(float, coverage.values())) >= floor,
        "candidatewise coverage is below its predeclared floor",
        failures,
    )
    for section in ("result_csv", "raw_example"):
        record = metadata.get(section)
        if not record:
            continue
        path = resolve_recorded_path(str(record["path"]), repository_root)
        require(path.is_file(), f"missing {section} file: {path}", failures)
        if path.is_file():
            require(
                sha256_file(path) == record["sha256"],
                f"SHA-256 mismatch for {section}: {path}",
                failures,
            )
    if require_full:
        settings = metadata.get("resolved_settings", {})
        require(metadata.get("rows") == 12_800, "full run must contain 12,800 rows", failures)
        require(settings.get("decay_grid_size") == 161, "full decay grid is not 161", failures)
        require(settings.get("variance_grid_size") == 101, "full variance grid is not 101", failures)
        require(settings.get("trials") == 200, "full run must use 200 trials", failures)
        require(len(coverage) == 64, "full run must contain 64 coverage cells", failures)
    return metadata, failures


def verify_paper_artifacts(
    paper_directory: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    manifest_path = paper_directory / "data" / "supportshift_artifact_manifest.json"
    manifest = load_json(manifest_path)
    failures: list[str] = []
    require(manifest.get("schema_version") == "1.1", "wrong manifest schema", failures)
    for name, record in manifest.get("inputs", {}).items():
        path = resolve_recorded_path(str(record["path"]), repository_root)
        require(path.is_file(), f"missing manifest input {name}: {path}", failures)
        if path.is_file():
            require(
                sha256_file(path) == record["sha256"],
                f"SHA-256 mismatch for manifest input {name}: {path}",
                failures,
            )
    for relative_path, expected_hash in manifest.get("outputs", {}).items():
        path = paper_directory / relative_path
        require(path.is_file(), f"missing manifest output: {path}", failures)
        if path.is_file():
            require(
                sha256_file(path) == expected_hash,
                f"SHA-256 mismatch for manifest output: {path}",
                failures,
            )
    for name, relative_path in manifest.get("input_output_aliases", {}).items():
        require(
            manifest.get("inputs", {}).get(name, {}).get("sha256")
            == manifest.get("outputs", {}).get(relative_path),
            f"aliased input/output hashes disagree for {name}",
            failures,
        )
    return manifest, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--paper-directory", type=Path, default=Path("paper"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    metadata, run_failures = verify_run(
        args.metadata,
        repository_root,
        allow_dirty=args.allow_dirty,
        require_full=args.require_full,
    )
    manifest, artifact_failures = verify_paper_artifacts(
        args.paper_directory,
        repository_root,
    )
    failures = run_failures + artifact_failures
    if failures:
        raise SystemExit("SupportShift release verification failed:\n- " + "\n- ".join(failures))
    coverage = metadata["validation_gates"]["empirical_uniform_bound_coverage"][
        "cell_coverage"
    ]
    print(
        "SupportShift release verified: "
        f"{metadata['rows']} rows, {len(coverage)} coverage cells, "
        f"{len(manifest['outputs'])} hashed paper artifacts."
    )


if __name__ == "__main__":
    main()
