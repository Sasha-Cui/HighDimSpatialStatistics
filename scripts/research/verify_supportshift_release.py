"""Verify a SupportShift run and its paper-artifact hash contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from verify_supportshift_claims import audit_claims


RELEASE_VERSION_PATTERN = re.compile(
    r"(?m)^version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*$"
)
RELEASE_TAG_PATTERN = re.compile(r"supportshift-geosim-v[0-9]+\.[0-9]+\.[0-9]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_seed(root_seed: int, common_random_number_group: str) -> int:
    digest = hashlib.sha256(
        f"{root_seed}:{common_random_number_group}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


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
    require(manifest.get("schema_version") == "1.3", "wrong manifest schema", failures)
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


def verify_finite_grid_artifact(
    manifest: dict[str, Any],
    paper_directory: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Reconstruct the 42-row finite-grid summary from all 8,400 fit records."""
    failures: list[str] = []
    required_inputs = ("finite_results", "finite_audit", "finite_manifest")
    inputs = manifest.get("inputs", {})
    for name in required_inputs:
        require(name in inputs, f"manifest input {name} is required", failures)
    if failures:
        return {}, failures

    paths = {
        name: resolve_recorded_path(str(inputs[name]["path"]), repository_root)
        for name in required_inputs
    }
    summary_path = paper_directory / "data" / "finite_summary.csv"
    for name, path in paths.items():
        require(path.is_file(), f"missing finite-grid {name}: {path}", failures)
    require(summary_path.is_file(), f"missing finite-grid summary: {summary_path}", failures)
    if failures:
        return {}, failures

    try:
        run_manifest = load_json(paths["finite_manifest"])
        reducer_audit = load_json(paths["finite_audit"])
        results = pd.read_csv(paths["finite_results"])
        summary = pd.read_csv(summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {}, [f"could not read finite-grid artifact: {error}"]

    configurations = run_manifest.get("configurations", [])
    defaults = run_manifest.get("defaults", {})
    expected_tasks = len(configurations)
    expected_rows = sum(
        2 * int((defaults | override).get("replicates", 0))
        for override in configurations
    )
    run_id = str(run_manifest.get("run_id", ""))
    require(run_id == "support_only_final_20260802_v2", "wrong finite-grid run id", failures)
    require(reducer_audit.get("run_id") == run_id, "finite audit run id mismatch", failures)
    require(
        reducer_audit.get("manifest_hash") == canonical_hash(run_manifest),
        "finite audit manifest hash mismatch",
        failures,
    )
    require(reducer_audit.get("complete") is True, "finite reducer audit is incomplete", failures)
    require(
        reducer_audit.get("missing_tasks") == [],
        "finite reducer audit reports missing tasks",
        failures,
    )
    require(
        reducer_audit.get("invalid_tasks") == [],
        "finite reducer audit reports invalid tasks",
        failures,
    )
    require(
        reducer_audit.get("valid_task_count") == expected_tasks == 21,
        "finite reducer task count mismatch",
        failures,
    )
    require(
        reducer_audit.get("aggregate_rows") == expected_rows == 8_400,
        "finite reducer row count mismatch",
        failures,
    )

    required_columns = {
        "run_id",
        "task_index",
        "config_id",
        "config_hash",
        "common_random_number_group",
        "dimension",
        "bandwidth",
        "smoothness",
        "number_of_inputs",
        "number_of_outputs",
        "git_commit",
        "git_dirty",
        "model",
        "replicate",
        "seed",
        "decay_estimate",
        "population_target",
        "decay_true",
        "signed_error",
        "absolute_error",
        "squared_error",
        "objective",
        "at_bound",
    }
    missing_columns = sorted(required_columns.difference(results.columns))
    require(not missing_columns, f"finite results missing columns: {missing_columns}", failures)
    if missing_columns:
        return {}, failures

    require(len(results) == expected_rows, "finite results row count mismatch", failures)
    require(results["run_id"].eq(run_id).all(), "finite results run id mismatch", failures)
    require(
        not results.duplicated(["task_index", "model", "replicate"]).any(),
        "finite results contain duplicate task/model/replicate keys",
        failures,
    )
    numeric_columns = [
        "decay_estimate",
        "population_target",
        "decay_true",
        "signed_error",
        "absolute_error",
        "squared_error",
        "objective",
    ]
    numeric = results[numeric_columns].apply(pd.to_numeric, errors="coerce")
    require(
        np.isfinite(numeric.to_numpy()).all(),
        "finite results contain non-finite numerical values",
        failures,
    )
    require(
        results["at_bound"].isin([True, False]).all(),
        "finite results contain invalid boundary-fit flags",
        failures,
    )
    require(
        results["git_dirty"].eq(False).all(),
        "finite results were generated from a dirty worktree",
        failures,
    )
    require(
        results["git_commit"].nunique() == 1,
        "finite results mix generation commits",
        failures,
    )
    require(
        np.allclose(
            numeric["signed_error"],
            numeric["decay_estimate"] - numeric["decay_true"],
            rtol=1e-12,
            atol=1e-12,
        ),
        "finite signed-error identity failed",
        failures,
    )
    require(
        np.allclose(
            numeric["absolute_error"],
            np.abs(numeric["signed_error"]),
            rtol=1e-12,
            atol=1e-12,
        ),
        "finite absolute-error identity failed",
        failures,
    )
    require(
        np.allclose(
            numeric["squared_error"],
            numeric["signed_error"] ** 2,
            rtol=1e-12,
            atol=1e-12,
        ),
        "finite squared-error identity failed",
        failures,
    )

    for task_index, override in enumerate(configurations):
        config = defaults | override
        task = results.loc[results["task_index"] == task_index]
        replicates = int(config["replicates"])
        expected_keys = {
            (model, replicate)
            for model in ("corrected", "naive")
            for replicate in range(replicates)
        }
        observed_keys = set(
            task[["model", "replicate"]].itertuples(index=False, name=None)
        )
        require(
            observed_keys == expected_keys,
            f"finite task {task_index} has incomplete model/replicate keys",
            failures,
        )
        expected_config_hash = canonical_hash(config)
        require(
            task["config_id"].eq(str(config["config_id"])).all(),
            f"finite task {task_index} config id mismatch",
            failures,
        )
        require(
            task["config_hash"].eq(expected_config_hash).all(),
            f"finite task {task_index} config hash mismatch",
            failures,
        )
        common_group = str(config.get("common_random_number_group", config["config_id"]))
        expected_seed = derive_seed(int(run_manifest["root_seed"]), common_group)
        require(
            task["seed"].eq(expected_seed).all(),
            f"finite task {task_index} deterministic seed mismatch",
            failures,
        )
        require(
            all(
                task.loc[task["model"] == model, "population_target"].nunique() == 1
                for model in ("corrected", "naive")
            ),
            f"finite task {task_index} has inconsistent population targets",
            failures,
        )

    recomputed: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for (config_id, model), group in results.groupby(["config_id", "model"], sort=True):
        estimates = group["decay_estimate"].to_numpy(dtype=float)
        truth = float(group["decay_true"].iloc[0])
        target = float(group["population_target"].iloc[0])
        standard_deviation = float(np.std(estimates, ddof=1))
        recomputed[(str(config_id), str(model))] = {
            "dimension": int(group["dimension"].iloc[0]),
            "bandwidth": float(group["bandwidth"].iloc[0]),
            "smoothness": float(group["smoothness"].iloc[0]),
            "number_of_inputs": int(group["number_of_inputs"].iloc[0]),
            "number_of_outputs": int(group["number_of_outputs"].iloc[0]),
            "replicates": len(group),
            "decay_true": truth,
            "population_target": target,
            "mean_estimate": float(np.mean(estimates)),
            "median_estimate": float(np.median(estimates)),
            "standard_deviation": standard_deviation,
            "bias_from_truth": float(np.mean(estimates) - truth),
            "bias_from_population_target": float(np.mean(estimates) - target),
            "rmse_from_truth": float(np.sqrt(np.mean((estimates - truth) ** 2))),
            "monte_carlo_standard_error_mean": float(
                standard_deviation / np.sqrt(len(estimates))
            ),
            "boundary_fits": int(group["at_bound"].sum()),
        }
    summary_keys = set(zip(summary["config_id"], summary["model"], strict=True))
    require(
        summary_keys == set(recomputed),
        "finite summary has missing or unexpected model/configuration cells",
        failures,
    )
    exact_fields = (
        "dimension",
        "number_of_inputs",
        "number_of_outputs",
        "replicates",
        "boundary_fits",
    )
    numeric_fields = tuple(
        field
        for field in next(iter(recomputed.values()))
        if field not in exact_fields
    )
    for record in summary.to_dict(orient="records"):
        key = (str(record["config_id"]), str(record["model"]))
        expected = recomputed.get(key)
        if expected is None:
            continue
        for field in exact_fields:
            require(
                int(record[field]) == expected[field],
                f"finite summary mismatch for {key} field {field}",
                failures,
            )
        for field in numeric_fields:
            require(
                bool(
                    np.isclose(
                        float(record[field]),
                        float(expected[field]),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ),
                f"finite summary mismatch for {key} field {field}",
                failures,
            )

    metrics = {
        "fit_rows": len(results),
        "configurations": len(configurations),
        "cells": len(recomputed),
        "boundary_fits": int(results["at_bound"].sum()),
        "generation_commit": str(results["git_commit"].iloc[0]),
    }
    return metrics, failures


def verify_release_identity(
    repository_root: Path,
    paper_directory: Path,
) -> tuple[str | None, list[str]]:
    """Require one release tag across the public citation and artifact surfaces."""
    failures: list[str] = []
    citation_path = repository_root / "CITATION.cff"
    require(citation_path.is_file(), f"missing release metadata: {citation_path}", failures)
    if not citation_path.is_file():
        return None, failures

    citation_text = citation_path.read_text(encoding="utf-8")
    version_match = RELEASE_VERSION_PATTERN.search(citation_text)
    require(version_match is not None, "CITATION.cff has no semantic version", failures)
    if version_match is None:
        return None, failures

    release_tag = f"supportshift-geosim-v{version_match.group(1)}"
    release_surfaces = [
        citation_path,
        repository_root / "README.md",
        paper_directory / "geosim2026.tex",
        paper_directory / "manuscript.tex",
        repository_root / "docs/research/ARTIFACT_DATA_CARD.md",
        repository_root / "docs/research/GEOSIM_SUBMISSION_CHECKLIST.md",
    ]
    for path in release_surfaces:
        require(path.is_file(), f"missing release-identity surface: {path}", failures)
        if not path.is_file():
            continue
        observed_tags = set(RELEASE_TAG_PATTERN.findall(path.read_text(encoding="utf-8")))
        require(
            observed_tags == {release_tag},
            f"release tag mismatch in {path}: observed {sorted(observed_tags)!r}; "
            f"required {release_tag}",
            failures,
        )
    return release_tag, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--paper-directory", type=Path, default=Path("paper"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    paper_directory = args.paper_directory
    if not paper_directory.is_absolute():
        paper_directory = repository_root / paper_directory
    metadata, run_failures = verify_run(
        args.metadata,
        repository_root,
        allow_dirty=args.allow_dirty,
        require_full=args.require_full,
    )
    manifest, artifact_failures = verify_paper_artifacts(
        paper_directory,
        repository_root,
    )
    finite_metrics, finite_failures = verify_finite_grid_artifact(
        manifest,
        paper_directory,
        repository_root,
    )
    claim_ledger = None
    claim_failures: list[str] = []
    release_tag = None
    identity_failures: list[str] = []
    if args.require_full:
        claim_ledger = audit_claims(repository_root, paper_directory)
        claim_failures = [
            f"paper claim {failure['claim']}: observed {failure['observed']!r}; "
            f"required {failure['requirement']}"
            for failure in claim_ledger["failures"]
        ]
        release_tag, identity_failures = verify_release_identity(
            repository_root,
            paper_directory,
        )
    failures = (
        run_failures
        + artifact_failures
        + finite_failures
        + claim_failures
        + identity_failures
    )
    if failures:
        raise SystemExit("SupportShift release verification failed:\n- " + "\n- ".join(failures))
    coverage = metadata["validation_gates"]["empirical_uniform_bound_coverage"][
        "cell_coverage"
    ]
    print(
        "SupportShift release verified: "
        f"{metadata['rows']} replicated fits, {finite_metrics['fit_rows']} finite-grid fits, "
        f"{len(coverage)} coverage cells, "
        f"{len(manifest['inputs'])} hashed source inputs, "
        f"{len(manifest['outputs'])} hashed paper artifacts"
        + (
            f", {claim_ledger['summary']['passed']} paper claims, release {release_tag}."
            if claim_ledger is not None
            else "."
        )
    )


if __name__ == "__main__":
    main()
