import csv
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_release_identity_requires_one_public_tag(tmp_path: Path) -> None:
    scripts_directory = REPO_ROOT / "scripts/research"
    sys.path.insert(0, str(scripts_directory))
    try:
        verifier = runpy.run_path(str(scripts_directory / "verify_supportshift_release.py"))
    finally:
        sys.path.remove(str(scripts_directory))
    verify_release_identity = verifier["verify_release_identity"]

    paper_directory = tmp_path / "paper"
    paper_directory.mkdir()
    release_tag = "supportshift-geosim-v1.2.3"
    surfaces = [
        tmp_path / "README.md",
        paper_directory / "geosim2026.tex",
        paper_directory / "manuscript.tex",
        tmp_path / "docs/research/ARTIFACT_DATA_CARD.md",
        tmp_path / "docs/research/GEOSIM_SUBMISSION_CHECKLIST.md",
    ]
    for path in surfaces:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(release_tag, encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text(
        f"version: 1.2.3\nurl: https://example.test/tree/{release_tag}\n",
        encoding="utf-8",
    )

    observed_tag, failures = verify_release_identity(tmp_path, paper_directory)
    assert observed_tag == release_tag
    assert failures == []

    surfaces[0].write_text("supportshift-geosim-v1.2.2", encoding="utf-8")
    _, failures = verify_release_identity(tmp_path, paper_directory)
    assert len(failures) == 1
    assert "release tag mismatch" in failures[0]


def test_finite_grid_fit_records_reconstruct_paper_summary(tmp_path: Path) -> None:
    scripts_directory = REPO_ROOT / "scripts/research"
    sys.path.insert(0, str(scripts_directory))
    try:
        verifier = runpy.run_path(str(scripts_directory / "verify_supportshift_release.py"))
    finally:
        sys.path.remove(str(scripts_directory))
    verify_finite_grid_artifact = verifier["verify_finite_grid_artifact"]

    manifest = json.loads(
        (REPO_ROOT / "paper/data/supportshift_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    metrics, failures = verify_finite_grid_artifact(
        manifest,
        REPO_ROOT / "paper",
        REPO_ROOT,
    )
    assert failures == []
    assert metrics["fit_rows"] == 8_400
    assert metrics["configurations"] == 21
    assert metrics["cells"] == 42

    corrupt_results = tmp_path / "results.csv"
    lines = (
        REPO_ROOT
        / "outputs/smoothing_bias/support_only_final_20260802_v2/results.csv"
    ).read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    first = lines[1].split(",")
    estimate_index = header.index("decay_estimate")
    first[estimate_index] = str(float(first[estimate_index]) + 0.25)
    lines[1] = ",".join(first)
    corrupt_results.write_text("\n".join(lines) + "\n", encoding="utf-8")
    corrupt_manifest = json.loads(json.dumps(manifest))
    corrupt_manifest["inputs"]["finite_results"]["path"] = str(corrupt_results)
    _, failures = verify_finite_grid_artifact(
        corrupt_manifest,
        REPO_ROOT / "paper",
        REPO_ROOT,
    )
    assert any("signed-error identity" in failure for failure in failures)
    assert any("finite summary mismatch" in failure for failure in failures)


def test_highdimensional_driver_shakedown_is_self_auditing(tmp_path: Path) -> None:
    result = tmp_path / "supportshift_shakedown.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run_supportshift_highdim.py",
            "--preset",
            "shakedown",
            "--output",
            str(result),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    metadata_path = result.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with result.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert metadata["benchmark_version"] == "1.1"
    assert metadata["rows"] == metadata["expected_rows"] == len(rows) == 16
    assert metadata["result_csv"]["sha256"] == _sha256(result)
    assert metadata["validation_gates"]["all_passed"]
    assert metadata["validation_gates"]["finite_grid_approximates_continuous_oracle"][
        "passed"
    ]
    coverage = metadata["validation_gates"]["empirical_uniform_bound_coverage"]
    assert coverage["event"] == "all candidates satisfy their own theorem radius"
    assert len(coverage["cell_coverage"]) == 4
    assert min(coverage["cell_coverage"].values()) >= 0.90
    assert {int(row["candidate_count"]) for row in rows} == {15}
    assert all(
        row["simultaneous_candidatewise_bound_holds"] == "True" for row in rows
    )
    assert max(
        float(row["max_candidatewise_deviation_to_bound_ratio"]) for row in rows
    ) <= 1.0


def test_paper_artifact_builder_records_input_and_output_hashes(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    result = tmp_path / "supportshift_shakedown.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run_supportshift_highdim.py",
            "--preset",
            "shakedown",
            "--output",
            str(result),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paper_directory = tmp_path / "paper"
    data_directory = paper_directory / "data"
    data_directory.mkdir(parents=True)
    phase = data_directory / "phase_oracle_d2.csv"
    phase_metadata = REPO_ROOT / "outputs/smoothing_bias/phase_oracle_d2_v2.metadata.json"
    finite_summary = data_directory / "finite_summary.csv"
    finite_results = (
        REPO_ROOT
        / "outputs/smoothing_bias/support_only_final_20260802_v2/results.csv"
    )
    finite_audit = (
        REPO_ROOT
        / "outputs/smoothing_bias/support_only_final_20260802_v2/audit.json"
    )
    finite_manifest = REPO_ROOT / "configs/smoothing_bias/support_only_20260802.json"
    transition = data_directory / "transition_stress.csv"
    dimension_kernel = (
        REPO_ROOT
        / "outputs/smoothing_bias/supportshift_dimension_kernel_robustness_20260804.csv"
    )
    shutil.copyfile(REPO_ROOT / "paper/data/phase_oracle_d2.csv", phase)
    shutil.copyfile(REPO_ROOT / "paper/data/finite_summary.csv", finite_summary)
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run_transition_stress_audit.py",
            "--smoothness-count",
            "5",
            "--bandwidths",
            "0.05",
            "--refinement-orders",
            "--output",
            str(transition),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    phase_hash_before = _sha256(phase)
    finite_hash_before = _sha256(finite_summary)
    subprocess.run(
        [
            sys.executable,
            "scripts/research/make_support_paper_artifacts.py",
            "--phase",
            str(phase),
            "--phase-metadata",
            str(phase_metadata),
            "--finite-summary",
            str(finite_summary),
            "--finite-results",
            str(finite_results),
            "--finite-audit",
            str(finite_audit),
            "--finite-manifest",
            str(finite_manifest),
            "--transition-stress",
            str(transition),
            "--transition-stress-metadata",
            str(transition.with_suffix(".metadata.json")),
            "--dimension-kernel-robustness",
            str(dimension_kernel),
            "--dimension-kernel-robustness-metadata",
            str(dimension_kernel.with_suffix(".metadata.json")),
            "--highdim",
            str(result),
            "--highdim-metadata",
            str(result.with_suffix(".metadata.json")),
            "--paper-directory",
            str(paper_directory),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_path = paper_directory / "data" / "supportshift_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.3"
    assert manifest["inputs"]["highdim"]["sha256"] == _sha256(result)
    assert manifest["inputs"]["highdim_metadata"]["sha256"] == _sha256(
        result.with_suffix(".metadata.json")
    )
    assert manifest["inputs"]["dimension_kernel_robustness"]["sha256"] == _sha256(
        dimension_kernel
    )
    assert manifest["inputs"]["finite_results"]["sha256"] == _sha256(finite_results)
    assert manifest["inputs"]["finite_audit"]["sha256"] == _sha256(finite_audit)
    assert manifest["inputs"]["finite_manifest"]["sha256"] == _sha256(finite_manifest)
    assert manifest["input_output_aliases"] == {
        "finite_summary": "data/finite_summary.csv",
        "phase": "data/phase_oracle_d2.csv",
        "transition_stress": "data/transition_stress.csv",
    }
    assert _sha256(phase) == phase_hash_before
    assert _sha256(finite_summary) == finite_hash_before
    assert manifest["inputs"]["phase"]["sha256"] == manifest["outputs"][
        "data/phase_oracle_d2.csv"
    ]
    assert manifest["inputs"]["finite_summary"]["sha256"] == manifest[
        "outputs"
    ]["data/finite_summary.csv"]
    assert "figures/supportshift_highdim.pdf" in manifest["outputs"]
    assert "figures/transition_stress.pdf" in manifest["outputs"]
    assert "tables/dimension_kernel_robustness.tex" in manifest["outputs"]
    assert manifest["outputs"]["data/dimension_kernel_robustness.csv"] == _sha256(
        dimension_kernel
    )
    for relative_path, expected_hash in manifest["outputs"].items():
        assert _sha256(paper_directory / relative_path) == expected_hash
    summary_header = (
        paper_directory / "data" / "supportshift_highdim_summary.csv"
    ).read_text(encoding="utf-8").splitlines()[0]
    assert "candidatewise_simultaneous_coverage" in summary_header
    verification = subprocess.run(
        [
            sys.executable,
            "scripts/research/verify_supportshift_release.py",
            "--metadata",
            str(result.with_suffix(".metadata.json")),
            "--paper-directory",
            str(paper_directory),
            "--repository-root",
            str(REPO_ROOT),
            "--allow-dirty",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SupportShift release verified" in verification.stdout


def test_transition_stress_driver_records_gates_and_hash(tmp_path: Path) -> None:
    result = tmp_path / "transition.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run_transition_stress_audit.py",
            "--smoothness-count",
            "5",
            "--bandwidths",
            "0.05",
            "--refinement-orders",
            "--output",
            str(result),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(
        result.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    with result.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert metadata["schema_version"] == "1.0"
    assert metadata["rows"] == len(rows) == 5
    assert metadata["validation_gates"]["all_passed"]
    assert metadata["result_csv"]["sha256"] == _sha256(result)
    assert max(float(row["transition_relative_error"]) for row in rows) <= 0.002


def test_dimension_kernel_driver_records_gates_and_hash(tmp_path: Path) -> None:
    result = tmp_path / "dimension_kernel.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run_dimension_kernel_robustness.py",
            "--dimensions",
            "1",
            "2",
            "--kernel-families",
            "epanechnikov",
            "uniform",
            "--smoothness",
            "0.5",
            "1.5",
            "--bandwidths",
            "0.004",
            "0.008",
            "--quadrature-order",
            "24",
            "--refinement-orders",
            "16",
            "32",
            "--allow-dirty",
            "--output",
            str(result),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(
        result.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    with result.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert metadata["schema_version"] == "1.0"
    assert metadata["rows"] == metadata["expected_rows"] == len(rows) == 16
    assert metadata["factor_grid"]["dimensions"] == [1, 2]
    assert metadata["factor_grid"]["kernel_families"] == [
        "epanechnikov",
        "uniform",
    ]
    assert metadata["validation_gates"]["all_passed"]
    assert metadata["result_csv"]["sha256"] == _sha256(result)
    assert min(float(row["decay_shift"]) for row in rows) > 0.0


def test_multilag_composite_driver_records_genuine_misspecification(
    tmp_path: Path,
) -> None:
    result = tmp_path / "multilag.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run_multilag_composite_audit.py",
            "--lags",
            "0.5",
            "1.0",
            "--smoothness",
            "0.5",
            "1.5",
            "--bandwidths",
            "0.005",
            "0.01",
            "--quadrature-order",
            "48",
            "--allow-dirty",
            "--output",
            str(result),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(
        result.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    with result.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert metadata["schema_version"] == "1.0"
    assert metadata["rows"] == metadata["expected_rows"] == len(rows) == 8
    assert metadata["validation_gates"]["all_passed"]
    assert metadata["validation_gates"]["genuine_multilag_misspecification"][
        "passed"
    ]
    assert metadata["result_csv"]["sha256"] == _sha256(result)
    assert min(float(row["minimum_composite_kl"]) for row in rows) > 0.0


def test_full_likelihood_driver_records_genuine_misspecification(
    tmp_path: Path,
) -> None:
    result = tmp_path / "full_likelihood.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run_full_likelihood_phase_audit.py",
            "--smoothness",
            "0.5",
            "1.5",
            "--bandwidths",
            "0.005",
            "0.01",
            "--quadrature-order",
            "48",
            "--allow-dirty",
            "--output",
            str(result),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(
        result.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    with result.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert metadata["rows"] == metadata["expected_rows"] == len(rows) == 4
    assert metadata["validation_gates"]["all_passed"]
    assert metadata["validation_gates"]["genuine_full_likelihood_misspecification"][
        "passed"
    ]
    assert metadata["result_csv"]["sha256"] == _sha256(result)


def test_joint_smoothness_driver_includes_partial_support_model(
    tmp_path: Path,
) -> None:
    result = tmp_path / "joint.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run_joint_smoothness_audit.py",
            "--input-side",
            "7",
            "--true-smoothness",
            "0.5",
            "--bandwidths",
            "0.35",
            "--sample-size",
            "2",
            "--replicates",
            "3",
            "--allow-dirty",
            "--output",
            str(result),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(
        result.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    summary = result.with_name(f"{result.stem}.summary.csv")
    with result.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert metadata["rows"] == len(rows) == 9
    assert metadata["summary_rows"] == 3
    assert metadata["validation_gates"]["all_passed"]
    assert metadata["validation_gates"][
        "partial_support_improves_population_criterion"
    ]["passed"]
    assert {row["model"] for row in rows} == {
        "support_aware",
        "partial_support",
        "point_support",
    }
    assert metadata["result_csv"]["sha256"] == _sha256(result)
    assert metadata["summary_csv"]["sha256"] == _sha256(summary)


def test_matched_boundary_driver_holds_output_dimension_fixed(
    tmp_path: Path,
) -> None:
    result = tmp_path / "matched_boundary.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/research/run_matched_boundary_audit.py",
            "--latent-side",
            "11",
            "--block-side",
            "3",
            "--interior-origin",
            "1.0",
            "--smoothness",
            "0.5",
            "--bandwidths",
            "0.5",
            "--allow-dirty",
            "--output",
            str(result),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(
        result.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    with result.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert metadata["rows"] == len(rows) == 6
    assert metadata["validation_gates"]["all_passed"]
    dimensions = {int(row["output_dimension"]) for row in rows}
    assert dimensions == {9}
    assert {row["region"] for row in rows} == {"boundary", "interior"}
    assert metadata["validation_gates"]["boundary_effect_at_matched_dimension"][
        "passed"
    ]
    assert metadata["result_csv"]["sha256"] == _sha256(result)


def test_paper_claim_audit_matches_promoted_artifacts() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/verify_supportshift_claims.py",
            "--repository-root",
            str(REPO_ROOT),
            "--paper-directory",
            str(REPO_ROOT / "paper"),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SupportShift paper claims verified" in result.stdout


def test_paper_claim_audit_rejects_changed_source_table(tmp_path: Path) -> None:
    paper_directory = tmp_path / "paper"
    shutil.copytree(REPO_ROOT / "paper" / "data", paper_directory / "data")
    phase = paper_directory / "data" / "phase_oracle_d2.csv"
    lines = phase.read_text(encoding="utf-8").splitlines()
    phase.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/verify_supportshift_claims.py",
            "--repository-root",
            str(REPO_ROOT),
            "--paper-directory",
            str(paper_directory),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "phase rows" in result.stderr
