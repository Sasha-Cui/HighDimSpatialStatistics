import csv
import hashlib
import json
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
    finite_summary = data_directory / "finite_summary.csv"
    transition = data_directory / "transition_stress.csv"
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
            "--finite-summary",
            str(finite_summary),
            "--transition-stress",
            str(transition),
            "--highdim",
            str(result),
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
    assert manifest["schema_version"] == "1.1"
    assert manifest["inputs"]["highdim"]["sha256"] == _sha256(result)
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
