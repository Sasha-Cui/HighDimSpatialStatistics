"""Run a legacy converted script with path remapping and magic stripping.

This avoids notebook execution while preserving legacy behavior.
"""
from __future__ import annotations

import argparse
import os
import re
import runpy
import sys
from pathlib import Path


def build_path_mappings(repo_root: Path) -> list[tuple[str, str]]:
    """Return ordered path mappings from legacy ~/project paths to repo-relative paths."""
    mappings = [
        ("~/project/41_1_train_data", str(repo_root / "data/interim/41_1_train_data")),
        ("~/project/41_2_test_locations", str(repo_root / "data/interim/41_2_test_locations")),
        ("~/project/41_3_test_cov", str(repo_root / "data/interim/41_3_test_cov")),
        ("~/project/42_randomly_subsampled_synthetic_data", str(repo_root / "data/interim/42_randomly_subsampled_synthetic_data")),
        ("~/project/42_subsampled_synthetic_data", str(repo_root / "data/interim/42_subsampled_synthetic_data")),
        ("~/project/51_1_hattie_data", str(repo_root / "data/interim/51_1_hattie_data")),
        ("~/project/52_subsampled_hattie_data", str(repo_root / "data/interim/52_subsampled_hattie_data")),
        ("~/project/43_randomly_subsampled_estimation_results", str(repo_root / "data/processed/43_randomly_subsampled_estimation_results")),
        ("~/project/43_estimation_results", str(repo_root / "data/processed/43_estimation_results")),
        ("~/project/44_cross_estimation_results", str(repo_root / "data/processed/44_cross_estimation_results")),
        ("~/project/53_hattie_marginal_estimation_results", str(repo_root / "data/processed/53_hattie_marginal_estimation_results")),
        ("~/project/python_processed_data", str(repo_root / "data/processed/python_processed_data")),
        ("~/project/R_processed_data", str(repo_root / "data/processed/R_processed_data")),
        ("~/project/synthetic_data", str(repo_root / "data/synthetic/synthetic_data")),
        ("~/project/ovary_Puck_230517_39.h5ad", str(repo_root / "data/raw/ovary_Puck_230517_39.h5ad")),
        ("~/project/mouse_ovary_slide_seq_young_estrus.h5ad", str(repo_root / "data/raw/mouse_ovary_slide_seq_young_estrus.h5ad")),
        ("~/project/archived_code", str(repo_root / "legacy/archived_code")),
        ("~/project/temporary_code", str(repo_root / "legacy/temporary_code")),
        ("~/project/ping_luo", str(repo_root / "legacy/ping_luo")),
        ("~/project/Code Packages", str(repo_root / "external/code-packages")),
        ("~/project/R", str(repo_root / "external/r")),
        ("~/project", str(repo_root)),
    ]
    return mappings


def map_legacy_path(path: str, mappings: list[tuple[str, str]]) -> str:
    for old, new in mappings:
        if path.startswith(old):
            return path.replace(old, new, 1)
    return path


def patch_expanduser(repo_root: Path, mappings: list[tuple[str, str]]):
    import os as _os

    _orig = _os.path.expanduser

    def _expanduser(path: str) -> str:
        if path.startswith("~/project"):
            return map_legacy_path(path, mappings)
        # Also handle bare "project/..." patterns
        if path.startswith("project/"):
            return str(repo_root / path)
        return _orig(path)

    _os.path.expanduser = _expanduser


def sanitize_script(source: str, repo_root: Path, mappings: list[tuple[str, str]]) -> str:
    lines = source.splitlines()
    out_lines = []

    run_re = re.compile(r"^\s*%run\s+-i\s+(.*)$")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%") or stripped.startswith("!"):
            m = run_re.match(line)
            if m:
                target = m.group(1).strip()
                target = target.replace("\"", "").replace("'", "")
                target = map_legacy_path(target, mappings)
                # map common shims
                if target.endswith("/preambles") or target.endswith("preambles"):
                    target = str(repo_root / "preambles.py")
                elif target.endswith("/helper_functions") or target.endswith("helper_functions"):
                    target = str(repo_root / "helper_functions.py")
                elif target.endswith("/fitting_functions") or target.endswith("fitting_functions"):
                    target = str(repo_root / "fitting_functions.py")
                elif target.endswith("/epilogue") or target.endswith("epilogue"):
                    target = str(repo_root / "epilogue.py")
                # ensure .py extension if missing and file exists
                if not target.endswith(".py") and Path(target + ".py").exists():
                    target = target + ".py"
                out_lines.append(f"exec(open(r'{target}').read(), globals())")
            else:
                # comment out other magics
                out_lines.append(f"# {line}")
            continue

        out_lines.append(line)

    return "\n".join(out_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Path to legacy script under scripts/legacy/ or legacy/scripts_legacy_old/")
    parser.add_argument("--output-dir", default="data/processed/logs/legacy_sanitized")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    script_path = (repo_root / args.script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy script not found: {script_path}")

    mappings = build_path_mappings(repo_root)
    patch_expanduser(repo_root, mappings)

    # Provide a minimal get_ipython stub for legacy scripts
    if "get_ipython" not in globals():
        def get_ipython():  # type: ignore
            class Dummy:
                def run_line_magic(self, *args, **kwargs):
                    return None
            return Dummy()

    source = script_path.read_text()
    sanitized = sanitize_script(source, repo_root, mappings)

    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sanitized_path = out_dir / script_path.name
    sanitized_path.write_text(sanitized)

    # Execute in a fresh globals dict with __file__
    runpy.run_path(str(sanitized_path), run_name="__main__")


if __name__ == "__main__":
    main()
