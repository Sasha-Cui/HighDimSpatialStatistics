"""Convert Jupyter notebooks to Python scripts.

Usage:
    python scripts/tools/convert_notebooks.py --output-root scripts/legacy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def notebook_to_script(nb_path: Path, out_path: Path) -> None:
    data = json.loads(nb_path.read_text())
    cells = data.get("cells", [])

    lines = []
    lines.append(f"# Generated from {nb_path.as_posix()} on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py")
    lines.append("")

    for cell in cells:
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        lines.append("# %%")
        if cell_type == "markdown":
            for line in source:
                line = line.rstrip("\n")
                lines.append(f"# {line}" if line else "#")
        elif cell_type == "code":
            if not source:
                lines.append("pass")
            else:
                lines.extend([s.rstrip("\n") for s in source])
        else:
            for line in source:
                line = line.rstrip("\n")
                lines.append(f"# {line}" if line else "#")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="scripts/legacy")
    parser.add_argument("--include-hidden", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / args.output_root
    legacy_notebooks_root = repo_root / "notebooks" / "legacy"

    notebooks = []
    for nb in repo_root.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in nb.parts:
            continue
        if not args.include_hidden and any(part.startswith(".") for part in nb.parts):
            continue
        notebooks.append(nb)

    for nb in notebooks:
        if legacy_notebooks_root in nb.parents:
            rel = nb.relative_to(legacy_notebooks_root)
        else:
            rel = nb.relative_to(repo_root)
        out_path = output_root / rel
        out_path = out_path.with_suffix(".py")
        notebook_to_script(nb, out_path)

    print(f"Converted {len(notebooks)} notebooks to {output_root}")


if __name__ == "__main__":
    main()
