# HighDimSpatialStatistics

HighDimSpatial is a cleaned, script-first implementation of the original notebook-based pipeline for high-dimensional spatial statistics.
All original notebooks are preserved under `notebooks/legacy/`, and one-to-one converted scripts live in `scripts/legacy/` for traceability.

## Quick Start

1. Create the conda environment:

```bash
conda env create -f environment.yml
conda activate research
```

2. Run a pipeline script:

```bash
python scripts/pipeline/generate_synthetic.py --n-locations 500
python scripts/pipeline/fit_marginals.py --input data/synthetic/genton_dataset.pt
```

## Repository Layout

- `src/HighDimSpatial/`: Core Python package (clean, importable, tested).
- `scripts/pipeline/`: Canonical CLI scripts for each pipeline stage.
- `scripts/legacy/`: Auto-converted notebook scripts (read-only).
- `notebooks/legacy/`: Original notebooks preserved for reference.
- `data/`: Repo-relative data root (raw/interim/processed/synthetic).
- `docs/`: Usage, migration, and audit notes.
- `references/`: Papers and books.
- `external/`: Third-party code/packages.
- `legacy/`: Archived or temporary code artifacts.

## Data Paths

All new code uses repo-relative paths via `HighDimSpatial.config`.
Set `HIGHDIMSPATIAL_DATA_DIR` if you want to point to a different data root.

## Legacy Notebooks

Notebooks are preserved under `notebooks/legacy/`. Their converted counterparts are in `scripts/legacy/`.
These legacy scripts are for reference and reproducibility only; new development happens in `src/HighDimSpatial/` and `scripts/pipeline/`.

## Environment

The primary environment spec is `environment.yml`.

## Status

The repository is in active cleanup. Core modules, conversion tooling, and initial pipeline scripts are in place.
Next: broaden pipeline coverage and add higher-level documentation.
