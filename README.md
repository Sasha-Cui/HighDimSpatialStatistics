# HighDimSpatialStatistics

This repository now has two clearly separated layers: a preserved legacy
high-dimensional spatial pipeline and the audited **SupportShift** research
artifact. SupportShift is a theorem-linked synthetic benchmark for range
distortion caused by fitting locally averaged Matérn observations as points.
The original notebooks remain under `notebooks/legacy/` for traceability; their
archived numerical claims are not paper evidence.

## Quick Start

1. Create the paper environment:

```bash
conda env create -f environment-research.yml
conda activate highdimspatial-research
```

2. Run the maintained checks or a pipeline script:

```bash
python -m pytest -q
python -m scripts.pipeline.generate_synthetic --n-locations 500
python -m scripts.pipeline.fit_marginals --input data/synthetic/genton_dataset.pt
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

The portable paper environment is `environment-research.yml`. The promoted
Bouchet snapshot and central-environment pointer are recorded in
`Environments/supportshift-bouchet-pip-snapshot.txt` and `environment.toml`.

## Research status

The original numerical results are not scientifically valid after an August 2026
audit found kernel-convention, cross-covariance-sign, vector-ordering, gradient,
likelihood, and smoothing-covariance defects. The paper artifact was developed
on `research/paper-audit`; consult `docs/research/FINAL_RESEARCH_ASSESSMENT.md`
and `docs/research/SUPPORTSHIFT_BENCHMARK.md` before using any result.

The canonical smoother now saves its linear operators, but the ordinary marginal
fitter is **not** a valid downstream fit for smoothed groups. Use the dedicated
smoothing-bias research workflow until a corrected production fitter is added.

The defensible paper studies the Matérn pseudo-range induced by ignored local
observation support. Its central novelty is the all-smoothness phase law and
directional coefficient. A standard finite-library Gaussian quadratic-form
certificate supplies the high-dimensional-probability framing for independent
replicated spatial fields; it is supporting machinery, not a new concentration
inequality. The technical and workshop manuscripts are `paper/manuscript.tex`
and `paper/geosim2026.tex`.

## SupportShift reproduction

Run the deterministic phase and directional oracles:

```bash
python scripts/research/run_matern_phase_oracle.py \
  --output outputs/smoothing_bias/phase_oracle_d2_v2.csv

python scripts/research/run_anisotropic_phase_oracle.py \
  --output outputs/smoothing_bias/supportshift_anisotropic.csv
```

Run a small end-to-end replicated-field check locally:

```bash
python scripts/research/run_supportshift_highdim.py \
  --preset shakedown \
  --output outputs/smoothing_bias/supportshift_shakedown.csv \
  --raw-example-output outputs/smoothing_bias/supportshift_raw_shakedown.csv
```

The promoted full run uses one CPU task, refuses a dirty worktree, and records
the authorized `pi_jss233` allocation:

```bash
sbatch scripts/slurm/supportshift_final.sbatch
```

Regenerate every paper figure, table, compact source-data extract, and SHA-256
manifest from validated inputs:

```bash
python scripts/research/make_support_paper_artifacts.py \
  --phase paper/data/phase_oracle_d2.csv \
  --finite-summary paper/data/finite_summary.csv \
  --anisotropy outputs/smoothing_bias/supportshift_anisotropic_final_20260803.csv \
  --highdim outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.csv \
  --raw-example outputs/smoothing_bias/supportshift_raw_final_v2_20260803.csv \
  --paper-directory paper
```

The promoted schema is 1.1: $p\in\{16,36,64,100\}$,
$N\in\{1,4,16,64\}$, 200 trials per design, and a fixed
$161\times101=16{,}261$-candidate variance--decay library. The clean final
run is Slurm job `21081491`, generated at commit `d5207fb` with all validation
gates passing. The audited paper package is frozen at tag
`supportshift-geosim-v1.0.0`.

Verify the promoted run and every paper-artifact hash in one command:

```bash
python scripts/research/verify_supportshift_release.py \
  --metadata outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.metadata.json \
  --paper-directory paper \
  --repository-root . \
  --require-full
```
