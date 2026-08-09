# HighDimSpatialStatistics

> **SupportShift reviewers and users:** start with the isolated
> [SupportShift entry point](supportshift/README.md). It lists only the paper,
> generators, promoted artifacts, and verification commands in the evidence
> contract; historical pipelines are out of scope.

This repository now has two clearly separated layers: a preserved legacy
high-dimensional spatial pipeline and the audited **SupportShift** research
artifact. SupportShift is a theorem-linked synthetic benchmark for range
distortion caused by fitting locally averaged Matérn observations as points.
The original notebooks remain under `notebooks/legacy/` for traceability; their
archived numerical claims are not paper evidence.

## SupportShift reviewer quick start

The audited artifact is frozen at
[`supportshift-geosim-v1.3.1`](https://github.com/Sasha-Cui/HighDimSpatialStatistics/tree/supportshift-geosim-v1.3.1).
Start with:

- the [GeoSim submission PDF](output/pdf/supportshift_geosim2026.pdf);
- the [technical manuscript](output/pdf/supportshift_technical_manuscript.pdf);
- the [benchmark contract](docs/research/SUPPORTSHIFT_BENCHMARK.md);
- the [synthetic-data card](docs/research/ARTIFACT_DATA_CARD.md); and
- the [submission checklist](docs/research/GEOSIM_SUBMISSION_CHECKLIST.md).

Create the portable environment and verify the promoted run, all paper
artifacts, and every manuscript-facing numerical claim:

```bash
conda env create -f environment-research.yml
conda activate highdimspatial-research
python scripts/research/verify_supportshift_release.py \
  --metadata outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.metadata.json \
  --paper-directory paper \
  --repository-root . \
  --require-full
```

A valid release reports 12,800 replicated-field fits, 8,400 finite-grid fits,
64 coverage cells, 23 hashed source inputs, 33 hashed paper artifacts, and 168
passed paper claims. Verification reconstructs the finite-grid summary from its
fit-level records; it does not require rerunning either Monte Carlo experiment.

## Preserved legacy pipeline

The commands below exercise the older general package and are not part of the
SupportShift paper evidence.

1. Create the maintained portable environment:

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
on `research/paper-audit` and subsequently consolidated onto `main`. Current
development, commits, tags, and pushes are main-only; consult
`docs/research/FINAL_RESEARCH_ASSESSMENT.md` and
`docs/research/SUPPORTSHIFT_BENCHMARK.md` before using any result.

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

python scripts/research/run_transition_stress_audit.py \
  --output outputs/smoothing_bias/supportshift_transition_stress_20260804.csv

python scripts/research/run_dimension_kernel_robustness.py \
  --output outputs/smoothing_bias/supportshift_dimension_kernel_robustness_20260804.csv
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
  --phase-metadata outputs/smoothing_bias/phase_oracle_d2_v2.metadata.json \
  --transition-stress outputs/smoothing_bias/supportshift_transition_stress_20260804.csv \
  --transition-stress-metadata outputs/smoothing_bias/supportshift_transition_stress_20260804.metadata.json \
  --dimension-kernel-robustness outputs/smoothing_bias/supportshift_dimension_kernel_robustness_20260804.csv \
  --dimension-kernel-robustness-metadata outputs/smoothing_bias/supportshift_dimension_kernel_robustness_20260804.metadata.json \
  --finite-summary paper/data/finite_summary.csv \
  --finite-results outputs/smoothing_bias/support_only_final_20260802_v2/results.csv \
  --finite-audit outputs/smoothing_bias/support_only_final_20260802_v2/audit.json \
  --finite-manifest configs/smoothing_bias/support_only_20260802.json \
  --anisotropy outputs/smoothing_bias/supportshift_anisotropic_final_20260803.csv \
  --anisotropy-metadata outputs/smoothing_bias/supportshift_anisotropic_final_20260803.metadata.json \
  --highdim outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.csv \
  --highdim-metadata outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.metadata.json \
  --raw-example outputs/smoothing_bias/supportshift_raw_final_v2_20260803.csv \
  --multilag outputs/smoothing_bias/supportshift_multilag_composite.csv \
  --multilag-metadata outputs/smoothing_bias/supportshift_multilag_composite.metadata.json \
  --full-likelihood outputs/smoothing_bias/supportshift_full_likelihood_phase.csv \
  --full-likelihood-metadata outputs/smoothing_bias/supportshift_full_likelihood_phase.metadata.json \
  --joint-smoothness outputs/smoothing_bias/supportshift_joint_smoothness.csv \
  --joint-smoothness-metadata outputs/smoothing_bias/supportshift_joint_smoothness.metadata.json \
  --matched-boundary outputs/smoothing_bias/supportshift_matched_boundary.csv \
  --matched-boundary-metadata outputs/smoothing_bias/supportshift_matched_boundary.metadata.json \
  --paper-directory paper
```

The promoted artifact-manifest schema is 1.4. Its replicated-field component remains
benchmark schema 1.1 with $p\in\{16,36,64,100\}$,
$N\in\{1,4,16,64\}$, 200 trials per design, and a fixed
$161\times101=16{,}261$-candidate variance--decay library. The clean final
run is Slurm job `21081491`, generated at commit `d5207fb` with all validation
gates passing. The threshold audit was generated from clean commit `34a2603`
and passed all sign, approximation-error, and quadrature-refinement gates. The
dimension--kernel audit was generated from clean commit `2fc7040` and passed its
predeclared sign, coefficient, and quadrature gates. The audited paper package
is frozen at tag `supportshift-geosim-v1.3.1`.

Verify the promoted run, every paper-artifact hash, and all 168 numerical claims
reported in the manuscripts in one command:

```bash
python scripts/research/verify_supportshift_release.py \
  --metadata outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.metadata.json \
  --paper-directory paper \
  --repository-root . \
  --require-full
```

The standalone claim ledger can also be inspected as JSON:

```bash
python scripts/research/verify_supportshift_claims.py \
  --repository-root . \
  --paper-directory paper \
  --json-output /tmp/supportshift_claim_ledger.json
```

## Artifact use and citation

The [synthetic-data card](docs/research/ARTIFACT_DATA_CARD.md) defines the
benchmark factors, files, dependence structure, validation contract, intended
uses, and limitations. The dated [prior-art search](docs/research/PRIOR_ART_SEARCH.md)
records the closest literature comparison without treating a negative search
as proof of priority. Original SupportShift code and generated synthetic data
use the scoped terms in `LICENSE-SUPPORTSHIFT`; preserved legacy and third-party
material is excluded. Citation metadata are in `CITATION.cff`.
