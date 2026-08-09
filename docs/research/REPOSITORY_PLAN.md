# Cleaned repository and reproducibility plan

> The broad legacy refactor below remains a long-term plan. For the scoped
> SupportShift paper release, the controlling status and remaining items are in
> `FINAL_RESEARCH_ASSESSMENT.md`.

## SupportShift release status (2026-08-08)

- The pre-audit `main` state remains identifiable at `2a6ef52`; all maintained
  paper work is now consolidated on the sole local and remote branch, `main`.
- The maintained theorem/benchmark code is script-first and covered by unit,
  regression, and end-to-end integration tests.
- `environment-research.yml`, `environment.toml`, and the promoted Bouchet pip
  snapshot declare the executable environment.
- Seeds, commands, Git cleanliness, Slurm job identity, configuration hashes,
  row counts, validation gates, and input/output SHA-256 hashes are recorded.
- Final synthetic artifacts include compact paper extracts, deterministic
  figures, the 12,800-fit replicated table, and the complete 8,400-fit finite
  table with its reducer audit and immutable run manifest.
- Legacy notebooks, converted scripts, and invalid outputs remain in place for
  forensic provenance and are explicitly excluded from paper evidence.
- Bulk movement or deletion of the 1.5 GB legacy tree is intentionally deferred;
  it would add release risk without strengthening the scoped paper.

## Current state

The preserved base is a 1.5 GB checkout with 5,598 tracked files, 330 tracked
notebooks when hidden checkpoint copies are included, 1,630 data artifacts, and
hundreds of tracked checkpoint/cache files. The four commits are bulk imports and
cleanup from one day. There was no portable package manifest, CI, license,
`CITATION.cff`, data registry, DVC/LFS policy, or reliable run manifest.

The existing `environment.yml` is a large platform-specific export. Other
environment files contain cluster-specific prefixes or inconsistent data
dependencies. Archived notebooks and converted scripts hard-code paths and often
cannot execute as ordinary Python.

## Preservation policy

- Keep the pre-audit base addressable by commit and preserve legacy artifacts.
- Develop, commit, tag, and push only on `main`; do not create auxiliary branches
  or worktrees for the paper workflow.
- Label old notebooks/results `forensic provenance -- scientifically invalid`.
- Store regenerated outputs under a versioned `outputs/` tree outside Git or in a
  declared artifact store.
- Never silently replace a historical file with a repaired result of the same
  name.
- Do not commit raw restricted biological data; record source, license, checksum,
  acquisition date, and preprocessing recipe.

## Target layout

```text
HighDimSpatialStatistics/
  pyproject.toml
  environment-research.yml
  configs/
    pilot/
    final/
  src/HighDimSpatial/
    covariance/
    observation/
    likelihood/
    composite/
    simulation/
    diagnostics/
  scripts/
    research/
    pipeline/
  tests/
    unit/
    integration/
    regression/
  docs/research/
  paper/
    manuscript.tex
    references.bib
    figures/
    tables/
  data/
    README.md
    registry.yml
  outputs/                  # ignored; immutable run directories
  legacy/                   # notebooks/results retained, not imported by package
```

Do not physically move thousands of legacy files until all references are mapped;
that cleanup is logically separate from scientific corrections and should be a
separate commit if authorized.

## Completed maintained scaffold

- canonical AGS decay-scale kernel and signed cross-covariance helper;
- location-major covariance/simulation/likelihood convention;
- corrected Bessel derivatives with finite-difference regression checks;
- corrected multivariate NLL dimension and shape validation;
- corrected location-major marginal covariance extraction;
- explicit smoothing matrices and covariance/cross-covariance transforms;
- regression tests for signs, stacking, gradients, transformed covariance, and
  scalar observation count;
- effective seed zero and optional deterministic Torch setting;
- minimal `pyproject.toml`, research environment, ignores, and a JSONL pilot;
- research assessment, proof map, protocol, log, paper, and venue documents.

These changes do not repair the nonidentifiable cross optimizer or validate any
archived output.

## Required scientific refactor

### 1. Parameter model

- Replace the free-\(W\), scalar-equicorrelation cross parameterization by an
  identified globally valid parameterization.
- Separate variance, standard deviation, covariance, and correlation types in API
  names and dataclasses.
- Enforce correlation-matrix constraints globally, not with one finite-design
  Cholesky test.
- Support mixed cross-covariance signs; the current positive \(W_i\) and scalar
  \(\rho_V\) force all off-diagonals to share one sign.
- Add mean and measurement-noise models.

### 2. Likelihoods

- Keep exact raw likelihood separate from block composite likelihood.
- Add `S K S.T` observation-operator likelihood, including transformed nugget.
- Redesign multiresolution estimation as bounded local blocks or a valid joint
  Gaussian model; do not treat entire overlapping grids as independent batches.
- Compute Godambe uncertainty with cross-block/cross-resolution score covariance.
- Remove the nearest-distance PSD repair from scientific fits unless a theorem
  justifies a replacement approximation.

### 3. Optimizer

- Use unconstrained transforms or manifold-safe parameterization rather than
  post-step clipping.
- Restore full optimizer state on rejected steps.
- Accumulate every block in epoch diagnostics and early stopping.
- Persist best parameters with the best objective.
- Bind checkpoints to config hash, input checksum, code commit, dtype/device, and
  RNG state; never auto-resume an unrelated default checkpoint.
- Expose convergence, gradient norm, covariance condition, and failure reason.

### 4. Experiment runner

- Typed immutable config with deterministic run identifier.
- One process writes one append-only metrics file.
- Record all seeds and failed fits.
- Save environment and Git metadata.
- Separate pilot/final paths and prevent accidental overwrite.
- Generate figures/tables from tidy metrics with explicit readable foreground and
  background colors and a screenshot/contrast check.

## Tests required before new results

### Unit

- Matérn values against SciPy/R for several \(\nu,\alpha\);
- cross-covariance signs and zero-lag semantics;
- location-major permutation and marginal extraction;
- gradients in \(\alpha,\nu,\sigma\) versus high-precision finite differences;
- exact AGS validity across randomized parameter/location sets;
- correlation-matrix/equicorrelation bounds;
- smoothing operator identity and transformed nugget;
- NLL against `torch.distributions.MultivariateNormal`;
- n=1/equal-distance/duplicate-location/rank-deficient-smoother cases.

### Integration

- documented module commands run from a fresh editable install;
- generate -> raw fit -> metric on a tiny configuration;
- generate -> smooth -> corrected fit -> metric once that fitter exists;
- checkpoint/resume equals uninterrupted run;
- CPU/GPU agreement within declared tolerance;
- seed zero is reproducible;
- real-data preprocessing dependency and schema check.

### Regression

- archived wrong-layout and swapped-alpha/nu fixtures are detected, not accepted;
- six archived `K_test` matrices are labelled legacy because their claimed
  parameters have relative errors up to roughly 0.54 while the swapped
  reconstruction matches at approximately \(10^{-8}\) or better;
- old 15,000/1,000 split is rejected as non-disjoint (93 coordinate overlaps).

## Data and artifact registry

Each dataset entry should contain:

```yaml
id: ovary_puck_230517_39
source_url: ...
license: ...
raw_sha256: ...
sample_id: ...
biological_unit: animal_or_slide
coordinates_units: ...
acquired_at: ...
preprocessing_config: configs/final/...
derived_artifacts:
  - path: ...
    sha256: ...
```

Large files should live in an institutional artifact store or a versioned data
system. Git should contain only small fixtures and metadata.

## Commit plan if/when commits are authorized

1. `fix: correct Matern semantics, signed covariance, stacking, and gradients`
2. `feat: retain smoothing operators and transformed covariance utilities`
3. `test: add kernel, likelihood, layout, and smoothing regression coverage`
4. `build: add portable package and research environment`
5. `docs: add mathematical audit and research protocol`
6. Later, separately: `refactor: replace cross parameterization and optimizer`

Do not combine the scientific model replacement with bulk legacy-file movement.

## Release criteria

- clean environment install on Linux and macOS;
- all unit/integration/regression tests pass;
- final simulation configuration frozen and checksummed;
- all figures/tables regenerate from archived metrics;
- either real-data source/license and independent samples are documented or
  the release is explicitly scoped as synthetic-only with no external-validity
  claim;
- manuscript claims linked to a theorem, test, figure, or table;
- no unresolved P0/P1 audit issue;
- archived invalid outputs cannot be mistaken for final outputs.
