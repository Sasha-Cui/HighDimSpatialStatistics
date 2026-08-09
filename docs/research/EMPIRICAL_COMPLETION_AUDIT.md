# SupportShift empirical completion audit

**Audit completed:** 2026-08-08

**Scope:** all synthetic experiments used by the GeoSim and technical manuscripts

**Decision:** empirical component complete for the scoped theorem-plus-simulation paper

## Completion criteria

An experiment is complete only if its factors match a paper claim, its target is
explicit, all expected cells are present, numerical failures are retained or
reported, source data and metadata are hash-bound, and an independent rerun or
raw-to-summary reconstruction agrees with the promoted artifact. Passing these
criteria establishes reproducibility and claim correspondence; it does not turn
finite numerical evidence into a theorem or imply external validity.

## Experiment inventory and disposition

| Track | Promoted size | Target and comparison | Independent audit | Status |
|---|---:|---|---|---|
| Continuous phase oracle | 108 rows | exact pair target versus the three regime-specific asymptotic laws | clean rerun; fitted inverse ranges agree within \(4.9\times10^{-15}\) | complete |
| Smoothness-one transition | 111 cells | exact target versus one-term and cancellation-aware two-term approximations | byte-identical rerun | complete |
| Dimension--kernel robustness | 72 cells | exact target versus theorem coefficient in \(d=1,2,3\) for two compact product kernels | byte-identical rerun | complete |
| Directional support | 2,128 rows | exact directional pair targets and major--minor coefficient | byte-identical rerun; corrected gate metadata | complete |
| Finite-grid likelihood | 8,400 fits, 42 summary cells | support-aware and naive estimates versus exact finite-design KL targets | every summary statistic reconstructed from fit-level rows | complete |
| Replicated high dimension | 12,800 fits, 64 cells | finite-library ERM versus exact population target and candidatewise concentration certificate | clean full rerun; every numerical field identical | complete |
| Raw supported-field illustration | 2,516 rows | latent fields versus exact application of the saved support operator | byte-identical rerun | complete |

## Finite-grid recovery and raw-to-summary validation

The original release tracked only `paper/data/finite_summary.csv`, although the
Slurm reducer had created all fit-level results. The complete support-only run
was recovered from Bouchet and promoted without recomputation:

- run: `support_only_final_20260802_v2`;
- 21 of 21 tasks valid, no missing or invalid shard;
- 8,400 rows: 21 configurations by 200 replicates by two fitted models;
- no duplicate task/model/replicate keys and no non-finite numerical values;
- deterministic seeds agree with the root seed and common-random-number groups;
- one clean generation commit, `1370be25f174b32a3357c2383014835c1485ad79`;
- 11 retained boundary fits over all designs, including the six reported core
  boundary fits; and
- all 42 summary cells reconstructed from estimates, targets, errors, and
  boundary flags.

The recovered reducer summary and the canonical paper summary differ only by
CSV serialization: the maximum numeric difference is
\(2.22\times10^{-16}\). Release verification now performs this reconstruction
and also checks configuration hashes, exact model/replicate keys, seeds, clean
provenance, finite objectives, and signed/absolute/squared-error identities.

The release-bound SHA-256 values are:

- recovered Bouchet fit table: `b18ced9feb4b114a727bdc67d915593401964a4937f96e0f3735ca65243745f5`;
- release fit table after line-ending normalization only:
  `ef9e97fdaddae7a44f98c9eca78317dd2d3aff02b1bd58ecb88f7389cb800bcb`;
- reducer audit: `7b488e3c5d54a54e7fbbbab1b6abb86529686dfadaf8e0aed43419f4f11afdce`;
- run manifest: `995b53156fcc7060c665829dcbfc87f0cbe74181b81df86a14c43b44137eb4e6`;
  and
- canonical summary: `15f535aeb85c5fbb160430da761ef75b710e3b6a8894c7a261131d9d6ca38b53`.

## Independent deterministic reruns

All deterministic drivers were rerun from clean commit
`b6c8ee294db9319746eb3a3869f4b1315f694ef9` in the declared Bouchet Python
3.12 environment.

- Transition stress reproduced SHA-256
  `61537589aa30ee8a67f7970125a2aec2ddc23742661e0d75abebfba0a00873a9`.
- Dimension--kernel robustness reproduced SHA-256
  `175ac2ca3417002f69dde1746165d4fd576ae80ae1e727f1da58913d89867516`.
- Directional support reproduced SHA-256
  `cdd7b0fdb760c8e3178c4b573b6f6d776ce2a13569cb0249cab8d06cf4e9f1e9`.
- The raw illustration reproduced SHA-256
  `ac5cd1543a77766789b1f4768185aa63aae0d6ae5c61f81a0fa6363139965c18`.

The phase-oracle CSV is sensitive at the final digits to the SciPy build. Its
clean rerun has the same 108-row schema and factor grid; the largest difference
in the fitted inverse range is \(4.9\times10^{-15}\), while a derived
small-bandwidth coefficient ratio differs by at most \(1.3\times10^{-9}\).
Both are far below the paper's numerical tolerances and do not change a rounded
claim, sign, rate, figure, or conclusion.

The directional CSV is unchanged, but its metadata was regenerated because an
older metadata key incorrectly said the endpoint coefficient error tolerance
was 0.10. The driver and research protocol use 0.15, the observed value is
0.1175, and the regenerated metadata now states the correct 0.15 gate. This was
a provenance-label defect, not a changed acceptance rule or numerical result.

## Independent replicated-field rerun

Bouchet Slurm job `21749885` reran the full CPU experiment under `pi_jss233`
from the same clean source commit used for the deterministic re-audit. It
completed with exit code zero in 41 seconds; the benchmark computation itself
took 20.60 seconds.

- all 12,800 rows and 58 columns are present;
- every numerical value is exactly equal to the promoted table;
- the only 12,800 textual differences are the intentionally recorded source
  commit (`d5207fb` in the promoted run and `b6c8ee2` in the re-audit);
- the 2,516-row raw illustration is byte-identical;
- all validation gates pass;
- all 64 candidatewise-coverage cells have empirical coverage one; and
- the run records a clean Git worktree.

This rerun verifies deterministic simulation, optimization, candidate-library
selection, certificate evaluation, and raw support application under the
current frozen scientific code.

## Factor and baseline adequacy

The promoted tracks collectively vary bandwidth, smoothness, dimension,
support-kernel family, support aspect ratio, lag angle, boundary retention,
coordinate jitter, spatial domain size, field dimension \(p\), independent
replicate count \(N\), and fitted model family. The baselines are matched to
the claims: no-support controls, naive point-support fits, support-aware fits,
one-term versus transition-aware expansions, equal-axis directional controls,
two kernel families, and exact population or finite-library oracles.

Monte Carlo uncertainty is reported as cellwise standard deviations and
standard errors in the finite track and as empirical target-specific RMSE and
candidatewise certificate coverage in the replicated track. Optimizer boundary
hits are counted rather than removed. The experiments do not claim that the
pairwise asymptotic coefficient equals a full-grid likelihood coefficient.

## Release contract

Artifact-manifest schema 1.3 hash-binds 15 promoted data/provenance inputs and
23 generated paper artifacts. One release command verifies:

- 12,800 replicated-field fits;
- 8,400 finite-grid fits and their 42 reconstructed summary cells;
- 64 concentration-coverage cells;
- all 15 source-input and 23 generated-artifact hashes; and
- all 113 manuscript-facing numerical claims.

The verifier has a negative mutation test: changing a fit-level estimate while
leaving the paper summary unchanged fails both the numerical-identity and
raw-to-summary checks.

## Empirical go/no-go decision

**Go for the scoped synthetic GeoSim/workshop paper.** No additional experiment
is required to support a claim currently made in either manuscript. The
remaining submission gates are non-empirical: independent proof review, manual
priority review, author attendance/registration, portal validation, and an
archival DOI.

Unknown support, unknown smoothness, nugget timing, nonstationarity, heavy-tailed
fields, broader covariance families, and continuous-parameter optimization are
reasonable journal extensions. They should not be added to the current paper
unless the claims and theory are deliberately expanded; doing so now would add
scope without repairing an empirical deficiency.

All evidence remains synthetic. The completed audit supports internal validity,
reproducibility, and theory--experiment correspondence, but not external
validity for a real spatial population.
