# Research log

## 2026-08-02 -- preservation and reconstruction

**State.** Preserved clean `main` at `2a6ef52`; created
`research/paper-audit`. No commit, push, reset, or deletion of original work.

**Repository evidence.** The checkout is approximately 1.5 GB with 5,598 tracked
files, 330 tracked notebooks including checkpoints, and hundreds of tracked
checkpoint/cache/junk paths. Four commits on one day do not record experiment or
proof evolution. There was no package manifest, CI, license, citation file, data
registry, or portable lock file at the preserved base.

**Mathematical reconstruction.** Identified the intended model as the flexible
multivariate Matérn of Apanasovich--Genton--Sun (2012), with marginal-first then
cross-parameter estimation and distance/group likelihood approximations.

**Negative findings.** No original theorem/proof/asymptotic regime exists. The
model and two-stage fitting scheme are established prior work. Synthetic
\(p=3\) and real \(p\approx22\) analyses do not constitute high-dimensional
statistics.

## 2026-08-02 -- fatal correctness audit

1. Original kernel used a length-scale convention while applying AGS decay-scale
   validity equations.
2. Cross amplitudes were squared, erasing negative signs.
3. Covariance was variable-major while simulation/likelihood treated data as
   location-major.
4. Bessel `kvp` was misused as a derivative in order rather than argument.
5. Multivariate NLL constant counted rows rather than scalar observations.
6. Cross parameterization overwrote the diagonal required by AGS validity and
   contained exact/product nonidentifiabilities.
7. Positive \(W_i\) and one scalar \(\rho_V\) force every off-diagonal
   cross-covariance to have the same sign, so the bundled \((-,-,+)\) truth is
   outside the fitted family.
8. Smoothed grids used the wrong covariance and were treated as independent.
9. Distance rounding can destroy PSD; design-dependent eigenvalue shifting is
   detached from gradients and costs a dense eigendecomposition.
10. Fixed-domain experiments claimed separate parameters without addressing
    microergodicity.

**Artifact falsification.** Thirty archived CSVs fit the induced permutation more
closely than the intended covariance in all 30 cases (mean NLL approximately
830.88 versus 869.09). Some R/Python covariance metrics compare estimates with a
truth matrix from unrelated coordinates. Two coordinate designs occur across the
300 CSVs while at least one aggregate calculation reuses one truth design.
Reported R/Python mean distances must be discarded.

## 2026-08-02 -- repairs made on audit branch

- Adopted AGS decay-scale Matérn convention.
- Separated marginal standard deviation from signed cross-covariance amplitude.
- Standardized location-major stacking and repaired marginal covariance extraction.
- Corrected Bessel argument/order derivatives and broadcast-gradient reduction.
- Corrected multivariate likelihood dimension and added shape validation.
- Corrected exact/approximate cross-kernel sign, stacking, and nugget placement.
- Saved explicit smoothing operators and added exact covariance/cross-covariance
  transformations.
- Made seed zero effective, added a minimal package/environment specification,
  cache/output ignores, and a seeded JSONL pilot experiment.
- Added regression tests. All tests passed after the changes at this point.
- Converted maintained Slurm entrypoints to module invocation; legacy jobs remain
  non-reproducible provenance snapshots for independent reasons.

**Important limit.** Repairs invalidate old outputs; they do not validate the
legacy cross optimizer, grouped likelihood, real-data model, or novelty claim.

## 2026-08-02 -- literature and direction decision

**Known.** \(SKS^\top\) is classical change of support; generic misspecified GP
MLE targets a KL pseudo-parameter; smoothing-induced gene-correlation inflation is
already discussed by Smoothie. Flexible multivariate Matérn construction and its
two-stage algorithm are from 2012; multivariate composite likelihood already has
increasing-domain theory.

**Possible gap.** No exact comparator found an explicit local-smoother,
Matérn-specific bias expansion/strict target shift covering boundary and fixed-\(p\)
cross-covariance parameters. This is the primary direction, conditional on a
four-week proof gate.

**Rejected as primary.** Pseudo-distance, boundary, permutation, and teleconnection
notebooks are exploratory and should be archived. Growing-\(p\) sparse theory is a
new project with direct 2026 competition. Distance lookup is unlikely to matter
unless factorization complexity is also reduced.

## Open questions

- Can a nonzero closed-form first bias coefficient be derived for
  Matérn-\(1/2\) plus Epanechnikov/Gaussian smoothing?
- What bandwidth/domain regime makes the boundary remainder lower order?
- Which multivariate microergodic combinations remain identifiable after
  component-common or component-specific smoothing?
- Can the correct operator covariance be embedded in Vecchia/SPDE without dense
  \(SKS^\top\) formation?
- Are multiple ovary slides/animals and reliable biological annotations available?
- Does the application smoother depend only on coordinates, or on response values
  after gene filtering/normalization?

## 2026-08-02 -- theorem gate passed and scope narrowed

**Decision.** Abandoned the high-dimensional and multivariate-paper framing.
Selected a fixed-dimensional Matérn change-of-support problem: characterize the
pseudo-decay selected when locally averaged observations are fitted with a
point-support pair likelihood.

**Derivation.** For every \(\nu>0\), obtained the exact pair pseudo-target and a
small-support expansion. The leading decay displacement has order
\(h^{2\nu}\) for \(0<\nu<1\), \(h^2\log(1/h)\) for \(\nu=1\), and \(h^2\) for
\(\nu>1\). A Bessel recurrence makes the smooth-regime normalized-correlation
coefficient strictly positive, including for anisotropic compact symmetric
kernels. Therefore the naive inverse range is eventually too small in all three
regimes. The exact one-dimensional exponential/Epanechnikov calculation is
retained only as a regression oracle because temporal aggregation is classical.

**Limitations frozen into the claim.** Known smoothness and support; whole-space
or interior convolution; fixed nonzero lag; pairwise pseudo-target. No
fixed-domain separate consistency, discrete-to-continuous theorem, analytic
full-grid target, joint nuisance fit, or universal boundary sign is claimed.

## 2026-08-02 -- deterministic phase oracle

**Configuration.** Two-dimensional product Epanechnikov smoothing,
\(\alpha=R=1\), quadrature order 96, 18 bandwidths from 0.003 to 0.3, and
\(\nu\in\{0.25,0.5,0.75,1,1.5,2.5\}\).

**Result.** All 108 computed decay shifts were positive. Log--log slopes over
the six smallest bandwidths were 0.515, 0.999, 1.468, 1.817, 1.994, and 2.000.
These match the predicted powers 0.5, 1, 1.5, the logarithmically modified
threshold behavior, and the two quadratic regimes.

## 2026-08-02 -- first promoted finite-grid experiment

**Configuration.** Immutable run identifier support_final_20260802_v1; 21
configurations; 200 replicates; 161-point log-decay profile; deterministic seed
root 20260802; Slurm array 21071686 and reducer 21071687 under pi_jss233.

**Audit.** All 21 array tasks and the reducer completed. The audit contains 8,400
estimates, no missing task, and no invalid shard. The submission rate stayed far
below the authorized limit.

**Population result.** The corrected target differed from the true decay by at
most \(6.2\times10^{-8}\) over the core designs. At \(h=0.7\), naive targets
were 0.148, 0.434, 0.610, and 0.772 as smoothness increased through 0.5, 1,
1.5, and 2.5. Boundary retention and coordinate jitter changed the rough-field
target modestly but did not repair naive misspecification.

**Monte Carlo result.** At \(h=0.7\), naive mean estimates were 0.146, 0.411,
0.603, and 0.737; corrected means were 1.093, 0.990, 0.990, and 0.991. Ten of
6,400 core fits reached declared parameter bounds; all were retained. No
optimizer or covariance failure was silently removed.

**Paper decision.** Continue as a focused theorem-plus-synthetic-validation
paper. Require independent review of the Bessel expansions and the
\(\nu=1\) remainder before submission.

**Post-audit correction.** This run used pre-smoothing noise in the generating
covariance but independent post-smoothing noise in the naive model. Its naive
target therefore combined support and noise-timing misspecification. The run is
retained as a stress artifact, but its numbers are not the paper's primary
support-only evidence.

## 2026-08-02 -- support-only promoted experiment

**Configuration.** Immutable run identifier support_only_final_20260802_v2;
the same 21 configurations, 200 replicates, bounds, and seed root, with nugget
set to zero. Slurm array 21073200 and reducer 21073201.

**Audit.** All 21 tasks completed and the reducer marked the run complete:
8,400 estimates, no missing task, and no invalid shard.

**Population result.** Corrected targets were within
\(1.3\times10^{-7}\) of one. At \(h=0.7\), support-only naive targets were
0.152, 0.414, 0.573, and 0.740 for smoothness 0.5, 1, 1.5, and 2.5. Their
Monte Carlo means were 0.151, 0.407, 0.570, and 0.738. Corrected means were
1.090, 0.990, 0.990, and 0.994. Six core fits reached declared bounds and were
retained.

**Decision.** Use this run for the main figures and table. Keep the noisy run
only to document why noise timing must be propagated through the observation
operator.

## 2026-08-02 -- final reproducibility and release audit

- Reran the 108-row deterministic phase oracle from clean commit `36cdec8`;
  its metadata records `git_dirty=false` and the committed source identifier.
- Made generated figure PDFs deterministic by removing volatile creation and
  modification timestamps; two consecutive regenerations had identical
  SHA-256 hashes.
- Compiled both the 13-page technical manuscript and the 5-page ACM GeoSim
  draft and visually inspected every rendered page. The pass found and fixed
  one missing LaTeX command escape in a displayed conclusion.
- The declared Bouchet environment passed all 62 tests. Ruff, Slurm shell
  syntax, and the support-only JSON manifest also passed validation.

## Entry template

```text
Date/time:
Question or claim:
Configuration / commit / data checksum:
Derivation or experiment:
Observed result (including failures):
Counterexample / diagnostic:
Decision and rationale:
Next falsifiable step:
```
