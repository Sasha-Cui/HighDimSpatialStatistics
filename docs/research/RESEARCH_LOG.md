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
