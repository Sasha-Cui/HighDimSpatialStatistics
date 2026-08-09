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

## 2026-08-03 -- SupportShift high-dimensional probability extension

**Scope decision.** Retained the Matérn change-of-support theorem as the central
contribution and added a separate replicated-field experiment. The extension
does not resurrect the invalid legacy multivariate claims. It uses independent
Gaussian field replicates with unrestricted within-field spatial dependence and
a deterministic finite covariance library.

**Probability statement.** For
$X_i\stackrel{\mathrm{iid}}{\sim}N_p(0,\Sigma_0)$, a union bound over the
Gaussian quadratic-form inequality gives, simultaneously over $M$ declared
covariances,

\[
 |\widehat L_N(\theta)-L(\theta)|\leq
 \frac{\|A_\theta\|_F}{p}\sqrt{\frac{\log(2M/\delta)}N}
 +\frac{\|A_\theta\|_{\mathrm{op}}}{p}
  \frac{\log(2M/\delta)}N.
\]

The ERM excess criterion is at most twice the largest radius. Under uniform
relative spectral control this is
$O\{\sqrt{\log M/(Np)}+\log M/(Np)\}$. This is standard supporting
machinery, not a novelty claim, and its oracle radius uses the generating
$\Sigma_0$.

## 2026-08-03 -- final schema-1.1 Monte Carlo run

**Configuration and provenance.** Slurm job `21081491` ran on allocation
`pi_jss233` from clean commit `d5207fb43cf63f5dfd68a443853a209e303f9aa2`.
It used $p\in\{16,36,64,100\}$, $N\in\{1,4,16,64\}$,
$\nu\in\{0.5,1.5\}$, 200 trials, and a fixed
$161\times101=16{,}261$-candidate variance--decay library. All 12,800 fits
completed. The result and raw-example SHA-256 hashes are `247db5e6f09f95f3`
and `ac5cd1543a777667` (prefixes).

**Acceptance gates.** Every gate passed. All 64 separately defined
candidatewise coverage cells had empirical coverage one; the global maximum
candidatewise deviation-to-radius ratio was 0.789341. The maximum and median
95th-percentile worst-envelope ratios were 0.401816 and 0.298185. The largest
normalized grid-to-continuous population-objective gap was
$2.6251\times10^{-5}$, below the fixed $5\times10^{-5}$ gate.

**Theorem--experiment correspondence.** The 95th-percentile criterion-deviation
log--$N$ slopes ranged from -0.528 to -0.454, with median -0.505. At
$p=100,\nu=1.5$, the naive finite-grid decay target was 0.594 versus physical
decay one. Its RMSE to that wrong target fell from 0.124 to 0.015 as $N$
increased from one to 64, while RMSE to the physical value remained 0.409. The
support-aware RMSE fell from 0.206 to 0.024. These parameter-error rates are
empirical; the theorem itself controls likelihood and excess KL criterion.

## 2026-08-03 -- independent audit and artifact-manifest repair

An independent recomputation found no discrepancy in the statistical outputs,
targets, slopes, or reported RMSE values. It did find that the artifact builder
rewrote `paper/data/phase_oracle_d2.csv` while also declaring that file an
immutable input, making the pre-fix manifest impossible to verify. Commit
`3a0b9bf` now skips writes when a source extract aliases its destination, records
the alias explicitly in manifest schema 1.1, and tests the same-directory case.
The complete figure set was regenerated under Python 3.12.8, NumPy 2.3.5,
pandas 2.2.3, and Matplotlib 3.10.9; every output and aliased input/output hash
then verified.

## 2026-08-03 -- paper decision

**Verdict.** Continue only as a focused theorem-plus-synthetic-benchmark paper.
The clean novelty claim is the explicit all-$\nu$ pairwise Matérn support phase
law, including the $\nu=1$ logarithmic transition, plus the directional
$h^2$ coefficient. Do not claim a new generic concentration inequality,
full-likelihood phase theorem, fixed-domain separate range consistency, or
data-driven confidence interval.

**Primary route.** Submit the eight-page SupportShift benchmark paper to GeoSim
2026, subject to independent human proof review and the official formatting
check. Retain the longer technical manuscript for a specialized journal only if
the workshop version receives constructive feedback or a broader
continuous-parameter result is completed.

## 2026-08-03 -- final independent release audit

Two independent read-only checks recomputed the theorem-linked summaries and
found no numerical discrepancy. They identified release-contract issues that
were repaired in commit `aa272c9`: the promoted fit-level CSVs and metadata are
now versioned so verification works from a clean clone; byte-preserving Git
attributes protect their recorded SHA-256 hashes; Fuentes (2007), the closest
block-Matérn likelihood comparison, is cited in the workshop paper; unresolved
priority language was weakened; the dependency map restores the
$\log(M)/(Np)$ term; and the raw illustration schema is described accurately.
The technical manuscript now labels increasing-domain pair consistency as an
unproved standard route rather than a completed theorem. Both PDFs were rebuilt
and visually rechecked after these changes.

## 2026-08-03 -- acceptance-risk and public-artifact hardening

A dated priority screen searched the Matérn, block-support, regularized-
variogram, aggregated-GP, and misspecified-likelihood literature for the exact
all-smoothness pseudo-range phase law. The closest analytic setup located was
Fuentes (2007), which explicitly gives the filtered spectrum of rectangular
block averages and a small-pixel approximation but does not derive the
next-order point-fit parameter displacement. The paper now states this precise
boundary and explicitly says the search is not proof of priority.

The release now supplies a consumer-facing benchmark contract, synthetic-data
card, `CITATION.cff`, scoped MIT terms that exclude legacy and third-party
content, and a GeoSim submission checklist. The promoted simulations and their
hashes are unchanged. Ruff's intended correctness-only rule set is now explicit
in `pyproject.toml`, avoiding version-dependent expansion of the default lint
scope. The intended patch tag is
`supportshift-geosim-v1.0.1`; it must be created only after manuscript rebuild,
visual inspection, clean-clone verification, local tests, and Bouchet
verification all pass.

## 2026-08-04 -- smoothness-one transition repair and stress audit

A fresh coefficient-by-coefficient proof audit confirmed the fixed-lag Hessian,
all three origin regimes, the integer-\(\nu\) logarithmic remainders, the
Bessel-recurrence sign reduction, the directional contrast, and the exact
normalization constants in the Gaussian finite-library certificate. The audit
also found and repaired one wording defect: the workshop manuscript had stated
pair-composite convergence too strongly even though the required uniform
ergodic law was not proved. Both manuscripts now describe that argument only
as a standard possible route.

The main substantive extension retains the analytic \(h^2\) term and the
fractional \(h^{2\nu}\) term simultaneously. Their coefficients are separately
singular as \(\nu\to1\), but the poles cancel and converge to the explicit
Matérn-one logarithmic expression. For fixed smoothness, the resulting pair
target has pseudo-decay error \(O(h^{2\nu+2})\) below one,
\(O\{h^4\log(1/h)\}\) at one, and \(O(h^4)\) between one and two. No joint
uniform remainder in \((\nu,h)\) is claimed.

The deterministic audit generated 111 cells from clean commit `34a2603`, using
\(d=2\), \(\alpha=R=1\), 37 smoothness values from 0.55 to 1.45, and
\(h\in\{0.01,0.02,0.05\}\). The minimum exact-to-one-term shift ratio was
0.153610, so the one-term prediction reached 6.51 times the exact shift near
the threshold. The transition-aware maximum relative shift error was
0.000984933, and its maximum relative variance-loss error was 0.000831938.
All exact and approximate shifts were positive. Quadrature orders 64 and 128
changed the order-96 target by at most \(3.9632\times10^{-9}\) and
\(3.0004\times10^{-10}\) relative to the exact shift. Metadata record a clean
worktree and CSV SHA-256
`61537589aa30ee8a67f7970125a2aec2ddc23742661e0d75abebfba0a00873a9`.
Bouchet Ruff checks passed; the full suite collected 102 tests, and the focused
continuous and artifact-driver groups passed 39/39 and 3/3.

## 2026-08-04 -- venue-fit red team and manuscript-claim audit

The current official GeoSim 2026 call was rechecked. It explicitly prioritizes
parameterizable, scalable benchmark data sets, community availability, and
verification and validation of spatial simulations. The 2025 program was
dominated by concrete generators and simulation frameworks. In response, the
GeoSim abstract now names the independent scale controls, released source
tables, seeds, metadata, and pass/fail gates; its ACM classification now leads
with modeling and simulation plus model verification and validation rather
than probability alone. The official EasyChair submission URL is recorded in
the submission checklist.

A new deterministic claim audit recomputes every paper-facing numerical result
from the promoted tables and metadata. Its 100 checks cover phase slopes and
coefficient ratios, transition errors, directional contrasts, finite-grid KL
targets and Monte Carlo summaries, raw-table dimensions, high-dimensional
certificate events, criterion-noise slopes, and wrong-target RMSE values. The
full release verifier now invokes this audit. A negative regression test removes
one phase row and confirms that the audit fails rather than silently accepting a
changed source table. The previously untracked clean phase-oracle CSV and
quadrature metadata are promoted so that the audit also works from a clean
clone. No reported number changed during this pass.

## 2026-08-04 -- hostile-referee proof and priority pass

The analytic chain was rederived from the exact pair target through the radial
Hessian, noninteger Bessel series, the two integer expansions, inverse-map
remainder orders, smooth-regime Bessel recurrence, directional cancellation,
and Gaussian quadratic-form normalization. No theorem coefficient, sign, or
remainder defect was found. This internal red team is evidence of consistency,
not a substitute for the still-required independent human proof review.

The older regularized-variogram search did find an omission in the literature
positioning. Clark (1977) proves that sampling support geometrically extends a
finite range of influence, and Bellehumeur and Legendre (1997) use analytical
change-of-support relationships with an exponential variogram and report an
increasing autocorrelation range. These are direct qualitative predecessors for
apparent-range inflation. They do not derive the free-variance, fixed-lag
Matérn KL pseudo-parameter: Matérn covariance has no finite range of influence,
and the classical practical-range rules do not give the
\(h^{2\nu}\), \(h^2\log(1/h)\), and \(h^2\) phase law. Both papers
are now cited and this distinction is explicit in the introduction, comparison
table, discussion, and prior-art audit.

The workshop paper now uses the call's optional post-reference allowance for a
one-page appendix. It supplies proof details for the phase theorem,
smoothness-one cancellation, directional contrast, and finite-library
certificate. The main paper remains ten pages before references; the appendix
is one page after them. No reported numerical value or promoted artifact
changed.

## 2026-08-04 -- reviewer-navigation and acceptance-rubric pass

A rendered-paper mock review found that the individual theory and experiments
were well supported but that the benchmark contract was distributed across
several sections. A new full-width table now maps each of the four synthetic
tracks to its nonrandom target, controlled factors, and explicit scoring or
falsification gate. The table fits within the existing ten-page main-text
limit and adds no new numerical claim.

The repository landing page now leads with the immutable paper PDFs, benchmark
specification, data card, checklist, and one-command verifier. Preserved
legacy-pipeline commands are labeled as non-paper evidence instead of appearing
as the primary quick start. An internal mock review records the likely
objections about classical range regularization, pairwise versus full-grid
claims, standard concentration machinery, synthetic-only validation, known
support, and boundary behavior, together with the manuscript's scoped answers.
The mock review gives a conditional weak-accept recommendation; it does not
replace independent expert proof and priority review.
A separate submission packet stages the portal-ready abstract, topic and
expertise selections, immutable PDF hash, and an author-reviewable
generative-AI disclosure while leaving all live-portal and attendance fields
explicitly unchecked.

## 2026-08-04 -- post-freeze independent identity audit and review packet

After freezing `supportshift-geosim-v1.2.0`, the central special-function
identities were recomputed at 100-decimal precision on an adversarial grid
spanning smoothness 0.05 through 10 and arguments 0.01 through 10. The Matérn
derivative identity and the two smooth-regime expressions for
\(G_\nu\) agreed to scaled errors below \(2\times10^{-101}\), and every
evaluated \(G_\nu\) was positive. The displayed integer-order remainders stayed
bounded after normalization, and the two singular terms in the
transition-aware expansion cancelled linearly as \(\nu\to1\).

A separate post-freeze diagnostic evaluated 216 one-dimensional exact
quadrature cells across two product kernels, 18 smoothness values (including
0.99, 1.01, 1.99, and 2.01), and six bandwidths. Every pseudo-decay shift had
the theorem's sign; the smallest was approximately \(5.25\times10^{-8}\).
These probes are diagnostic rather than promoted confirmatory evidence and do
not alter the frozen artifact.

`EXTERNAL_PROOF_REVIEW_PACKET.md` now gives an independent reviewer nine
equation-level checks, authoritative identity sources, and explicit fallback
actions for each possible failure. This closes the internal preparation for
proof review but does not mark the human-review gate complete.

## 2026-08-04 -- GeoSim proceedings and topic-routing audit

The official ACM GeoSim 2025 proceedings report 11 accepted papers from 13
submissions. The accepted program spans short and full papers on mobility,
flow, agent-based wildfire, surrogate modeling, and spatial-data generation;
full-paper proceedings spans are roughly eight to thirteen pages. The 2026
call explicitly solicits parameterized and scaled benchmark data and
verification and validation of spatial simulations. This supports the current
benchmark-first title, abstract, contribution order, and 13-page
main--references--appendix structure. No manuscript rewrite was justified by
the comparison.

The portal packet now makes Verifying and Validating Spatial Simulations the
primary topic, with data generators and simulation-based spatial analysis as
secondary topics. Big Spatial Data Simulation was removed: varying dimension
and lattice size is not, by itself, evidence of high-throughput computational
scalability. This routing change reduces the chance of review against an
unsupported systems claim while preserving the frozen paper.

## 2026-08-04 -- dimension and kernel robustness audit

**Question.** The phase theorem permits fixed arbitrary dimension and compact
symmetric kernels, but the main deterministic display used only (d=2) and a
product Epanechnikov kernel. This left a reviewer-facing implementation-coverage
gap even though the proof itself is kernel-general.

**Predeclared audit.** Before examining the promoted results, the driver fixed
(d\in\{1,2,3\}), product Epanechnikov and product uniform kernels,
(\nu\in\{0.5,1,1.5,2.5\}),
(h\in\{0.002,0.004,0.008\}), and (alpha=R=1). Tensor quadrature used order
48 per coordinate with orders 32 and 64 as refinements. Acceptance required a
complete 72-cell grid, positive shifts in every cell, maximum
smallest-bandwidth coefficient error at most 0.20, and maximum relative
quadrature-refinement change at most (2\times10^{-4}).

**Clean result.** Bouchet generated the promoted table from clean commit
`2fc7040c0f3f82b8b6c196a0b05ac9dad3ccb791` under Python 3.12.8, NumPy 2.3.5,
and SciPy 1.17.1. All 72 shifts were positive. The worst coefficient error was
0.168688, and the worst relative refinement change was
(3.9131\times10^{-7}). The CSV SHA-256 is
`175ac2ca3417002f69dde1746165d4fd576ae80ae1e727f1da58913d89867516`.

**Decision.** Promote the result as a finite robustness audit in the second
GeoSim appendix page and the technical manuscript, not as proof of the theorem
or as a uniform result over kernels or dimensions. Artifact schema 1.2 copies
source tables byte-for-byte, records 23 output hashes, and expands the claim
ledger from 100 to 113 checks. The configured Bouchet environment passes 138
tests and the maintained Ruff scope.

## 2026-08-08 -- final empirical completion audit

**Recovered finite evidence.** The complete support-only finite run remained on
Bouchet even though only its 42-row summary had been tracked. The recovered
artifact contains all 8,400 fits, its 21-task reducer audit, clean generation
commit, exact configuration hashes, and deterministic seeds. There are no
missing tasks, invalid rows, duplicates, or non-finite objectives. Recomputing
the summary from fit-level estimates changes no value beyond
(2.22\times10^{-16}), attributable to CSV serialization.

**Independent reruns.** Clean commit `b6c8ee2` reproduced the transition,
dimension--kernel, and anisotropy CSVs byte-for-byte. The phase fitted inverse
ranges agree within (4.9\times10^{-15}); only a derived coefficient ratio
differs by as much as (1.3\times10^{-9}) under the newer SciPy build. Full CPU
Slurm job `21749885` reran all 12,800 replicated fits under `pi_jss233` with
exit code zero. Every numerical field equals the promoted table exactly; the
only row-level differences are the recorded source commit. Its 2,516-row raw
illustration is byte-identical, all gates pass, and all 64 coverage cells equal
one.

**Metadata repair.** The anisotropy CSV did not change. Its metadata was
regenerated because an old key named a 0.10 endpoint-error gate even though the
driver and predeclared rule used 0.15; the observed 0.1175 passes the actual
rule. The corrected metadata now names the 0.15 tolerance.

**Release decision.** Artifact-manifest schema 1.3 binds 15 data/provenance
inputs and 23 generated paper artifacts. The release verifier reconstructs all
finite summary cells from raw fits and checks 12,800 replicated fits, 8,400
finite fits, 64 coverage cells, and 113 paper claims. The scoped empirical
component is complete; no new simulation is required for a current manuscript
claim. Remaining gates are external proof/priority review, attendance,
submission-portal validation, and archival DOI creation.

## 2026-08-09 -- referee-directed likelihood and nuisance audit

**Question.** Does SupportShift create genuine misspecification beyond the
saturated two-point construction, and do the conclusions survive joint
smoothness fitting, an intermediate support model, and a dimension-matched
boundary comparison?

**Theory repair.** Commit `dbd1013` added an information-weighted multi-lag
pair-composite projection with a nonzero residual-KL coefficient and a general
finite-design Fisher-tangent projection for full Gaussian likelihood. The
pairwise phase law is now explicitly a building block, not the full
misspecification claim.

**Predeclared synthetic audits.** Commits `4f05f14` and `8505a37` added four
clean-run drivers. The promoted outputs contain 48 multi-lag cells, 12
full-likelihood projection cells, 2,400 joint smoothness--decay fits, and 48
matched-boundary cells. At the smallest bandwidth, the maximum multi-lag shift
relative error is 0.0141 and the maximum residual-KL relative error is 0.0985.
All 12 evaluated full-likelihood designs have positive residual KL; the maximum
smallest-bandwidth errors are 0.0542 for decay shift, 0.1268 for variance shift,
and 0.0177 for residual KL.

**Nuisance and intermediate-model result.** Exact support selects the physical
joint target in all eight design cells. Point support increases the smoothness
target in all eight and selects inverse range above one in all eight, reversing
the fixed-smoothness decay direction. The 75%-bandwidth support model has no
larger population KL than point support in all eight cells. For matched
interior/boundary blocks with (p=16), point-target differences are
0.030--0.062 and the partial-to-point KL ratio is 0.167--0.323.

**Decision.** Promote all four audits into artifact-manifest schema 1.4. The
manifest now binds 23 source inputs and 33 generated outputs; the claim ledger
contains 168 passing checks. Retain the universal sign only for fixed
smoothness and translation-invariant interior pair targets. Treat joint
smoothness and boundary behavior as finite-design evidence, and treat the
full-likelihood result as a local projection with a design-specific sign.

## 2026-08-09 -- GeoSim acceptance-hardening preflight

**Reviewer-facing repair.** The GeoSim abstract was reduced to a direct
theory-to-benchmark narrative. The finite-design proposition now states the
Fisher-orthogonality equations defining its residual, and the fixed-smoothness
scope is positioned against published evidence that Mat\'ern smoothness
information can be appreciable and design-dependent. A new regression test
confirms that both the analytic projection and exact numerical KL target are
invariant to site ordering.

**Artifact usability.** The isolated reviewer README now includes a custom
synthetic-generation example with explicit controls and row semantics. The
exact command was executed with two lattice sizes, two replicate counts, two
smoothness values, 10 trials, and a 21-by-15 variance--decay library. It wrote
160 fit rows and 1,044 raw-field rows in under one second and passed every
declared validation gate. The GeoSim appendix now explains how the release
verifier, claim ledger, source hashes, and generators divide responsibility.

**Preflight.** All 156 tests pass in the portable research environment. Both
papers compile without unresolved citations, unresolved references, or
overfull boxes. The submission PDF remains 13 letter-size pages: 10 main-text
pages, one references page, and two appendix pages. Ghostscript parses both
PDFs and every listed font is embedded. A page-by-page visual audit found no
clipping, overlap, invisible text, or illegible figure labels; bibliography
spacing in the technical manuscript was tightened to avoid a nearly empty
27th page.

**Decision.** Cut patch release `supportshift-geosim-v1.3.1`. No new Monte
Carlo experiment is required for a manuscript claim. The remaining scientific
gate is independent expert proof and priority review; the remaining submission
gates require author metadata and the live EasyChair form.

## 2026-08-09 -- mathematical notation and presentation audit

**Problem.** The paper used Latin \(v\) adjacent to Mat\'ern smoothness
\(\nu\), plain epsilon for vector noise, and overloaded \(a\), \(C\), \(J\),
and \(b\) across covariance, composite-likelihood, and projection arguments.
Although the derivations were correct, these collisions made the mathematics
look less disciplined and increased the cost of checking proofs.

**Notation repair.** Both manuscripts now use \(\sigma^2\) for physical
variance, \(\boldsymbol\varepsilon\) for measurement noise, \(\eta\) for a
generic inverse-range candidate, \(\kappa_j\) for lag displacement
coefficients, and \(\alpha_h^{\mathrm C}\) for the composite KL target.
Gaussian laws use \(\mathcal N\), and KL notation, matrix norms, roman
subscripts, and multiplication spacing are consistent. The finite-design
result now defines the Fisher inner product explicitly and uses
\((\mathcal J,g_\nu,\beta_\nu)\) for information, perturbation score, and
projected shift, with the residual stated directly as Fisher-orthogonal to the
covariance tangent space.

**Verification.** All 156 tests and all 168 numerical claim checks pass. Both
papers compile without unresolved citations, unresolved references, or
overfull boxes. A fresh page-by-page render preserves the 13-page GeoSim and
26-page technical layouts with no clipped equations, collisions, or illegible
symbols.

**Decision.** Publish the presentation-only patch as
`supportshift-geosim-v1.3.2`. The theorem assumptions, coefficients, rates,
simulation outputs, and acceptance claims are unchanged.

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
