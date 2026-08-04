# GeoSim 2026 internal mock review

- **Review date:** 2026-08-04
- **Artifact reviewed:** frozen tag `supportshift-geosim-v1.2.0`
- **Status:** internal red-team assessment, not an official review or an
  acceptance prediction

## Overall assessment

**Provisional recommendation: weak accept, conditional on independent proof
review and author attendance.** The paper is unusually reproducible for a
workshop submission, is directly aligned with GeoSim's call for
parameterizable and scalable benchmark data, and has a mathematically
specific contribution. Its principal acceptance risk is not experimental
execution; it is whether a spatial-statistics reviewer regards the
fixed-lag Matérn pseudo-parameter expansion as sufficiently distinct from
classical semivariogram regularization and sufficiently connected to the
full-grid benchmark.

## Venue calibration from the latest proceedings

The official ACM record for
[GeoSim 2025](https://doi.org/10.1145/3764921) reports 11 acceptances from 13
submissions (85%). The same record lists 64% for 2020 and 70% for 2019. These
are small annual denominators and are context, not an acceptance guarantee.

The 2025 proceedings contain 11 papers covering mobility simulation,
experiential geosimulation, adaptive flow modeling, agent-based wildfire,
surrogate modeling, point and trajectory generation, hyperspectral generation,
and land-cover simulation. Their proceedings spans are four pages for short
papers and roughly eight to thirteen pages for full papers. The current
SupportShift package has the expected full-paper footprint and unusually strong
verification infrastructure. Its main venue-fit risk is instead topical: it is
more statistical and less application-driven than most 2025 papers. The
[2026 call](https://geosim.org/2026/cfp/) directly mitigates that risk by
soliciting parameterizable and scalable benchmark data, spatial analysis based
on simulation, and verification and validation of spatial simulations.

Accordingly, the submission should use **Verifying and Validating Spatial
Simulations** as its primary topic. **Spatial Data/Trajectory Generators** and
**Spatial Analysis based on Simulation** are defensible secondary topics.
**Big Spatial Data Simulation** should not be selected merely because the
benchmark varies dimension: the paper makes no large-system throughput claim.

| Criterion | Internal score | Evidence and remaining risk |
|---|---:|---|
| GeoSim relevance | 5/5 | Four parameterized spatial-simulation tracks, released source data, and explicit verification/validation gates match the call closely. |
| Technical correctness | 4/5 | The derivations, integer transitions, signs, and concentration constants survived an internal hostile-referee audit and numerical checks. Independent expert review is still required. |
| Originality | 3.5/5 | The exact Matérn KL estimand, smoothness-dependent order, coefficient, and directional contrast were not located in the screened literature. The qualitative range-inflation direction is classical and is now cited explicitly. |
| Experimental support | 4/5 | Deterministic oracles now cover two compact kernels in dimensions one through three; full Gaussian KL targets, Monte Carlo fits, anisotropy, boundary/irregular stress cases, and growing-p,N trials correspond to declared claims. No external-validity claim is possible without real data. |
| Reproducibility | 5/5 | Clean-clone verification covers 12,800 fits, 64 coverage cells, 23 hashes, and 113 manuscript claims, with immutable inputs and metadata. |
| Clarity and presentation | 4/5 | The model, asymptotic scope, and limitations are explicit. The new benchmark-contract table removes the main navigation burden; the paper remains mathematically dense. |

## Likely reviewer objections and current answers

### 1. "Range inflation under aggregation is already known."

This objection is correct qualitatively. Clark (1977) gives a geometric
finite-range-of-influence rule, and Bellehumeur and Legendre (1997) report
increasing practical autocorrelation range under aggregation. The paper now
states this in the introduction, comparison table, and discussion. The
narrower addition is a variance-refitted fixed-nonzero-lag Gaussian KL
parameter for an infinite-range Matérn covariance, with explicit
$h^{2\nu}$, $h^2\log(1/h)$, and $h^2$ regimes and a directional
coefficient. The paper must continue to avoid "first" or universal novelty
language.

### 2. "The theorem is pairwise, but the experiments fit full grids."

The paper discloses this distinction in the introduction, Section 4, the
benchmark contract, and the limitations. Pairwise quadrature tests the exact
theorem; full-grid KL minimizers are computed rather than assumed to inherit
its coefficient. The finite-grid track tests model containment and
misspecification empirically. Acceptance should not depend on an unproved
pair-to-full-grid transfer.

### 3. "The high-dimensional theorem is standard or tacked on."

The paper labels the Gaussian quadratic-form certificate as standard
supporting machinery. Its role is operational: it certifies the exact
finite-library experiment without assuming independent spatial coordinates
and separates stochastic error from KL approximation error. The
benchmark-contract table makes this role explicit. The paper does not claim a
new generic concentration inequality or a continuous-parameter minimax rate.

### 4. "This is not a useful benchmark without real data or many baselines."

SupportShift is a diagnostic benchmark, not an application leaderboard. Its
purpose is to expose whether a procedure distinguishes the physical
parameter, its own KL target, and finite-sample error under a known support
operator. The support-aware and point-support fits are mechanism controls,
not claims to cover the universe of spatial estimators. The synthetic-only
scope should be viewed as appropriate for theorem and simulator validation,
not as evidence of field performance.

### 5. "Known smoothness and support make the problem too easy."

Those assumptions are deliberate identifiability controls. They isolate
support misspecification before introducing joint uncertainty in support,
smoothness, nugget, variance, and range. The paper explicitly lists unknown
support and joint estimation as future problems. It should not imply that the
support-aware model is automatically deployable when the operator is unknown.

### 6. "Boundary behavior may reverse the conclusion."

The theorem is restricted to translation-invariant interior averaging.
Section 4 displays the first-order term created by location-dependent
truncation, and the paper disclaims a universal boundary sign. Boundary and
irregular outputs are retained as stress illustrations rather than evidence
for the interior coefficient.

## Required external checks before submission

1. A spatial-statistics or applied-probability expert should verify every
   theorem and the two-page appendix, especially the Bessel expansions at
   ν=1 and ν=2, the inverse-map remainders, and the smooth-regime sign.
   The claim-by-claim worksheet is in `EXTERNAL_PROOF_REVIEW_PACKET.md`.
2. That reviewer should inspect the older regularized-variogram monographs
   listed in `PRIOR_ART_SEARCH.md` for an exact free-variance Matérn
   pseudo-range result.
3. A listed author must confirm affiliation/contact metadata, registration,
   attendance, and presentation availability.
4. The final PDF must pass the submission portal and any ACM preflight check;
   the exact uploaded file and receipt should be archived.

## Acceptance-preserving fallback

If independent review finds a repairable theorem issue, weaken the statement
and retain only experiment panels tied to the repaired result. If the
all-smoothness expansion fails materially, preserve the exact exponential
case and reproducibility artifact but reframe the submission as a conservative
simulation-validation note rather than submitting unsupported generality.
