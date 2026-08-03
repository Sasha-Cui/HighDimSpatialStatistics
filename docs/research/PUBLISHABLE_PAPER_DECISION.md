# Publishable-paper decision

Date: 2026-08-02

## Decision

Continue, but abandon the original high-dimensional/multivariate framing. The
strongest defensible paper is a focused spatial-statistics result about the
pseudo-parameter induced when locally averaged Matérn observations are fitted as
point observations. No empirical dataset is required for the core claim.

The recommended paper is:

> **How Local Averaging Moves Matérn Pairwise Range Targets: A
> Smoothness-Dependent Phase Law**

For a stationary Matérn field in fixed spatial dimension, continuous convolution
by a known compact symmetric kernel moves the naive point-pair decay target at
rate

\[
h^{2\nu}\quad(0<\nu<1),\qquad
h^2\log(1/h)\quad(\nu=1),\qquad
h^2\quad(\nu>1).
\]

Explicit coefficients show that inverse range is underestimated, hence range is
overestimated, for every smoothness and any nondegenerate symmetric compact
kernel when the support is sufficiently small. The smooth-field sign follows
from a positive Bessel-function identity and remains valid for anisotropic
averaging kernels.

This is a plausible specialized theorem-plus-simulation paper. It is not a
top-tier high-dimensional-statistics paper, and it should not be advertised as
one.

## Formal reconstruction

### Original project

The original repository attempted marginal-first and cross-parameter fitting of
the Apanasovich--Genton--Sun multivariate Matérn model, followed by smoothing and
spatial-transcriptomics analyses. The audit found no high-dimensional asymptotic
theorem, no original estimator with guarantees, and multiple fatal
implementation inconsistencies involving Matérn scale, signs, stacking, Bessel
derivatives, likelihood dimension, identifiability, and smoothed-grid
dependence. Those results cannot support a paper.

### Salvaged model

Let

\[
C(r)=v\mathcal M_\nu(\alpha\|r\|),\qquad
\mathcal M_\nu(x)=\frac{2^{1-\nu}}{\Gamma(\nu)}x^\nu K_\nu(x),
\]

and observe

\[
Z_h(t)=\int k_h(u)Y(t-u)\,du,\qquad k_h(u)=h^{-d}k(u/h),
\]

where \(k\) is a known symmetric compact probability kernel. If \(U,V\) are
independent draws from \(k\) and \(D=U-V\), then

\[
C_h(r)=v\,\mathbb E\mathcal M_\nu\{\alpha\|r+hD\|\}.
\]

At a fixed nonzero lag \(R\), the naive two-point Gaussian model fits variance
\(v_h^\dagger\) and decay \(\alpha_h^\dagger\) as though the observations had
point support. Its exact KL target is

\[
v_h^\dagger=C_h(0),\qquad
\mathcal M_\nu(\alpha_h^\dagger R)=C_h(R)/C_h(0).
\]

The proposed correction uses the known observation support. On a finite grid
with rectangular smoothing matrix \(S\), raw-site noise variance \(\tau^2\), and
latent covariance \(K_\alpha\), it fits

\[
\Sigma_{\mathrm{corr}}(\alpha)=S(K_\alpha+\tau^2I)S^\top
\]

instead of the naive centroid model

\[
\Sigma_{\mathrm{naive}}(\alpha)=K_\alpha(X_{\mathrm{out}},X_{\mathrm{out}})
  +\tau^2I.
\]

## Theorem and proof audit

| Result | Status | Remaining risk |
|---|---|---|
| Exact continuous support covariance | Complete | Standard change-of-support identity; not novel |
| Uniform fixed-nonzero-lag \(h^2\) expansion | Complete | Requires compact kernel and lag bounded away from zero |
| Origin expansion for \(0<\nu<1\) | Complete | Moment/remainder statement needs independent review |
| Transition expansion at \(\nu=1\) | Complete | Logarithmic constants need independent review |
| Smooth regime \(\nu>1\) | Complete | Bessel recurrence and sign identity independently checkable |
| Exact naive pair KL target and uniqueness | Complete | Applies to the declared pair criterion |
| Universal small-\(h\) range-inflation sign | Complete | Only an eventual small-support statement |
| Increasing-domain pair-estimator consistency | Standard argument is defensible | Add full uniform-ergodic details for a journal |
| Exact separable anisotropic exponential benchmark | Complete | Tensorizes classical temporal aggregation |
| Discrete-to-continuous uniform approximation | Not proved | Do not claim it |
| Boundary-row-normalization expansion | Not proved | Finite-design stress test only |
| Full-grid analytic pseudo-target | Not proved | Compute finite-design KL target numerically |
| Joint nuisance estimation | Not studied | Fix nuisance parameters in this paper |

No independence assumption is used for overlapping spatial pairs. Consistency
uses ergodicity/mixing; uncertainty would require a Godambe or long-run
covariance. No separate fixed-domain consistency claim for variance and range is
made.

## Literature and novelty map

- Change of support and \(SKS^\top\) are established spatial-statistics ideas;
  see Gelfand, Zhu and Carlin (2001), Gotway and Young (2002), Kyriakidis and
  Yoo (2005), and Chacón-Montalván et al. (2024).
- Fuentes (2007) is the closest methodological predecessor found: it develops
  approximate likelihood for irregularly placed block averages and includes a
  Matérn example. It models the blocks correctly; it does not derive the
  pseudo-range selected when those averages are instead fitted as points, nor
  the three smoothness-dependent small-support regimes and their signs.
- Aggregated AR/ARIMA and one-dimensional OU calculations are classical; see
  Amemiya and Wu (1972), Stram and Wei (1986), and Folia and Rattray (2018).
- Misspecified Gaussian likelihood selecting a KL target is established; see
  Bachoc (2018).
- Fixed-domain Matérn parameters require microergodic care; see Zhang (2004).
- Recent finite-grid Matérn likelihood work handles discretization and edge
  effects but does not provide this preprocessing-support pseudo-target phase
  law; see Simons et al. (2026).

Targeted searches did not identify the conjunction of an all-\(\nu\),
\(d\)-dimensional Matérn small-support expansion, an explicit pseudo-decay
coefficient, and a universal sign result. The novelty claim should remain
scoped:

> We derive an explicit smoothness-dependent small-support expansion for the
> naive Matérn pair-likelihood target and prove that ignored symmetric local
> support inflates inferred range in every smoothness regime.

Do not claim novelty for change of support, covariance propagation, generic KL
projection, the one-dimensional OU formula, or support-aware likelihood by
itself.

## Candidate papers

### 1. Matérn phase law plus finite-grid validation — primary

- **Main theorem:** the three bandwidth regimes and strict range-inflation sign.
- **Assumptions:** fixed \(d\), known \(\nu\), known symmetric compact support,
  fixed nonzero fitted lag, stationary zero-mean Gaussian field.
- **Experiments:** deterministic 2D quadrature; finite 2D exact-versus-naive KL
  targets; 200-replicate synthetic likelihood study; boundary and irregular
  stress panels.
- **Baselines:** naive centroid likelihood, support-aware likelihood, exact pair
  target, and zero-bandwidth selector.
- **Principal risk:** a close unlocated theorem or an error found by external
  proof review, especially at the \(\nu=1\) transition.
- **Minimum publishable result:** theorem, exact OU benchmark, deterministic
  quadrature, and concise finite-grid validation.
- **Ambitious extension:** full-likelihood spectral and boundary expansions.

### 2. Exact anisotropic exponential correction — fallback

- **Contribution:** exact \(d\)-axis pseudo-target, universal signs, and two-lag
  correction for separable exponential covariance.
- **Risk:** mathematically a tensor product of classical one-dimensional
  aggregation.
- **Minimum outlet:** low-tier workshop or short methodological note.

### 3. Discrete boundary-aware computational study — later extension

- **Contribution:** finite-design KL maps for irregular and boundary-normalized
  smoothing, with a practical correction.
- **Risk:** without a real application or boundary theorem it may look like a
  simulation exercise.

Direction 1 has the best novelty/feasibility/significance balance.

## Simulation and validation plan

The reproducible design is synthetic-only.

1. **Continuous theorem oracle.** Two-dimensional product Epanechnikov
   quadrature at six smoothness values and 18 bandwidths.
2. **Finite 2D core.** A \(19\times19\) raw lattice, rectangular output support,
   four smoothness values, and four bandwidths. Fit decay only, with variance
   and smoothness known and nugget set to zero to isolate support.
3. **Stress tests.** Retain boundary outputs, jitter raw inputs, and compare 1D
   increasing-domain sizes 100, 225, and 400.
4. **Population check.** Compute the nonrandom finite-design KL target for every
   configuration. Compare empirical estimates with that target.
5. **Controls.** At \(h=0\), exact and naive covariances, objectives, and
   estimates coincide. Corrected population decay equals the truth.
6. **Reproducibility.** Immutable manifest; SHA-256 hashes; deterministic common
   random numbers; atomic task shards; after-any reducer; no silent jitter or
   dropped failures.

The shakedown used 21 array elements and one reducer under pi_jss233. The
paper-grade run used the same 21 configurations with 200 replicates. Each stage
required only two sbatch submissions, far below the 180/hour limit.

The initial promoted run (Slurm array 21071686, reducer 21071687) exposed a
design confound: it combined latent support misspecification with incorrect
noise timing in the naive model. It is retained as a stress artifact, not as
primary evidence.

The support-only promoted run completed on 2026-08-02 under Slurm array 21073200
and reducer 21073201. All 21 tasks completed successfully; the reducer audited
8,400 aggregate fitted estimates with no missing or invalid shard. Across the
core two-dimensional designs, the support-aware population target was within
\(1.3\times10^{-7}\) of the true decay \(1\). At \(h=0.7\), the naive targets
were \(0.152,0.414,0.573,0.740\) for
\(\nu=0.5,1,1.5,2.5\), respectively. The corresponding Monte Carlo means were
\(0.151,0.407,0.570,0.738\). All declared boundary hits were retained.

## Repository and proof-completion plan

Completed:

- original main preserved at the pre-audit commit and work developed on
  research/paper-audit;
- corrected Matérn convention, cross signs, stacking, gradients, and likelihood;
- theory, design, KL, estimator, and experiment modules;
- closed-form OU/Epanechnikov software oracle;
- rectangular smoothing and transformed-noise likelihood;
- manifests, CPU arrays, atomic shards, reducer, central environment, and logs;
- 56 passing tests on Bouchet before the promoted run;
- LaTeX manuscript source, checked bibliography, generated tables, and
  publication-resolution figures;
- deterministic phase oracle with 108 positive shifts and the predicted three
  smoothness regimes;
- promoted 200-replicate finite-grid experiment with a complete shard audit.

Before submission:

1. Have the external reviewer verify the theorem, especially the Bessel series
   signs and \(\nu=1\) remainder.
2. Run a second targeted literature search around block Matérn covariance and
   pseudo-true range.
3. Reproduce the frozen figures and tables from a clean clone of the release
   commit.
4. Fill affiliations, author order, acknowledgments, funding, and disclosure
   statements.
5. Convert to the selected venue template and check page limits.

## Venue and submission roadmap

### Fast primary target

[GeoSim 2026](https://geosim.org/2026/cfp/) is the only credible immediate
archival workshop target found. It accepts peer-reviewed full papers up to 10
pages and short papers up to 4 pages in the ACM Digital Library. The deadline is
**2026-08-15**, notification is **2026-09-15**, and the workshop is
**2026-11-03**. Frame the paper around verifying spatial simulation and the
inferential consequence of support.

Internal schedule:

- Aug 2–5: proof review and final synthetic artifacts;
- Aug 5–8: ACM sigconf conversion and related-work tightening;
- Aug 9–11: independent reproduction and author review;
- Aug 12: internal freeze;
- Aug 13–14: upload/checksum buffer;
- Aug 15: official deadline.

### Rolling journal route

- **Statistics & Probability Letters:** best for a concise rigorous theorem.
- **Statistics:** plausible for the full theorem-plus-simulation manuscript.
- **Communications in Statistics—Simulation and Computation** or **Journal of
  Statistical Computation and Simulation:** realistic simulation-heavy targets.
- **Austrian Journal of Statistics:** lower-prestige fallback without a real
  application.
- **Spatial Statistics:** strong fit but less realistic without an application.

If GeoSim is missed or rejects, the verified lower-tier archival fallback is
[GEOProcessing 2027](https://www.iaria.org/conferences2027/GEOProcessing27.html),
with deadline **2026-12-28** and conference dates **2027-04-18 to 2027-04-22**.

## Candid go/no-go assessment

**Continue the narrowed project.** The core theorem is materially stronger than
the original repository and the deterministic rate experiment supports it. The
project is worth finishing for a specialized journal or archival workshop.

**Abandon these claims:** high dimensionality, a new multivariate Matérn model,
originality of \(SKS^\top\), separate fixed-domain range/variance consistency,
and any spatial-transcriptomics conclusion.

**Stop or downgrade to the exact exponential fallback** if independent proof
review finds the all-\(\nu\) sign identity wrong, or if a close prior paper
already states the same pseudo-target expansion.
