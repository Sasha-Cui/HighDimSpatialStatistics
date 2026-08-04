# Final research assessment: SupportShift

**Assessment date:** 2026-08-02

**Repository:** HighDimSpatialStatistics

**Preserved base:** main at 2a6ef52

**Development branch audited:** research/paper-audit; final benchmark generated
at d5207fb and release hardening continued on the same branch

**Recommended paper type:** narrow theory-plus-synthetic-benchmark paper

**Recommended immediate target:** GeoSim 2026, conditional on proof review and an
author being able to register and present

## Executive decision

The project should be **continued, but permanently narrowed**. The original
multivariate/high-dimensional Matérn project does not support a paper: its model
implementation, likelihood, archived experiments, and claimed asymptotic
interpretation contain fatal errors. Those results must not be rehabilitated by
relabeling them.

A different and mathematically coherent paper has, however, been recovered:

> **SupportShift: Matérn Range Distortion under Ignored Observation Support,
> with a Theory-Linked Synthetic Benchmark**

Its principal result is a smoothness-dependent phase law for the pseudo-inverse
range obtained when locally averaged Matérn observations are fitted as if they
had point support. For a fixed nonzero fitted lag and a known compact symmetric
averaging kernel of bandwidth \(h\),

\[
\alpha-\alpha_h^\dagger
\asymp
\begin{cases}
h^{2\nu}, & 0<\nu<1,\\
h^2\log(1/h), & \nu=1,\\
h^2, & \nu>1.
\end{cases}
\]

The leading coefficient is positive in this display: ignored local support
decreases inverse range and therefore inflates inferred range for every fixed
\(\nu>0\), once \(h\) is sufficiently small. An explicit order-\(h^2\)
directional contrast describes apparent range anisotropy caused by elongated
observation support.

A separate finite-library proposition supplies the requested
high-dimensional-probability framing. For \(N\) independent
\(p\)-dimensional Gaussian spatial fields and \(M\) deterministic covariance
candidates, it controls normalized likelihood error and empirical-risk excess.
Under uniform relative spectral control, its radius is

\[
O\!\left\{
\sqrt{\frac{\log M}{Np}}+\frac{\log M}{Np}
\right\}.
\]

This proposition allows arbitrary spatial dependence among the \(p\)
coordinates. It is a correct and useful bridge from theory to the synthetic
experiment, but it is based on standard Gaussian quadratic-form concentration
and is **not** itself a new high-dimensional theorem. The project is therefore
not a top-tier high-dimensional-statistics or theoretical-ML paper. It is a
plausible specialized spatial-statistics theorem, an archival simulation
workshop paper, or a concise probability/statistics letter.

The absence of a real dataset is not fatal for this version. It does rule out a
convincing application paper and makes Spatial Statistics a less realistic
first journal because that journal rarely accepts purely theoretical work.

## 1. Evidence base and provenance

The audit covered the original notebooks and converted scripts, package source,
LaTeX manuscripts and appendices, proof notes, experiment configurations,
synthetic artifacts, reference library, tests, Git history, and the successive
research-branch commits. It also reconstructed several archived simulations
directly enough to falsify their original interpretation.

The original main branch is preserved at 2a6ef52. All scientific salvage work
was developed on research/paper-audit. The paper-grade SupportShift benchmark
was executed from clean commit
d5207fb43cf63f5dfd68a443853a209e303f9aa2. The final schema-1.1 run records:

- Slurm job 21081491 under pi_jss233;
- host a1132u31n04.mghpcc.ycrc.yale.edu;
- Python 3.12.8, NumPy 2.3.5, and SciPy 1.17.1;
- 12,800 finite, uniquely keyed likelihood fits;
- 16,261 variance-decay candidates in every fit;
- a root seed of 20260803;
- a clean Git state;
- exact SHA-256 hashes for the result and raw-example files; and
- all predeclared validation gates passing.

The audited numerical release was frozen under tag
`supportshift-geosim-v1.0.0`. Documentation, artifact metadata, the explicit
benchmark-consumer contract, and the more precise Fuentes comparison were added
without changing promoted numerical inputs in patch release
`supportshift-geosim-v1.0.1`.
Release `supportshift-geosim-v1.1.0` adds the transition-aware theorem,
the clean-commit threshold stress audit, and its manifest-linked source data and
figure; it does not alter the earlier finite-grid or replicated-field inputs.
Patch release `supportshift-geosim-v1.1.1` promotes the phase-oracle provenance,
adds a 100-check manuscript-claim audit with a negative mutation test, and
foregrounds simulation and model verification in the ACM classification and
abstract. It changes no reported numerical value.
Patch release `supportshift-geosim-v1.1.2` adds the complete one-page proof
appendix permitted by the GeoSim call and repairs the closest-prior-art map:
Clark (1977) and Bellehumeur and Legendre (1997) are now cited as qualitative
predecessors for support-induced apparent-range inflation and distinguished
from the variance-refitted, infinite-range Matérn KL target. It changes no
theorem statement or reported numerical value.

The pre-audit history consisted of four bulk commits made on one day and did not
record a theorem or experiment-development trail. The research branch now has
separate commits for the mathematical salvage, simulation workflow,
anisotropic oracle, high-dimensional benchmark, candidatewise audit, and
reproducibility hardening. That later history is scientifically useful; the
original history is provenance rather than evidence.

The final benchmark files are:

- outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.csv;
- outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.metadata.json;
- outputs/smoothing_bias/supportshift_raw_final_v2_20260803.csv;
- outputs/smoothing_bias/supportshift_anisotropic_final_20260803.csv; and
- outputs/smoothing_bias/supportshift_anisotropic_final_20260803.metadata.json;
- outputs/smoothing_bias/supportshift_transition_stress_20260804.csv; and
- outputs/smoothing_bias/supportshift_transition_stress_20260804.metadata.json.

The artifact manifest at paper/data/supportshift_artifact_manifest.json links
the final inputs to generated source data, figures, and tables. Manifest schema
1.1 explicitly records any source file that is also a paper output and forbids
rewriting that aliased input. Every current input/output hash verifies.

The maintained suite was verified with 102 passing tests in the configured
research environment, and the maintained SupportShift Ruff scope is clean.
Reproduction instructions point to the declared Python 3.12 environment and
promoted package snapshot rather than an arbitrary system Python.

## 2. Formal reconstruction

### 2.1 The original project

The repository originally attempted to fit a stationary, isotropic,
\(q\)-variate Gaussian random field

\[
Z(s)=\{Z_1(s),\ldots,Z_q(s)\}^{\mathsf T}
\]

using a flexible multivariate Matérn cross-covariance

\[
C_{ij}(r)
=\sigma_{ij}
\frac{2^{1-\nu_{ij}}}{\Gamma(\nu_{ij})}
\{\alpha_{ij}\lVert r\rVert\}^{\nu_{ij}}
K_{\nu_{ij}}(\alpha_{ij}\lVert r\rVert).
\]

The intended sufficient construction was the
Apanasovich--Genton--Sun parameterization, followed by its established
marginal-first, cross-parameter-second fitting scheme. The intended exact
Gaussian negative log likelihood for \(n\) locations was

\[
\ell_n(\theta)
=\frac12 y^{\mathsf T}K_\theta^{-1}y
{}+\frac12\log\det K_\theta
{}+\frac{nq}{2}\log(2\pi),
\]

with one fixed and documented stacking convention.

That original project is not salvageable as a scientific result. The audit
found all of the following:

1. The code used a length-scale Matérn convention while applying validity
   formulas for an inverse-range convention.
2. Squaring cross amplitudes erased negative covariance.
3. Covariance matrices were generated in variable-major order while responses
   were reshaped and fitted as location-major vectors.
4. The derivative of \(K_\nu(x)\) in its argument was used as if it were the
   derivative in Bessel order.
5. The multivariate likelihood constant counted rows rather than scalar
   observations.
6. The fitted cross-parameter family was nonidentified and could not represent
   the mixed-sign synthetic truth.
7. Distance rounding plus detached eigenvalue shifts did not define a fixed
   differentiable positive-definite covariance family.
8. Smoothed and overlapping grids were assigned point-support covariance and
   treated as independent batches, even though their correct covariance and
   cross-covariance are nonzero.
9. Fixed-box densification was described without accounting for Matérn
   microergodicity; separate variance and range consistency was unsupported.
10. Archived benchmark truth matrices sometimes used unrelated coordinates or
    the legacy swapped-parameter kernel.
11. No growing-\(q\) theorem, minimax result, support-recovery guarantee, valid
    uncertainty calculation, or scalable estimator was present.
12. The real-data model omitted a defensible mean, measurement model,
    biological replication, and independent validation.

The original \(q=3\) simulations and approximately \(q=22\) gene analysis were
not high-dimensional statistics. Repairs made on the research branch are useful
software regression guards, but they invalidate rather than rescue the old
numerical conclusions.

### 2.2 The salvaged continuous-support model

Let \(Y=\{Y(t):t\in\mathbb R^d\}\) be a mean-zero stationary Gaussian field with

\[
C(r)=v\mathcal M_\nu(\alpha\lVert r\rVert),\qquad
\mathcal M_\nu(x)
=\frac{2^{1-\nu}}{\Gamma(\nu)}x^\nu K_\nu(x),
\]

where \(v>0\), inverse range \(\alpha>0\), and smoothness \(\nu>0\). Larger
\(\alpha\) means shorter range.

Let \(k\) be a nonnegative symmetric probability density supported on
\(B(0,L)\), let \(k_h(u)=h^{-d}k(u/h)\), and observe the local average

\[
Z_h(t)=\int_{\mathbb R^d}k_h(u)Y(t-u)\,du.
\]

For independent \(U,V\sim k\), write

\[
D=U-V,\qquad
\Sigma_k=\mathbb E(UU^{\mathsf T}),\qquad
T_k=\operatorname{tr}(\Sigma_k).
\]

Stationarity and Fubini's theorem give the exact covariance

\[
C_h(r)=v\,\mathbb E\!\left[
\mathcal M_\nu\{\alpha\lVert r+hD\rVert\}
\right].
\]

This identity is classical change of support and is not a novelty claim.

### 2.3 Inferential objective and exact pair target

Fix a nonzero lag \(r=Re\), where \(R>0\) and \(\lVert e\rVert=1\). A naive
analyst fits the smoothed pair with a point-support Matérn covariance, treating
smoothness as known and allowing variance and inverse range to vary. The exact
Gaussian KL minimizer is

\[
v_h^\dagger=C_h(0),\qquad
\mathcal M_\nu(\alpha_h^\dagger R)
=\frac{C_h(Re)}{C_h(0)}.
\]

Strict monotonicity,

\[
\mathcal M_\nu'(x)
=-\frac{2^{1-\nu}}{\Gamma(\nu)}x^\nu K_{\nu-1}(x)<0,
\]

identifies \(\alpha_h^\dagger\). The inferential target is therefore the
population parameter induced by misinterpreting observation support, not the
finite-sample bias of an otherwise correct estimator.

### 2.4 Finite-design corrected and naive procedures

On raw input locations, let \(S_h\in\mathbb R^{p\times n}\) be the known
rectangular smoothing or aggregation matrix. If raw-site noise is present
before smoothing,

\[
y=S_h\{Y(X_{\mathrm{in}})+\varepsilon\},\qquad
\varepsilon\sim N_n(0,\tau^2I_n).
\]

The support-aware covariance is

\[
\Sigma_{\mathrm{corr}}(v,\alpha)
=vS_hR_\alpha S_h^{\mathsf T}
{}+\tau^2S_hS_h^{\mathsf T}.
\]

The naive point-support family is

\[
\Sigma_{\mathrm{naive}}(v,\alpha)
=vR_\alpha(X_{\mathrm{out}},X_{\mathrm{out}})
{}+\tau^2I_p.
\]

The primary experiments set \(\tau=0\), so support misspecification is not
confounded with incorrect noise timing. At fixed finite design, the population
objective is

\[
Q_p(v,\alpha)
=\frac{1}{2p}
\left[
\log\det\Sigma(v,\alpha)
{}+\operatorname{tr}\{\Sigma(v,\alpha)^{-1}\Sigma_0\}
\right].
\]

The support-aware family contains \(\Sigma_0\) when the support and nuisance
quantities are known. The naive family generally does not. Full-grid
pseudo-targets are computed by numerical KL minimization; no theorem says that
they have the pairwise phase coefficient.

### 2.5 The high-dimensional-probability experiment

Here \(p\) is the number of spatial output coordinates in one field, not the
number of response variables from the original multivariate model. For

\[
X_1,\ldots,X_N\stackrel{\mathrm{iid}}{\sim}N_p(0,\Sigma_0),
\]

and a deterministic finite library
\(\{\Sigma_\theta:\theta\in\mathcal G\}\) of size \(M\), define

\[
\widehat\Sigma_N=\frac1N\sum_{i=1}^N X_iX_i^{\mathsf T},
\]

\[
\widehat L_N(\theta)
=\frac1{2p}\left\{
\log\det\Sigma_\theta+
\operatorname{tr}(\Sigma_\theta^{-1}\widehat\Sigma_N)
\right\}.
\]

The estimator is exact finite-library empirical risk minimization:

\[
\widehat\theta\in\arg\min_{\theta\in\mathcal G}\widehat L_N(\theta).
\]

Spatial coordinates inside \(X_i\) are correlated. Independence is assumed only
across replicated fields \(i=1,\ldots,N\).

### 2.6 Assumptions that must appear explicitly

The phase theorem requires:

- a mean-zero stationary Matérn field with the declared inverse-range
  convention;
- fixed spatial dimension \(d\);
- fixed known \(\nu>0\);
- a known, nonnegative, symmetric, compactly supported probability kernel;
- nondegenerate kernel second moment \(T_k>0\);
- a fixed lag in an annulus bounded away from zero;
- \(h\le R/(4L)\) for the stated uniform Taylor neighborhood;
- \(\alpha\) in a compact subset of \((0,\infty)\); and
- free pair variance and inverse range.

The finite-library result requires:

- genuinely independent Gaussian fields across \(N\);
- known zero mean;
- strictly positive-definite truth and candidates;
- a deterministic candidate library;
- the exact \(1/(2p)\) likelihood normalization; and
- a separation or curvature condition before likelihood error can be converted
  into selection or parameter error.

The simplified \(Np\) rate additionally requires uniform relative spectral
control

\[
c\Sigma_0\preceq\Sigma_\theta\preceq C\Sigma_0.
\]

That condition is assumed, not implied by spatial stationarity.

### 2.7 Internally consistent asymptotic regimes

The revised project separates five regimes that the original work conflated.

| Question | Regime | Defensible conclusion | Explicit nonclaim |
|---|---|---|---|
| Continuous support distortion | \(h\downarrow0\), fixed \(d,\nu,\alpha,R\) | Pair pseudo-target phase law | No sample-size consistency statement |
| Pair-composite estimation | Increasing rectangular Følner domains at fixed lattice spacing | Consistency for the pair KL target under ergodicity and uniformity assumptions | Overlapping pairs are not independent |
| Finite full likelihood | Fixed finite lattice and fixed \(S_h\) | Numerical finite-design KL target | No universal full-grid phase law |
| Replicated-field probability | \(p\) and/or \(N\) increase; fields independent, coordinates dependent | Criterion and excess-risk concentration, conditional \(Np\) simplification | No automatic parameter rate or minimax theorem |
| Fixed-domain infill | Bounded domain, spacing shrinks | Only appropriately microergodic combinations may be estimable | No separate variance/range consistency |

In the growing-\(p\) benchmark, grid spacing is fixed while the output square
grows from side four to side ten. This is increasing-domain geometry. Increasing
that domain does not make a fixed discrete smoother converge to the continuous
convolution theorem; the continuous and discrete tracks remain deliberately
separate.

## 3. Theorem and proof audit

### 3.1 Status summary

“Complete” below means internally derived, written, numerically checked, and
covered by regression tests. It does not replace independent expert review.

| Statement | Status | Audit judgment |
|---|---|---|
| Exact continuous support covariance | Complete, classical | Correct by Fubini and stationarity; not novel |
| Exact pair KL target and uniqueness | Complete | Correct for known \(\nu\), free variance and decay, fixed \(R>0\) |
| Uniform fixed-nonzero-lag expansion | Complete | Correct only away from the origin and with bounded support |
| Origin expansion for \(0<\nu<1\) | Complete | Fractional \(h^{2\nu}\) term and remainder are consistent |
| Threshold expansion at \(\nu=1\) | Complete | The \(h^2\log(1/h)\) coefficient is consistent |
| Transition-aware two-term approximation | Complete | Singular quadratic and fractional terms cancel continuously at one; error orders remain pointwise in \(\nu\) |
| Smooth expansion for \(\nu>1\) | Complete | Includes separate remainders at \(1<\nu<2\), \(\nu=2\), and \(\nu>2\) |
| Universal small-support range-inflation sign | Complete | Bessel recurrence reduces the coefficient to a positive function |
| Directional support contrast for all \(\nu>0\) | Complete | Common origin term cancels, leaving order \(h^2\) |
| Exact finite-design support-aware containment | Complete | True only when \(S_h\), mean, noise timing, and nuisance values are correct |
| Finite-library likelihood certificate | Complete, standard | Constants match the \(1/(2p)\) normalization |
| ERM excess-KL inequality | Complete, standard | Parameter error still needs a margin |
| Increasing-domain pair consistency | Defensible proof sketch | Expand uniform ergodic argument for journal submission |
| Boundary behavior | Partial | A local order-\(h\) term is identified; no universal boundary theorem or sign |
| Full-grid analytic pseudo-target | Not proved | Keep it numerical |
| Continuous joint variance-range estimator rate | Not proved | Finite-library result is not a substitute |
| Unknown support, bandwidth, or smoothness | Not studied | Do not imply adaptivity |
| Nonstationarity or latent anisotropy theory | Not studied | Anisotropic observation support is the only directional theorem |
| Minimax rate or lower bound | Absent | Do not use minimax language |
| Original multivariate/high-dimensional theorem | Failed/absent | Abandon the claim |

### 3.2 Central phase theorem

Put \(z=\alpha R\), \(a_k(e)=e^{\mathsf T}\Sigma_ke\), and
\(m_q=\mathbb E\lVert U-V\rVert^q\). For \(0<\nu<1\), define

\[
c_\nu
=-\frac{\Gamma(-\nu)}{2^{2\nu}\Gamma(\nu)}
=\frac{\Gamma(1-\nu)}
{\nu\,2^{2\nu}\Gamma(\nu)}>0.
\]

The variance expansion is

\[
\frac{C_h(0)}{v}
=
\begin{cases}
1-c_\nu m_{2\nu}(\alpha h)^{2\nu}+O(h^2),
&0<\nu<1,\\[3pt]
1-\dfrac{m_2}{2}(\alpha h)^2\log\{1/(\alpha h)\}+O(h^2),
&\nu=1,\\[6pt]
1-\dfrac{m_2}{4(\nu-1)}(\alpha h)^2+o(h^2),
&\nu>1.
\end{cases}
\]

The corresponding inverse-range displacement is

\[
\alpha_h^\dagger-\alpha
=
\frac{\mathcal M_\nu(z)c_\nu m_{2\nu}\alpha^{2\nu}}
{R\mathcal M_\nu'(z)}
h^{2\nu}
{}+O\!\left(h^{\min(4\nu,2)}\right),
\qquad 0<\nu<1,
\]

\[
\alpha_h^\dagger-\alpha
\sim
\frac{m_2\mathcal M_1(z)\alpha^2}
{2R\mathcal M_1'(z)}
h^2\log(1/h),
\qquad \nu=1,
\]

and

\[
\alpha_h^\dagger-\alpha
=\frac{A_{\nu,k}(Re)}
{R\mathcal M_\nu'(z)}h^2+\mathcal R_\nu(h),
\qquad \nu>1,
\]

where

\[
A_{\nu,k}(Re)
=\alpha^2\{(2\nu-2)a_k(e)+T_k\}G_\nu(z)>0,
\]

\[
G_\nu(z)
=\frac{2^{1-\nu}}{\Gamma(\nu)}
\frac{z^\nu K_{\nu-2}(z)}{2\nu-2}>0.
\]

Because \(\mathcal M_\nu'(z)<0\), every displayed leading coefficient for
\(\alpha_h^\dagger-\alpha\) is negative. This yields

\[
v_h^\dagger<v,\qquad
\alpha_h^\dagger<\alpha,\qquad
\alpha/\alpha_h^\dagger>1
\]

for sufficiently small positive \(h\).

The smooth-regime remainder is \(O(h^{2\nu})\) for \(1<\nu<2\),
\(O\{h^4\log(1/h)\}\) at \(\nu=2\), and \(O(h^4)\) for \(\nu>2\).
Uniformity is not claimed as \(\nu\) approaches one or two.

### 3.3 Directional corollary

For two unit lag directions \(e_1,e_2\), with equal fixed lag length \(R\), let
\(\Delta_e(h)=\alpha-\alpha_h^\dagger(e)\). Then

\[
\Delta_{e_1}(h)-\Delta_{e_2}(h)
=
\frac{\alpha^2\{a_k(e_1)-a_k(e_2)\}}{R}
\frac{K_{\nu-2}(\alpha R)}{K_{\nu-1}(\alpha R)}
h^2+o(h^2).
\]

This holds for every fixed \(\nu>0\). For rough fields it is lower order than
the direction-independent phase term. For the paper's aspect-\(\rho\)
two-dimensional support family, the major/minor leading-coefficient ratio for
\(\nu>1\) is

\[
\frac{(2\nu-1)\rho^2+1}{\rho^2+2\nu-1}.
\]

### 3.4 Finite-library probability proposition

Let

\[
A_\theta=\Sigma_0^{1/2}\Sigma_\theta^{-1}\Sigma_0^{1/2},
\qquad
t=\log(2M/\delta).
\]

With probability at least \(1-\delta\), simultaneously over the fixed library,

\[
\left|\widehat L_N(\theta)-L(\theta)\right|
\le
\frac{\lVert A_\theta\rVert_{\mathrm F}}{p}
\sqrt{\frac{t}{N}}
{}+
\frac{\lVert A_\theta\rVert_{\mathrm{op}}}{p}
\frac{t}{N}
=:u_\theta.
\]

If \(\widehat\theta\) and \(\theta^\star\) minimize empirical and population
risk, respectively,

\[
L(\widehat\theta)-L(\theta^\star)
\le 2\max_{\theta\in\mathcal G}u_\theta.
\]

Moreover, \(L(\theta)\) differs by a candidate-independent constant from

\[
\frac1pD_{\mathrm{KL}}
\{N_p(0,\Sigma_0)\,\|\,N_p(0,\Sigma_\theta)\}.
\]

The proof stacks the \(N\) whitened fields, applies the two-sided Gaussian
quadratic-form inequality to \(I_N\otimes A_\theta\), divides by \(2Np\), and
uses a union bound. The factors of two and \(p\) have been audited. No spatial
coordinate independence is used.

The theorem does not directly give
\(\lVert\widehat\theta-\theta^\star\rVert\). A local or global margin such as

\[
L(\theta)-L(\theta^\star)
\ge c_0 d(\theta,\theta^\star)^\kappa
\]

is needed for a parameter rate. A positive library gap is needed for exact
model selection. The current paper states those limitations.

### 3.5 Proof dependency map

The dependency chain is:

\[
\begin{aligned}
&\text{exact support covariance}
\longrightarrow \text{pair KL target},\\
&\text{fixed-lag Taylor expansion}
{}+\text{Matérn origin expansion}
\longrightarrow \text{normalized correlation},\\
&\text{normalized correlation}
{}+\text{inverse map}
{}+\text{Bessel sign identity}
\longrightarrow \text{phase theorem},\\
&\text{fixed-lag directional difference}
{}+\text{inverse map}
\longrightarrow \text{directional corollary},\\
&\text{Gaussian quadratic-form tail}
{}+\text{finite union bound}
\longrightarrow \text{likelihood certificate}
\longrightarrow \text{ERM excess KL}.
\end{aligned}
\]

The continuous phase theorem and the finite-library proposition are logically
independent. The finite-design covariance \(S_hKS_h^{\mathsf T}\) connects them
only at the synthetic generator.

### 3.6 Specific theoretical hazards and their disposition

**Independence.** Overlapping spatial pairs are dependent. The pair consistency
sketch uses ergodicity on increasing domains; it does not count pairs as
independent. The high-dimensional proposition instead requires independent
replicated fields and permits dependence within a field.

**Concentration.** The radius is exact for a fixed Gaussian covariance library.
It is not valid for a library selected on the same samples. It is also an oracle
diagnostic because \(A_\theta\) contains unknown \(\Sigma_0\).

**Boundary effects.** Row-normalized truncated support can have a nonzero first
moment. For two local kernels with means \(\mu_s,\mu_t\),

\[
C_{h,s,t}(r)
=C(r)+h\nabla C(r)^{\mathsf T}(\mu_s-\mu_t)+O(h^2).
\]

Thus boundaries can create order-\(h\) effects, nonstationarity, and a sign
different from the interior theorem. The boundary panel is illustrative, not a
boundary theorem.

**Nonstationarity and latent anisotropy.** Neither is covered. The anisotropic
track changes the observation support while leaving the latent Matérn field
isotropic.

**Covariance misspecification.** The naive family is deliberately
misspecified. The corrected family is correct only because the benchmark knows
\(S_h\), the zero mean, \(\nu\), noise timing, and the candidate library.

**Identifiability.** The pair target is identified because the Matérn
correlation is strictly decreasing at fixed \(R>0\). Full-grid parameter
conclusions require an interior unique KL minimizer. Unknown support and range
may be weakly confounded and are outside scope.

**Fixed versus increasing domain.** The revised text no longer claims separate
fixed-domain variance and range consistency. The conditional \(Np\) certificate
is not a way around Gaussian-measure equivalence or microergodicity.

**Pointwise versus uniform claims.** The phase result is uniform over compact
inverse-range sets and fixed-lag annuli for each fixed \(\nu\), but not uniform
in \(\nu\) near one or two. The likelihood certificate is uniform only over a
finite declared library. No minimax or fully uniform continuous-parameter
statement is available.

**Dimension, sparsity, and resolution.** There is no sparsity parameter,
graph-support target, or sparse covariance estimator in the revised paper.
Spatial output dimension grows at fixed lattice spacing. It would be incorrect
to reinterpret \(p\) as independent sample size or to claim a sparsity rate.

## 4. Correspondence between theory and experiments

### 4.1 Claim-to-artifact map

| Theory claim | Synthetic track | Result | Interpretation |
|---|---|---|---|
| Three phase regimes | 2D product-Epanechnikov deterministic quadrature | 108 targets over six \(\nu\) and 18 bandwidths | Direct coefficient and rate check |
| Smoothness-one transition | Fine-grid deterministic quadrature | 111 targets; two-term error below 0.1% | Resolves finite-bandwidth cancellation without claiming a joint uniform theorem |
| Universal small-\(h\) sign | Same phase track | All 108 inverse-range shifts positive | Supports, but does not replace, proof |
| Directional \(h^2\) contrast | Elongated-support quadrature | 2,128 rows; all gates pass | Tests exact directional coefficient |
| Correct family contains truth | Finite-grid population KL | Corrected target numerically one | Model-containment control |
| Naive full-grid target is displaced | Finite-grid likelihood | 8,400 fits in 21 configurations | Numerical full-likelihood evidence only |
| Finite-library concentration | Schema-1.1 replicated fields | 12,800 fits in 64 cells | Exact theorem event evaluated candidate by candidate |
| Sampling error can vanish around a wrong target | Growing \(N,p\) track | RMSE to naive KL target falls while error to physical truth persists | Central empirical phenomenon |
| Exact observation map is reproducible | Raw latent/averaged CSV | Four fields, 529 latent and 100 output sites each | Reconstructable illustration |

### 4.2 Continuous phase results

The phase oracle used \(d=2\), \(\alpha=R=1\), a product Epanechnikov kernel,
96-point tensor quadrature, 18 bandwidths from 0.003 to 0.3, and

\[
\nu\in\{0.25,0.5,0.75,1,1.5,2.5\}.
\]

All 108 shifts were positive. Log-log slopes over the six smallest bandwidths
were

\[
0.515,\ 0.999,\ 1.468,\ 1.817,\ 1.994,\ 2.000.
\]

These match theoretical powers \(0.5,1,1.5\), the logarithmically modified
threshold behavior, and the two quadratic regimes. At the smallest bandwidth,
exact-shift/leading-term ratios were

\[
1.021,\ 1.000,\ 0.958,\ 1.059,\ 0.997,\ 1.000.
\]

Quadrature refinement changed pseudo-decay by at most
\(4.31\times10^{-8}\) at order 64 and
\(5.12\times10^{-9}\) at order 128 relative to order 96. The two largest
bandwidths, 0.229 and 0.3, lie outside the sufficient Taylor neighborhood and
are correctly labeled stress points.

### 4.3 Directional results

The anisotropic track evaluated aspect ratios one and four, four smoothness
values, 14 bandwidths, and lag angles in five-degree increments. All 2,128
shifts were positive. At aspect four, the major/minor smooth-regime coefficient
ratios are 1.833 for \(\nu=1.5\) and 3.25 for \(\nu=2.5\), exactly as predicted
by the closed formula. The smallest-bandwidth relative directional-contrast
error is at most \(4.22\times10^{-6}\). Quadrature refinement changes the
contrast by less than \(7.7\times10^{-10}\) relatively.

At \(h=0.15\), the rough-field apparent range inflation is 13.46% on the major
support axis and 11.47% on the minor axis. For \(\nu=1.5\), the corresponding
values are 1.56% and 0.76%. These are illustrations, not universal finite-\(h\)
bounds.

### 4.4 Finite full-likelihood results

The support-only run used a \(19\times19\) raw lattice, rectangular output
support, four smoothness values, four bandwidths, 200 replicates, and 161
profiled decay candidates. Across core designs, corrected population targets
were within \(1.3\times10^{-7}\) of the true inverse range one.

At \(h=0.7\), naive population targets were

\[
0.152,\ 0.414,\ 0.573,\ 0.740
\]

for \(\nu=0.5,1,1.5,2.5\), while Monte Carlo means were

\[
0.151,\ 0.407,\ 0.570,\ 0.738.
\]

Corrected means were \(1.090,0.990,0.990,0.994\). All declared boundary fits
were retained. Boundary retention and coordinate jitter changed the naive
finite-design target while the corrected target remained one. These results
show correspondence to each model's finite-grid KL target; they do not prove a
full-likelihood phase theorem.

### 4.5 Final schema-1.1 high-dimensional track

The final track used:

- output grid sides \(4,6,8,10\), hence \(p=16,36,64,100\);
- \(\nu=0.5\) and \(1.5\);
- \(N=1,4,16,64\) independent fields;
- 200 nested Monte Carlo sequences per \((p,\nu)\);
- true \(v=\alpha=1\);
- bandwidth \(h=0.5\) and input spacing 0.25;
- a joint \(101\times161=16{,}261\) variance-decay library;
- corrected candidates \(vS_hR_\alpha S_h^{\mathsf T}\);
- naive candidates \(vR_\alpha\) at declared centers; and
- \(\delta=0.05\).

All 12,800 fits satisfied the exact deterministic ERM inequality. Every one of
the 64 model-design-sample-size cells had 100% empirical coverage of the
candidatewise simultaneous event. Across all trials,

\[
\max_{\theta}
\frac{|\widehat L_N(\theta)-L(\theta)|}{u_\theta}
\le 0.789.
\]

Each 95% statement applies separately to its fixed cell, not jointly to all 64
cells. The result confirms implementation of the inequality; 100% observed
coverage does not show that the bound is tight. Indeed, smooth-field covariance
conditions reached approximately \(1.67\times10^5\), and worst-candidate
envelopes can be very conservative.

The maximum per-coordinate population-objective gap between the finite library
and the continuously profiled oracle was

\[
2.6251\times10^{-5},
\]

below the predeclared \(5\times10^{-5}\) tolerance. All population oracles were
interior. Reapplying the stored smoothing operator to the raw illustration
reproduced averaged values with maximum difference zero.

At \(p=100,\nu=0.5\), the corrected finite-grid decay target is one while the
naive target is 0.320. At \(N=64\), the naive estimator's RMSE is 0.015 to its
own target but 0.681 to the physical value one; the corrected RMSE is 0.041.
At \(p=100,\nu=1.5\), the naive target is 0.594, its \(N=64\) RMSE is 0.015 to
that target but about 0.409 to physical truth, and the corrected RMSE is 0.024.

For fixed model, \(p\), and \(\nu\), empirical criterion-deviation slopes
against \(N\) range from \(-0.528\) to \(-0.454\), with median \(-0.505\).
This matches the leading \(N^{-1/2}\) behavior. Parameter RMSE summaries are
empirical because a curvature theorem has not been supplied.

### 4.6 Remaining experiment-theory gaps

The final evidence does not yet vary latent inverse range, kernel family,
nugget, or nonstationary misspecification in the schema-1.1 track. Smoothness
changes dependence strength, but it is not a complete dependence-strength
study. The truth is deliberately included in the corrected finite library,
making exact containment an oracle control rather than a deployable
grid-selection result.

The 200 replicates per cell are sufficient for the theorem-linked workshop
figures and Monte Carlo means, but any claim about calibrated 95% coverage
should include binomial Monte Carlo uncertainty. No simulation should be
described as a real-data application.

## 5. Related work and novelty map

The closest literature establishes most ingredients separately. The paper must
distinguish those ingredients from the surviving contribution.

| Area and nearest work | What is established | What may be new here |
|---|---|---|
| Change of support: [Gelfand, Zhu and Carlin (2001)](https://doi.org/10.1093/biostatistics/2.1.31), [Gotway and Young (2002)](https://doi.org/10.1198/016214502760047140), [Kyriakidis and Yoo (2005)](https://doi.org/10.1111/j.1538-4632.2005.00633.x), Chilès and Delfiner (2012) | Block covariance, areal-to-point prediction, and support integration | Explicit point-fit pseudo-range phase law and sign |
| Classical apparent-range regularization: [Clark (1977)](https://doi.org/10.1016/0098-3004(77)90010-3), [Bellehumeur and Legendre (1997)](https://doi.org/10.1111/j.1538-4632.1997.tb00961.x) | Support can enlarge a finite range of influence or a model-specific practical variogram range | Variance-refitted fixed-lag Matérn KL parameter, its smoothness-dependent order, and its coefficient; Matérn covariance has no finite range of influence |
| Closest Matérn block likelihood: [Fuentes (2007)](https://doi.org/10.1198/016214506000000852) | Correct approximate likelihood for irregular block averages, including Matérn examples | Parameter selected when those averages are incorrectly fitted as points |
| Modern aggregated GPs: [Tanaka et al. (2019)](https://proceedings.neurips.cc/paper/2019/hash/a941493eeea57ede8214fd77d41806bc-Abstract.html), [Chacón-Montalván et al. (2024)](https://doi.org/10.48550/arXiv.2403.08514), [Zheng et al. (2026)](https://doi.org/10.1016/j.spasta.2026.100998) | Observation operators, aggregation, and point-grid fusion | No identified all-\(\nu\) ignored-support coefficient in these works |
| Temporal aggregation: [Amemiya and Wu (1972)](https://doi.org/10.1080/01621459.1972.10481264), [Stram and Wei (1986)](https://doi.org/10.1111/j.1467-9892.1986.tb00495.x), [Folia and Rattray (2018)](https://doi.org/10.1007/s11222-017-9779-x) | Aggregation changes time-series and OU parameters | The one-dimensional exponential example is a check, not a novelty claim |
| Misspecified GP estimation: [Bachoc (2018)](https://doi.org/10.3150/16-BEJ906) | Increasing-domain KL pseudo-target framework | Closed Matérn support-specific direction, rate, and coefficient |
| Fixed-domain Matérn inference: [Zhang (2004)](https://doi.org/10.1198/016214504000000241), [Kaufman and Shaby (2013)](https://doi.org/10.1093/biomet/ass079) | Microergodicity and limits of separate range estimation | The current work avoids, rather than extends, these results |
| Convolution and anisotropy: [Clifford et al. (2006)](https://doi.org/10.1017/S0021859606005892), [Paciorek and Schervish (2006)](https://doi.org/10.1002/env.785) | Convolution/support can create anisotropic covariance | Explicit leading apparent-range directional contrast |
| Finite sampled Matérn grids: [Simons et al. (2026)](https://doi.org/10.1093/gji/ggag044) | Finite-grid likelihood, discretization, and edge effects | Ignored preprocessing-support pseudo-target phase law |
| Quadratic-form concentration: [Laurent and Massart (2000)](https://doi.org/10.1214/aos/1015957395), [Hsu, Kakade and Zhang (2012)](https://doi.org/10.1214/ECP.v17-2079) | Gaussian/sub-Gaussian quadratic tails | Direct theorem-matched certificate for the benchmark library only |
| Spatial benchmarks: [Heaton et al. (2019)](https://doi.org/10.1007/s13253-018-00348-w) | Auditable shared spatial simulation comparisons | Analytic support-specific targets and pass/fail gates |

The cleanest novelty claim that may survive review is:

> For a Matérn field observed through known compact symmetric local support, we
> derive an explicit all-smoothness, fixed-nonzero-lag expansion for the
> point-support Gaussian pair pseudo-parameter. The inverse-range displacement
> changes from \(h^{2\nu}\) to \(h^2\log(1/h)\) to \(h^2\) as \(\nu\) crosses
> one, has a universal range-inflation direction for sufficiently small
> support, and admits an explicit order-\(h^2\) directional contrast.

The dated screening protocol and source-by-source comparisons are recorded in
`docs/research/PRIOR_ART_SEARCH.md`. Targeted searches did not locate this exact
conjunction of rate transition, coefficient, sign, and directional contrast.
The search did locate direct qualitative predecessors for apparent-range
inflation, which are now cited and distinguished by estimand. That is not an
exhaustive proof of priority. A human reviewer should still
search older change-of-support monographs,
block-variogram expansions, image-resolution geostatistics, and nonstandard
Matérn parameter-estimation papers before the manuscript uses “first” or
“previously unknown.” The recommended wording is “we derive,” not “we are the
first.”

The following are explicitly **not novel**:

- \(S_hKS_h^{\mathsf T}\);
- support-aware Gaussian likelihood;
- generic KL projection under misspecification;
- temporal aggregation of OU/exponential covariance;
- the qualitative support/anisotropy connection;
- Gaussian quadratic-form concentration and a finite union bound; and
- the claim that more data can concentrate an estimator around a misspecified
  population target.

## 6. Candidate paper directions

### Direction 1 — primary: phase law plus theorem-linked SupportShift benchmark

**Main contribution.** The all-\(\nu\) pair pseudo-range phase theorem,
universal small-support sign, directional contrast, and a synthetic benchmark
that separates physical truth, population KL target, and finite-sample error.

**Exact assumptions.** Fixed \(d\); stationary mean-zero Gaussian Matérn field;
known \(\nu\); compact nonnegative symmetric kernel with \(T_k>0\); fixed
nonzero lag; small \(h\); known finite-design support matrix; and independent
replicated fields for the probability track.

**Required proofs.**

1. Exact support covariance and pair KL target.
2. Uniform fixed-lag Taylor expansion.
3. Matérn origin expansions in the rough, threshold, and smooth regimes.
4. Inverse-map expansion with explicit coefficients and remainders.
5. Bessel recurrence proving the sign.
6. Directional cancellation and coefficient.
7. Finite-library Gaussian concentration and ERM consequence.

Items 1–7 are internally complete. Independent review and a fuller pair
ergodic-consistency paragraph remain.

**Necessary synthetic experiments.**

- deterministic continuous phase oracle;
- anisotropic-support directional oracle;
- finite-grid corrected versus naive population KL and Monte Carlo fits;
- schema-1.1 replicated-field library track; and
- one raw latent/averaged field illustration.

All are complete. For a journal extension, add dependence-strength and
misspecification panels.

**Baselines.**

- physical unsmoothed Matérn truth;
- exact analytic pair target;
- naive centroid/point-support likelihood;
- support-aware \(S_hKS_h^{\mathsf T}\) likelihood;
- finite-grid continuous-profile oracle; and
- zero-support \(h=0\) identity control.

No Vecchia or SPDE baseline is needed unless the paper claims computational
scalability.

**Expected figures.**

1. Raw latent field and locally averaged output.
2. Phase shifts and leading terms on log scales.
3. Directional range inflation and coefficient convergence.
4. Finite-grid corrected/naive KL targets and Monte Carlo means.
5. Candidatewise likelihood envelope, target RMSE, and wrong-target
   concentration as \(Np\) increases.

**Expected tables.**

- assumptions and theorem scope;
- finite-design target and Monte Carlo summaries;
- schema-1.1 configurations and validation gates;
- nearest-literature comparison; and
- reproducibility hashes and failure counts.

**Principal technical risk.** An error in the Bessel expansion or \(\nu=1\)
remainder, or discovery of an old theorem giving the same pseudo-target law.

**Minimum publishable result.** The phase theorem, directional corollary,
deterministic quadrature, finite-grid containment control, and transparent
synthetic evidence. This is already close to a GeoSim full paper or a concise
statistics letter.

**Ambitious extension.** A full-grid spectral KL phase theorem with boundaries,
unknown nuisance parameters, and parameter curvature.

### Direction 2 — higher risk: a genuine high-dimensional lattice theorem

**Main contribution.** Prove structured likelihood and parameter rates for
support-aware and naive Matérn libraries as lattice size \(p\to\infty\),
explicitly relating spatial spectral geometry, support filtering, and the
effective-rank factor in the finite-library certificate.

**Exact assumptions.** Increasing periodic or rectangular lattice at fixed
spacing; \(N\) independent fields; known mean and \(\nu\); deterministic support
filter; compact variance-decay library; a positive nugget or another condition
that bounds spectra away from zero; and a quantitative KL margin.

**Required proofs.**

1. Toeplitz/circulant or Fourier representation of truth and candidates.
2. Uniform relative spectral bounds for the declared parameter/support class.
3. Effective-rank lower bounds and boundary approximation.
4. KL curvature or a finite-grid separation lower bound.
5. Parameter or exact-selection rate derived from the criterion certificate.
6. Optional information-theoretic lower bound showing when \(p\) cannot help.

**Necessary synthetic experiments.** FFT-scaled lattices from \(p=64\) to at
least \(p=4096\); several correlation ranges, nuggets, support bandwidths, and
kernel shapes; empirical spectral constants; parameter RMSE versus theorem
rates; and cases where relative spectral control fails.

**Baselines.** Corrected and naive finite-library ERM, continuous-profile MLE,
and a Fourier/Whittle implementation. Vecchia is relevant only if computation
is part of the claim.

**Expected figures/tables.** Effective rank and operator norm versus \(p\);
criterion and parameter errors versus \(Np\); spectral-density ratios; phase
diagram for valid/invalid relative-spectral assumptions; and lower-bound
comparison.

**Principal technical risk.** Spatial correlation can make spectral constants
deteriorate, especially without a nugget or under aggressive smoothing.
Dimension alone does not create \(p\) independent observations. Establishing a
uniform margin may be harder than the current phase theorem.

**Minimum publishable result.** A one-dimensional periodic-lattice theorem with
a fixed positive nugget and finite candidate library. That would be narrower
than a top-tier ML result but genuinely high-dimensional.

**Ambitious extension.** Multidimensional Toeplitz boundaries, continuous
parameter recovery, matching lower bounds, and adaptive support uncertainty.

This is a future project, not a result presently supported by the repository.

### Direction 3 — lowest-risk fallback: synthetic benchmark and computational note

**Main contribution.** Release SupportShift as a reproducible spatial-simulation
benchmark with analytic pair targets, exact finite-design KL targets,
anisotropic support, and explicit validation gates.

**Exact assumptions.** Fully synthetic Gaussian fields; known support operator;
declared Matérn convention; fixed benchmark candidate libraries and seeds; no
claim of general parameter consistency.

**Required theory.** The exact covariance identity, pair target, at least one
sound phase theorem or exact exponential benchmark, and the standard
finite-library certificate.

**Necessary experiments.** The four completed tracks plus a small
misspecification suite: latent anisotropy, wrong \(\nu\), wrong support
bandwidth, nonstationary range, and heavy-tailed replicated fields.

**Baselines.** Naive point support, exact support-aware likelihood, physical
truth, and the population oracle. Add external methods only if the benchmark
claims method ranking.

**Expected figures/tables.** Generator schematic, raw/averaged fields, target
decomposition, phase and anisotropy panels, criterion coverage, and a benchmark
data card.

**Principal technical risk.** Without the all-\(\nu\) theorem, reviewers may
view the benchmark as a polished demonstration of classical change of support.

**Minimum publishable result.** The current GeoSim-formatted benchmark paper,
provided proof review passes and artifacts are publicly archived.

**Ambitious extension.** A community benchmark with multiple support geometries,
non-Gaussian fields, missingness, boundary masks, competing approximate
likelihoods, and public leaderboards.

## 7. Recommended primary direction

Direction 1 has the strongest combination of novelty, feasibility, and
significance.

- It has a precise mathematical object and an explicit theorem.
- The proof does not depend on unavailable real data.
- Every central statement maps to a completed synthetic experiment.
- The probability track gives a legitimate high-dimensional lens without
  pretending spatial coordinates are independent.
- The current code and artifacts already enforce deterministic seeds, exact
  finite-library optimization, candidatewise diagnostics, and provenance.
- Its limitations can be stated cleanly.

Direction 2 would better match the repository name but requires new spectral
and margin theory. It should not delay the present paper. Direction 3 is a
credible salvage path if the broad theorem is weakened or a close predecessor
is found, but it is less mathematically distinctive.

The paper's contribution contract should be:

1. Claim novelty for the Matérn pair pseudo-target phase law, sign, and explicit
   directional contrast only.
2. Present the finite-library inequality as established probability machinery
   specialized to the benchmark.
3. Present full-grid targets as numerical population calculations.
4. Call \(p\) spatial vector dimension and \(N\) independent field replication.
5. Avoid “high-dimensional covariance estimation,” “minimax,” “adaptive,”
   “full-likelihood phase law,” and “real-data validation.”

## 8. Proof-completion plan

### Gate P0 — independent theorem review

An external spatial-statistics or applied-probability reviewer should verify:

- the noninteger Bessel series and sign of \(c_\nu\);
- the exact \(m_{2\nu}\), \(m_2\), and inverse-map constants;
- the \(\nu=1\) logarithmic coefficient and remainder;
- the separate \(\nu=2\) logarithmic fourth-order remainder;
- the recurrence reducing \(G_\nu\) to \(K_{\nu-2}\);
- the uniformity statement in \(\alpha\) and the lag annulus;
- positivity and uniqueness of the smoothed-pair correlation target; and
- the all-\(\nu\) directional difference.

Any substantive failure at the sign or threshold step is a stop condition for
the primary theorem claim.

### Gate P1 — turn the proof into reviewer-proof lemmas

The appendix should explicitly separate:

1. covariance transformation;
2. pair KL minimization;
3. fourth-order fixed-lag Taylor lemma;
4. origin expansions with dominated expectation;
5. inverse-function lemma and remainder propagation;
6. Bessel positivity lemma;
7. directional cancellation; and
8. Gaussian library concentration.

Each lemma should list exactly which compactness or moment assumption it uses.
The proof should state that bounded support, rather than a formal interchange
over unbounded \(D\), justifies the remainders.

### Gate P2 — complete the standard consistency statement or remove it

If pair-composite consistency remains in the journal manuscript, supply:

- the increasing rectangular Følner design;
- compact parameter set and interior unique target;
- an integrable quadratic envelope;
- uniform continuity/equicontinuity of the criterion;
- an ergodic theorem or mixing condition; and
- a clear statement that uncertainty needs long-run/Godambe covariance.

For GeoSim, this may remain a tightly worded remark because consistency is not
the headline. If these details are not written, do not promote the remark to a
numbered theorem.

### Gate P3 — protect the high-dimensional statement

Retain the exact candidate-specific radius. State the \(Np\) form only after the
relative-spectral assumption. Add one sentence showing how a margin would
convert excess KL to parameter distance, but do not assert that the Matérn
library satisfies such a margin uniformly. The simulation must continue to call
parameter RMSE empirical.

### Optional Gate P4 — future full-grid theory

Only after submission should work begin on a Fourier/Toeplitz full-grid
pseudo-target theorem, boundary remainder, or unknown-support inference. None is
required for the minimum paper.

## 9. Synthetic-data and empirical-validation plan

### 9.1 Completed datasets

The paper already contains six useful synthetic products.

1. **Continuous phase table.** Exact quadrature targets across \(h\) and
   \(\nu\), with leading coefficients and refinement diagnostics.
2. **Anisotropic support table.** Targets over lag angle, aspect ratio,
   bandwidth, and smoothness.
3. **Finite full-likelihood table.** Corrected and naive population KL targets
   and 200-replicate estimates, including boundary, jitter, and
   increasing-domain stress designs.
4. **Schema-1.1 replicated-field table.** Joint variance-decay finite-library
   fits over \(p,N,\nu\), with exact candidatewise theorem radii and ERM
   diagnostics.
5. **Raw illustration data.** Four latent \(23\times23\) fields and their exact
   \(10\times10\) supported outputs, sufficient to reconstruct the
   before/after figure.
6. **Threshold stress table.** Exact, one-term, and transition-aware targets on
   111 cells around \(\nu=1\), with quadrature refinements and predeclared sign
   and relative-error gates.

These are empirical experiments on synthetic data, not empirical application
data.

### 9.2 Minimum submission matrix

| Claim | Factors | Metric | Required gate |
|---|---|---|---|
| Phase exponent | six \(\nu\), 18 \(h\) | small-\(h\) slope and exact/leading ratio | refinement and sign pass |
| Threshold cancellation | 37 \(\nu\), three \(h\) | one-term ratio and two-term relative error | sign and 0.2% error gates pass |
| Directional coefficient | four \(\nu\), two aspects, angles, 14 \(h\) | contrast divided by theorem term | relative error below declared threshold |
| Corrected containment | \(p,\nu\), corrected family | population target and objective gap | truth exactly represented |
| Full-grid misspecification | \(h,\nu\), boundary/jitter | naive target and MC mean | all failures retained |
| Probability certificate | \(p,N,\nu\), both families | maximum candidatewise ratio and ERM excess | event checked over all 16,261 candidates |
| Wrong-target concentration | \(p,N,\nu\) | RMSE to physical truth and own KL target | both reported together |

### 9.3 Journal-extension experiments

If the paper is extended beyond GeoSim, add a deliberately small but diagnostic
suite:

- inverse range \(\alpha\in\{0.5,1,2\}\) to vary spatial dependence;
- bandwidth/effective-range ratios at three levels;
- compact box, radial Epanechnikov, and one misspecified kernel;
- correct and misspecified \(\nu\);
- nugget-to-signal ratios \(0,0.1,0.5\), with noise propagated through \(S_h\);
- latent geometric anisotropy separated from support anisotropy;
- one smoothly nonstationary range field;
- Gaussian versus standardized Student-\(t\) replicated fields; and
- support bandwidth perturbed by \(\pm10\%\) and \(\pm25\%\).

The goal is sensitivity, not an enormous factorial design. Use a fixed
fractional design, predeclare metrics, and report Monte Carlo standard errors.
No additional panel should be added unless it tests a theorem assumption or a
plausible failure.

### 9.4 Baselines and reporting

Report the physical parameter, pair target, finite-grid KL target, naive
estimate, and corrected estimate as distinct objects. At \(h=0\), corrected
and naive covariance, objective, and estimate must agree. For every run report:

- attempted, completed, failed, and boundary-hit fits;
- parameter bias/RMSE to both truth and the relevant KL target;
- normalized population excess risk;
- exact candidatewise envelope ratio;
- operator and Frobenius relative-spectrum norms;
- covariance condition number;
- objective-grid approximation gap;
- seed, configuration hash, code commit, environment, and elapsed time; and
- Monte Carlo uncertainty for reported means and coverage.

Do not silently jitter a covariance, discard an “unreasonable” estimate, or
replace a failed candidate.

### 9.5 Computational plan

The final schema-1.1 run took about 18 seconds on the Bouchet configuration and
does not justify GPU use. CPU jobs are sufficient at \(p\le100\). Future
extensions should batch many cells per Slurm job, use deterministic task seeds,
write atomic shards, and reduce only after all dependencies succeed. Submission
must remain below the authorized 180 jobs per hour under pi_jss233. GPU jobs
should be reserved for a future workload that demonstrably uses them.

## 10. Repository cleanup and reproducibility

### 10.1 Completed

- main is preserved and the research work is isolated on
  research/paper-audit;
- the Matérn convention, signed covariance, stacking, Bessel derivatives, and
  likelihood dimension have regression tests;
- explicit smoothing matrices and transformed covariance utilities exist;
- the SupportShift theory, finite-design, KL, experiment, reducer, and plotting
  modules are separated;
- deterministic phase and anisotropy oracles exist;
- the schema-1.1 candidate grid, seeds, validation gates, and metadata are
  frozen;
- final runs record clean commit and environment provenance;
- promoted fit-level inputs and metadata are versioned release assets, so the
  verifier runs from a clean clone of the frozen tag;
- paper source data and figures have a hash manifest;
- the release verifier checks statistical gates and every artifact hash; and
- both a technical manuscript and an ACM-format manuscript exist.

### 10.2 Release state and submission blockers

The internal artifact gates are complete: the GeoSim paper has nine main-text
pages plus one reference page and is within the 10-page main-text limit, the
technical manuscript is 22 pages, every page has
been rendered and inspected,
the maintained test suite passes, scoped Ruff is clean, and the release verifier
checks 12,800 rows, 64 coverage cells, 21 hashed paper artifacts, and 100
manuscript claims. The remaining
tasks are proof, priority, publication, and release chores rather than further
statistical computation:

1. Obtain an independent human review of the Bessel expansions, threshold
   remainder, sign, and directional corollary.
2. Have a human expert complete the residual manual priority check in older
   change-of-support and regularized-variogram monographs. The dated electronic
   screening and closest-source comparison are complete.
3. Confirm that one author can register for GeoSim and the main SIGSPATIAL
   conference and present in person.
4. Perform the final ACM submission-system validation and disclosure review.
5. Create an archival DOI and insert it in the camera-ready artifact. The scoped
   license, `CITATION.cff`, synthetic-data card, and submission checklist are
   complete.

Any change resulting from external proof review should receive a new patch
release and a repeated PDF/artifact verification, rather than mutating the
frozen tag.

Whole-repository formatting is not a release criterion because archived
notebooks and converted legacy scripts are intentionally unmaintained. Checks
must be scoped to the package, research scripts, tests, and paper generator.

### 10.3 Suggested release layout

\[
\begin{array}{ll}
\text{src/HighDimSpatial/smoothing\_bias/} & \text{theory and supported code},\\
\text{scripts/research/} & \text{reproducible entry points},\\
\text{configs/smoothing\_bias/} & \text{immutable pilot/final settings},\\
\text{tests/} & \text{unit, integration, regression gates},\\
\text{outputs/smoothing\_bias/} & \text{versioned external artifacts},\\
\text{paper/data, figures, tables/} & \text{compact manuscript inputs},\\
\text{docs/research/} & \text{audit, proof map, protocol, log}.
\end{array}
\]

Large legacy movement should be a later, separate cleanup. It should not be
mixed with scientific release commits.

## 11. Proposed paper

### 11.1 Working title

For a statistics journal:

> **Matérn Range Distortion under Local Averaging: A
> Smoothness-Dependent Phase Law**

For GeoSim:

> **SupportShift: A Theory-Linked Spatial Simulation Benchmark for Ignored
> Matérn Observation Support**

“High-dimensional” should not appear in the title. It may appear in a
subsection describing the finite-library probability track.

### 11.2 Outline

1. **Introduction.** Supported spatial measurements, the point-support shortcut,
   the quantitative gap, contributions, and explicit nonclaims.
2. **Model and exact pair target.** Matérn convention, local support, exact
   covariance, KL target, and identifiability.
3. **Smoothness-dependent phase law.** Fixed-lag expansion, origin regimes,
   sign, directional contrast, and scope.
4. **Finite-grid support-aware likelihood.** Rectangular observation operator,
   noise timing, numerical KL target, and boundary caveat.
5. **Finite-library probability certificate.** Candidatewise concentration,
   ERM excess KL, conditional \(Np\) form, and missing margin.
6. **Synthetic validation.** Phase, anisotropy, finite-grid, replicated-field,
   and raw-illustration tracks.
7. **Related work.** Change of support, block Matérn likelihood, aggregation,
   misspecification, anisotropy, probability tools, and benchmarks.
8. **Discussion.** What more data can and cannot repair, limitations, and
   extensions.
9. **Appendices.** Full proofs, exact exponential regression oracle,
   configurations, validation gates, hashes, and failure accounting.

For GeoSim, related work should be partly integrated into the introduction and
proof details moved to the allowed appendix pages. For a statistics letter, the
benchmark must be compressed to one main figure and supporting material.

### 11.3 Proposed abstract

Spatial measurements often represent pixels, footprints, or local averages but
are subsequently fitted as point observations. This changes the population
target of covariance estimation rather than merely adding finite-sample bias.
We study a stationary Matérn Gaussian field in \(\mathbb R^d\) observed through
a known compact symmetric averaging kernel of bandwidth \(h\). For a fixed
nonzero-lag Gaussian pair likelihood with known smoothness \(\nu\), we derive
the exact pseudo-variance and pseudo-inverse-range and their small-support
expansions. The inverse-range displacement has order \(h^{2\nu}\) for
\(0<\nu<1\), \(h^2\log(1/h)\) at \(\nu=1\), and \(h^2\) for \(\nu>1\).
Explicit coefficients show that ignored support underestimates inverse range,
and hence inflates inferred range, for every fixed \(\nu>0\) at sufficiently
small support. An all-smoothness directional contrast quantifies apparent
range anisotropy caused by elongated observation support. For \(N\) independent
\(p\)-dimensional Gaussian fields and a deterministic library of \(M\)
covariances, we also specialize a standard quadratic-form inequality to obtain
a simultaneous likelihood certificate and excess-KL bound; under relative
spectral control its radius is
\(\sqrt{\log(M)/(Np)}+\log(M)/(Np)\). Deterministic quadrature verifies the
phase and directional coefficients, while exact finite-grid simulations show
that a support-aware likelihood targets the physical covariance and a
point-support likelihood concentrates around a displaced KL target as
information grows. The covariance transformation and concentration inequality
are established tools; the contribution is the explicit Matérn
support-distortion law and its theorem-linked synthetic benchmark.

## 12. Venue assessment

Venue information below was checked on 2026-08-02. Journal submissions are
rolling; conference dates must be rechecked immediately before submission.

| Venue | Date | Why it fits | Standard and current blocker | Must be finished |
|---|---|---|---|---|
| [GeoSim 2026](https://geosim.org/2026/cfp/) | Full-paper deadline **2026-08-15**; notification 2026-09-15; workshop 2026-11-03 | Exact fit to parameterized spatial simulation, data generation, and validation | Peer-reviewed, single-blind archival ACM workshop; current risks are proof review and attendance | External theorem check and one author able to register for both the workshop and main conference and present |
| [GISTAM 2027](https://gistam.scitevents.org/ImportantDates.aspx) | Regular paper **2026-11-17**; position/regular second deadline 2026-12-21; conference 2027-04-20–22 | Accepts theoretical or practical GIS/spatial-analysis contributions | Lower-tier proceedings; double blind; the work must be framed as spatial data analysis rather than pure probability | Anonymized paper, broader support/misspecification experiments, public artifact |
| [GEOProcessing 2027](https://www.iaria.org/conferences2027/GEOProcessing27.html) | Submission **2026-12-28**; notification 2027-02-24; conference 2027-04-18–22 | Broad geospatial fundamentals, uncertainty, and processing scope | Realistic low-tier archival fallback; weaker statistical audience and impact | Complete benchmark paper, artifact, and conservative claims |
| [Statistics & Probability Letters](https://www.sciencedirect.com/journal/statistics-and-probability-letters) | Rolling | Best journal form for a concise theorem; official scope emphasizes short theoretical results and rapid publication | Six journal pages; current manuscript is too long and novelty needs independent confirmation | Proof review, severe compression, one decisive figure, supplementary proofs/artifacts |
| [Journal of Statistical Computation and Simulation](https://www.tandfonline.com/journals/gscs20) | Rolling | Synthetic benchmark and theorem-linked Monte Carlo fit its computation/simulation scope | Current experiments compare two model formulations rather than a broad computational method | Add compact sensitivity study, formal simulation design, MC uncertainty, reproducible software |
| [Spatial Statistics](https://shop.elsevier.com/journals/spatial-statistics/2211-6753) | Rolling | Strongest topical match for support, Matérn covariance, and spatial inference | Official scope says purely theoretical studies are only rarely accepted; no real application is available | Either add a substantial full-grid methodological theorem or later obtain a meaningful application |

### 12.1 Theoretical ML and top statistics

NeurIPS, ICML, ICLR, AISTATS, COLT, JASA Theory and Methods, JRSS-B, Biometrika,
and Annals of Statistics are not realistic targets for the present result. The
finite-library inequality is standard, the procedure is not a new scalable ML
algorithm, and there is no minimax lower bound or continuous-parameter
high-dimensional rate. A future workshop with an explicitly open call on
Gaussian processes, uncertainty, or scientific ML could provide feedback, but
no unverified deadline should drive the current schedule.

### 12.2 Recommended venue order

For fastest publication:

1. Submit the full benchmark paper to GeoSim 2026 if the proof and attendance
   gates pass by 2026-08-09.
2. If GeoSim is missed or unsuitable, choose either:
   - Statistics & Probability Letters for a compressed theorem-first paper; or
   - GISTAM 2027 for the full synthetic benchmark.
3. Use GEOProcessing 2027 only as the low-tier archival fallback.
4. Treat Journal of Statistical Computation and Simulation as the
   simulation-heavy journal route after sensitivity extensions.
5. Do not target Spatial Statistics without a stronger full-grid theorem or
   future application.

An archival GeoSim full paper cannot be submitted simultaneously to a journal.
A later journal version must be substantially extended, cite the workshop
paper, and follow both publishers' prior-publication policies.

## 13. Submission roadmap

### Immediate GeoSim schedule

- **Aug 2–4:** freeze theorem statements and send proof appendix to the external
  reviewer.
- **Aug 3–6:** complete the targeted prior-art search and bibliography audit.
- **Aug 4–7:** fit the current text to ACM sigconf limits; keep full-paper
  content within ten pages excluding references and no more than two appendix
  pages after references.
- **Aug 6–9:** regenerate every figure/table from the clean release candidate,
  run all tests, and visually inspect the PDF.
- **Aug 9:** go/no-go gate. Stop GeoSim work if proof review or presentation
  logistics are unresolved.
- **Aug 10–12:** author review, disclosure, artifact data card, and final claim
  cross-check.
- **Aug 12:** internal freeze.
- **Aug 13–14:** upload, metadata, PDF, and checksum buffer.
- **Aug 15:** official deadline.

### If GeoSim is not ready

Do not submit a rushed or overclaimed version. Use August and September for the
external proof revision and the small dependence/misspecification suite.
Prepare the six-page theorem letter in parallel with a longer benchmark paper,
then choose one archival route. A GISTAM regular-paper freeze by 2026-11-07
leaves ten days for double-blind formatting and upload. GEOProcessing provides
an additional December fallback.

## 14. Candid final verdict

### Continue

Continue the SupportShift phase-law project. The theorem is narrow but
substantive, the proof chain is coherent, and the final synthetic evidence is
unusually well aligned with the mathematical statements. A specialized
workshop or modest statistics journal is realistic.

### Narrow permanently

Remove or subordinate all of the following:

- the original flexible multivariate Matérn estimator;
- spatial-transcriptomics conclusions;
- a claim of new change-of-support methodology;
- high-dimensional covariance-estimation language;
- fixed-domain separate variance/range consistency;
- arbitrary full-grid phase-law language;
- data-driven confidence-certificate language;
- minimax or adaptive claims; and
- any implication that synthetic experiments are a real application.

The honest high-dimensional-probability message is:

> With independent replicated fields, finite-library likelihood noise can
> contract with spectral effective information while spatial dependence inside
> each field remains unrestricted. That contraction can make a
> support-misspecified estimator increasingly precise about the wrong KL target.

### Abandon conditions

Abandon the primary theorem paper if:

1. independent review finds the all-\(\nu\) sign or \(\nu=1\) expansion
   incorrect and no comparably strong weaker theorem remains;
2. a close predecessor already states the same pseudo-target phase law and
   directional coefficient; or
3. the final artifacts cannot be reproduced from a clean release commit.

If condition 1 occurs but the exact exponential result and benchmark remain
correct, the work may still be salvaged as a low-tier simulation workshop note,
but it must be explicitly reframed as a benchmark around classical
change-of-support behavior.

### Bottom line

The project is not the paper originally envisioned, and it should not be pushed
toward a top-tier venue by adding breadth. Its strongest form is a focused
Matérn pseudo-parameter theorem with a rigorous synthetic benchmark and a
carefully delimited high-dimensional-probability section. On those terms, it is
worth finishing now.
