# SupportShift synthetic benchmark

Promoted schema: **1.1**. Version 1.1 adds the resolution-certified
\(161\times101\) library and exact candidatewise simultaneous-coverage fields;
earlier 1.0 files are pre-release artifacts.

## Purpose

SupportShift is a theorem-linked synthetic benchmark for covariance inference
when each recorded spatial value is a local average but the fitted model treats
it as a point observation. It is not a substitute for a real-data application.
Its purpose is to separate three quantities that are often conflated:

1. the physical Matérn parameter used to generate the latent field;
2. the population Kullback--Leibler target of the fitted covariance family; and
3. the random finite-sample estimate.

The benchmark is successful only when every plotted curve has a declared
mathematical target and every theorem-facing numerical approximation has an
accuracy audit.

## Statistical experiment

For a deterministic latent grid (X_{\rm in}) and row-stochastic support
operator (S_h\), generate independent replicated fields

\[
  X_i\sim N_p(0,\Sigma_0),\qquad
  \Sigma_0=S_hK_{\nu,\alpha_0,v_0}(X_{\rm in})S_h^\top,
  \qquad i=1,\ldots,N.
\]

Coordinates within an (X_i) are spatially dependent. Independence is assumed
only across the (N) replicated fields. The primary design is
increasing-domain: lattice spacing and the local support rule remain fixed while
the output side length, and hence (p), grows.

Two deterministic finite covariance libraries are fitted:

\[
  \Sigma^{\rm SA}_{v,\alpha}=vS_hK_{\nu,\alpha,1}S_h^\top,
  \qquad
  \Sigma^{\rm NV}_{v,\alpha}=vK_{\nu,\alpha,1}(X_{\rm out}).
\]

The support-aware library contains the truth when its variance and decay grids
contain ((v_0,\alpha_0)). The naive library generally has a nonzero KL
approximation error. Both grids are fixed before seeing the simulated fields;
this is necessary for the finite-library concentration theorem.
For benchmark validation the Cartesian grid is deliberately anchored at the
known generating pair \((v_0,\alpha_0)\). This oracle-informed control verifies
model containment; it is not proposed as a deployable data-analysis recipe.

## Four benchmark tracks

### A. Continuous phase oracle

Deterministic product Gauss--Legendre quadrature computes the exact pairwise
pseudo-target over bandwidth. It tests the three predicted regimes

\[
  h^{2\nu},\qquad h^2\log(1/h),\qquad h^2
\]

for (0<\nu<1), (\nu=1), and (\nu>1), respectively. The plotted leading
terms include their theorem coefficients. Orders 64, 96, and 128 provide a
quadrature audit.

A separate 111-cell threshold audit uses 37 smoothness values from 0.55 to
1.45 and bandwidths 0.01, 0.02, and 0.05. It compares the exact target with
both the pointwise one-term law and the cancellation-aware two-term target.
The audit is deliberately labeled finite-grid evidence: it does not promote
the fixed-smoothness theorem into a joint uniform result.

### B. Finite-grid support likelihood

The existing finite study compares support-aware and naive full Gaussian
likelihoods under interior, boundary, and fixed irregular designs. This track
shows that the continuous pair theorem is a local oracle rather than an
unproved universal statement about every full-grid likelihood.

### C. High-dimensional replicated fields

The output dimension \(p\), replicate count \(N\), and Matérn smoothness are
varied while support bandwidth is fixed at 0.5. Tracks A and B vary bandwidth.
For every library and design, the benchmark records:

- the physical parameter;
- the exact finite-design KL oracle;
- the empirical joint variance--decay minimizer;
- total error to the physical parameter and stochastic error to the KL oracle;
- the maximum empirical-to-population criterion deviation;
- the exact finite-library Gaussian concentration radius; and
- whether every candidate satisfies its own simultaneous certificate and
  whether the ERM excess-risk consequence holds. The output separately records
  the weaker worst-envelope comparison
  \(\max_\theta|\widehat L-L|\leq\max_\theta u_\theta\).

This track tests the high-dimensional probability statement directly. Under
uniform relative spectral control, criterion noise is of order

\[
  \sqrt{\frac{\log M}{Np}}+\frac{\log M}{Np},
\]

where \(M\) is the fixed library size. The full preset enumerates a
\(161\times101=16{,}261\)-candidate Cartesian decay--variance library. This
resolution was selected in a pre-run audit: across all eight \((p,\nu)\)
designs, the normalized population NLL at the discrete oracle is within
\(2.7\times10^{-5}\) of the continuously profiled oracle. The run aborts if
the gap exceeds the predeclared \(5\times10^{-5}\) tolerance.

The key misspecification phenomenon is that stochastic error can shrink while
the naive KL target remains separated
from the physical parameter: more information can make the wrong target more
precise.

### D. Directional support

A fixed-trace linear transform of the product Epanechnikov kernel changes its
principal support aspect ratio. The experiment varies lag angle and Matérn
smoothness. It tests the corollary

\[
  \Delta_{e_1}(h)-\Delta_{e_2}(h)
  =D_\nu(A;e_1,e_2)h^2+o(h^2),
\]

where \(\Delta_e(h)=\alpha-\alpha_h^\dagger(e)\). The untransformed product
kernel has equal coordinate axes but is not fully rotation-invariant; figures
and captions must not label it an isotropic support.

## Acceptance gates

The final benchmark run is accepted only if all of the following hold:

- every support-aware population oracle whose truth lies on the grid selects
  the true candidate;
- every theorem-facing covariance is positive definite without adaptive jitter;
- all expected rows and unique configuration--replicate--model keys are present;
- the zero-support control has no pseudo-target displacement;
- continuous phase coefficients agree with the smallest-bandwidth oracle to the
  predeclared tolerance;
- every transition-audit shift has the predicted sign, and the two-term shift
  and variance-loss relative errors stay below the predeclared 0.2% tolerance;
- directional contrasts agree with their (h^2) coefficient and remain stable
  under quadrature refinement;
- the finite-library population objective is within its predeclared tolerance
  of the continuously profiled population objective;
- empirical simultaneous-certificate coverage is at least its nominal level
  within a predeclared Monte Carlo tolerance, or any shortfall is reported rather
  than hidden;
- estimates at parameter-grid endpoints are retained and counted;
- figures are regenerated only from validated aggregate tables; and
- final metadata identify a clean immutable Git commit, seed, environment,
  command, and output checksums.

## Synthetic data products

The artifact contains both aggregate benchmark tables and a small raw example.
The raw table has 2,516 rows: 2,116 `latent_input` records and 400
`averaged_output` records. Rows are keyed by configuration, replicate,
`field_stage`, and `location_index`; they store coordinates, field value,
support bandwidth, smoothness, variance, decay, and seed. It is intended for
inspection and method development, not as an additional independent experiment.

Aggregate tables contain one row per Monte Carlo fit or one row per validated
configuration summary. Raw latent grids from large runs are not duplicated when
the seed and exact covariance generator reproduce them.

## Claim-to-result map

| Paper claim | Benchmark evidence | What would falsify or weaken it |
|---|---|---|
| Three-regime pairwise phase law | Track A exact oracle and coefficient ratio | wrong slope, sign, or nonconvergent quadrature |
| Two-term threshold approximation | Track A fine-grid transition audit | failed sign gate, error above 0.2%, or quadrature instability |
| Support anisotropy creates directional range inflation | Track D angular oracle and contrast ratio | wrong contrast sign or coefficient |
| Support-aware finite library contains the physical covariance | Tracks B and C population KL oracle | corrected oracle misses an on-grid truth |
| Likelihood concentration depends on \(N,p,M\) and relative spectra | Track C varies \(N,p\) at fixed \(M\), reports matrix geometry, and compares deviations with the theorem radius | certificate undercoverage or uncontrolled spectra |
| Naive precision can increase around a wrong target | Track C stochastic error to oracle versus total error to truth | pseudo-target gap vanishes or stochastic error does not shrink |

## Nonclaims

SupportShift does not establish separate Matérn range and variance consistency
under fixed-domain asymptotics, does not justify data-dependent candidate grids,
and does not replace biological or environmental validation. The
high-dimensional concentration proposition is standard Gaussian
quadratic-form machinery and is used to certify the benchmark; it is not claimed
as the paper's central novelty.
