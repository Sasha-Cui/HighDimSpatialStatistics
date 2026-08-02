# Proposed paper

## Working title

**What Spatial Smoothing Makes a Matérn Model Estimate: Target Shift,
Information Loss, and Corrected Multivariate Inference**

Avoid “high-dimensional” in the title unless a genuine growing-\(p\) theorem is
later added.

## Draft abstract

Spatial measurements are often smoothed before covariance parameters are
estimated, after which the smoothed values are treated as point observations.
Although linear smoothing is a standard change-of-support operation, its effect
on the inferential target of Matérn likelihood estimation is not usually made
explicit. We study a stationary Matérn field observed through a known local
smoothing operator. The exact observation covariance is
\(S(K_\theta+\tau^2I)S^\top\), including correlated transformed measurement
noise, whereas the common point-level analysis uses a misspecified Matérn
covariance at smoothing centers. Under increasing-domain sampling, we characterize
the latter estimator's Kullback--Leibler pseudo-target and [derive an explicit
bandwidth-dependent target-shift expansion / prove strict displacement for the
specified Matérn and smoother classes]. We construct a local-block
smoothing-aware composite estimator and establish consistency and asymptotic
normality with Godambe uncertainty that accounts for overlapping resolutions.
Simulations examine bandwidth, smoothness, range, noise, boundaries, irregular
designs, and fixed-dimensional multivariate cross-covariances, and compare the
predicted bias with finite-sample behavior. [A replicated spatial-transcriptomics
application evaluates whether correcting the observation support changes
scientific conclusions.] The results distinguish the standard covariance
transformation from the new target-shift analysis and show when smoothing bias is
negligible, first-order, or dominant relative to sampling error.

Bracketed text is a completion condition, not a current claim.

## Paper contribution contract

The introduction may claim only contributions that survive all three gates:

1. explicit Matérn target shift/bias expansion;
2. valid corrected estimator and uncertainty under a declared regime;
3. theorem-matched simulations and independently replicated application.

It must say explicitly that covariance propagation under linear observation
operators, generic KL pseudo-targeting, and multivariate Matérn validity are prior
work.

## Outline

### 1. Introduction

- Scientific practice: spatial smoothing followed by covariance fitting.
- Why point-observation likelihood changes the estimand.
- Distinguish change-of-support correction (known) from explicit target shift
  (proposed contribution).
- Contributions and nonclaims.

### 2. Observation model and motivating failure

- Latent Matérn field, mean, nugget, and AGS multivariate convention.
- Location-major stacking.
- Smoothing operator and exact transformed covariance.
- One analytic exponential/Matérn-\(1/2\) example; for Epanechnikov smoothing,
  the marginal variance has first-order decrease
  \(1-(18/35)\alpha h+O(h^2)\).

### 3. Naive pseudo-target

- Finite-\(n\) expected Gaussian criterion.
- KL projection and expected-score diagnostic.
- Interior spectral form.
- Small-bandwidth expansion and strict nonzero coefficient.
- Fixed versus vanishing bandwidth phase regimes.

### 4. Boundaries, irregular designs, and multivariate fields

- Row-normalized boundary kernels and induced nonstationarity.
- Boundary remainder/rate.
- Spectral-matrix formulation for fixed \(p\).
- Signed cross-covariance and identified valid parameterization.

### 5. Corrected local-block composite inference

- Exact block covariance and overlapping resolutions.
- Consistency theorem under increasing domain.
- Godambe CLT and analytic Gaussian score covariance.
- Optional optimal block/resolution weighting.

### 6. Fixed-domain limitations

- Gaussian-measure equivalence and microergodic targets.
- Data-processing proposition: smoothing cannot recover nonidentifiable separate
  variance/range parameters.
- Prediction versus parameter inference.

### 7. Simulation study

- Preregistered claim-to-panel design.
- Bias slopes, coverage, boundary/irregularity, multivariate signs.
- Failure and convergence reporting.
- Runtime only if implementation is competitive.

### 8. Replicated application

- Scientific estimand and sample metadata.
- Mean/noise model and train-only preprocessing.
- Naive/corrected/current-baseline comparison.
- Sample-level validation and sensitivity.

### 9. Discussion

- When smoothing bias is negligible or dominant.
- What is specific to Matérn versus generic misspecification.
- Limitations: fixed \(p\), deterministic smoother, Gaussian residual model,
  bandwidth choice, and boundary complexity.

### Appendices

- AGS validity and parameter conversion.
- All proofs.
- Gaussian quadratic-form/Godambe calculations.
- Additional simulations and failed-fit table.
- Reproducibility manifest.

## Minimum figure sequence

1. Schematic of latent observations, smoother, naive model, and corrected model.
2. Analytical target-shift curve with simulated convergence.
3. Bias/coverage phase diagram in \((n,h,\nu)\).
4. Boundary/irregular-design effects.
5. Multivariate cross-parameter recovery.
6. Application replication/holdout result.

All visual artifacts must assign explicit readable foreground/background colors to
every text-bearing element, legend, panel, table, badge, and annotation and pass a
rendered contrast/clipping review.
