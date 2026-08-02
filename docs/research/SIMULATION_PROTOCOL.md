# Simulation and empirical-validation protocol

## Purpose

The simulations must test specific theorem statements, not merely show that an
optimizer runs. Pilot results are exploratory. Final grids, metrics, seeds, and
exclusion rules must be frozen before large runs.

## Claim-to-experiment matrix

| Claim | Required experiment | Failure criterion |
|---|---|---|
| Naive smoothing fit has target shift of order \(h^q\) | Increasing-domain bias versus bandwidth on log--log axes; overlay theorem coefficient | Slope/coefficient does not converge or changes sign unexplained |
| Corrected likelihood targets \(\theta_0\) | Bias/RMSE/coverage for exact \(SKS^\top\) fit as domain grows | Persistent bias or undercoverage beyond Monte Carlo error |
| Boundary remainder is negligible at stated rate | Interior-only, normalized-boundary, and exact irregular-boundary comparisons | Boundary effect remains first order in claimed regime |
| Composite estimator has Godambe CLT | Quantile/coverage study with sandwich variance and spatial blocks | Standardized estimates non-Gaussian or coverage outside tolerance |
| Multivariate cross parameters are affected | Signed cross-correlation and cross-decay recovery across \(p\) and smoothing | Proposed coefficient is zero/unstable or model cannot represent sign pattern |
| Computation is useful | Matched-accuracy runtime/memory against exact and established approximations | No material end-to-end advantage |
| Biological conclusion replicates | Sample-level train/test and independent slide/dataset | Effect disappears across samples or is baseline-dependent |

## Data-generating processes

### Univariate core

Use

\[
Y(s)=m(s)+Z(s)+\epsilon(s),\qquad
Z\sim GP(0,C_{\theta_0}),\quad
\epsilon\stackrel{iid}{\sim}N(0,\tau_0^2),
\]

where \(C_{\theta_0}\) uses the AGS decay-scale convention. The theorem-matching
design begins with \(m=0\), a known nugget, a regular lattice, and a deterministic
symmetric smoother. Misspecification panels add unknown mean, anisotropy,
nonstationarity, heavy tails, and count-like observation noise one at a time.

### Multivariate core

Use an identified globally valid multivariate Matérn with fixed
\(p\in\{2,3,5,10\}\). Include all-positive, mixed-sign, sparse, and weak
cross-dependence. The original scalar \(\rho_VW_iW_j\) parameterization cannot
represent a \((-,-,+)\) sign pattern and must not be used as the final DGP or fit.

### Domain regimes

1. **Increasing domain:** fixed sampling density, domain side length grows as
   \(n^{1/d}\); this is the primary consistency/CLT regime.
2. **Fixed domain:** domain remains bounded and resolution increases; report
   microergodic combinations and prediction, not separate range/variance
   consistency.
3. **Independent replicates:** fixed design with \(R\) independent fields; useful
   for application-like multi-slide data and finite-design identifiability.

## Factors and minimum final grid

| Factor | Levels |
|---|---|
| \(n\), increasing domain | 100, 225, 400, 900, 1,600 (reduce only after power/runtime calculation) |
| \(n\), fixed domain | 100, 400, 1,600 |
| Matérn \(\nu\) | 0.5, 1.0, 1.5, 2.5 |
| decay \(\alpha\) relative to domain | short, medium, long effective range |
| marginal SD | 0.5, 1, 2 |
| nugget-to-signal ratio | 0, 0.05, 0.25, 1 |
| bandwidth/effective range | 0, 0.05, 0.1, 0.25, 0.5 |
| bandwidth regime | fixed; \(h_n\downarrow0\) at two theorem-compatible rates; one incompatible rate |
| boundary | periodic/interior oracle, rectangle, concave tissue mask, holes |
| design | lattice, jittered lattice, uniform irregular, clustered |
| \(p\) | 1, 2, 3, 5, 10 for fixed-\(p\) paper |
| cross strength | 0, weak, moderate, near validity boundary; mixed signs |
| misspecification | anisotropy, trend, nonstationarity, Student-\(t\), zero inflation/count transform |

Use a fractional factorial pilot to identify redundant combinations. The final
theorem panels must retain the full factors needed to test rates. Target at least
500 replicates for bias/coverage cells; compute and report Monte Carlo standard
errors, and increase replicates when a coverage difference is comparable to its
Monte Carlo error.

## Estimators and baselines

1. Oracle exact likelihood on unsmoothed observations.
2. Naive point-level likelihood on smoothed observations.
3. Exact observation-operator likelihood with
   \(S(K_\theta+\tau^2I)S^\top\).
4. Correct local-block composite likelihood with sandwich uncertainty.
5. Same block scheme without smoothing correction, isolating the correction.
6. Vecchia/nearest-neighbor approximation with the same observation model.
7. SPDE/change-of-support implementation where feasible.
8. Current distance-quantized implementation, labelled legacy and excluded if it
   needs design-dependent eigenvalue shifting.
9. For SRT: COVET, Smoothie, SpaceX, spMOCA, CellCharter or other task-appropriate
   current baselines; do not force every method into an unrelated task.

## Metrics

- parameter bias, RMSE, median absolute error, and failure rate;
- microergodic-parameter error under fixed domain;
- empirical coverage and interval length using correct Godambe or likelihood
  uncertainty;
- predictive log score, RMSE, and calibrated interval coverage on spatial blocks;
- covariance/spectral-matrix operator and Frobenius error;
- cross-sign/support recovery when it is an estimand;
- minimum eigenvalue and Cholesky failure count;
- objective, score, and gradient error relative to exact calculations;
- wall time, peak memory, special-function time, factorization time, and total
  end-to-end time;
- biological replicate stability and sample-level held-out performance.

Always report the number of attempted, converged, failed, and excluded runs.
Never remove “unreasonable” estimates without a predeclared rule and a sensitivity
analysis treating failure as an outcome.

## Expected figures

1. Theorem figure: \(\widehat\theta_{\rm naive}-\theta_0\) versus \(h\), with
   predicted slope and coefficient.
2. Pseudo-target surface/score vector for one analytically tractable Matérn case.
3. Corrected versus naive bias and coverage across \(n\) and bandwidth.
4. Fixed-domain plot for microergodic versus separate parameter estimates.
5. Boundary/irregular-design effect map.
6. Multivariate signed cross-covariance recovery.
7. Error--runtime and memory--\(n\) frontiers.
8. Application map with uncertainty and sample-level replication, with text,
   labels, backgrounds, and legends assigned explicit readable colors.

## Expected tables

1. DGPs and assumptions matched to theorem clauses.
2. Bias/RMSE/coverage with Monte Carlo standard errors.
3. Optimization failures and covariance-validity diagnostics.
4. Runtime/memory at matched statistical accuracy.
5. Misspecification sensitivity.
6. Biological dataset/sample metadata and independent validation.

## Spatial transcriptomics validation

The existing single-puck, 22-gene analysis is insufficient. The final analysis
requires:

- multiple independent animals/slides or a second public dataset;
- a scientific estimand chosen before fitting (prediction, region contrast,
  reproducible network edge, or niche association);
- cell-type/tissue-region covariates or residualization fitted on training data;
- train-only normalization and parameter selection;
- sample-level or geographically separated holdout;
- explicit treatment of count/zero-inflated measurement noise;
- sensitivity to bandwidth, anisotropy, nonstationarity, and gene filtering;
- appropriate multiplicity control for gene pairs/regions;
- biological interpretation reviewed by a domain collaborator.

The archived processed response has unit SD but nonzero means, 30.6%--99.88% zeros
across selected genes, and maxima roughly 14--53. Extremely rare genes include
examples with only tens of nonzero observations among 15,907 locations. A
zero-mean homoskedastic Gaussian field is not a credible raw observation model.

## Reproducibility contract

- Root seed family: `20260802`; replicate seed is a documented deterministic
  function of configuration and replicate index.
- Every output begins with run metadata: timestamp, code commit, dirty-state hash,
  environment versions, device, dtype, configuration, and input checksums.
- Final configurations are immutable YAML/TOML files; command-line overrides are
  recorded.
- Metrics are append-only JSONL or Parquet; figures are regenerated from those
  tables, never edited manually.
- Pilot and final outputs use separate directories.
- Checkpoints contain data/config identity and RNG state.
- Unit tests cover stacking, signs, gradients, transformed covariance, and
  likelihood dimensions; integration tests cover every documented command.

## Existing executable pilot

The branch contains a small logged comparison:

```bash
python -m scripts.research.run_smoothing_bias_study \
  --n-locations 40 \
  --bandwidth 0.5 \
  --replicates 10 \
  --output outputs/smoothing_bias/pilot.jsonl
```

This script varies only the univariate decay parameter on a grid. It demonstrates
the corrected and naive objectives and provides infrastructure; it is not a final
estimator, does not establish novelty, and must not be cited as paper evidence.
