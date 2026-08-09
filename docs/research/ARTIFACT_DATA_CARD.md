# SupportShift artifact and synthetic-data card

## Summary

SupportShift is a fully synthetic, theorem-linked benchmark for a specific
spatial-model misspecification: observations are local averages of a stationary
Matérn Gaussian field, while the naive fitted covariance treats those averages
as point observations. The release separates physical parameters, population
KL targets, finite-library estimates, and nonasymptotic likelihood certificates.

No human-subject, proprietary, environmental, or empirical data are included.
The benchmark is not calibrated to a particular real population and should not
be used to make scientific claims about one.

## Versioned release

- Repository: <https://github.com/Sasha-Cui/HighDimSpatialStatistics>
- Artifact release: `supportshift-geosim-v1.2.1`
- Artifact-manifest schema: `1.3`
- Replicated-field benchmark schema: `1.1`
- Promoted high-dimensional run: Slurm job `21081491`
- Authorized allocation: `pi_jss233`
- Promoted generation commit recorded in metadata: `d5207fb`
- Independent empirical re-audit: Slurm job `21749885`, clean commit `b6c8ee2`
- Release verifier: `scripts/research/verify_supportshift_release.py`

The promoted numerical inputs are immutable even though later release commits
add documentation and manuscript corrections. Their metadata retain the exact
generation commit, environment, arguments, validation gates, and hashes.

## Scientific factors and targets

| Track | Controlled factors | Oracle or population target | Main reported diagnostic |
|---|---|---|---|
| Continuous phase | bandwidth, Matérn smoothness | exact two-point pseudo-decay from deterministic quadrature | rate and coefficient ratio |
| Smoothness-one transition audit | bandwidth and a fine smoothness grid around one | exact pair target and cancellation-aware two-term target | one-term failure and two-term relative error |
| Dimension--kernel robustness audit | dimension, compact product-kernel family, bandwidth, smoothness | exact pair target and theorem coefficient | sign, coefficient ratio, and quadrature stability |
| Directional support | bandwidth, smoothness, lag angle, support aspect ratio | exact directional two-point pseudo-decay | apparent-range contrast |
| Finite grid | bandwidth, smoothness, boundary/irregular/domain design, fitted support model | exact finite-design Gaussian KL minimizer | bias and Monte Carlo error relative to target |
| Replicated high dimension | \(p\), \(N\), smoothness, support-aware/naive model | exact finite-library population minimizer | criterion certificate, excess risk, and target-specific RMSE |

The principal scaling variables in the promoted high-dimensional track are
\(p\in\{16,36,64,100\}\) and \(N\in\{1,4,16,64\}\), with two smoothness
values, two fitted model families, and 200 trials per cell. The fixed joint
variance--decay library contains 16,261 candidates.

## Files and schemas

### Promoted source data

- `outputs/smoothing_bias/phase_oracle_d2_v2.csv`: 108 deterministic
  continuous-oracle rows used by the phase-law figure and coefficient checks.
- `outputs/smoothing_bias/phase_oracle_d2_v2.metadata.json`: clean-commit
  provenance, environment, resolved factors, and order-64/order-128 quadrature
  refinements for the phase oracle.
- `outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.csv`:
  12,800 fit rows and 58 columns. It includes factor values, seeds, estimates,
  physical and grid targets, criterion deviations, certificate radii, ERM
  checks, and spectral diagnostics.
- `outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.metadata.json`:
  complete factor grids, environment, provenance, hashes, and predeclared
  validation gates for the preceding CSV.
- `outputs/smoothing_bias/supportshift_raw_final_v2_20260803.csv`:
  2,516 rows for one latent/smoothed field illustration, with coordinates,
  stage labels, generating parameters, and seed.
- `outputs/smoothing_bias/supportshift_anisotropic_final_20260803.csv`:
  2,128 rows over aspect ratio, angle, smoothness, and bandwidth, including
  theorem coefficients and quadrature comparisons.
- `outputs/smoothing_bias/supportshift_anisotropic_final_20260803.metadata.json`:
  generation arguments, environment, quadrature-refinement diagnostics, and
  validation gates.
- `outputs/smoothing_bias/supportshift_transition_stress_20260804.csv`: 111
  deterministic cells over 37 smoothness values and three bandwidths, with
  exact, one-term, and transition-aware targets and errors.
- `outputs/smoothing_bias/supportshift_transition_stress_20260804.metadata.json`:
  clean-commit provenance, environment, SHA-256, quadrature refinement, and
  predeclared sign and approximation-error gates.
- `outputs/smoothing_bias/supportshift_dimension_kernel_robustness_20260804.csv`:
  72 deterministic cells in dimensions one through three for product
  Epanechnikov and product uniform kernels.
- `outputs/smoothing_bias/supportshift_dimension_kernel_robustness_20260804.metadata.json`:
  clean-commit provenance, SHA-256, factor grid, quadrature refinement, and
  predeclared sign and coefficient-accuracy gates.
- `outputs/smoothing_bias/support_only_final_20260802_v2/results.csv`: all
  8,400 finite-grid fit records, including task/configuration hashes, model and
  replicate keys, deterministic seeds, estimates, targets, errors, objectives,
  boundary-fit flags, and clean generation commit.
- `outputs/smoothing_bias/support_only_final_20260802_v2/audit.json`: reducer
  certificate for all 21 tasks, with no missing or invalid shard.
- `configs/smoothing_bias/support_only_20260802.json`: immutable 21-design
  manifest and root seed used to validate the preceding records.

### Compact paper data

- `paper/data/phase_oracle_d2.csv`: 108 continuous-oracle rows.
- `paper/data/finite_summary.csv`: 42 model/configuration summaries from 8,400
  finite-grid fits.
- `paper/data/anisotropic_phase.csv`: compact directional plot data.
- `paper/data/supportshift_highdim_summary.csv`: compact replicated-track
  summaries.
- `paper/data/supportshift_raw_example.csv`: plot-ready field extract.
- `paper/data/transition_stress.csv`: threshold-audit plot data.
- `paper/data/dimension_kernel_robustness.csv`: exact robustness-audit source
  data for the appendix table.
- `paper/data/supportshift_artifact_manifest.json`: SHA-256 contract for 15
  promoted data/provenance inputs and 23 generated tables, extracts, and figures.

CSV files use header rows, period decimal separators, and no implicit row index.
Boolean fields are textual `True`/`False`. Coordinates and parameter values use
the units of the synthetic lattice; there is no physical-unit interpretation.

## Generation and dependence structure

Within a replicate, spatial coordinates are jointly Gaussian and dependent
according to the declared Matérn covariance. The high-dimensional probability
result does **not** assume coordinate independence. Replicated fields indexed by
\(N\) are independent. Smoothing is a recorded linear map \(S_h\), so the
support-aware covariance is \(S_h\Sigma S_h^\top\). The naive covariance is
evaluated directly at output centers.

The final high-dimensional experiment uses growing output domains at fixed
lattice spacing. It is not a fixed-domain infill experiment. The pair theorem
is a continuous interior result; the finite-grid targets are calculated rather
than assumed to share the pairwise coefficient.

## Reproduction and validation

Create the declared environment with `environment-research.yml`, then run:

```bash
python scripts/research/verify_supportshift_release.py \
  --metadata outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.metadata.json \
  --paper-directory paper \
  --repository-root . \
  --require-full
```

A valid full release has 12,800 replicated-field fits, 8,400 finite-grid fits,
64 candidatewise-coverage cells, all predeclared statistical gates passing, 15
matching source-input hashes, 23 matching paper-artifact hashes, and 113
machine-checked manuscript claims. The verifier independently reconstructs all
42 finite-grid summary rows from the fit-level table and checks exact task keys,
configuration hashes, deterministic seeds, numerical identities, and clean
generation provenance.
The verifier fails on missing files, hash changes, incomplete grids, failed
gates, dirty generation provenance, or a mismatch between any reported number
and its released source table. The public artifact also supplies the exact
regeneration command in `README.md`.

## Known limitations

- All data are synthetic; there is no external-validity claim.
- Smoothness and the support operator are known.
- The finite-grid track mostly fixes nuisance parameters; the replicated track
  jointly selects variance and decay only over a fixed oracle-containing
  library.
- The exact concentration radius depends on the generating covariance and is a
  benchmark certificate, not a deployable confidence interval.
- Boundary normalization, nonstationarity, unknown support, measurement error,
  and arbitrary full-likelihood signs are outside the theorem.
- The release does not establish uniform consistency over a continuous
  parameter set or separate fixed-domain consistency of Matérn variance and
  range.
- The transition-aware error orders are pointwise in smoothness. The fine-grid
  audit does not establish a joint remainder uniform in smoothness and
  bandwidth.
- The dimension--kernel audit covers only (d\in\{1,2,3\}) and two compact
  product kernels. It is not a proof or a uniform guarantee over dimensions or
  kernel families.

## Intended and unintended uses

Intended uses are regression testing for support-aware spatial simulators,
checking whether a method distinguishes physical parameters from KL targets,
studying finite-sample concentration around a misspecified target, and extending
the generator to other support shapes or covariance families.

The data should not be presented as observational evidence, used to rank broad
spatial methods outside the declared candidate families, or interpreted as a
universal guarantee for real-data range estimation.

## Licensing and citation

Original SupportShift code and generated synthetic data are covered by the
scoped terms in `LICENSE-SUPPORTSHIFT`. Third-party and legacy material is
excluded. Cite the artifact using `CITATION.cff` and cite the paper separately
when a proceedings record becomes available.
