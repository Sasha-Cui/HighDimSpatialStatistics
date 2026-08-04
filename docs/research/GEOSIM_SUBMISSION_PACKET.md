# GeoSim 2026 submission packet

**Prepared:** 2026-08-04  
**Official call:** <https://geosim.org/2026/cfp/>  
**Paper type:** Full paper  
**Artifact candidate:** `supportshift-geosim-v1.2.0`

This file stages copy-ready submission metadata and records the fields that
still require the author or the live EasyChair portal. It is not evidence that
a submission has been made.

## Title

SupportShift: A Theory-Linked Spatial Simulation Benchmark for Ignored Matérn
Observation Support

## Plain-text abstract

Spatial measurements often represent pixels, footprints, or local averages
but are fitted as point observations. We introduce SupportShift, a
parameterized and scalable synthetic benchmark that separates the physical
covariance parameter, the population Kullback-Leibler (KL) target, and
finite-sample estimation error under this support misspecification. Users can
vary support width and geometry, smoothness, lattice size, output dimension,
replicate count, and covariance-library resolution. Its analytic anchor is a
stationary Matérn Gaussian field in R^d observed through a compact symmetric
averaging kernel of bandwidth h. For a fixed nonzero-lag two-point likelihood
with known smoothness nu, the naive inverse-range shift has order h^(2 nu) for
0 < nu < 1, h^2 log(1/h) at nu = 1, and h^2 for nu > 1. Explicit coefficients
prove range inflation for every fixed nu > 0 at sufficiently small support. A
transition-aware two-term approximation resolves the slow finite-bandwidth
crossover at nu = 1, while a directional h^2 contrast handles elongated
support. For N independent p-dimensional Gaussian fields and a deterministic
finite covariance library, a simultaneous likelihood certificate scales as
sqrt(log(M)/(N p)) + log(M)/(N p) under relative spectral control; spatial
dependence within each field is unrestricted. Four reproducible tracks pair
these statements with deterministic quadrature, finite-grid likelihoods,
directional support, and growing-p replicated fields. Released generators,
source tables, seeds, metadata, and pass/fail gates make every reported number
auditable. The benchmark exposes a sharp failure mode: likelihood noise can
vanish while a point-support fit concentrates around the wrong range target.

## Keywords

Matérn covariance; change of support; Gaussian random field;
high-dimensional probability; spatial simulation benchmark; misspecified
likelihood

## Suggested topic and reviewer assignments

If the portal presents the topics from the official call, select:

1. Verifying and Validating Spatial Simulations
2. Big Spatial Data Simulation
3. Spatial Analysis based on Simulation
4. Spatial Data/Trajectory Generators

Suggested reviewer expertise:

- spatial statistics and change of support;
- Matérn Gaussian processes and covariance misspecification;
- high-dimensional probability and Gaussian quadratic forms; and
- reproducible simulation benchmark design.

## Artifact and PDF record

- Repository: <https://github.com/Sasha-Cui/HighDimSpatialStatistics>
- Candidate tag: `supportshift-geosim-v1.2.0`
- Submission PDF: `output/pdf/supportshift_geosim2026.pdf`
- PDF structure: 10 main pages, one reference page, two appendix pages
- Current PDF SHA-256:
  `27c9e349111f66a30df8235e1e930a4af258a6e19e84677d1f736d6d24bd98ff`
- Current PDF size: 702,520 bytes

Recompute the hash from the exact uploaded file and replace these values if the
portal copy differs from the tagged artifact.

## Suggested generative-AI disclosure for author review

> OpenAI Codex was used to assist with repository audit, code and prose editing,
> test construction, reproducibility infrastructure, and manuscript
> preparation. All mathematical claims, literature comparisons, experimental
> design choices, reported results, and the final submitted text remain the
> author's responsibility. Reported numerical results were generated and
> checked by the released deterministic or seeded code and are covered by the
> artifact's automated verification gates.

Use this text only after checking the venue's actual disclosure question and
editing it to match the author's assessment. Do not state that a disclosure
was made until the live form has been completed.

## Author and portal fields requiring confirmation

- corresponding-author email;
- exact affiliation and postal metadata;
- any ORCID identifier;
- conflicts and suggested/excluded reviewers, if requested;
- registration and in-person presentation commitment;
- the portal's PDF-size and supplementary-material limits;
- the live generative-AI disclosure field; and
- final submission identifier, timestamp, and receipt.

## Final upload record

- [ ] Exact tagged PDF uploaded
- [ ] Uploaded PDF downloaded and hash-matched
- [ ] Portal metadata matches the PDF
- [ ] Submission identifier recorded
- [ ] Receipt saved outside the repository
