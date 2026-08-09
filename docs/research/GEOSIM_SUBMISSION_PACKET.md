# GeoSim 2026 submission packet

- **Prepared:** 2026-08-04; referee-directed revision 2026-08-09
- **Official call:** <https://geosim.org/2026/cfp/>
- **Paper type:** Full paper
- **Artifact candidate:** `supportshift-geosim-v1.3.0`

This file stages copy-ready submission metadata and records the fields that
still require the author or the live EasyChair portal. It is not evidence that
a submission has been made.

## Title

SupportShift: A Theory-Linked Spatial Simulation Benchmark for Ignored Matérn
Observation Support

## Plain-text abstract

Spatial measurements often represent pixels, footprints, or local averages
but are fitted as point observations. We introduce SupportShift, a
parameterized synthetic benchmark that separates physical covariance
parameters, population Kullback-Leibler (KL) targets, and finite-sample error
under this support misspecification. Its analytic anchor is a stationary Matérn
Gaussian field observed through a compact averaging kernel of bandwidth h. For
fixed smoothness nu, ignoring support shifts the fitted inverse range at order
h^(2 nu) for 0 < nu < 1, h^2 log(1/h) at nu = 1, and h^2 for nu > 1, with
explicit coefficients proving small-support range inflation. Because a
variance-refitted two-point model is saturated, we extend the law to a weighted
multi-lag composite and obtain both its information-weighted displacement and
a strictly positive first nonzero residual-KL term. Under finite-design
identifiability, a second result gives the Fisher-metric projection of the
support perturbation onto the variance-decay tangent space and its irreducible
full-likelihood KL component.

The released tracks test these claims by deterministic quadrature, exact
finite-design targets, joint smoothness-range fitting, a partially corrected
support model, matched-size boundary comparisons, and growing-p replicated
fields. Joint fitting can reverse the fixed-smoothness decay direction, while
increasing replication can concentrate a point-support estimator around a
displaced target. A finite-library likelihood certificate permits arbitrary
spatial dependence within each field. Code, synthetic generators, source
tables, seeds, hashes, and pass/fail gates make every reported number auditable.

## Keywords

Matérn covariance; change of support; Gaussian random field;
high-dimensional probability; spatial simulation benchmark; misspecified
likelihood

## Suggested topic and reviewer assignments

If the portal presents the topics from the official call, select:

1. Verifying and Validating Spatial Simulations
2. Spatial Data/Trajectory Generators
3. Spatial Analysis based on Simulation

Do not select Big Spatial Data Simulation solely because the benchmark varies
\(p\), \(N\), and lattice size. SupportShift controls statistical scale but
does not claim large-system throughput or subcubic dense-covariance
computation.

Suggested reviewer expertise:

- spatial statistics and change of support;
- Matérn Gaussian processes and covariance misspecification;
- high-dimensional probability and Gaussian quadratic forms; and
- reproducible simulation benchmark design.

If EasyChair offers an optional note to the program chairs, use:

> This paper is submitted primarily under Verifying and Validating Spatial
> Simulations. It releases a parameterized synthetic benchmark with analytic
> and finite-grid targets for testing whether spatial covariance procedures
> distinguish physical parameters, misspecified KL targets, and sampling
> error. The paper is intentionally theory-plus-simulation rather than a
> real-data application.

## Artifact and PDF record

- Repository: <https://github.com/Sasha-Cui/HighDimSpatialStatistics>
- Candidate tag: `supportshift-geosim-v1.3.0`
- Submission PDF: `output/pdf/supportshift_geosim2026.pdf`
- PDF structure: 10 main pages, one reference page, two appendix pages
- Current PDF SHA-256:
  `e22ffbfb4a682b92932c8ca4acfd51097715453fec00f18f91371811f5558cc8`
- Current PDF size: 694,863 bytes

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
