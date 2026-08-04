# GeoSim 2026 submission checklist

**Target:** 9th ACM SIGSPATIAL International Workshop on Geospatial Simulation<br>
**Workshop:** 2026-11-03, Riverside, California<br>
**Submission deadline:** 2026-08-15<br>
**Notification:** 2026-09-15<br>
**Official call:** <https://geosim.org/2026/cfp/><br>
**Submission site:** <https://easychair.org/conferences/?conf=geosim2026><br>
**Artifact candidate:** `supportshift-geosim-v1.1.3`

The official call lists parameterizable and scalable spatial-simulation data
sets made available to the community as an in-scope topic. Full papers may use
up to 10 pages excluding references, with up to two appendix pages after the
references. Review is single-blind and uses the ACM `sigconf` format.

## Scientific gates

- [x] Formal model and estimand are explicit.
- [x] Pairwise theorem is separated from numerical full-grid claims.
- [x] Fixed-domain and increasing-domain statements are not conflated.
- [x] Within-field spatial dependence is retained in the probability result.
- [x] Concentration machinery is described as standard supporting machinery.
- [x] Synthetic factors map directly to theorem or limitation statements.
- [x] Monte Carlo uncertainty and boundary fits are reported.
- [x] Closest modern and classical change-of-support work is compared directly.
- [x] Clark's finite range-of-influence rule and the
  Bellehumeur--Legendre practical-range result are distinguished from the
  Matérn KL pseudo-parameter.
- [x] Priority search is documented without treating a negative search as proof.
- [ ] An independent spatial-statistics/probability expert has checked every
  theorem and proof.
- [ ] A human expert has manually checked older regularized-variogram monographs
  for the exact pseudo-range result.

## Artifact gates

- [x] Full synthetic inputs required by the paper are tracked.
- [x] Seeds, factors, environment, generation commit, and validation gates are
  recorded in machine-readable metadata.
- [x] One command verifies 12,800 fits, 64 coverage cells, and 21 hashes.
- [x] The same command recomputes all 100 reported numerical claims from the
  promoted tables and fails on any mismatch.
- [x] Clean-clone reproduction has passed.
- [x] Local and Bouchet checks have passed.
- [x] `CITATION.cff`, scoped license, and synthetic-data card are present.
- [x] Legacy outputs are explicitly excluded from paper evidence.
- [ ] A DOI-backed archive (for example Zenodo) has been created and inserted in
  the camera-ready paper. A Git tag is adequate for review but weaker than a DOI.

## Manuscript and ACM gates

- [x] Draft uses `\documentclass[sigconf]{acmart}`.
- [x] Authors and affiliations are visible for single-blind review.
- [x] Current main text fits the full-paper limit.
- [x] A one-page proof appendix follows the references and remains within the
  call's optional two-page appendix allowance.
- [x] ACM classification, keywords, figure descriptions, and references are
  present.
- [x] ACM classification foregrounds modeling, simulation, and model
  verification in addition to probability.
- [x] Claims avoid “first,” “novel,” and unsupported universal language.
- [x] Every plotted result is generated from hash-verified source data.
- [x] A benchmark-contract table maps every synthetic track to its target,
  controls, and falsification gate.
- [x] An internal mock review records likely objections, current answers, and
  acceptance-preserving fallbacks.
- [x] A portal-ready metadata packet stages the title, plain-text abstract,
  keywords, topic selections, reviewer expertise, artifact hash, and disclosure
  draft without claiming that the live form is complete.
- [ ] Run the official ACM TAPS/preflight checks available to submitters.
- [x] The official call identifies the GeoSim 2026 EasyChair submission site.
- [ ] Confirm the file-size limit and supplementary-material interface inside
  EasyChair before the final upload.
- [ ] Complete any required use-of-generative-AI disclosure in the venue's
  current author form.
- [ ] Replace placeholder copyright/DOI/ISBN fields only at the stage directed
  by the organizers.

## Attendance and administrative gates

- [ ] A listed author commits to register for both the workshop and ACM
  SIGSPATIAL if accepted.
- [ ] A listed author can attend and present in Riverside on 2026-11-03.
- [ ] Author names, affiliation, title, and contact information are confirmed.
- [ ] Submission is uploaded before 2026-08-15 with a local copy of the exact
  submitted PDF and portal receipt.

## Go/no-go rule

Submit to GeoSim only if independent proof review finds no major theorem defect
and attendance is confirmed. If a correctable proof issue is found, weaken the
statement and regenerate only the experiments still tied to the repaired
theorem. If the all-smoothness theorem fails but the exact exponential case and
artifact remain valid, reframe the paper as a conservative simulation benchmark
and use a lower-risk workshop or computational venue rather than submitting the
current claim set unchanged.
