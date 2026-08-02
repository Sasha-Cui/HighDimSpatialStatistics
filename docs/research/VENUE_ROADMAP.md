# Venue and submission roadmap

Information was checked on 2026-08-02. Deadlines change; verify on the official
site immediately before scheduling work. Journal submissions listed here are
ordinary rolling submissions, not annual deadlines.

## Realistic journals

| Venue | Fit | What blocks submission now | Minimum before submission |
|---|---|---|---|
| [Spatial Statistics](https://www.sciencedirect.com/journal/spatial-statistics) | Best fit for smoothing-aware spatial methodology, theory, simulation, and an application | Corrected likelihood alone is standard; no new theorem or valid evidence | Explicit Matérn target-shift theorem, corrected estimator, comprehensive simulations, meaningful application |
| [Journal of Computational and Graphical Statistics](https://www.tandfonline.com/journal/ucgs20) | Fit only if observation-operator/composite computation is genuinely scalable | Current lookup still needs dense eigendecomposition/Cholesky and lacks modern baselines | Strong numerical method, error guarantees, matched-accuracy speed/memory advantage, software |
| [Biometrics](https://academic.oup.com/biometrics/pages/general-instructions) | Fit if ovary/SRT biology drives a new biometrical method | One puck, no biological estimand/replication, implausible zero-mean Gaussian observation model | Multiple biological samples, scientific collaboration, valid mean/noise model, uncertainty and independent validation |
| [Bioinformatics](https://academic.oup.com/BIOINFORMATICS/pages/author-guidelines) | Fit if a distinct SRT algorithm and biological task emerge | Strong current competition from COVET, Smoothie, SpaceX, spMOCA; no task-level evidence | New algorithm, public software, multiple datasets, current baselines, biological validation |
| [JRSS Series C](https://academic.oup.com/JRSSSC/pages/general-instructions) | Fit for a substantial applied problem with innovative methodology | Present work is simulation/exploration without substantive findings | Deep applied question, real insights, complete method and uncertainty; routine adaptation is insufficient |
| [Annals of Applied Statistics](https://imstat.org/journals-and-publications/annals-of-applied-statistics/annals-of-applied-statistics-manuscript-submission/) | Fit for a deep scientific application with broadly interesting methodology | Current real analysis is illustrative only | Multiple real datasets/replicates and scientific conclusions beyond the numerical method |
| [Journal of Multivariate Analysis](https://www.sciencedirect.com/journal/journal-of-multivariate-analysis) | Fit for a focused multivariate theorem | No original theorem; fixed-\(p\) extension not complete | Substantial matrix-spectral/cross-parameter theorem and rigorous proofs |
| [Journal of Agricultural, Biological and Environmental Statistics](https://link.springer.com/journal/13253) | Plausible for focused spatial composite methodology plus application | Existing multivariate composite-likelihood theory is close | New smoothing/block contribution, Godambe theory, careful applied study |

Top general theory venues (JASA Theory and Methods, JRSS-B, Biometrika) are not
realistic unless the target-shift or multivariate microergodic theory becomes much
more general and substantial than the minimum paper proposed here.

## ML conferences

| Venue | Verified status on 2026-08-02 | Fit assessment |
|---|---|---|
| [NeurIPS 2026](https://neurips.cc/Conferences/2026/CallForPapers) | Abstract 2026-05-04 and paper 2026-05-06; passed | Not realistic without a new broadly relevant ML method/theorem. Suggested workshop contribution date was 2026-08-29, but individual accepted-workshop calls govern and are generally nonarchival. |
| [UAI 2026](https://www.auai.org/uai2026/call_for_papers) | Deadline 2026-02-25; passed | Possible only for substantial probabilistic inference methodology, not current work. |
| [AISTATS 2027](https://aistats.org/other.html) | No official 2027 paper deadline verified | Potential only after a clear statistical-ML theorem and competitive scalable algorithm. |
| [COLT 2027](https://learningtheory.org/colt2027/) | Meeting 2027-06-28 to 2027-07-02; submission date listed as TBD | Poor fit unless the project becomes a general rate/lower-bound paper. |
| [ICLR 2027](https://iclr.cc/Conferences/FutureMeetings) | Future meeting listed; official paper dates not verified | Poor fit without an ML-specific learned model and strong empirical/theoretical case. |
| NeurIPS/ICML 2027 | Official paper dates not verified | Do not use unofficial deadline aggregators. Current project is not competitive. |

## Nonarchival feedback opportunities

| Event | Verified date | Use |
|---|---|---|
| [ISI World Statistics Congress 2027](https://www.isi-next.org/conferences/isi-wsc2027/key-dates/) | Contributed abstract deadline 2026-10-08; meeting 2027-07-11 to 2027-07-15 | Too early for a finished paper, but suitable for a precise theorem/pilot if Gate 1 passes before the deadline. |
| [JEMS 2027](https://www.jems2027.eu/key-dates-deadlines/) | Abstracts 2027-02-01 to 2027-03-21; meeting begins 2027-08-29 | Realistic feedback target after proof and main simulations. |
| [JSM 2027](https://www.amstat.org/meetings/joint-statistical-meetings) | Meeting 2027-08-08 to 2027-08-12; abstract deadline not posted at audit time | Useful methods session/poster after a complete draft. |

## Recommended timeline

### August 2026

- Finish analytical proof gate for explicit nonzero target shift.
- Freeze kernel/parameter convention and identified model.
- Do not submit an ISI abstract unless the theorem exists by internal review.

### September--November 2026

- Correct local-block estimator and Godambe calculation.
- Run theorem-matching pilot and adversarial diagnostics.
- Secure multiple application samples and domain collaborator.

### December 2026--March 2027

- Final simulation grid and independent application validation.
- Internal proof review and reproducibility audit.
- JEMS/JSM abstract only if the core manuscript is coherent.

### Spring--Summer 2027

- Complete manuscript and artifact.
- Primary journal target: Spatial Statistics.
- Switch to JCGS only if the computational contribution becomes central and
  convincingly competitive; switch to Biometrics/Bioinformatics only if the
  replicated scientific study becomes central.

## Submission stop conditions

Do not submit if any of the following remains true:

- novelty consists only of \(SKS^\top\) or generic KL misspecification;
- cross-covariance validity is checked only on finite observed coordinates;
- old synthetic artifacts or metrics are used;
- variance/range consistency is claimed under unqualified fixed-domain asymptotics;
- whole overlapping grids are treated as independent samples;
- the application still has one biological sample and no scientific validation;
- current baselines are absent;
- failed fits, seeds, or preprocessing are not fully reproducible.
