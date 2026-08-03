# SupportShift prior-art search

**Search date:** 2026-08-03<br>
**Purpose:** identify the closest published or publicly available result to the
SupportShift small-support pseudo-range expansion.<br>
**Status:** a documented screening search, not proof of priority.

## Exact claim screened

The paper studies a stationary Matérn field observed through a compact,
symmetric local average of bandwidth \(h\), but fitted at point support. For a
fixed nonzero lag, known smoothness \(\nu\), and jointly fitted variance and
inverse range, the exact two-point Gaussian KL target satisfies

\[
\alpha-\alpha_h^\dagger \asymp
\begin{cases}
h^{2\nu}, & 0<\nu<1,\\
h^2\log(1/h), & \nu=1,\\
h^2, & \nu>1,
\end{cases}
\]

with explicit coefficients and positive range inflation for sufficiently small
support. The directional extension isolates an \(h^2\) major-minus-minor
apparent-range contrast.

The search therefore required more than a paper that integrates a covariance
over a block. A direct predecessor would need to analyze the parameter selected
when that support operator is ignored, and ideally give the Matérn
smoothness-dependent order or coefficient.

## Search protocol

Searches used combinations of the following phrases, both unrestricted and
restricted to publisher, Project Euclid, arXiv, and proceedings domains:

- `Matérn block support covariance range estimation`
- `Matérn change of support range parameter`
- `Matérn aggregated observations covariance misspecification range`
- `Matérn local averages pseudo true range covariance`
- `regularized variogram range parameter block support bias`
- `regularization effect variogram range block support`
- `block covariance Matérn asymptotic small block`
- `change of support Matérn covariance parameter estimation`
- `Matérn averaged process range parameter bias`
- `Matérn small-support expansion local average covariance`
- exact-rate searches for `h^(2 nu)` and `h^2 log(1/h)` together with Matérn,
  averaging, covariance, and support terms.

The title, abstract, available full text, theorem statements, and relevant
equations were checked where accessible. Citation trails were screened through
the standard monographs and the closest change-of-support articles already
cited by the manuscript. Search-engine ranking is not an exhaustive index, and
older monographs remain a human-review item.

## Closest sources and exact comparison

| Source | What it establishes | Why it is not the screened claim |
|---|---|---|
| [Journel and Huijbregts (1978), *Mining Geostatistics*](https://www.sciencedirect.com/book/9780123910509/mining-geostatistics) and [Chilès and Delfiner (2012), *Geostatistics: Modeling Spatial Uncertainty*](https://doi.org/10.1002/9781118136188) | Classical point-to-block and block-to-block regularization, including the reduction of within-block variability. | Establishes the support transform, not the KL parameter obtained by ignoring it, and no all-\(\nu\) Matérn pseudo-range phase law was located. |
| [Gelfand, Zhu and Carlin (2001)](https://doi.org/10.1093/biostatistics/2.1.31) and [Gotway and Young (2002)](https://doi.org/10.1198/016214502760047140) | Statistical change-of-support models and the dangers of combining incompatible supports. | Motivates support-aware modeling; it does not provide the fixed-lag Matérn pseudo-parameter expansion screened here. |
| [Kyriakidis and Yoo (2005)](https://doi.org/10.1111/j.1538-4632.2005.00633.x) | Prediction and simulation of point values from areal data using coherent change-of-support geostatistics. | Solves a support-aware prediction/simulation problem rather than quantifying a point-support likelihood target under ignored averaging. |
| [Fuentes (2007)](https://doi.org/10.1198/016214506000000852) | For irregular spatial data, explicitly represents a rectangular block average and its filtered spectrum, then notes that the spectrum approaches the point-support spectrum for a small pixel. | This is the nearest located analytic setup. It does not compute the next-order range displacement, the smoothness transition at \(\nu=1\), or the directional coefficient. |
| [Tanaka et al. (2019)](https://proceedings.neurips.cc/paper/2019/hash/a941493eeea57ede8214fd77d41806bc-Abstract.html) | A spatially aggregated Gaussian-process model that places the areal support in the observation model. | Provides a modern support-aware computational method, not the pseudo-range asymptotics of an ignored operator. |
| [Chacón-Montalván et al. (2024)](https://arxiv.org/abs/2403.08514) and [Zheng et al. (2026)](https://doi.org/10.1016/j.spasta.2026.100998) | Latent-Gaussian and INLA--SPDE methods for data on differing point/grid/rectilinear supports. | Treat the support explicitly; neither located statement gives the screened misspecified Matérn phase law. |
| [Bachoc (2018)](https://doi.org/10.3150/16-BEJ906) | KL pseudo-targets and asymptotics for covariance estimation under Gaussian-process misspecification. | Supplies the generic misspecification framework, but not the support-specific target, direction, or rate. |
| [Clifford et al. (2006)](https://doi.org/10.1017/S0021859606005892) and [Paciorek and Schervish (2006)](https://doi.org/10.1002/env.785) | Convolution/support effects and flexible anisotropic or nonstationary covariance constructions. | Establish the qualitative possibility of support-induced directional structure; the explicit directional apparent-range contrast was not located. |
| [Simons and Olhede (2026)](https://doi.org/10.1093/gji/ggag044) | Finite sampled-grid Matérn maximum likelihood, discretization, parameterization, and edge effects. | Addresses grid sampling rather than an ignored local averaging operator and does not state the screened pseudo-target expansion. |

## Search conclusion and wording rule

No exact predecessor for the all-smoothness point-fit pseudo-range expansion or
its directional coefficient was identified in this search. That negative result
does **not** prove priority. The paper must therefore use comparative wording:
“the object studied here,” “the addition studied here,” and “we did not locate,”
not “first,” “novel,” or “previously unknown.”

The strongest defensible comparison is:

> Classical work derives the support-transformed covariance, and Fuentes
> (2007) gives the small-pixel spectral approximation. SupportShift studies the
> next-order parameter selected when that transform is ignored, including the
> Matérn smoothness transition and a directional coefficient.

## Residual priority risk

Before submission, a spatial-statistics expert should manually inspect the
relevant sections and bibliographies of Journel and Huijbregts (1978), Chilès
and Delfiner (2012), Webster and Oliver, and older regularized-variogram and
dispersion-variance literature. The question is specifically whether any source
fits a point-support Matérn model to block support and derives the asymptotic
range shift, not merely whether it describes regularization. If such a result is
found, the theorem can remain useful as a benchmark anchor, but the paper must
reframe it as a sharpened synthesis or special-case derivation and cite the
predecessor directly.
