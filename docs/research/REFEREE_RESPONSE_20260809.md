# Response to referee report (2026-08-09)

This document records how the revision responds to the report on
*SupportShift: A Theory-Linked Spatial Simulation Benchmark for Ignored Matérn
Observation Support*. “Addressed” means that the manuscript now contains a
mathematical result or hash-audited experiment tied to the point. It does not
mean that the revision proves a broader claim than its stated assumptions.

## Major comment 1: the two-point construction is saturated

**Assessment: correct and addressed.** The revision now labels the two-point
target as an analytic building block rather than evidence of genuine
misspecification. It adds:

- a weighted multi-lag pair-composite theorem. If the lag-specific phase
  coefficients are $C_j$ and the information weights are $\lambda_j$, the
  pseudo-range shift has coefficient
  
  \[
  \bar C=\frac{\sum_j\lambda_j C_j}{\sum_j\lambda_j},
  \]
  
  and the first nonzero minimum-KL term is
  
  \[
  \frac{s_\nu(h)^2}{2}\sum_j\lambda_j(C_j-\bar C)^2.
  \]
  
  Thus the composite is genuinely misspecified whenever the lag coefficients
  are not all equal.
- a finite-design full-Gaussian local projection. For
  $\theta=(\log v,\log\alpha)$, the covariance perturbation is projected onto
  the variance--decay tangent space with Fisher matrix $J$. The local target
  shift is $J^{-1}b\,s_\nu(h)$, and the irreducible KL coefficient is the
  squared Fisher norm of the residual perturbation.

The 48-cell multi-lag audit and 12-cell full-likelihood audit evaluate both
results against exact numerical KL minimizers. At the smallest bandwidth, the
largest relative errors are 0.0141 for the multi-lag shift, 0.0985 for its
residual KL, 0.0542 for the full-likelihood decay shift, 0.1268 for the variance
shift, and 0.0177 for the full-likelihood residual KL.

## Major comment 2: smoothness is treated as known

**Assessment: empirically addressed; a general theorem remains open.** The
revision preserves the universal sign only at fixed smoothness. It adds a
2,400-fit finite-library experiment that jointly selects smoothness, decay, and
profiled variance. Exact support selects the physical population target in all
eight cells. Ignored support changes the smoothness target in all eight,
always upward on this design, and selects inverse range above one in all eight.
Consequently, joint nuisance movement can reverse the fixed-smoothness decay
direction.

This is reported as a finite-design phenomenon, not a universal joint-nuisance
theorem. Continuous joint estimation of smoothness, support width, variance,
and range remains a stated limitation.

## Major comment 3: exact versus naive is self-validating

**Assessment: addressed.** The revision adds a third model using the correct
radial support shape but 75% of the true bandwidth. Its population KL is no
larger than the point-support model in all eight joint-fit cells. In the
matched-boundary experiment, its KL is 16.7%--32.3% of point-support KL. This
gives a nontrivial partial-correction ordering while retaining exact support as
the containment control.

## Minor comments

### Boundary comparison changed dimension

**Addressed.** The new comparison uses translated interior and boundary
$4\times4$ blocks with the same $p=16$ and identical relative geometry.
Their point-support population targets differ by 0.030--0.062 across the four
smoothness values, isolating boundary normalization without a dimension change.

### Lag sensitivity was not quantified

**Addressed.** The new multi-lag theorem makes lag dependence explicit through
$C_j$ and $\lambda_j$. For $\nu=0.5$, the reported coefficient changes
from 1.61 at $\alpha R=0.5$ to 0.40 at $\alpha R=2$; unequal coefficients are
exactly what produces the positive residual-KL term.

### Fixed-$N$ scaling in $p$ was not shown

**Addressed with a scoped empirical result.** Regressing the 95th-percentile
criterion deviation on $p$ at fixed $N$ gives slopes from $-0.509$ to
$-0.297$, with median $-0.413$. The paper does not claim a universal
$p^{-1/2}$ law: the certificate depends on relative operator norms and
effective rank, not coordinate count alone.

### Continuous-to-library resolution used only objective distance

**Addressed.** The revision reports parameter distances as well. Maximum
continuous-to-library distances are 0.00499 in decay and 0.0297 in variance for
point support, and below $4.1\times10^{-7}$ for both coordinates under exact
support.

### SupportShift was not isolated from legacy code

**Addressed.** `supportshift/README.md` is the clean reviewer entry point. It
lists the paper sources, mathematical implementation, promoted inputs,
generators, and one-command verifier, and explicitly excludes historical
notebooks and the `legacy/` directory from the evidence contract.

## Remaining non-negotiable limitations

- The all-smoothness phase law and its sign are translation-invariant interior,
  fixed-smoothness results.
- The full-likelihood proposition is local and design specific; it supplies no
  universal sign.
- The joint smoothness result is a finite-library experiment, not a continuous
  asymptotic theorem.
- The replicated-field concentration certificate is standard supporting
  machinery and uses independent fields, though coordinates within a field may
  be arbitrarily dependent.
- Synthetic experiments are validation evidence, not a real-data application.

These restrictions are now stated in the abstract, theorem discussion,
experiment interpretation, conclusion, artifact card, and release verifier.
