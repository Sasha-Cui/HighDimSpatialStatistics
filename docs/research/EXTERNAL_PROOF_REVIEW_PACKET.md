# External proof-review packet: SupportShift

- **Prepared:** 2026-08-04
- **Frozen scientific artifact:** `supportshift-geosim-v1.2.1`
- **Commit reviewed:** `b6c8ee294db9319746eb3a3869f4b1315f694ef9`
- **Requested reviewer expertise:** spatial statistics, Gaussian-process
  misspecification, special functions, or applied probability

## Purpose and decision rule

This packet is for an independent human audit of the mathematical claims in
*SupportShift: Matérn Range Distortion under Ignored Observation Support, with
a Theory-Linked Synthetic Benchmark*. It deliberately separates analytic
claims from numerical evidence. The code, simulations, and checks below may
help locate an error, but they are not substitutes for verifying the displayed
identities and remainders.

Please classify each item as **verified**, **correct after a stated minor
repair**, or **not verified**. GeoSim submission should proceed only if items
P1--P8 are verified or repaired without changing the paper's central phase-law
claim. Item P9 is standard supporting machinery but its constants and
normalization must still be correct.

The primary source is `paper/geosim2026.tex`; the longer derivation is in
`paper/manuscript.tex`. The theorem intentionally concerns the exact
two-location Gaussian KL pseudo-parameter at a fixed nonzero lag. It is not a
claim about unrestricted full-grid maximum likelihood.

## Model and notation to hold fixed

Let

\[
C(r)=v\mathcal M_\nu(\alpha\lVert r\rVert),\qquad
\mathcal M_\nu(x)=\frac{2^{1-\nu}}{\Gamma(\nu)}x^\nu K_\nu(x),
\]

where \(v,\alpha,\nu>0\). Let \(k\) be a nonnegative symmetric probability
density supported on \(B(0,L)\), let \(U,V\) be independent with density
\(k\), and put

\[
D=U-V,\quad \Sigma_k=\mathbb E(UU^{\mathsf T}),\quad
T_k=\operatorname{tr}(\Sigma_k),\quad
m_q=\mathbb E\lVert D\rVert^q.
\]

For \(Z_h(t)=\int h^{-d}k(u/h)Y(t-u)\,du\),

\[
C_h(r)=v\mathbb E\mathcal M_\nu
   \{\alpha\lVert r+hD\rVert\}.
\]

At \(r=Re\), \(R>0\), the claimed point-support pair target is

\[
v_h^\dagger=C_h(0),\qquad
\mathcal M_\nu(\alpha_h^\dagger R)=
\rho_h(r)=\frac{C_h(r)}{C_h(0)}.
\]

The asymptotics fix \(\nu\), keep \(R\) in an annulus bounded away from zero,
keep \(\alpha\) in a compact subset of \((0,\infty)\), and send \(h\downarrow0\).
No remainder is asserted jointly uniform as \(\nu\to1\) or \(\nu\to2\).

## Analytic audit ledger

### P1. Exact pair KL target and identifiability

Check that the smoothed two-by-two covariance belongs to the naive
variance--decay family and that \(0<\rho_h(r)<1\) for \(r\ne0\). One route to
strictness is the smoothed spectral density
\(|\widehat k(h\omega)|^2f_Y(\omega)\), which is positive on a neighborhood of
the origin. Then

\[
C_h(0)-C_h(r)\propto\int\{1-\cos(\omega^{\mathsf T}r)\}
  |\widehat k(h\omega)|^2f_Y(\omega)\,d\omega>0.
\]

Check also

\[
\mathcal M_\nu'(z)
=-\frac{2^{1-\nu}}{\Gamma(\nu)}z^\nu K_{\nu-1}(z)<0,
\]

which makes the decay target unique.

**If this fails:** the KL interpretation or target uniqueness fails and the
current paper must not be submitted.

### P2. Fixed-nonzero-lag Taylor expansion

For \(f(x)=\mathcal M_\nu(\alpha\lVert x\rVert)\), verify that
\(h\le R/(4L)\) keeps \(r+hD\) uniformly away from the origin, and that
central symmetry of \(D\) gives

\[
\mathbb E f(r+hD)=f(r)+h^2B_{\nu,k}(r)+O(h^4),
\]

where, with \(z=\alpha R\), \(a_k(e)=e^{\mathsf T}\Sigma_ke\),

\[
B_{\nu,k}(r)=\alpha^2\left[
a_k(e)\mathcal M_\nu''(z)
{}+\{T_k-a_k(e)\}\frac{\mathcal M_\nu'(z)}z
\right].
\]

The factor of two must come from
\(\mathbb E(DD^{\mathsf T})=2\Sigma_k\). The expansion must not be reused at
the origin.

**If this fails:** repair the coefficient before any submission; simulations
and the directional claim must then be regenerated.

### P3. Noninteger origin expansion and phase constants

For noninteger \(\nu\), independently derive

\[
\mathcal M_\nu(x)=
\sum_{j\ge0}\frac{(x^2/4)^j}{j!(1-\nu)_j}
+b_\nu x^{2\nu}
\sum_{j\ge0}\frac{(x^2/4)^j}{j!(1+\nu)_j},
\qquad
b_\nu=\frac{\Gamma(-\nu)}{2^{2\nu}\Gamma(\nu)}.
\]

Compact support of \(D\) should justify termwise expectation. In particular,
check

\[
c_\nu=-b_\nu
=\frac{\Gamma(1-\nu)}{\nu\,2^{2\nu}\Gamma(\nu)}>0,
\qquad 0<\nu<1,
\]

and the quadratic coefficient \(-1/\{4(\nu-1)\}\) for \(\nu>1\).

**If this fails only outside \(\nu=1/2\):** downgrade to the exact exponential
fallback described in the technical manuscript rather than retaining an
all-smoothness claim.

### P4. Integer expansions and remainder thresholds

Check directly from the integer-order expansion of \(K_n\) that

\[
\mathcal M_1(x)=1+\frac{x^2}{2}
\{\log(x/2)+\gamma_{\mathrm E}-1/2\}
+O\{x^4(1+|\log x|)\},
\]

and

\[
\mathcal M_2(x)=1-\frac{x^2}{4}
-\frac{x^4}{16}
\{\log(x/2)+\gamma_{\mathrm E}-3/4\}
+O\{x^6(1+|\log x|)\}.
\]

These should imply origin remainders \(O(h^2)\) below one,
\(O\{h^4\log(1/h)\}\) at two after the leading quadratic term, and
\(O(h^4)\) above two, with the fractional \(O(h^{2\nu})\) term retained for
\(1<\nu<2\).

**If the \(\nu=1\) constant or sign fails:** the main three-regime theorem
requires repair. **If only the \(\nu=2\) remainder fails:** repair the stated
secondary remainder; the leading \(h^2\) law may remain intact.

### P5. Normalization and inverse-map orders

Starting from P2--P4, expand \(C_h(r)/C_h(0)\), then use the local inverse of
\(\alpha\mapsto\mathcal M_\nu(\alpha R)\). Check all three leading shifts and
the nonlinear inverse remainder:

\[
\alpha-\alpha_h^\dagger\asymp
\begin{cases}
h^{2\nu},&0<\nu<1,\\
h^2\log(1/h),&\nu=1,\\
h^2,&\nu>1.
\end{cases}
\]

For \(0<\nu<1\), the remainder should be
\(O\{h^{\min(4\nu,2)}\}\); this minimum combines the quadratic term in the
covariance expansion with the second-order Taylor term of the inverse map.
For \(\nu>1\), check the remainders \(O(h^{2\nu})\),
\(O\{h^4\log(1/h)\}\), and \(O(h^4)\) below, at, and above two.

**If this fails:** weaken the theorem to the order and coefficient actually
proved; do not use simulation agreement to fill an analytic gap.

### P6. Smooth-regime sign identity

For \(\nu>1\), verify that the order-\(h^2\) correlation coefficient is

\[
A_{\nu,k}(r)=
\alpha^2\{(2\nu-2)a_k(e)+T_k\}G_\nu(z),
\]

where

\[
G_\nu(z)=\frac{\mathcal M_\nu(z)}{2\nu-2}
+\frac{\mathcal M_\nu'(z)}z
=\frac{2^{1-\nu}}{\Gamma(\nu)}
\frac{z^\nu K_{\nu-2}(z)}{2\nu-2}>0.
\]

This uses
\(K_\nu(z)=K_{\nu-2}(z)+2(\nu-1)K_{\nu-1}(z)/z\). Since
\(R\mathcal M_\nu'(z)<0\), the displayed coefficient must imply
\(\alpha_h^\dagger-\alpha<0\).

**If this identity fails:** remove the universal smooth-regime sign claim and
reassess the paper's central message.

### P7. Transition cancellation at \(\nu=1\)

For

\[
W_{\nu,k}(h)=
\frac{m_2(\alpha h)^2}{4(1-\nu)}
+b_\nu m_{2\nu}(\alpha h)^{2\nu},
\]

check, with \(\nu=1+\varepsilon\),

\[
b_{1+\varepsilon}=rac1{4\varepsilon}
+\frac{2\gamma_{\mathrm E}-1-2\log2}{4}+O(\varepsilon)
\]

and

\[
m_{2+2\varepsilon}(\alpha h)^{2+2\varepsilon}
=(\alpha h)^2\left[m_2+2\varepsilon
\{\ell_{2,k}+m_2\log(\alpha h)\}+O(\varepsilon^2)\right].
\]

The poles should cancel and leave

\[
W_{1,k}(h)=\frac{(\alpha h)^2}{2}\left[
m_2\{\log(\alpha h/2)+\gamma_{\mathrm E}-1/2\}
+\ell_{2,k}\right].
\]

Check the fixed-\(\nu\) approximation errors; do not infer joint uniformity in
\((\nu,h)\).

**If this fails:** remove the transition-aware proposition and its threshold
figure. The pointwise phase theorem should be assessed separately.

### P8. Directional support contrast

After subtracting two lag directions, verify cancellation of the common
zero-lag normalization and use

\[
\mathcal M_\nu''(z)-\frac{\mathcal M_\nu'(z)}z
=\frac{2^{1-\nu}}{\Gamma(\nu)}z^\nu K_{\nu-2}(z)
\]

to obtain, for every fixed \(\nu>0\),

\[
\Delta_{e_1}(h)-\Delta_{e_2}(h)
=\frac{\alpha^2\{a_k(e_1)-a_k(e_2)\}}R
\frac{K_{\nu-2}(z)}{K_{\nu-1}(z)}h^2+o(h^2).
\]

Check the sign convention \(\Delta_e(h)=\alpha-\alpha_h^\dagger(e)\).

**If this fails:** remove the anisotropy proposition and corresponding panel;
the isotropic phase theorem can remain if P1--P7 pass.

### P9. Finite-library Gaussian certificate

For \(N\) independent \(N_p(0,\Sigma_0)\) vectors and deterministic positive
definite candidate matrices, set
\(A_\theta=\Sigma_0^{1/2}\Sigma_\theta^{-1}\Sigma_0^{1/2}\). Check that the
two-sided quadratic-form bound, division by \(2Np\), and union bound with
\(t=\log(2M/\delta)\) give

\[
|\widehat L_N(\theta)-L(\theta)|\le
\frac{\lVert A_\theta\rVert_{\mathrm F}}p\sqrt{\frac tN}
+\frac{\lVert A_\theta\rVert_{\mathrm{op}}p\frac tN
\]

simultaneously. Verify the factor two in the ERM excess-risk bound and that a
parameter-error claim would require an additional margin condition. Under
\(c\Sigma_0\preceq\Sigma_\theta\), check
\(A_\theta\preceq c^{-1}I_p\) and hence the conditional \(Np\) rate.

**If a constant fails:** repair the proposition, verifier, and reported
coverage together. **If only parameter identifiability is questioned:** the
paper already makes no parameter-rate claim from this certificate alone.

## Independent references for the identities

The reviewer should consult the formulas directly rather than treating this
list as verification:

- NIST DLMF [modified Bessel equation, Eq. 10.25.1](https://dlmf.nist.gov/10.25.E1);
- NIST DLMF [connection formula, Eq. 10.27.4](https://dlmf.nist.gov/10.27.E4);
- NIST DLMF [recurrences and derivatives, Section 10.29](https://dlmf.nist.gov/10.29);
- NIST DLMF [integer-order power series, Section 10.31](https://dlmf.nist.gov/10.31);
- Laurent and Massart (2000), as cited in the paper, for Gaussian quadratic
  forms; and
- Hsu, Kakade, and Zhang (2012), as cited in the paper, for a modern matrix
  formulation.

## Adversarial numerical probes already performed

These checks are implementation diagnostics, not proof.

1. At 100 decimal digits, the derivative identity and the two expressions for
   \(G_\nu(z)\) were compared for
   \(\nu\in\{0.05,0.25,0.5,0.99,1,1.01,1.5,1.99,2,2.01,4,10\}\) and
   \(z\in\{0.01,0.1,1,10\}\). Maximum scaled discrepancies were
   \(1.96\times10^{-101}\) and \(9.83\times10^{-102}\), respectively; every
   evaluated \(G_\nu\) was positive.
2. After subtracting the displayed \(\nu=1\) and \(\nu=2\) terms, division by
   the claimed logarithmic remainder scales stayed bounded over
   \(x=10^{-2},10^{-4},\ldots,10^{-12}\).
3. For a one-dimensional uniform kernel and \(\alpha h=0.037\), the difference
   \(|W_{1+\varepsilon,k}-W_{1,k}|\) was proportional to
   \(|\varepsilon|\) from \(10^{-2}\) through \(10^{-12}\), after the two
   singular terms were evaluated at 100-digit precision.
4. An additional 216 exact-quadrature cells crossed product Epanechnikov and
   uniform kernels, 18 smoothness values from 0.05 through 10 (including
   0.99, 1.01, 1.99, and 2.01), and six bandwidths from 0.1 to 0.002. Every
   exact pseudo-decay shift was positive. The smallest shift was
   \(5.25\times10^{-8}\).
5. The promoted, predeclared audit remains the paper evidence: 72 cells across
   dimensions one through three, two kernels, four smoothnesses, and three
   bandwidths, with independent quadrature refinement. The broader probes in
   this packet were performed after release and are not presented as a new
   confirmatory experiment.

## Reviewer response template

Please return the following, with equation-level notes for every non-verified
item.

| Item | Verified / minor repair / not verified | Notes |
|---|---|---|
| P1 Pair target |  |  |
| P2 Fixed-lag Taylor expansion |  |  |
| P3 Noninteger origin expansion |  |  |
| P4 Integer expansions |  |  |
| P5 Normalization and inverse map |  |  |
| P6 Smooth-regime sign |  |  |
| P7 Transition cancellation |  |  |
| P8 Directional contrast |  |  |
| P9 Gaussian certificate |  |  |

Also answer:

1. Is any assumption used but absent from the theorem statement?
2. Is any remainder stated uniformly over a set not supported by the proof?
3. Does any displayed coefficient have a sign, factor-of-two, or inverse-range
   convention error?
4. Does the paper accidentally promote a pairwise pseudo-parameter statement
   into a full-grid or consistency claim?
5. Is the exact result already present in older change-of-support or
   regularized-variogram literature known to the reviewer?

The review should name the source and page or equation for any priority concern.
Silence or numerical agreement is not a verification.
