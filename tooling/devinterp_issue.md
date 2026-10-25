# Proposal: regression tests against exact learning coefficients, and a code-length utility

Thanks for maintaining `devinterp`. Two pieces of work from a research project on
description lengths may be useful upstream. This is a proposal for discussion, not a PR;
I can turn either part into one if wanted (see the end).

## (i) Regression tests with exact targets

**Why.** `λ̂ = nβ (E_β[L_n] − L_n(w*))` depends on step size, localization, chain length and
`β`. On real networks `λ` is unknown, so a biased setting cannot be told from a true value.
Models with closed-form `(λ, m)` make bias visible, and two-parameter models are the
cheapest possible fixtures.

**Targets** (exact values, cross-checked numerically):

- *Monomial germs* on `[-1, 1]^2`, uniform prior: `x² + y²` (λ = 1, m = 1); `x² + y⁴`
  (3/4, 1); `x² y²` (1/2, 2); `(ab − 1/2)²` (1/2, 1). `F(n)` reduces to one-dimensional
  integrals, so the exact free energy, Bayes learning curve and Gibbs moments of `K` are
  available at any `n`. The last two share `λ` and differ in `m`, isolating the
  `log log n` bias.
- *Reduced-rank regression* `y = UVx + noise` with the Aoyagi–Watanabe (2005) formula,
  e.g. `(M, N, H) = (2, 2, 2)`, true rank 0: λ = 3/2; `(4, 4, 4)`, rank 0: λ = 6. With
  `H ≥ min(M, N)` least squares has the regular coefficient `MN/2`, so this family also
  checks that an estimator reports the Bayesian `λ`, not a parameter count.

**Assertions**, with the sampler configuration (`β`, step size, `γ`, chains, draws,
burn-in, seed) recorded in the test:

- `|λ̂ − λ| ≤ 0.05` for the `m = 1` models on a stated `n` grid;
- for `m = 2`, assert `λ̂ < λ` and report the gap: the exact Bayes code of `x² y²` fitted
  without the `log log n` term gives 0.39 at `n ≤ 2^14`, so a downward bias is expected;
- `λ̂` from two chain lengths (or step sizes) agrees within Monte Carlo error.

One subtlety: on these germs the estimator should target the tempered posterior near the
global minimum set under the box prior, so localization must be weak or absent.

## (ii) A code-length utility

`λ log n` is the redundancy of the Bayes mixture code, so code lengths are the natural
companion to LLC estimates. A small module could provide:

- **Prequential estimators**, per-sample and block: `DL(n) = Σ −log p(y_i | y_<i)` with
  the learner refit per prefix, a `Schedule` of block boundaries, raw per-item losses kept.
- **Batching-factor correction.** Geometric blocks `n_j = ⌈r^j⌉` multiply any logarithmic
  coefficient by `(r − 1)/ln r`, since each block contributes `(r − 1)κ` and there are
  `ln n / ln r` blocks. For a Gaussian mean (`κ = 1/2`) exact sums give 0.7213 with
  doubling blocks instead of 0.5 (0.6166 at `r = 1.5`, 1.0820 at `r = 4`); dividing by the
  factor recovers 0.5.
- **WBIC** (Watanabe, *JMLR* 14(27):867–897, 2013): `E_β[nL_n]` at `β = 1/log n` as a
  free-energy estimate, comparable to a prequential code length; `devinterp` already samples
  at this `β`.
- **Loss-fluctuation thermometer.** Under a Gibbs law `∝ exp(−K/T)` at a minimum of
  multiplicity one, the excess loss is asymptotically `Gamma(λ, T)`, so
  `T̂ = Var(K)/E(K)` and `λ̂ = E(K)²/Var(K)`. Exact for the `m = 1` germs; for `x² y²` it
  gives `λ̂ = 0.37, 0.40, 0.42` at `T = 1e-2, 1e-3, 1e-4`. Caveat: Gibbs samplers only. SGD's
  stationary law is not Gibbs (Mandt et al. 2017) and there it returns half the Hessian's
  participation ratio instead of `λ`. Its use is checking a sampler's effective temperature
  against the nominal `1/(nβ)`.

## Offer

A numpy/scipy reference implementation exists in my `bitsev` package
(`bitsev.exact.germs`, `rrr`, `thermometer`; `bitsev.codes.schedule`, `prequential`,
`gain`), torch-free and tested against the numbers above. I would be glad to adapt it to
`devinterp`'s conventions and open a PR in roughly months 6–9 of my project (spring to
summer 2027) if you want it. Open questions: fixtures under `tests/` or a `devinterp.exact`
module; which tolerances and sampler settings to pin; whether torch-free code is acceptable
for the utility layer.
