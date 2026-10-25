# Reduced-rank regression: three codes against Aoyagi–Watanabe (2026-10-21)

Family `y = U V x + e`, `U` (M×H), `V` (H×N), unit Gaussian noise and inputs, N(0,1) prior; true rank r.
`bitsev-rrr-codes` (testbeds/rrr.py): Bayes code by HMC on the sufficient statistics (64 chains × 200
samples per block), least-squares plug-in code, Langevin local learning coefficient at the MAP.
Regret against the true law, geometric blocks r = 2 from 8, slopes on n ≥ 256, n = 65536, 16 seeds.

## Learner coefficients (codes_summary.txt, job 1813528)

| (M,N,H,r) | λ (AW) | m | MN/2 | plug-in slope | Bayes slope |
|---|---|---|---|---|---|
| (2,2,2,2) | 2.0 | 1 | 2.0 | 2.11 ± 0.18 | 2.09 ± 0.19 |
| (2,2,2,0) | 1.5 | 1 | 2.0 | 2.11 ± 0.18 | 1.53 ± 0.15 |
| (3,3,3,0) | 3.5 | 2 | 4.5 | 4.31 ± 0.26 | 3.05 ± 0.21 |
| (3,3,3,1) | 4.0 | 1 | 4.5 | 4.31 ± 0.26 | 3.72 ± 0.23 |
| (4,4,4,0) | 6.0 | 1 | 8.0 | 8.29 ± 0.31 | 6.04 ± 0.27 |
| (4,4,4,2) | 7.5 | 1 | 8.0 | 8.29 ± 0.31 | 7.58 ± 0.29 |
| (4,2,4,0) | 3.5 | 1 | 4.0 | 3.81 ± 0.17 | 3.22 ± 0.16 |

Rank-restricted families on the same data (gain_summary.txt, job 1813605): (3,3,1,1) 2.48 ± 0.15 (λ 2.5),
(4,4,2,2) 6.06 ± 0.23 (6), (4,4,1,1) 3.51 ± 0.15 (3.5), (4,4,4,1) 6.91 ± 0.27 (7, m = 2), (4,4,1,0)
1.99 ± 0.20 (2, m = 2).

**Reading.** The Bayes code's coefficient is Aoyagi–Watanabe's λ in every case (the two m = 2 cases sit
0.1–0.45 below λ, the sign and size of the −(m−1) ln ln n term at these n); the plug-in's is the
regular MN/2 whatever the rank. This is the plan's open statement T4 made numerical: κ_plug-in ≠ λ
whenever the true rank is deficient, and the gap MN/2 − λ is exactly the predicted one (0.5, 1.0,
2.0, 0.5, 0.5). Gains between two families that both contain the truth grow like Δλ ln n
(gain_between_families.txt).

## Schedule check (job 1813692; n = 4096, 8 seeds, slopes on n ≥ 128)

Corrected Bayes slopes under per-sample coding (growth 1) / doubling blocks / quadrupling blocks:
(3,3,3,0) 3.15 / 3.13 / 2.84; (4,4,4,0) 5.75 / 6.00 / 6.24; (2,2,2,2) 2.11 / 1.94 / 1.98;
(4,4,4,2) 7.18 / 7.44 / 7.68 (s.e. 0.25–1.0). The batching correction (r−1)/ln r holds for a
singular Bayes code, as the plan's T3 asserts.

## Langevin LLC

At n = 4096 the estimator (step 1e-4, pull 1, 4000 steps) reads 3.38 / 6.25 / 2.16 / 7.91 for
λ = 3.5 / 6 / 2 / 7.5. At n = 65536 the same step overshoots or diverges (9.4 for λ = 2, NaN for the
4×4 shapes): the drift n β ∇L grows with n and the step must shrink with it. The LLC is reported at
n = 4096 only.
