# Gate G-A0 report

Generated 2026-10-15 by `bitsev-exact` at commit `2b4cf28` from `preregistration/G-A0.md`.

**Gate verdict: PASS** (criteria 1, 2, 3 and 5 met; criterion 4 run).

| criterion                                                      | status |
|----------------------------------------------------------------|--------|
| 1. Batching factor reproduced (Gaussian mean)                  | PASS   |
| 2. Bayes redundancy isolated (per-sample codes of exact germs) | PASS   |
| 3. Batching isolated (block codes of exact germs)              | PASS   |
| 4. Optimizer effect (plug-in κ against λ)                      | PASS   |
| 5. Carried-over checks (thermometer, identifiability)          | PASS   |

## 1. Batching factor reproduced (Gaussian mean)

**PASS**

| learner | r   | fitted | (r-1)/(2 ln r) | error    |
|---------|-----|--------|----------------|----------|
| plug-in | 1   | 0.5000 | 0.5000         | 2.2e-06  |
| plug-in | 1.5 | 0.6163 | 0.6166         | -2.8e-04 |
| plug-in | 2   | 0.7213 | 0.7213         | -2.2e-16 |
| plug-in | 4   | 1.0820 | 1.0820         | 2.2e-16  |
| bayes   | 1   | 0.5000 | 0.5000         | -8.6e-06 |
| bayes   | 2   | 0.7208 | 0.7213         | -5.8e-04 |

## 2. Bayes redundancy isolated (per-sample codes of exact germs)

**PASS**

| germ       | λ    | m | slope (no log log) | slope (m fixed) | criterion |
|------------|------|---|--------------------|-----------------|-----------|
| x^2+y^2    | 1    | 1 | 1.0000 ± 0.0000    | 1.0000 ± 0.0000 | PASS      |
| x^2+y^4    | 0.75 | 1 | 0.7500 ± 0.0000    | 0.7500 ± 0.0000 | PASS      |
| x^2y^2     | 0.5  | 2 | 0.3944 ± 0.0030    | 0.5282 ± 0.0018 | reported  |
| (ab-0.5)^2 | 0.5  | 1 | 0.5012 ± 0.0003    | 0.5012 ± 0.0003 | PASS      |

## 3. Batching isolated (block codes of exact germs)

**PASS**

| germ       | r | λ    | fitted | λ (r-1)/ln r | corrected | criterion        |
|------------|---|------|--------|--------------|-----------|------------------|
| x^2+y^2    | 1 | 1    | 1.0000 | 1.0000       | 1.0000    | PASS             |
| x^2+y^2    | 2 | 1    | 1.4419 | 1.4427       | 0.9994    | PASS             |
| x^2+y^2    | 4 | 1    | 2.1623 | 2.1640       | 0.9992    | PASS             |
| x^2+y^4    | 1 | 0.75 | 0.7500 | 0.7500       | 0.7500    | PASS             |
| x^2+y^4    | 2 | 0.75 | 1.0814 | 1.0820       | 0.7496    | PASS             |
| x^2+y^4    | 4 | 0.75 | 1.6217 | 1.6230       | 0.7494    | PASS             |
| x^2y^2     | 1 | 0.5  | 0.4010 | 0.5000       | 0.4010    | reported (m = 2) |
| x^2y^2     | 2 | 0.5  | 0.5629 | 0.7213       | 0.3902    | reported (m = 2) |
| x^2y^2     | 4 | 0.5  | 0.8345 | 1.0820       | 0.3856    | reported (m = 2) |
| (ab-0.5)^2 | 1 | 0.5  | 0.5006 | 0.5000       | 0.5006    | PASS             |
| (ab-0.5)^2 | 2 | 0.5  | 0.7233 | 0.7213       | 0.5014    | PASS             |
| (ab-0.5)^2 | 4 | 0.5  | 1.0863 | 1.0820       | 0.5020    | PASS             |

## 4. Optimizer effect (plug-in κ against λ)

**PASS**

| (M,N,H)   | r | λ (Bayes) | m | κ (least squares) | κ − λ |
|-----------|---|-----------|---|-------------------|-------|
| (2, 2, 2) | 0 | 3/2       | 1 | 2                 | 1/2   |
| (2, 2, 2) | 1 | 2         | 2 | 2                 | 0     |
| (2, 2, 2) | 2 | 2         | 1 | 2                 | 0     |
| (3, 3, 3) | 0 | 7/2       | 2 | 9/2               | 1     |
| (3, 3, 3) | 1 | 4         | 1 | 9/2               | 1/2   |
| (3, 3, 3) | 2 | 9/2       | 2 | 9/2               | 0     |
| (3, 3, 3) | 3 | 9/2       | 1 | 9/2               | 0     |
| (4, 4, 4) | 0 | 6         | 1 | 8                 | 2     |
| (4, 4, 4) | 1 | 7         | 2 | 8                 | 1     |
| (4, 4, 4) | 2 | 15/2      | 1 | 8                 | 1/2   |
| (4, 4, 4) | 3 | 8         | 2 | 8                 | 0     |
| (4, 4, 4) | 4 | 8         | 1 | 8                 | 0     |
| (3, 2, 3) | 0 | 5/2       | 1 | 3                 | 1/2   |
| (3, 2, 3) | 1 | 3         | 2 | 3                 | 0     |
| (3, 2, 3) | 2 | 3         | 1 | 3                 | 0     |
| (4, 2, 4) | 0 | 7/2       | 1 | 4                 | 1/2   |
| (4, 2, 4) | 1 | 4         | 2 | 4                 | 0     |
| (4, 2, 4) | 2 | 4         | 1 | 4                 | 0     |

The least-squares learner of reduced-rank regression with `H >= min(M, N)` has the regular coefficient `MN/2` (exact finite-sample excess `MN / (2(m - N - 1))`), while the Bayes code of the same family has the Aoyagi–Watanabe `λ`. Positive gaps are exact optimizer effects. In the two-parameter germs the natural plug-in learners have `κ = λ`, so no effect appears there.

## 5. Carried-over checks (thermometer, identifiability)

**PASS**

| germ    | T      | T̂/T     | plan | λ̂       | plan | criterion |
|---------|--------|----------|------|----------|------|-----------|
| x^2y^2  | 0.01   | 0.933    | 0.93 | 0.373    | 0.37 | PASS      |
| x^2y^2  | 0.001  | 0.967    | 0.97 | 0.400    | 0.40 | PASS      |
| x^2y^2  | 0.0001 | 0.980    | 0.98 | 0.419    | 0.42 | PASS      |
| x^2+y^4 | 0.01   | 1.000000 | 1    | 0.750000 | 0.75 | PASS      |
| x^2+y^4 | 0.001  | 1.000000 | 1    | 0.750000 | 0.75 | PASS      |
| x^2+y^4 | 0.0001 | 1.000000 | 1    | 0.750000 | 0.75 | PASS      |
| x^2+y^2 | 0.01   | 1.000000 | 1    | 1.000000 | 1    | PASS      |
| x^2+y^2 | 0.001  | 1.000000 | 1    | 1.000000 | 1    | PASS      |
| x^2+y^2 | 0.0001 | 1.000000 | 1    | 1.000000 | 1    | PASS      |

Identifiability of `(log n)/n` against `1/n` (log-spaced grids):

| decades | n range        | points | VIF | SE inflation |
|---------|----------------|--------|-----|--------------|
| 1       | [1e+02, 1e+03] | 21     | 260 | 16.1×        |
| 2       | [1e+02, 1e+04] | 41     | 134 | 11.6×        |
| 3       | [1e+02, 1e+05] | 61     | 112 | 10.6×        |
