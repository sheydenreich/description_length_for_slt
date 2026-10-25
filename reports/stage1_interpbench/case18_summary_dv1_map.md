# Stage 1 on InterpBench case 18

Targets hash `39c60f6b5e9ca439`, schedule geometric(r=2), label-only code 1341.8 bits on 1016 coded targets.

| configuration                          | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/resid_1/subspace               | 3     | -496.9 ± 197.5              | 0.8 ± 0.0 | 0.8 ± 0.0   | 0.8 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/subspace/locus=resid_0 | 3     | -2030.5 ± 13.0              | 0.4 ± 0.0 | 0.4 ± 0.0   | 2.3 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/subspace/map=coupling  | 3     | -2021.5 ± 7.8               | 0.4 ± 0.0 | 0.4 ± 0.0   | 2.5 ± 0.1    | 8.7e+07    | -   | -            | -    | -          |
| trained/resid_1/subspace/untrained     | 3     | -294.0 ± 0.0                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/subspace               | 3     | 345.0 ± 4.1                 | 0.9 ± 0.0 | 0.9 ± 0.0   | 0.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/subspace/locus=resid_0 | 3     | -2030.5 ± 13.0              | 0.4 ± 0.0 | 0.4 ± 0.0   | 2.3 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/subspace/map=coupling  | 3     | -1982.5 ± 0.0               | 0.7 ± 0.1 | 0.7 ± 0.1   | 1.2 ± 0.5    | 1.4e+06    | -   | -            | -    | -          |
| trained/resid_2/subspace/untrained     | 3     | -293.9 ± 0.0                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/resid_2/coupling             | 3     | -1069.0 ± 103.9             | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.2 ± 0.1    | 1.2e+06    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis               | control                | seeds | gain G(n)      |
|--------------------------|------------------------|-------|----------------|
| trained/resid_1/subspace | subspace/locus=resid_0 | 3     | 1533.6 ± 192.0 |
| trained/resid_1/subspace | subspace/map=coupling  | 3     | 1524.6 ± 196.2 |
| trained/resid_1/subspace | subspace/untrained     | 3     | -202.9 ± 197.5 |
| trained/resid_2/subspace | subspace/locus=resid_0 | 3     | 2375.5 ± 9.0   |
| trained/resid_2/subspace | subspace/map=coupling  | 3     | 2327.5 ± 4.1   |
| trained/resid_2/subspace | subspace/untrained     | 3     | 638.9 ± 4.1    |