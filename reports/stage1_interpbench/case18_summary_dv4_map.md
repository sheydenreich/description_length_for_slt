# Stage 1 on InterpBench case 18

Targets hash `39c60f6b5e9ca439`, schedule geometric(r=2), label-only code 1341.8 bits on 1016 coded targets.

| configuration                          | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/resid_1/subspace               | 3     | 356.6 ± 97.1                | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.3 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/subspace/locus=resid_0 | 3     | -1961.7 ± 41.2              | 0.5 ± 0.0 | 0.5 ± 0.0   | 2.1 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/subspace/map=coupling  | 3     | -1927.8 ± 36.5              | 0.4 ± 0.0 | 0.4 ± 0.0   | 2.3 ± 0.0    | 1.9e+07    | -   | -            | -    | -          |
| trained/resid_1/subspace/untrained     | 3     | -294.6 ± 0.0                | 0.3 ± 0.0 | 0.3 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/subspace               | 3     | 1051.5 ± 33.8               | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.2 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/subspace/locus=resid_0 | 3     | -1961.7 ± 41.2              | 0.5 ± 0.0 | 0.5 ± 0.0   | 2.1 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/subspace/map=coupling  | 3     | -1982.5 ± 0.0               | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.2 ± 0.0    | 4.3e+05    | -   | -            | -    | -          |
| trained/resid_2/subspace/untrained     | 3     | -295.8 ± 0.0                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/resid_2/coupling             | 3     | -1693.4 ± 106.3             | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.8 ± 0.3    | 4.3e+05    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis               | control                | seeds | gain G(n)      |
|--------------------------|------------------------|-------|----------------|
| trained/resid_1/subspace | subspace/locus=resid_0 | 3     | 2318.3 ± 99.0  |
| trained/resid_1/subspace | subspace/map=coupling  | 3     | 2284.4 ± 119.4 |
| trained/resid_1/subspace | subspace/untrained     | 3     | 651.2 ± 97.2   |
| trained/resid_2/subspace | subspace/locus=resid_0 | 3     | 3013.1 ± 75.0  |
| trained/resid_2/subspace | subspace/map=coupling  | 3     | 3033.9 ± 33.8  |
| trained/resid_2/subspace | subspace/untrained     | 3     | 1347.2 ± 33.8  |