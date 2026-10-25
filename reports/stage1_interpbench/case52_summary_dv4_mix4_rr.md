# Stage 1 on InterpBench case 52

Targets hash `5db132df67c67ae7`, schedule geometric(r=2), label-only code 3535.2 bits on 1016 coded targets.

| configuration                          | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/resid_1/rotation               | 3     | -1244.1 ± 30.7              | 0.2 ± 0.1 | 0.2 ± 0.1   | 3.1 ± 0.5    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/map=coupling  | 3     | -2682.5 ± 48.6              | 0.1 ± 0.0 | 0.1 ± 0.0   | 4.3 ± 0.1    | 1.1e+05    | -   | -            | -    | -          |
| trained/resid_1/rotation/untrained     | 3     | 35.5 ± 0.1                  | 0.1 ± 0.0 | 0.1 ± 0.0   | 2.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation               | 3     | 1186.8 ± 126.4              | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.6 ± 0.1    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/locus=resid_1 | 3     | -1244.1 ± 30.7              | 0.2 ± 0.1 | 0.2 ± 0.1   | 3.1 ± 0.5    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/map=coupling  | 3     | 1985.8 ± 267.6              | 0.8 ± 0.2 | 0.8 ± 0.2   | 0.6 ± 0.2    | 3.5e+03    | -   | -            | -    | -          |
| trained/resid_2/rotation/untrained     | 3     | 35.0 ± 0.1                  | 0.1 ± 0.0 | 0.1 ± 0.0   | 2.4 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/resid_2/coupling             | 3     | -93.1 ± 26.4                | 0.6 ± 0.0 | 0.6 ± 0.0   | 1.2 ± 0.0    | 1.4e+06    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

`peak evidence` is the maximum of the running log Bayes factor (an e-process under the control); `rejects α=0.01 by n` is the coded index by which every seed's running evidence first exceeds log(100) ≈ 6.6 bits, or `never`.

| hypothesis               | control                | seeds | gain G(n)      | peak evidence  | rejects α=0.01 by n |
|--------------------------|------------------------|-------|----------------|----------------|---------------------|
| trained/resid_1/rotation | rotation/map=coupling  | 3     | 1438.4 ± 76.2  | 1441.3 ± 74.1  | 24                  |
| trained/resid_1/rotation | rotation/untrained     | 3     | -1279.6 ± 30.7 | -2.2 ± 0.5     | never               |
| trained/resid_2/rotation | rotation/locus=resid_1 | 3     | 2430.9 ± 151.1 | 2432.3 ± 150.7 | 345                 |
| trained/resid_2/rotation | rotation/map=coupling  | 3     | -799.0 ± 288.0 | 26.1 ± 21.7    | never               |
| trained/resid_2/rotation | rotation/untrained     | 3     | 1151.8 ± 126.4 | 1153.0 ± 126.7 | 634                 |