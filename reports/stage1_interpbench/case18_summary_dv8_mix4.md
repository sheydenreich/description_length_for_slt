# Stage 1 on InterpBench case 18

Targets hash `39c60f6b5e9ca439`, schedule geometric(r=2), label-only code 1341.8 bits on 1016 coded targets.

| configuration                          | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/resid_1/rotation               | 3     | -16.9 ± 45.0                | 0.9 ± 0.0 | 0.9 ± 0.0   | 0.6 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/locus=resid_0 | 3     | -4358.0 ± 155.3             | 0.5 ± 0.0 | 0.5 ± 0.0   | 4.1 ± 0.3    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/map=coupling  | 3     | -203.2 ± 103.5              | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.4 ± 0.0    | 1.6e+07    | -   | -            | -    | -          |
| trained/resid_1/rotation/untrained     | 3     | -296.2 ± 0.4                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation               | 3     | -92.7 ± 8.4                 | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.9 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/locus=resid_0 | 3     | -4358.0 ± 155.3             | 0.5 ± 0.0 | 0.5 ± 0.0   | 4.1 ± 0.3    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/map=coupling  | 3     | -383.2 ± 24.4               | 1.0 ± 0.0 | 1.0 ± 0.0   | 1.0 ± 0.0    | 5.6e+04    | -   | -            | -    | -          |
| trained/resid_2/rotation/untrained     | 3     | -296.9 ± 0.6                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/resid_2/coupling             | 3     | -3396.7 ± 219.8             | 0.6 ± 0.0 | 0.6 ± 0.0   | 1.0 ± 0.0    | 5.4e+06    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis               | control                | seeds | gain G(n)      |
|--------------------------|------------------------|-------|----------------|
| trained/resid_1/rotation | rotation/locus=resid_0 | 3     | 4341.1 ± 112.4 |
| trained/resid_1/rotation | rotation/map=coupling  | 3     | 186.3 ± 132.7  |
| trained/resid_1/rotation | rotation/untrained     | 3     | 279.3 ± 45.4   |
| trained/resid_2/rotation | rotation/locus=resid_0 | 3     | 4265.3 ± 160.9 |
| trained/resid_2/rotation | rotation/map=coupling  | 3     | 290.4 ± 23.0   |
| trained/resid_2/rotation | rotation/untrained     | 3     | 204.2 ± 8.1    |