# Stage 1 on InterpBench case 7

Targets hash `6fd5ea486de804ae`, schedule geometric(r=2), label-only code 2700.0 bits on 1016 coded targets.

| configuration                          | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/resid_1/rotation               | 3     | 1874.7 ± 51.5               | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.5 ± 0.1    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/locus=resid_0 | 3     | -2009.1 ± 23.1              | 0.2 ± 0.0 | 0.2 ± 0.0   | 3.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/map=coupling  | 3     | -1010.4 ± 461.6             | 0.3 ± 0.1 | 0.3 ± 0.1   | 2.4 ± 0.6    | 7.8e+05    | -   | -            | -    | -          |
| trained/resid_1/rotation/untrained     | 3     | -791.9 ± 1.3                | 0.1 ± 0.0 | 0.1 ± 0.0   | 2.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation               | 3     | 2012.0 ± 4.0                | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.3 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/locus=resid_0 | 3     | -2009.1 ± 23.1              | 0.2 ± 0.0 | 0.2 ± 0.0   | 3.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/map=coupling  | 3     | 1700.0 ± 23.2               | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.3 ± 0.0    | 2.6e+04    | -   | -            | -    | -          |
| trained/resid_2/rotation/untrained     | 3     | -802.3 ± 0.7                | 0.1 ± 0.0 | 0.1 ± 0.0   | 2.4 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/resid_2/coupling             | 3     | -865.3 ± 67.0               | 0.6 ± 0.0 | 0.6 ± 0.0   | 1.2 ± 0.0    | 3e+06      | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis               | control                | seeds | gain G(n)      |
|--------------------------|------------------------|-------|----------------|
| trained/resid_1/rotation | rotation/locus=resid_0 | 3     | 3883.8 ± 74.6  |
| trained/resid_1/rotation | rotation/map=coupling  | 3     | 2885.1 ± 466.0 |
| trained/resid_1/rotation | rotation/untrained     | 3     | 2666.6 ± 50.6  |
| trained/resid_2/rotation | rotation/locus=resid_0 | 3     | 4021.1 ± 23.2  |
| trained/resid_2/rotation | rotation/map=coupling  | 3     | 312.0 ± 23.1   |
| trained/resid_2/rotation | rotation/untrained     | 3     | 2814.3 ± 4.6   |