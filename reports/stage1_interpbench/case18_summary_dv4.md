# Stage 1 on hierarchical equality

Targets hash `39c60f6b5e9ca439`, schedule geometric(r=2), label-only code 1341.8 bits on 1016 coded targets.

| configuration                          | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/resid_1/rotation               | 3     | -1970.7 ± 176.5             | 0.9 ± 0.0 | 0.9 ± 0.0   | 1.9 ± 0.3    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/locus=resid_0 | 3     | -19792.6 ± 443.9            | 0.5 ± 0.0 | 0.5 ± 0.0   | 16.0 ± 0.9   | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/map=coupling  | 3     | -18797.1 ± 5264.9           | 0.8 ± 0.1 | 0.8 ± 0.1   | 2.4 ± 0.9    | 4e+07      | -   | -            | -    | -          |
| trained/resid_1/rotation/untrained     | 3     | -296.8 ± 1.6                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation               | 3     | -599.1 ± 9.8                | 1.0 ± 0.0 | 1.0 ± 0.0   | 1.2 ± 0.1    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/locus=resid_0 | 3     | -19792.6 ± 443.9            | 0.5 ± 0.0 | 0.5 ± 0.0   | 16.0 ± 0.9   | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/map=coupling  | 3     | -5472.2 ± 1746.8            | 1.0 ± 0.0 | 1.0 ± 0.0   | 1.8 ± 0.1    | 7.2e+04    | -   | -            | -    | -          |
| trained/resid_2/rotation/untrained     | 3     | -297.7 ± 1.3                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/resid_2/coupling             | 3     | -14261.8 ± 1347.0           | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.4 ± 0.0    | 2.9e+06    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis               | control                | seeds | gain G(n)        |
|--------------------------|------------------------|-------|------------------|
| trained/resid_1/rotation | rotation/locus=resid_0 | 3     | 17821.9 ± 308.6  |
| trained/resid_1/rotation | rotation/map=coupling  | 3     | 16826.4 ± 5415.8 |
| trained/resid_1/rotation | rotation/untrained     | 3     | -1673.9 ± 175.1  |
| trained/resid_2/rotation | rotation/locus=resid_0 | 3     | 19193.6 ± 436.5  |
| trained/resid_2/rotation | rotation/map=coupling  | 3     | 4873.1 ± 1738.7  |
| trained/resid_2/rotation | rotation/untrained     | 3     | -301.4 ± 9.6     |