# Stage 1 on InterpBench case 18

Targets hash `39c60f6b5e9ca439`, schedule geometric(r=2), label-only code 1341.8 bits on 1016 coded targets.

| configuration                          | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/resid_1/rotation               | 3     | 718.9 ± 21.8                | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.2 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/locus=resid_0 | 3     | -894.3 ± 22.3               | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/map=coupling  | 3     | -642.0 ± 78.1               | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.2 ± 0.1    | 3.3e+07    | -   | -            | -    | -          |
| trained/resid_1/rotation/untrained     | 3     | -294.3 ± 0.3                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation               | 3     | 995.9 ± 7.7                 | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.2 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/locus=resid_0 | 3     | -894.3 ± 22.3               | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/map=coupling  | 3     | 182.7 ± 54.8                | 0.8 ± 0.2 | 0.8 ± 0.2   | 0.3 ± 0.1    | 9.9e+07    | -   | -            | -    | -          |
| trained/resid_2/rotation/untrained     | 3     | -295.5 ± 0.7                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/resid_2/coupling             | 3     | -429.5 ± 32.9               | 0.6 ± 0.0 | 0.6 ± 0.0   | 0.9 ± 0.0    | 5.8e+06    | -   | -            | -    | -          |
| untrained/resid_2/coupling/twin=8      | 3     | -413.9 ± 74.3               | 0.6 ± 0.0 | 0.6 ± 0.0   | 0.8 ± 0.0    | 7.1e+06    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

`peak evidence` is the maximum of the running log Bayes factor (an e-process under the control); `rejects α=0.01 by n` is the coded index by which every seed's running evidence first exceeds log(100) ≈ 6.6 bits, or `never`.

| hypothesis                 | control                | seeds | gain G(n)     | peak evidence | rejects α=0.01 by n |
|----------------------------|------------------------|-------|---------------|---------------|---------------------|
| trained/resid_1/rotation   | rotation/locus=resid_0 | 3     | 1613.2 ± 18.2 | 1615.0 ± 17.2 | 119                 |
| trained/resid_1/rotation   | rotation/map=coupling  | 3     | 1360.9 ± 80.9 | 1362.2 ± 79.7 | 90                  |
| trained/resid_1/rotation   | rotation/untrained     | 3     | 1013.2 ± 21.6 | 1014.9 ± 20.9 | 228                 |
| trained/resid_2/rotation   | rotation/locus=resid_0 | 3     | 1890.2 ± 22.7 | 1892.6 ± 22.4 | 49                  |
| trained/resid_2/rotation   | rotation/map=coupling  | 3     | 813.2 ± 62.4  | 815.7 ± 63.4  | 44                  |
| trained/resid_2/rotation   | rotation/untrained     | 3     | 1291.3 ± 7.0  | 1293.6 ± 6.8  | 69                  |
| untrained/resid_2/coupling | coupling/twin=8        | 3     | -15.7 ± 42.0  | 37.7 ± 22.1   | never               |