# Stage 1 on InterpBench case 18

Targets hash `39c60f6b5e9ca439`, schedule geometric(r=2), label-only code 1341.8 bits on 1016 coded targets.

| configuration                          | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/resid_1/rotation               | 3     | 635.6 ± 9.3                 | 0.9 ± 0.0 | 0.9 ± 0.0   | 0.3 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/locus=resid_0 | 3     | -963.4 ± 21.1               | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.6 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_1/rotation/map=coupling  | 3     | -412.4 ± 84.4               | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.3 ± 0.0    | 1.5e+07    | -   | -            | -    | -          |
| trained/resid_1/rotation/untrained     | 3     | -295.2 ± 0.5                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation               | 3     | 982.6 ± 11.2                | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.2 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/locus=resid_0 | 3     | -963.4 ± 21.1               | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.6 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/resid_2/rotation/map=coupling  | 3     | 53.7 ± 146.5                | 0.8 ± 0.1 | 0.8 ± 0.1   | 0.5 ± 0.1    | 1.3e+06    | -   | -            | -    | -          |
| trained/resid_2/rotation/untrained     | 3     | -295.5 ± 0.7                | 0.4 ± 0.0 | 0.4 ± 0.0   | 1.1 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/resid_2/coupling             | 3     | -437.3 ± 31.3               | 0.6 ± 0.0 | 0.6 ± 0.0   | 0.9 ± 0.0    | 2.9e+06    | -   | -            | -    | -          |
| untrained/resid_2/coupling/twin=8      | 3     | -504.2 ± 8.5                | 0.6 ± 0.0 | 0.6 ± 0.0   | 0.9 ± 0.0    | 2.7e+06    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

`peak evidence` is the maximum of the running log Bayes factor (an e-process under the control); `rejects α=0.01 by n` is the coded index by which every seed's running evidence first exceeds log(100) ≈ 6.6 bits, or `never`.

| hypothesis                 | control                | seeds | gain G(n)     | peak evidence | rejects α=0.01 by n |
|----------------------------|------------------------|-------|---------------|---------------|---------------------|
| trained/resid_1/rotation   | rotation/locus=resid_0 | 3     | 1599.0 ± 20.3 | 1599.0 ± 20.3 | 135                 |
| trained/resid_1/rotation   | rotation/map=coupling  | 3     | 1048.0 ± 82.8 | 1048.0 ± 82.8 | 199                 |
| trained/resid_1/rotation   | rotation/untrained     | 3     | 930.8 ± 9.0   | 930.8 ± 9.0   | 229                 |
| trained/resid_2/rotation   | rotation/locus=resid_0 | 3     | 1946.1 ± 10.7 | 1946.1 ± 10.7 | 35                  |
| trained/resid_2/rotation   | rotation/map=coupling  | 3     | 928.9 ± 148.2 | 932.7 ± 148.2 | 24                  |
| trained/resid_2/rotation   | rotation/untrained     | 3     | 1278.1 ± 10.5 | 1280.5 ± 10.4 | 83                  |
| untrained/resid_2/coupling | coupling/twin=8        | 3     | 67.0 ± 23.6   | 73.0 ± 24.8   | 969                 |