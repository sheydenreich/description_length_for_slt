# Stage 1 on IOI in gpt2

Targets hash `927e660f3575b1b1`, schedule geometric(r=2), label-only code 4324.5 bits on 1016 coded targets.

| configuration                                | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/layer0/subspace                      | 3     | -3517.9 ± 0.0               | 0.1 ± 0.0 | 0.1 ± 0.0   | 5.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer0/subspace/map=coupling         | 3     | -43.2 ± 5.1                 | 0.1 ± 0.0 | 0.0 ± 0.0   | 2.9 ± 0.0    | 3.3e+12    | -   | -            | -    | -          |
| trained/layer0/subspace/untrained            | 3     | -848.0 ± 0.0                | 0.0 ± 0.0 | 0.0 ± 0.0   | 3.6 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer11/subspace                     | 3     | 3384.6 ± 4.9                | 1.0 ± 0.0 | 0.1 ± 0.0   | 0.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer11/subspace/locus=layer0        | 3     | -3517.9 ± 0.0               | 0.1 ± 0.0 | 0.1 ± 0.0   | 5.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer11/subspace/map=coupling        | 3     | 2459.3 ± 16.3               | 1.0 ± 0.0 | 0.1 ± 0.0   | 0.5 ± 0.0    | 1e+06      | -   | -            | -    | -          |
| trained/layer11/subspace/untrained           | 3     | 59.2 ± 8.1                  | 0.2 ± 0.0 | 0.0 ± 0.0   | 2.9 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/layer10/coupling                   | 3     | -1985.5 ± 602.2             | 0.8 ± 0.0 | 0.0 ± 0.0   | 1.1 ± 0.0    | 9.9e+10    | -   | -            | -    | -          |
| untrained/layer10/coupling/steps=4000/twin=8 | 3     | -1774.4 ± 554.7             | 0.8 ± 0.0 | 0.0 ± 0.0   | 1.3 ± 0.1    | 1.6e+11    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

`peak evidence` is the maximum of the running log Bayes factor (an e-process under the control); `rejects α=0.01 by n` is the coded index by which every seed's running evidence first exceeds log(100) ≈ 6.6 bits, or `never`.

| hypothesis                 | control                    | seeds | gain G(n)       | peak evidence | rejects α=0.01 by n |
|----------------------------|----------------------------|-------|-----------------|---------------|---------------------|
| trained/layer0/subspace    | subspace/map=coupling      | 3     | -3474.7 ± 5.1   | 2.9 ± 0.7     | never               |
| trained/layer0/subspace    | subspace/untrained         | 3     | -2669.9 ± 0.0   | -3.9 ± 0.0    | never               |
| trained/layer11/subspace   | subspace/locus=layer0      | 3     | 6902.5 ± 4.9    | 6902.5 ± 4.9  | 13                  |
| trained/layer11/subspace   | subspace/map=coupling      | 3     | 925.3 ± 15.2    | 926.5 ± 15.2  | 11                  |
| trained/layer11/subspace   | subspace/untrained         | 3     | 3325.4 ± 5.7    | 3326.3 ± 5.4  | 20                  |
| untrained/layer10/coupling | coupling/steps=4000/twin=8 | 3     | -211.1 ± 1156.4 | 780.8 ± 632.0 | 78                  |