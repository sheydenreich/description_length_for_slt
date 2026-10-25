# Stage 1 on IOI in gpt2

Targets hash `927e660f3575b1b1`, schedule geometric(r=2), label-only code 4324.5 bits on 1016 coded targets.

| configuration                         | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|---------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/layer0/subspace               | 5     | -3517.9 ± 0.0               | 0.1 ± 0.0 | 0.1 ± 0.0   | 5.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer0/subspace/map=coupling  | 5     | -26.7 ± 3.7                 | 0.1 ± 0.0 | 0.0 ± 0.0   | 2.9 ± 0.0    | 2e+12      | -   | -            | -    | -          |
| trained/layer0/subspace/untrained     | 5     | -848.0 ± 0.0                | 0.0 ± 0.0 | 0.0 ± 0.0   | 3.6 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer10/subspace              | 5     | 2974.6 ± 3.0                | 1.0 ± 0.0 | 0.1 ± 0.0   | 0.6 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer10/subspace/locus=layer0 | 5     | -3517.9 ± 0.0               | 0.1 ± 0.0 | 0.1 ± 0.0   | 5.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer10/subspace/map=coupling | 5     | 2902.7 ± 17.2               | 1.0 ± 0.0 | 0.1 ± 0.0   | 0.4 ± 0.0    | 2.3e+06    | -   | -            | -    | -          |
| trained/layer10/subspace/untrained    | 5     | 22.4 ± 1.4                  | 0.2 ± 0.0 | 0.0 ± 0.0   | 2.9 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer11/subspace              | 5     | 3301.7 ± 3.0                | 1.0 ± 0.0 | 0.1 ± 0.0   | 0.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer11/subspace/locus=layer0 | 5     | -3517.9 ± 0.0               | 0.1 ± 0.0 | 0.1 ± 0.0   | 5.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer11/subspace/map=coupling | 5     | 2896.0 ± 9.3                | 1.0 ± 0.0 | 0.1 ± 0.0   | 0.5 ± 0.0    | 2.6e+06    | -   | -            | -    | -          |
| trained/layer11/subspace/untrained    | 5     | 45.5 ± 1.4                  | 0.2 ± 0.0 | 0.0 ± 0.0   | 2.8 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer12/subspace              | 5     | 3145.0 ± 2.6                | 1.0 ± 0.0 | 0.1 ± 0.0   | 0.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer12/subspace/locus=layer0 | 5     | -3517.9 ± 0.0               | 0.1 ± 0.0 | 0.1 ± 0.0   | 5.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer12/subspace/map=coupling | 5     | 1812.7 ± 19.2               | 0.9 ± 0.0 | 0.0 ± 0.0   | 0.6 ± 0.0    | 1.6e+05    | -   | -            | -    | -          |
| trained/layer12/subspace/untrained    | 5     | 63.2 ± 1.8                  | 0.2 ± 0.0 | 0.0 ± 0.0   | 2.8 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer6/subspace               | 5     | -2492.1 ± 12.1              | 0.1 ± 0.0 | 0.0 ± 0.0   | 4.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer6/subspace/locus=layer0  | 5     | -3517.9 ± 0.0               | 0.1 ± 0.0 | 0.1 ± 0.0   | 5.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer6/subspace/map=coupling  | 5     | -2292.9 ± 110.9             | 0.1 ± 0.0 | 0.0 ± 0.0   | 3.0 ± 0.0    | 3.1e+10    | -   | -            | -    | -          |
| trained/layer6/subspace/untrained     | 5     | -84.1 ± 1.6                 | 0.2 ± 0.0 | 0.0 ± 0.0   | 3.0 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer9/subspace               | 5     | -2481.6 ± 1.7               | 0.1 ± 0.0 | 0.1 ± 0.0   | 4.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer9/subspace/locus=layer0  | 5     | -3517.9 ± 0.0               | 0.1 ± 0.0 | 0.1 ± 0.0   | 5.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/layer9/subspace/map=coupling  | 5     | -883.1 ± 22.8               | 0.6 ± 0.0 | 0.0 ± 0.0   | 1.6 ± 0.0    | 5.3e+08    | -   | -            | -    | -          |
| trained/layer9/subspace/untrained     | 5     | -3.1 ± 1.3                  | 0.2 ± 0.0 | 0.0 ± 0.0   | 2.9 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/layer6/coupling             | 5     | 60.5 ± 137.9                | 0.9 ± 0.0 | 0.0 ± 0.0   | 1.0 ± 0.1    | 1e+16      | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis               | control               | seeds | gain G(n)      |
|--------------------------|-----------------------|-------|----------------|
| trained/layer0/subspace  | subspace/map=coupling | 5     | -3491.2 ± 3.7  |
| trained/layer0/subspace  | subspace/untrained    | 5     | -2669.9 ± 0.0  |
| trained/layer10/subspace | subspace/locus=layer0 | 5     | 6492.5 ± 3.0   |
| trained/layer10/subspace | subspace/map=coupling | 5     | 71.9 ± 17.2    |
| trained/layer10/subspace | subspace/untrained    | 5     | 2952.1 ± 2.8   |
| trained/layer11/subspace | subspace/locus=layer0 | 5     | 6819.6 ± 3.0   |
| trained/layer11/subspace | subspace/map=coupling | 5     | 405.7 ± 10.0   |
| trained/layer11/subspace | subspace/untrained    | 5     | 3256.2 ± 2.6   |
| trained/layer12/subspace | subspace/locus=layer0 | 5     | 6662.9 ± 2.6   |
| trained/layer12/subspace | subspace/map=coupling | 5     | 1332.4 ± 20.2  |
| trained/layer12/subspace | subspace/untrained    | 5     | 3081.9 ± 3.6   |
| trained/layer6/subspace  | subspace/locus=layer0 | 5     | 1025.8 ± 12.1  |
| trained/layer6/subspace  | subspace/map=coupling | 5     | -199.2 ± 116.2 |
| trained/layer6/subspace  | subspace/untrained    | 5     | -2408.0 ± 12.7 |
| trained/layer9/subspace  | subspace/locus=layer0 | 5     | 1036.3 ± 1.7   |
| trained/layer9/subspace  | subspace/map=coupling | 5     | -1598.4 ± 21.7 |
| trained/layer9/subspace  | subspace/untrained    | 5     | -2478.5 ± 2.1  |