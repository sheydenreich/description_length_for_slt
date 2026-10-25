# Stage 1 on IOI in meta-llama/Llama-3.1-8B

Targets hash `927e660f3575b1b1`, schedule geometric(r=2), label-only code 4324.5 bits on 1016 coded targets.

| configuration                         | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|---------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/layer0/subspace               | 1     | -3526.5 ± nan               | 0.1 ± nan | 0.1 ± nan   | 5.4 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer0/subspace/map=coupling  | 1     | -924.8 ± nan                | 0.1 ± nan | 0.0 ± nan   | 2.9 ± nan    | 2e+13      | -   | -            | -    | -          |
| trained/layer0/subspace/untrained     | 1     | -1719.4 ± nan               | 0.0 ± nan | 0.0 ± nan   | 4.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer28/subspace              | 1     | 3018.2 ± nan                | 0.9 ± nan | 0.1 ± nan   | 0.6 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer28/subspace/locus=layer0 | 1     | -3526.5 ± nan               | 0.1 ± nan | 0.1 ± nan   | 5.4 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer28/subspace/map=coupling | 1     | 1634.8 ± nan                | 0.9 ± nan | 0.1 ± nan   | 0.7 ± nan    | 4.3e+09    | -   | -            | -    | -          |
| trained/layer28/subspace/untrained    | 1     | -537.1 ± nan                | 0.3 ± nan | 0.0 ± nan   | 3.0 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer30/subspace              | 1     | 3101.1 ± nan                | 0.9 ± nan | 0.1 ± nan   | 0.7 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer30/subspace/locus=layer0 | 1     | -3526.5 ± nan               | 0.1 ± nan | 0.1 ± nan   | 5.4 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer30/subspace/map=coupling | 1     | 1435.9 ± nan                | 0.9 ± nan | 0.1 ± nan   | 0.7 ± nan    | 2.2e+10    | -   | -            | -    | -          |
| trained/layer30/subspace/untrained    | 1     | -638.8 ± nan                | 0.3 ± nan | 0.0 ± nan   | 2.9 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer32/subspace              | 1     | 1708.8 ± nan                | 0.7 ± nan | 0.1 ± nan   | 1.3 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer32/subspace/locus=layer0 | 1     | -3526.5 ± nan               | 0.1 ± nan | 0.1 ± nan   | 5.4 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer32/subspace/map=coupling | 1     | 826.5 ± nan                 | 0.9 ± nan | 0.1 ± nan   | 0.6 ± nan    | 9.7e+09    | -   | -            | -    | -          |
| trained/layer32/subspace/untrained    | 1     | -605.5 ± nan                | 0.3 ± nan | 0.0 ± nan   | 2.8 ± nan    | 1          | -   | -            | -    | -          |
| untrained/layer30/coupling            | 1     | -1829.5 ± nan               | 0.5 ± nan | 0.0 ± nan   | 2.8 ± nan    | 2.3e+15    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

`peak evidence` is the maximum of the running log Bayes factor (an e-process under the control); `rejects α=0.01 by n` is the coded index by which every seed's running evidence first exceeds log(100) ≈ 6.6 bits, or `never`.

| hypothesis               | control               | seeds | gain G(n)     | peak evidence | rejects α=0.01 by n |
|--------------------------|-----------------------|-------|---------------|---------------|---------------------|
| trained/layer0/subspace  | subspace/map=coupling | 1     | -2601.6 ± nan | 61.1 ± nan    | 39                  |
| trained/layer0/subspace  | subspace/untrained    | 1     | -1807.1 ± nan | 0.6 ± nan     | never               |
| trained/layer28/subspace | subspace/locus=layer0 | 1     | 6544.6 ± nan  | 6544.6 ± nan  | 13                  |
| trained/layer28/subspace | subspace/map=coupling | 1     | 1383.4 ± nan  | 1391.3 ± nan  | 11                  |
| trained/layer28/subspace | subspace/untrained    | 1     | 3555.3 ± nan  | 3555.3 ± nan  | 18                  |
| trained/layer30/subspace | subspace/locus=layer0 | 1     | 6627.6 ± nan  | 6627.6 ± nan  | 9                   |
| trained/layer30/subspace | subspace/map=coupling | 1     | 1665.2 ± nan  | 1673.7 ± nan  | 9                   |
| trained/layer30/subspace | subspace/untrained    | 1     | 3739.9 ± nan  | 3739.9 ± nan  | 12                  |
| trained/layer32/subspace | subspace/locus=layer0 | 1     | 5235.3 ± nan  | 5235.3 ± nan  | 17                  |
| trained/layer32/subspace | subspace/map=coupling | 1     | 882.3 ± nan   | 1104.4 ± nan  | 11                  |
| trained/layer32/subspace | subspace/untrained    | 1     | 2314.3 ± nan  | 2314.3 ± nan  | 37                  |