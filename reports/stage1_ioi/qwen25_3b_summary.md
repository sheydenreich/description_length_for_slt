# Stage 1 on IOI in Qwen/Qwen2.5-3B

Targets hash `927e660f3575b1b1`, schedule geometric(r=2), label-only code 4324.5 bits on 1016 coded targets.

| configuration                         | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|---------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/layer0/subspace               | 1     | -3548.4 ± nan               | 0.1 ± nan | 0.1 ± nan   | 5.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer0/subspace/map=coupling  | 1     | -918.3 ± nan                | 0.1 ± nan | 0.0 ± nan   | 3.1 ± nan    | 2.3e+09    | -   | -            | -    | -          |
| trained/layer0/subspace/untrained     | 1     | -1200.9 ± nan               | 0.0 ± nan | 0.0 ± nan   | 3.8 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer32/subspace              | 1     | -444.7 ± nan                | 0.2 ± nan | 0.1 ± nan   | 3.0 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer32/subspace/locus=layer0 | 1     | -3548.4 ± nan               | 0.1 ± nan | 0.1 ± nan   | 5.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer32/subspace/map=coupling | 1     | -666.9 ± nan                | 0.7 ± nan | 0.0 ± nan   | 1.6 ± nan    | 6.8e+08    | -   | -            | -    | -          |
| trained/layer32/subspace/untrained    | 1     | 183.0 ± nan                 | 0.4 ± nan | 0.0 ± nan   | 2.4 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer34/subspace              | 1     | 3129.2 ± nan                | 0.9 ± nan | 0.1 ± nan   | 0.7 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer34/subspace/locus=layer0 | 1     | -3548.4 ± nan               | 0.1 ± nan | 0.1 ± nan   | 5.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer34/subspace/map=coupling | 1     | 1689.8 ± nan                | 0.9 ± nan | 0.1 ± nan   | 0.8 ± nan    | 9e+07      | -   | -            | -    | -          |
| trained/layer34/subspace/untrained    | 1     | 222.7 ± nan                 | 0.4 ± nan | 0.0 ± nan   | 2.4 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer36/subspace              | 1     | 2701.9 ± nan                | 0.9 ± nan | 0.1 ± nan   | 0.7 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer36/subspace/locus=layer0 | 1     | -3548.4 ± nan               | 0.1 ± nan | 0.1 ± nan   | 5.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer36/subspace/map=coupling | 1     | 1384.9 ± nan                | 0.9 ± nan | 0.0 ± nan   | 0.7 ± nan    | 8.5e+06    | -   | -            | -    | -          |
| trained/layer36/subspace/untrained    | 1     | 188.4 ± nan                 | 0.4 ± nan | 0.0 ± nan   | 2.4 ± nan    | 1          | -   | -            | -    | -          |
| untrained/layer34/coupling            | 1     | -1722.8 ± nan               | 0.6 ± nan | 0.0 ± nan   | 2.3 ± nan    | 6.6e+13    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

`peak evidence` is the maximum of the running log Bayes factor (an e-process under the control); `rejects α=0.01 by n` is the coded index by which every seed's running evidence first exceeds log(100) ≈ 6.6 bits, or `never`.

| hypothesis               | control               | seeds | gain G(n)     | peak evidence | rejects α=0.01 by n |
|--------------------------|-----------------------|-------|---------------|---------------|---------------------|
| trained/layer0/subspace  | subspace/map=coupling | 1     | -2630.0 ± nan | 10.1 ± nan    | 11                  |
| trained/layer0/subspace  | subspace/untrained    | 1     | -2347.5 ± nan | -0.8 ± nan    | never               |
| trained/layer32/subspace | subspace/locus=layer0 | 1     | 3103.7 ± nan  | 3103.7 ± nan  | 13                  |
| trained/layer32/subspace | subspace/map=coupling | 1     | 222.2 ± nan   | 535.0 ± nan   | 19                  |
| trained/layer32/subspace | subspace/untrained    | 1     | -627.7 ± nan  | 4.5 ± nan     | never               |
| trained/layer34/subspace | subspace/locus=layer0 | 1     | 6677.6 ± nan  | 6677.6 ± nan  | 13                  |
| trained/layer34/subspace | subspace/map=coupling | 1     | 1439.3 ± nan  | 1439.3 ± nan  | 11                  |
| trained/layer34/subspace | subspace/untrained    | 1     | 2906.4 ± nan  | 2906.4 ± nan  | 20                  |
| trained/layer36/subspace | subspace/locus=layer0 | 1     | 6250.3 ± nan  | 6250.3 ± nan  | 9                   |
| trained/layer36/subspace | subspace/map=coupling | 1     | 1317.1 ± nan  | 1318.8 ± nan  | 9                   |
| trained/layer36/subspace | subspace/untrained    | 1     | 2513.5 ± nan  | 2513.5 ± nan  | 11                  |