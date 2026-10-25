# Stage 1 on MIB two-digit addition (carry) in meta-llama/Llama-3.1-8B

Targets hash `494aa2b643579f2e`, schedule geometric(r=2), label-only code 7602.7 bits on 1016 coded targets.

| configuration                         | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|---------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/layer0/subspace               | 1     | 2715.2 ± nan                | 0.5 ± nan | 0.5 ± nan   | 3.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer0/subspace/map=coupling  | 1     | 2862.6 ± nan                | 0.0 ± nan | 0.0 ± nan   | 5.1 ± nan    | 1.6e+14    | -   | -            | -    | -          |
| trained/layer0/subspace/untrained     | 1     | -1038.2 ± nan               | 0.0 ± nan | 0.0 ± nan   | 6.0 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer16/subspace              | 1     | 3374.8 ± nan                | 0.5 ± nan | 0.4 ± nan   | 2.7 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer16/subspace/locus=layer0 | 1     | 2715.2 ± nan                | 0.5 ± nan | 0.5 ± nan   | 3.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer16/subspace/map=coupling | 1     | 4429.4 ± nan                | 0.7 ± nan | 0.7 ± nan   | 1.4 ± nan    | 1.3e+11    | -   | -            | -    | -          |
| trained/layer16/subspace/untrained    | 1     | -929.3 ± nan                | 0.0 ± nan | 0.0 ± nan   | 5.9 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer20/subspace              | 1     | 3767.7 ± nan                | 0.5 ± nan | 0.5 ± nan   | 2.6 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer20/subspace/locus=layer0 | 1     | 2715.2 ± nan                | 0.5 ± nan | 0.5 ± nan   | 3.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer20/subspace/map=coupling | 1     | 1875.7 ± nan                | 1.0 ± nan | 1.0 ± nan   | 0.8 ± nan    | 3.5e+11    | -   | -            | -    | -          |
| trained/layer20/subspace/untrained    | 1     | -997.2 ± nan                | 0.0 ± nan | 0.0 ± nan   | 5.9 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer24/subspace              | 1     | 3764.4 ± nan                | 0.5 ± nan | 0.5 ± nan   | 2.4 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer24/subspace/locus=layer0 | 1     | 2715.2 ± nan                | 0.5 ± nan | 0.5 ± nan   | 3.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer24/subspace/map=coupling | 1     | 5063.2 ± nan                | 0.9 ± nan | 0.9 ± nan   | 0.8 ± nan    | 3.8e+11    | -   | -            | -    | -          |
| trained/layer24/subspace/untrained    | 1     | -1035.0 ± nan               | 0.0 ± nan | 0.0 ± nan   | 6.0 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer28/subspace              | 1     | 3528.4 ± nan                | 0.5 ± nan | 0.5 ± nan   | 2.6 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer28/subspace/locus=layer0 | 1     | 2715.2 ± nan                | 0.5 ± nan | 0.5 ± nan   | 3.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer28/subspace/map=coupling | 1     | 4084.6 ± nan                | 0.9 ± nan | 0.9 ± nan   | 0.9 ± nan    | 3.1e+11    | -   | -            | -    | -          |
| trained/layer28/subspace/untrained    | 1     | -1055.6 ± nan               | 0.0 ± nan | 0.0 ± nan   | 6.0 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer32/subspace              | 1     | 3156.4 ± nan                | 0.4 ± nan | 0.5 ± nan   | 3.0 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer32/subspace/locus=layer0 | 1     | 2715.2 ± nan                | 0.5 ± nan | 0.5 ± nan   | 3.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer32/subspace/map=coupling | 1     | 414.7 ± nan                 | 0.4 ± nan | 0.4 ± nan   | 3.3 ± nan    | 2.1e+11    | -   | -            | -    | -          |
| trained/layer32/subspace/untrained    | 1     | -1083.3 ± nan               | 0.0 ± nan | 0.0 ± nan   | 6.0 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer8/subspace               | 1     | 2786.8 ± nan                | 0.5 ± nan | 0.5 ± nan   | 3.4 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer8/subspace/locus=layer0  | 1     | 2715.2 ± nan                | 0.5 ± nan | 0.5 ± nan   | 3.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer8/subspace/map=coupling  | 1     | 2817.5 ± nan                | 0.0 ± nan | 0.0 ± nan   | 5.3 ± nan    | 6.9e+15    | -   | -            | -    | -          |
| trained/layer8/subspace/untrained     | 1     | -904.0 ± nan                | 0.0 ± nan | 0.0 ± nan   | 5.8 ± nan    | 1          | -   | -            | -    | -          |
| untrained/layer24/coupling            | 1     | -2159.8 ± nan               | 0.1 ± nan | 0.0 ± nan   | 6.1 ± nan    | 7.2e+14    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

`peak evidence` is the maximum of the running log Bayes factor (an e-process under the control); `rejects α=0.01 by n` is the coded index by which every seed's running evidence first exceeds log(100) ≈ 6.6 bits, or `never`.

| hypothesis               | control               | seeds | gain G(n)     | peak evidence | rejects α=0.01 by n |
|--------------------------|-----------------------|-------|---------------|---------------|---------------------|
| trained/layer0/subspace  | subspace/map=coupling | 1     | -147.5 ± nan  | 1261.6 ± nan  | 9                   |
| trained/layer0/subspace  | subspace/untrained    | 1     | 3753.4 ± nan  | 3754.1 ± nan  | 9                   |
| trained/layer16/subspace | subspace/locus=layer0 | 1     | 659.6 ± nan   | 660.2 ± nan   | 66                  |
| trained/layer16/subspace | subspace/map=coupling | 1     | -1054.5 ± nan | 156.6 ± nan   | 9                   |
| trained/layer16/subspace | subspace/untrained    | 1     | 4304.1 ± nan  | 4304.3 ± nan  | 10                  |
| trained/layer20/subspace | subspace/locus=layer0 | 1     | 1052.5 ± nan  | 1052.5 ± nan  | 41                  |
| trained/layer20/subspace | subspace/map=coupling | 1     | 1892.0 ± nan  | 1892.0 ± nan  | 9                   |
| trained/layer20/subspace | subspace/untrained    | 1     | 4764.9 ± nan  | 4764.9 ± nan  | 9                   |
| trained/layer24/subspace | subspace/locus=layer0 | 1     | 1049.2 ± nan  | 1049.2 ± nan  | 112                 |
| trained/layer24/subspace | subspace/map=coupling | 1     | -1298.9 ± nan | 158.1 ± nan   | 33                  |
| trained/layer24/subspace | subspace/untrained    | 1     | 4799.3 ± nan  | 4799.3 ± nan  | 9                   |
| trained/layer28/subspace | subspace/locus=layer0 | 1     | 813.2 ± nan   | 813.2 ± nan   | 145                 |
| trained/layer28/subspace | subspace/map=coupling | 1     | -556.3 ± nan  | 377.4 ± nan   | 18                  |
| trained/layer28/subspace | subspace/untrained    | 1     | 4584.0 ± nan  | 4584.0 ± nan  | 9                   |
| trained/layer32/subspace | subspace/locus=layer0 | 1     | 441.2 ± nan   | 441.2 ± nan   | 39                  |
| trained/layer32/subspace | subspace/map=coupling | 1     | 2741.7 ± nan  | 2744.5 ± nan  | 9                   |
| trained/layer32/subspace | subspace/untrained    | 1     | 4239.6 ± nan  | 4239.6 ± nan  | 9                   |
| trained/layer8/subspace  | subspace/locus=layer0 | 1     | 71.6 ± nan    | 71.6 ± nan    | 189                 |
| trained/layer8/subspace  | subspace/map=coupling | 1     | -30.7 ± nan   | 1154.6 ± nan  | 9                   |
| trained/layer8/subspace  | subspace/untrained    | 1     | 3690.8 ± nan  | 3692.1 ± nan  | 9                   |