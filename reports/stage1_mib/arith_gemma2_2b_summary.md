# Stage 1 on MIB two-digit addition (carry) in google/gemma-2-2b

Targets hash `15515a62f70acec9`, schedule geometric(r=2), label-only code 2280.9 bits on 1016 coded targets.

| configuration                         | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|---------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| trained/layer0/subspace               | 1     | 718.8 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer0/subspace/map=coupling  | 1     | 685.7 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 0.6 ± nan    | 2.3e+11    | -   | -            | -    | -          |
| trained/layer0/subspace/untrained     | 1     | -1354.4 ± nan               | 0.1 ± nan | 0.1 ± nan   | 2.5 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer12/subspace              | 1     | 770.4 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.0 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer12/subspace/locus=layer0 | 1     | 718.8 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer12/subspace/map=coupling | 1     | 918.5 ± nan                 | 0.9 ± nan | 0.9 ± nan   | 0.6 ± nan    | 9.7e+09    | -   | -            | -    | -          |
| trained/layer12/subspace/untrained    | 1     | -541.9 ± nan                | 0.5 ± nan | 0.5 ± nan   | 1.7 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer16/subspace              | 1     | 887.9 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 0.9 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer16/subspace/locus=layer0 | 1     | 718.8 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer16/subspace/map=coupling | 1     | 814.0 ± nan                 | 0.9 ± nan | 0.8 ± nan   | 0.6 ± nan    | 2.9e+09    | -   | -            | -    | -          |
| trained/layer16/subspace/untrained    | 1     | -765.4 ± nan                | 0.4 ± nan | 0.4 ± nan   | 1.8 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer20/subspace              | 1     | 1472.7 ± nan                | 1.0 ± nan | 0.9 ± nan   | 0.4 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer20/subspace/locus=layer0 | 1     | 718.8 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer20/subspace/map=coupling | 1     | 1218.1 ± nan                | 1.0 ± nan | 1.0 ± nan   | 0.4 ± nan    | 5.3e+09    | -   | -            | -    | -          |
| trained/layer20/subspace/untrained    | 1     | -877.9 ± nan                | 0.4 ± nan | 0.4 ± nan   | 1.9 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer24/subspace              | 1     | 1116.1 ± nan                | 0.8 ± nan | 0.8 ± nan   | 0.7 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer24/subspace/locus=layer0 | 1     | 718.8 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer24/subspace/map=coupling | 1     | 1053.8 ± nan                | 1.0 ± nan | 0.9 ± nan   | 0.4 ± nan    | 2.8e+10    | -   | -            | -    | -          |
| trained/layer24/subspace/untrained    | 1     | -1081.7 ± nan               | 0.3 ± nan | 0.3 ± nan   | 2.2 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer26/subspace              | 1     | 527.7 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer26/subspace/locus=layer0 | 1     | 718.8 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer26/subspace/map=coupling | 1     | 642.8 ± nan                 | 0.9 ± nan | 0.9 ± nan   | 0.6 ± nan    | 9.6e+09    | -   | -            | -    | -          |
| trained/layer26/subspace/untrained    | 1     | -1320.0 ± nan               | 0.3 ± nan | 0.3 ± nan   | 2.3 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer8/subspace               | 1     | 743.5 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer8/subspace/locus=layer0  | 1     | 718.8 ± nan                 | 0.8 ± nan | 0.8 ± nan   | 1.1 ± nan    | 1          | -   | -            | -    | -          |
| trained/layer8/subspace/map=coupling  | 1     | 799.9 ± nan                 | 0.9 ± nan | 0.9 ± nan   | 0.6 ± nan    | 9.5e+08    | -   | -            | -    | -          |
| trained/layer8/subspace/untrained     | 1     | -438.7 ± nan                | 0.5 ± nan | 0.5 ± nan   | 1.7 ± nan    | 1          | -   | -            | -    | -          |
| untrained/layer20/coupling            | 1     | -1074.8 ± nan               | 0.6 ± nan | 0.6 ± nan   | 1.5 ± nan    | 1.6e+18    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

`peak evidence` is the maximum of the running log Bayes factor (an e-process under the control); `rejects α=0.01 by n` is the coded index by which every seed's running evidence first exceeds log(100) ≈ 6.6 bits, or `never`.

| hypothesis               | control               | seeds | gain G(n)    | peak evidence | rejects α=0.01 by n |
|--------------------------|-----------------------|-------|--------------|---------------|---------------------|
| trained/layer0/subspace  | subspace/map=coupling | 1     | 33.1 ± nan   | 264.2 ± nan   | 52                  |
| trained/layer0/subspace  | subspace/untrained    | 1     | 2073.2 ± nan | 2076.1 ± nan  | 10                  |
| trained/layer12/subspace | subspace/locus=layer0 | 1     | 51.6 ± nan   | 52.1 ± nan    | 132                 |
| trained/layer12/subspace | subspace/map=coupling | 1     | -148.1 ± nan | 110.9 ± nan   | 23                  |
| trained/layer12/subspace | subspace/untrained    | 1     | 1312.4 ± nan | 1316.2 ± nan  | 11                  |
| trained/layer16/subspace | subspace/locus=layer0 | 1     | 169.1 ± nan  | 171.8 ± nan   | 27                  |
| trained/layer16/subspace | subspace/map=coupling | 1     | 73.9 ± nan   | 155.9 ± nan   | 9                   |
| trained/layer16/subspace | subspace/untrained    | 1     | 1653.4 ± nan | 1656.8 ± nan  | 11                  |
| trained/layer20/subspace | subspace/locus=layer0 | 1     | 753.9 ± nan  | 753.9 ± nan   | 16                  |
| trained/layer20/subspace | subspace/map=coupling | 1     | 254.6 ± nan  | 255.9 ± nan   | 18                  |
| trained/layer20/subspace | subspace/untrained    | 1     | 2350.6 ± nan | 2350.6 ± nan  | 11                  |
| trained/layer24/subspace | subspace/locus=layer0 | 1     | 397.3 ± nan  | 397.3 ± nan   | 59                  |
| trained/layer24/subspace | subspace/map=coupling | 1     | 62.3 ± nan   | 166.4 ± nan   | 9                   |
| trained/layer24/subspace | subspace/untrained    | 1     | 2197.8 ± nan | 2197.8 ± nan  | 11                  |
| trained/layer26/subspace | subspace/locus=layer0 | 1     | -191.1 ± nan | 5.8 ± nan     | never               |
| trained/layer26/subspace | subspace/map=coupling | 1     | -115.1 ± nan | 320.0 ± nan   | 9                   |
| trained/layer26/subspace | subspace/untrained    | 1     | 1847.7 ± nan | 1848.5 ± nan  | 10                  |
| trained/layer8/subspace  | subspace/locus=layer0 | 1     | 24.7 ± nan   | 25.2 ± nan    | 189                 |
| trained/layer8/subspace  | subspace/map=coupling | 1     | -56.4 ± nan  | 105.5 ± nan   | 9                   |
| trained/layer8/subspace  | subspace/untrained    | 1     | 1182.2 ± nan | 1186.3 ± nan  | 11                  |