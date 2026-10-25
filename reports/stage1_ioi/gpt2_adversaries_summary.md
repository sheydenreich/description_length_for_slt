# Stage 1 on IOI in gpt2

Targets hash `927e660f3575b1b1`, schedule geometric(r=2), label-only code 4324.5 bits on 1016 coded targets.

| configuration                                | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| untrained/layer10/coupling/steps=4000        | 3     | 704.2 ± 62.6                | 0.9 ± 0.0 | 0.0 ± 0.0   | 0.9 ± 0.0    | 4.1e+16    | -   | -            | -    | -          |
| untrained/layer10/coupling/steps=4000/twin=8 | 3     | 355.0 ± 142.1               | 0.8 ± 0.0 | 0.0 ± 0.0   | 1.0 ± 0.0    | 3.2e+10    | -   | -            | -    | -          |
| untrained/layer11/coupling/steps=4000        | 3     | 234.0 ± 161.5               | 0.9 ± 0.0 | 0.0 ± 0.0   | 0.8 ± 0.0    | 1.8e+11    | -   | -            | -    | -          |
| untrained/layer12/coupling/steps=4000        | 3     | -1007.6 ± 337.7             | 0.8 ± 0.0 | 0.0 ± 0.0   | 0.9 ± 0.0    | 1e+09      | -   | -            | -    | -          |
| untrained/layer6/coupling/steps=16000        | 2     | -743.6 ± 95.0               | 0.8 ± 0.1 | 0.0 ± 0.0   | 1.4 ± 0.2    | 1.3e+20    | -   | -            | -    | -          |
| untrained/layer9/coupling/steps=4000         | 3     | 290.2 ± 43.2                | 0.9 ± 0.0 | 0.0 ± 0.0   | 0.9 ± 0.0    | 7.2e+11    | -   | -            | -    | -          |
| untrained/layer9/coupling/steps=4000/twin=8  | 3     | 279.8 ± 110.1               | 0.8 ± 0.0 | 0.0 ± 0.0   | 1.0 ± 0.0    | 1e+11      | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis                            | control                    | seeds | gain G(n)     |
|---------------------------------------|----------------------------|-------|---------------|
| untrained/layer10/coupling/steps=4000 | coupling/steps=4000/twin=8 | 3     | 349.1 ± 194.8 |
| untrained/layer9/coupling/steps=4000  | coupling/steps=4000/twin=8 | 3     | 10.5 ± 82.3   |