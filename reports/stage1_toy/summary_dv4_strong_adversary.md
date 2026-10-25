# Stage 1 on hierarchical equality

Targets hash `ef53ae7ab6d7890b`, schedule geometric(r=2), label-only code 1021.1 bits on 1016 coded targets.

| configuration                | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE       | unmapped err | D1 p | D2 share≤0 |
|------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----------|--------------|------|------------|
| untrained/h2/coupling        | 5     | -607.6 ± 47.2               | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.0 ± 0.1    | 4.5e+14    | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| untrained/h2/coupling/twin=8 | 5     | -453.9 ± 58.6               | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.0 ± 0.0    | 6.3e+13    | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis            | control         | seeds | gain G(n)     |
|-----------------------|-----------------|-------|---------------|
| untrained/h2/coupling | coupling/twin=8 | 5     | -153.7 ± 71.1 |