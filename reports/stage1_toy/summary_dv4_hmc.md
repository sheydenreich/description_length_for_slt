# Stage 1 on hierarchical equality

Targets hash `ef53ae7ab6d7890b`, schedule geometric(r=2), label-only code 1021.1 bits on 1016 coded targets.

| configuration                    | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE       | unmapped err | D1 p | D2 share≤0 |
|----------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----------|--------------|------|------------|
| trained/h1/rotation              | 5     | 733.0 ± 0.8                 | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.2 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained/h1/rotation/locus=input  | 5     | -76.3 ± 47.5                | 0.7 ± 0.0 | 0.7 ± 0.0   | 0.7 ± 0.1    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained/h1/rotation/map=coupling | 5     | -322.2 ± 36.0               | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.9 ± 0.0    | 1.1        | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained/h1/rotation/untrained    | 5     | -13.8 ± 0.0                 | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained/h2/rotation              | 5     | 80.7 ± 43.1                 | 0.7 ± 0.0 | 0.6 ± 0.0   | 0.5 ± 0.0    | 1          | 0.4 ± 0.0 | 0.4 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained/h2/rotation/locus=input  | 5     | -76.3 ± 47.5                | 0.7 ± 0.0 | 0.7 ± 0.0   | 0.7 ± 0.1    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained/h2/rotation/map=coupling | 5     | -295.5 ± 28.4               | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.9 ± 0.1    | 1.1        | 0.4 ± 0.0 | 0.4 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained/h2/rotation/untrained    | 5     | -14.2 ± 0.0                 | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@1/h1/rotation            | 5     | -1.5 ± 0.0                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@1/h2/rotation            | 5     | -1.7 ± 0.0                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@1024/h1/rotation         | 5     | 729.7 ± 0.4                 | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.2 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained@1024/h2/rotation         | 4     | 60.1 ± 51.1                 | 0.7 ± 0.0 | 0.7 ± 0.0   | 0.7 ± 0.1    | 1          | 0.4 ± 0.0 | 0.4 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained@128/h1/rotation          | 5     | 184.3 ± 0.4                 | 0.8 ± 0.0 | 0.7 ± 0.0   | 0.5 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained@128/h2/rotation          | 4     | -269.8 ± 0.3                | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.8 ± 0.0    | 1          | 0.5 ± 0.0 | 0.4 ± 0.0    | 0.01 | 0.9 ± 0.0  |
| trained@16/h1/rotation           | 5     | 2.3 ± 0.0                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@16/h2/rotation           | 4     | 2.3 ± 0.0                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@2/h1/rotation            | 5     | 5.2 ± 0.0                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@2/h2/rotation            | 5     | 4.8 ± 0.0                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@2048/h1/rotation         | 5     | 731.8 ± 0.6                 | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.2 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained@2048/h2/rotation         | 4     | 80.3 ± 44.5                 | 0.7 ± 0.1 | 0.6 ± 0.0   | 0.7 ± 0.1    | 1          | 0.4 ± 0.0 | 0.4 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained@256/h1/rotation          | 5     | 697.0 ± 0.8                 | 1.0 ± 0.0 | 0.9 ± 0.0   | 0.2 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained@256/h2/rotation          | 4     | -140.7 ± 81.8               | 0.7 ± 0.0 | 0.6 ± 0.0   | 0.7 ± 0.1    | 1          | 0.4 ± 0.0 | 0.4 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained@32/h1/rotation           | 5     | 0.9 ± 0.0                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@32/h2/rotation           | 4     | 0.5 ± 0.0                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@4/h1/rotation            | 5     | 4.2 ± 0.1                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@4/h2/rotation            | 4     | 4.0 ± 0.0                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@512/h1/rotation          | 5     | 726.8 ± 0.3                 | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.2 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained@512/h2/rotation          | 4     | 38.2 ± 42.7                 | 0.7 ± 0.0 | 0.7 ± 0.0   | 0.6 ± 0.0    | 1          | 0.4 ± 0.0 | 0.4 ± 0.0    | 0.01 | 1.0 ± 0.0  |
| trained@64/h1/rotation           | 5     | 4.2 ± 0.1                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@64/h2/rotation           | 4     | 1.7 ± 0.1                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@8/h1/rotation            | 5     | -1.4 ± 0.0                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| trained@8/h2/rotation            | 4     | -1.3 ± 0.0                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
| untrained/h2/coupling            | 5     | -380.9 ± 34.6               | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.9 ± 0.1    | 1.1        | 0.5 ± 0.0 | 0.5 ± 0.0    | 0.01 | 0.0 ± 0.0  |
## Online probe codes

A logistic probe codes the *variable* V from the locus (noise-free, no interventions, no readout), so these lengths are a different quantity from the Stage 1 codes above and are not gains over the label-only code.

| probe              | code (bits) | coded items |
|--------------------|-------------|-------------|
| probe/trained/h1   | 26.3        | 1016        |
| probe/trained/h2   | 1.6         | 1016        |
| probe/untrained/h1 | 1243.4      | 1016        |
| probe/untrained/h2 | 1043.9      | 1016        |

## Gains against matched controls (bits, mean ± s.e. over seeds)

`peak evidence` is the maximum of the running log Bayes factor (an e-process under the control); `rejects α=0.01 by n` is the coded index by which every seed's running evidence first exceeds log(100) ≈ 6.6 bits, or `never`.

| hypothesis          | control               | seeds | gain G(n)     | peak evidence | rejects α=0.01 by n |
|---------------------|-----------------------|-------|---------------|---------------|---------------------|
| trained/h1/rotation | rotation/locus=input  | 5     | 809.3 ± 47.3  | 809.3 ± 47.3  | 35                  |
| trained/h1/rotation | rotation/map=coupling | 5     | 1055.3 ± 35.4 | 1055.3 ± 35.4 | 75                  |
| trained/h1/rotation | rotation/untrained    | 5     | 746.9 ± 0.8   | 746.9 ± 0.8   | 73                  |
| trained/h2/rotation | rotation/locus=input  | 5     | 157.0 ± 55.0  | 160.5 ± 53.3  | 976                 |
| trained/h2/rotation | rotation/map=coupling | 5     | 376.2 ± 23.6  | 376.6 ± 23.8  | 637                 |
| trained/h2/rotation | rotation/untrained    | 5     | 94.9 ± 43.1   | 110.4 ± 29.0  | never               |