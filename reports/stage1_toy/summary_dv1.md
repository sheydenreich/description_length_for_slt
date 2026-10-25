# Stage 1 on hierarchical equality

Targets hash `ef53ae7ab6d7890b`, schedule geometric(r=2), label-only code 1021.1 bits on 1016 coded targets.

| configuration                    | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| probe/trained/h1                 | 1     | 994.8 ± nan                 | -         | -           | -            | -          | -   | -            | -    | -          |
| probe/trained/h2                 | 1     | 1019.6 ± nan                | -         | -           | -            | -          | -   | -            | -    | -          |
| probe/untrained/h1               | 1     | -222.3 ± nan                | -         | -           | -            | -          | -   | -            | -    | -          |
| probe/untrained/h2               | 1     | -22.7 ± nan                 | -         | -           | -            | -          | -   | -            | -    | -          |
| trained/h1/rotation              | 5     | -456.6 ± 104.1              | 1.0 ± 0.0 | 1.0 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/h1/rotation/locus=input  | 5     | -6099.8 ± 139.4             | 0.7 ± 0.0 | 0.6 ± 0.0   | 3.8 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/h1/rotation/map=coupling | 5     | -676.4 ± 43.6               | 1.0 ± 0.0 | 0.9 ± 0.0   | inf ± nan    | 1.9e+03    | -   | -            | -    | -          |
| trained/h1/rotation/untrained    | 5     | -15.0 ± 0.3                 | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/h2/rotation              | 5     | -8296.8 ± 62.8              | 0.5 ± 0.0 | 0.5 ± 0.0   | 6.1 ± 0.1    | 1          | -   | -            | -    | -          |
| trained/h2/rotation/locus=input  | 5     | -6099.8 ± 139.4             | 0.7 ± 0.0 | 0.6 ± 0.0   | 3.8 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/h2/rotation/map=coupling | 5     | -1818.2 ± 198.3             | 1.0 ± 0.0 | 0.9 ± 0.0   | 0.6 ± 0.0    | 5.9e+03    | -   | -            | -    | -          |
| trained/h2/rotation/untrained    | 5     | -16.1 ± 0.7                 | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@1/h2/rotation            | 5     | -2.9 ± 0.5                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@1024/h2/rotation         | 5     | -5823.6 ± 42.2              | 0.5 ± 0.0 | 0.5 ± 0.0   | 4.5 ± 0.1    | 1          | -   | -            | -    | -          |
| trained@128/h2/rotation          | 5     | -389.0 ± 4.5                | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.9 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@16/h2/rotation           | 5     | 2.0 ± 0.3                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@2/h2/rotation            | 5     | 4.8 ± 0.4                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@2048/h2/rotation         | 5     | -7001.2 ± 47.0              | 0.5 ± 0.0 | 0.5 ± 0.0   | 5.3 ± 0.1    | 1          | -   | -            | -    | -          |
| trained@256/h2/rotation          | 5     | -3271.3 ± 10.6              | 0.5 ± 0.0 | 0.5 ± 0.0   | 2.8 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@32/h2/rotation           | 5     | -1.2 ± 0.3                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@4/h2/rotation            | 5     | 5.0 ± 0.4                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@512/h2/rotation          | 5     | -4793.1 ± 27.9              | 0.5 ± 0.0 | 0.5 ± 0.0   | 3.8 ± 0.1    | 1          | -   | -            | -    | -          |
| trained@64/h2/rotation           | 5     | -4.1 ± 0.6                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@8/h2/rotation            | 5     | -1.1 ± 0.3                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/h2/coupling            | 5     | -11551.1 ± 842.7            | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.9 ± 0.1    | 2.7e+04    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis          | control               | seeds | gain G(n)       |
|---------------------|-----------------------|-------|-----------------|
| trained/h1/rotation | rotation/locus=input  | 5     | 5643.1 ± 85.7   |
| trained/h1/rotation | rotation/map=coupling | 5     | 219.8 ± 111.9   |
| trained/h1/rotation | rotation/untrained    | 5     | -441.6 ± 104.0  |
| trained/h2/rotation | rotation/locus=input  | 5     | -2197.1 ± 177.9 |
| trained/h2/rotation | rotation/map=coupling | 5     | -6478.6 ± 210.2 |
| trained/h2/rotation | rotation/untrained    | 5     | -8280.8 ± 63.4  |