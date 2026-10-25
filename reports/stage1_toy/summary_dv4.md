# Stage 1 on hierarchical equality

Targets hash `ef53ae7ab6d7890b`, schedule geometric(r=2), label-only code 1021.1 bits on 1016 coded targets.

| configuration                    | seeds | gain over label-only (bits) | IIA iid   | IIA shifted | log loss iid | Dist bound | CAE | unmapped err | D1 p | D2 share≤0 |
|----------------------------------|-------|-----------------------------|-----------|-------------|--------------|------------|-----|--------------|------|------------|
| probe/trained/h1                 | 1     | 994.8 ± nan                 | -         | -           | -            | -          | -   | -            | -    | -          |
| probe/trained/h2                 | 1     | 1019.6 ± nan                | -         | -           | -            | -          | -   | -            | -    | -          |
| probe/untrained/h1               | 1     | -222.3 ± nan                | -         | -           | -            | -          | -   | -            | -    | -          |
| probe/untrained/h2               | 1     | -22.7 ± nan                 | -         | -           | -            | -          | -   | -            | -    | -          |
| trained/h1/rotation              | 5     | -494.5 ± 52.2               | 1.0 ± 0.0 | 0.9 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/h1/rotation/locus=input  | 5     | -1214.3 ± 130.0             | 0.9 ± 0.0 | 0.8 ± 0.0   | 1.1 ± 0.2    | 1          | -   | -            | -    | -          |
| trained/h1/rotation/map=coupling | 5     | -1015.8 ± 96.2              | 1.0 ± 0.0 | 0.9 ± 0.0   | 0.8 ± 0.0    | 3.8e+02    | -   | -            | -    | -          |
| trained/h1/rotation/untrained    | 5     | -11.7 ± 0.3                 | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained/h2/rotation              | 5     | -8271.6 ± 54.3              | 0.5 ± 0.0 | 0.5 ± 0.0   | 6.0 ± 0.1    | 1          | -   | -            | -    | -          |
| trained/h2/rotation/locus=input  | 5     | -1214.3 ± 130.0             | 0.9 ± 0.0 | 0.8 ± 0.0   | 1.1 ± 0.2    | 1          | -   | -            | -    | -          |
| trained/h2/rotation/map=coupling | 5     | -668.5 ± 87.4               | 1.0 ± 0.0 | 0.9 ± 0.0   | 0.6 ± 0.0    | 4.3e+03    | -   | -            | -    | -          |
| trained/h2/rotation/untrained    | 5     | -16.0 ± 0.7                 | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@1/h1/rotation            | 5     | 0.8 ± 0.4                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@1/h2/rotation            | 5     | -2.9 ± 0.5                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@1024/h1/rotation         | 5     | -147.6 ± 61.5               | 1.0 ± 0.0 | 0.9 ± 0.0   | 0.6 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@1024/h2/rotation         | 5     | -5810.4 ± 31.3              | 0.5 ± 0.0 | 0.5 ± 0.0   | 4.5 ± 0.1    | 1          | -   | -            | -    | -          |
| trained@128/h1/rotation          | 5     | 191.9 ± 3.4                 | 0.8 ± 0.0 | 0.7 ± 0.0   | 0.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@128/h2/rotation          | 5     | -385.9 ± 6.4                | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.9 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@16/h1/rotation           | 5     | 5.1 ± 0.3                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@16/h2/rotation           | 5     | 2.0 ± 0.2                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@2/h1/rotation            | 5     | 8.6 ± 0.2                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@2/h2/rotation            | 5     | 4.5 ± 0.4                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@2048/h1/rotation         | 5     | -329.8 ± 40.3               | 1.0 ± 0.0 | 0.9 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@2048/h2/rotation         | 5     | -6980.7 ± 45.6              | 0.5 ± 0.0 | 0.5 ± 0.0   | 5.2 ± 0.1    | 1          | -   | -            | -    | -          |
| trained@256/h1/rotation          | 5     | 365.1 ± 12.8                | 1.0 ± 0.0 | 0.9 ± 0.0   | 0.4 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@256/h2/rotation          | 5     | -3255.2 ± 18.8              | 0.5 ± 0.0 | 0.5 ± 0.0   | 2.8 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@32/h1/rotation           | 5     | 1.7 ± 0.1                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@32/h2/rotation           | 5     | -1.2 ± 0.3                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@4/h1/rotation            | 5     | 9.1 ± 0.4                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@4/h2/rotation            | 5     | 4.7 ± 0.3                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@512/h1/rotation          | 5     | 15.6 ± 34.8                 | 1.0 ± 0.0 | 0.9 ± 0.0   | 0.5 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@512/h2/rotation          | 5     | -4769.0 ± 17.8              | 0.5 ± 0.0 | 0.5 ± 0.0   | 3.8 ± 0.1    | 1          | -   | -            | -    | -          |
| trained@64/h1/rotation           | 5     | 5.0 ± 1.3                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@64/h2/rotation           | 5     | -4.3 ± 0.8                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@8/h1/rotation            | 5     | 1.5 ± 0.3                   | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| trained@8/h2/rotation            | 5     | -1.1 ± 0.3                  | 0.5 ± 0.0 | 0.5 ± 0.0   | 0.7 ± 0.0    | 1          | -   | -            | -    | -          |
| untrained/h2/coupling            | 5     | -11974.6 ± 1426.5           | 0.5 ± 0.0 | 0.5 ± 0.0   | 1.1 ± 0.1    | 1.6e+05    | -   | -            | -    | -          |

## Gains against matched controls (bits, mean ± s.e. over seeds)

| hypothesis          | control               | seeds | gain G(n)       |
|---------------------|-----------------------|-------|-----------------|
| trained/h1/rotation | rotation/locus=input  | 5     | 719.8 ± 172.2   |
| trained/h1/rotation | rotation/map=coupling | 5     | 521.3 ± 87.7    |
| trained/h1/rotation | rotation/untrained    | 5     | -482.8 ± 52.1   |
| trained/h2/rotation | rotation/locus=input  | 5     | -7057.3 ± 177.2 |
| trained/h2/rotation | rotation/map=coupling | 5     | -7603.1 ± 97.2  |
| trained/h2/rotation | rotation/untrained    | 5     | -8255.6 ± 54.4  |