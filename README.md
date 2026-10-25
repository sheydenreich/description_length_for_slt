# bitsev: Bits of Evidence

Code for the research plan *Bits of Evidence: when description lengths expose
manufactured explanations, and what singular learning theory says about the codes*
(`research_plan_synthesis_v7_2.tex`). The implementation plan for the first six
weeks is in `PLAN.md`.

## Layout

| Layer | Package | Depends on |
|---|---|---|
| 1 | `bitsev.codes` — prequential schedules, learners, Bayes and plug-in predictors, gains, e-values | numpy, scipy |
| 1b | `bitsev.exact` — exactly solvable models (Gaussian mean, singular germs by quadrature, reduced-rank regression) | layer 1 |
| 2 | `bitsev.alignment`, `bitsev.baselines`, `bitsev.diagnostics` | torch (+ gudhi for D3) |
| 3 | `bitsev.testbeds` | transformer_lens, pyvene |
| 4 | `bitsev.experiments`, `bitsev.report` | everything above |

No layer imports from a layer above it.

## Setup

```bash
uv venv --python 3.12 ~/venvs/bitsev && source ~/venvs/bitsev/bin/activate
uv pip install -e ".[dev]"          # layer 1 and tests
uv pip install -e ".[ml]"           # + torch for layer 2
bitsev-check-env
pytest
```

## Compute

Small tests run directly on the THIN login node; anything heavy goes through SLURM
(`slurm/`). CPU work: `epyc` (many cores) or `genoa` (large memory). GPU work: `dgx`
(A100) by default, `h100` if needed. The torch build in the `bitsev` env (cu130) ships
kernels for sm_75 and newer only, so it does **not** run on the V100 `gpu` partition.
Environments on CephFS are shared across nodes, but conda envs may be node-specific;
Apptainer images live in `/u/lade/sheydenreich/fast/apptainer`.

Results are written under `results/`, which is a symlink to scratch and is not
committed. Reports under `results/<gate>/report.md` are generated from result
files, never edited by hand.

## Experiments

| Command | What it does |
|---|---|
| `bitsev-exact` | Gate G-A0 tables and verdict (`results/G-A0/report.md`, record in `reports/G-A0/`) |
| `bitsev-stage1-toy --smoke` | Stage 1 on hierarchical equality, tiny settings, on the login node |
| `sbatch slurm/stage1_toy.sbatch --seeds 5 --d-v 4 --members 4 --out results/<name>` | Full Stage 1 toy run with the H-A2 trajectory (EPYC) |
| `bitsev-stage1-toy --seeds 5 --no-diagnostics --out results/<name>` | Regenerate `summary.md` of a finished run (skips existing records) |
| `bitsev-stage1-figures results/<name>` | Gain curves, trajectory and per-block loss figures under `results/<name>/figures/` |
| `sbatch slurm/stage1_interpbench.sbatch --case 18 --variable freq --d-v 4 --members 4 --out results/<name>` | Stage 1 on an InterpBench case (verified cases: 52, 7, 18) |

Records are one JSON per (configuration, seed) under `results/<name>/records/`; every
record carries the targets hash, the schedule, the raw per-item losses, faithfulness on
each split, the map size and distortion bound, and (unless `--no-diagnostics`) the causal
abstraction error and the D1/D2 diagnostics. Summaries and figures are regenerated from
records, never edited.
