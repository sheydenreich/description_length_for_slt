"""Which map specification reaches a faithful IOI alignment at all?

The first GPT-2 run coded every subspace hypothesis at about 7 bits per target: the maps
never moved the prediction off the base name. This sweep fits single maps (no prequential
code) on a fixed target set for a grid of locus, subspace width, learning rate and steps,
and reports held-out faithfulness, so the learner specification of ``bitsev-stage1-ioi``
can be fixed from evidence. Run on a GPU:
  sbatch --export=ALL,HF_HOME=$HOME/scratch/hf_cache slurm/gpu.sbatch \
      python -m bitsev.experiments.ioi_map_sweep --out results/ioi_map_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from bitsev.alignment import maps as M
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import GradientAlignmentLearner, TrainingBudget
from bitsev.alignment.score import faithfulness
from bitsev.experiments import run_stage1_ioi as R


def _factory(family: str, width: int, d_v: int):
    if family == "subspace":
        return lambda: M.SubspaceMap(width, d_v)
    return lambda: M.AdditiveCoupling(width, d_v, n_blocks=4, width=512)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--layers", type=int, nargs="+", default=[6, 9, 10, 11, 12])
    p.add_argument("--d-vs", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-2])
    p.add_argument("--steps", type=int, nargs="+", default=[400, 2000])
    p.add_argument("--n-targets", type=int, default=512)
    p.add_argument("--out", type=Path, default=Path("results/ioi_map_sweep"))
    args = p.parse_args(argv)
    ioi_args = R.parse(
        [
            "--model", args.model, "--dtype", args.dtype, "--layers", *map(str, args.layers),
            "--n-targets", str(args.n_targets), "--n-eval", "500",
        ]
    )  # fmt: skip
    net, alg, pool, targets, evals, _ = R.build(ioi_args)
    device = ioi_args.device
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for locus in net.loci:
        cache = HiddenCache.build(net, locus, pool, device)
        specs = [
            ("subspace", d_v, lr, steps)
            for d_v in args.d_vs
            for lr in args.lrs
            for steps in args.steps
        ]
        specs += [("coupling", 16, lr, 400) for lr in args.lrs]
        for family, d_v, lr, steps in specs:
            t0 = time.time()
            learner = GradientAlignmentLearner(
                cache,
                _factory(family, cache.width, d_v),
                TrainingBudget(steps, lr),
                seed=0,
                family_name=family,
                channel=targets.channel,
            )
            pred = learner.fit(targets.pairs, targets.y)
            train = float(-pred.log_prob(targets.pairs, targets.y).mean())
            f = {k: faithfulness(pred, t) for k, t in evals.items()}
            row = {
                "locus": locus,
                "family": family,
                "d_v": d_v,
                "lr": lr,
                "steps": steps,
                "train_nats": train,
                "seconds": time.time() - t0,
            }
            for k, v in f.items():
                row |= {
                    f"{k}_log_loss": v.log_loss,
                    f"{k}_iia": v.iia,
                    f"{k}_ece": v.expected_calibration_error,
                }
            rows.append(row)
            print(
                f"{locus:<8} {family:<8} d_v={d_v:<3} lr={lr:<6g} steps={steps:<5} "
                f"train={train:.2f} nats  iid: loss={f['iid'].log_loss:.2f} "
                f"IIA={f['iid'].iia:.2f} ECE={f['iid'].expected_calibration_error:.2f}  "
                f"shifted IIA={f['shifted'].iia:.2f}  ({row['seconds']:.0f}s)",
                flush=True,
            )
            (args.out / "sweep.json").write_text(json.dumps(rows, indent=1))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
