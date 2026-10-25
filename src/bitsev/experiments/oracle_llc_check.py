"""Cross-check: SGLD local learning coefficient vs the prequential regret slope on the oracle.

For each ``k`` (readable directions), fit the MAP subspace on the full target set and
estimate the LLC from the loss landscape; compare with ``k / 2`` and with the regret
slopes of ``oracle_lambda_sweep`` (``reports/oracle/effective_lambda_sweep.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from bitsev.alignment import maps as M
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import LaplaceAlignmentLearner, LaplaceBudget
from bitsev.alignment.llc import local_learning_coefficient
from bitsev.alignment.targets import make_targets
from bitsev.testbeds.oracle import make_oracle


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=12)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 7, 11])
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 1e-4])
    p.add_argument("--gammas", type=float, nargs="+", default=[10.0, 100.0])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/oracle_llc_check"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    rows = []
    for k in args.ks:
        world = make_oracle(args.d, 3 * args.n, args.eps, seed=0, k_relevant=k)
        cache = HiddenCache.build(world.network, "h1", world.pool)
        t = make_targets(world.algorithm, world.pool, args.n, "V", args.eps, seed=300)
        learner = LaplaceAlignmentLearner(
            cache,
            lambda d=args.d: M.SubspaceMap(d, 1, seed=0),
            LaplaceBudget(steps=300, prior_precision=1.0, n_samples=0),
            warm_start=False,
            seed=0,
        )
        phi = learner.fit(t.pairs, t.y).phi
        for lr in args.lrs:
            for gamma in args.gammas:
                est = local_learning_coefficient(
                    cache, phi, t.pairs, t.y, None, n_chains=args.chains, steps=args.steps,
                    lr=lr, gamma=gamma,
                )  # fmt: skip
                rows.append(
                    {"k": k, "lambda_eff": world.lam, "lr": lr, "gamma": gamma, "llc": est.lam,
                     "llc_se": est.lam_se, "chains": est.chains, "loss_star": est.loss_star,
                     "mean_loss": est.mean_loss}
                )  # fmt: skip
                print(
                    f"d={args.d} k={k:2d} lambda_eff={world.lam:4.1f} lr={lr:g} gamma={gamma:g}: "
                    f"LLC {est.lam:6.2f} +- {est.lam_se:4.2f} (L* {est.loss_star:.4f}, "
                    f"mean {est.mean_loss:.4f}, beta {est.beta:.3f})",
                    flush=True,
                )
                (args.out / "llc.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
