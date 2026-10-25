"""Does the code's regret slope track the oracle's learning coefficients?

On the oracle with ``k_relevant`` of the ``d - 1`` Grassmannian directions readable (the data
vary along them), the remaining ``d - 1 - k`` directions are quartic, not flat
(``make_oracle``): the plug-in coefficient ``(1/2) tr(F H^+)`` is ``k / 2`` and the Bayes
learning coefficient is ``k / 2 + (d - 1 - k) / 4``; the nominal ``(d - 1) / 2`` holds only at
``k = d - 1``. The MAP subspace code or the Bayes code by HMC (regret against the known law,
batching-corrected slope on n >= n_min) is fitted for several ``k`` at fixed ``d``.
  python -m bitsev.experiments.oracle_lambda_sweep --out results/oracle_lambda_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bitsev.alignment import maps as M
from bitsev.alignment.hmc import HmcAlignmentLearner, HmcBudget
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import LaplaceAlignmentLearner, LaplaceBudget
from bitsev.alignment.targets import make_targets
from bitsev.codes import schedule as S
from bitsev.codes.gain import fit_log_slope
from bitsev.codes.prequential import prequential_code
from bitsev.testbeds.oracle import make_oracle


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=12)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 7, 11])
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--growth", type=float, default=1.25)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--n-min", type=int, default=256)
    p.add_argument(
        "--channel",
        choices=["none", "rr"],
        default="none",
        help="none: the oracle readout already carries the noise (the validated setting); rr: the "
        "learner applies the randomized-response channel on top (double noise unless beta-scale "
        "is large enough to saturate the readout)",
    )
    p.add_argument("--beta-scale", type=float, default=1.0, help="readout sharpness multiplier")
    p.add_argument("--v-margin", type=float, default=0.0, help="exclude |v| < margin")
    p.add_argument(
        "--v-continuous",
        action="store_true",
        help="V coordinate ~ N(0, 1) instead of +-1; regret is then measured against the oracle's "
        "own counterfactual law instead of the fixed randomized-response formula",
    )
    p.add_argument(
        "--learner",
        choices=["map", "hmc"],
        default="map",
        help="map: MAP subspace code (learner coefficient); hmc: Bayes code by HMC (lambda)",
    )
    p.add_argument("--hmc-chains", type=int, default=4)
    p.add_argument("--hmc-warmup", type=int, default=200)
    p.add_argument("--hmc-samples", type=int, default=100)
    p.add_argument("--hmc-leapfrog", type=int, default=10)
    p.add_argument("--hmc-step", type=float, default=1e-2)
    p.add_argument("--prior-var", type=float, default=1.0, help="hmc: Gaussian prior on the frame")
    p.add_argument("--out", type=Path, default=Path("results/oracle_lambda_sweep"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    rows = []
    for k in args.ks:
        world = make_oracle(
            args.d,
            3 * args.n,
            args.eps,
            seed=0,
            k_relevant=k,
            beta_scale=args.beta_scale,
            v_continuous=args.v_continuous,
            v_margin=args.v_margin,
        )
        cache = HiddenCache.build(world.network, "h1", world.pool)
        sched = S.geometric(args.n, args.growth, start=16)
        curves, ends = [], None
        for seed in range(args.seeds):
            t = make_targets(world.algorithm, world.pool, args.n, "V", args.eps, seed=300 + seed)
            channel_used = t.channel if args.channel == "rr" else None
            if args.learner == "hmc":
                learner = HmcAlignmentLearner(
                    cache,
                    lambda d=args.d: M.SubspaceMap(d, 1, seed=0),
                    HmcBudget(
                        chains=args.hmc_chains,
                        warmup=args.hmc_warmup,
                        samples=args.hmc_samples,
                        leapfrog=args.hmc_leapfrog,
                        step=args.hmc_step,
                    ),
                    prior_var=args.prior_var,
                    seed=seed,
                    channel=channel_used,
                )
            else:
                learner = LaplaceAlignmentLearner(
                    cache,
                    lambda d=args.d: M.SubspaceMap(d, 1, seed=0),
                    LaplaceBudget(steps=100, prior_precision=1.0, n_samples=0),
                    warm_start=True,
                    seed=seed,
                    channel=channel_used,
                )
            code = prequential_code(learner, t.pairs, t.y, sched)
            if args.learner == "hmc":
                print(
                    f"   seed {seed}: HMC acceptance {learner.last_accept:.2f}, final step "
                    f"{learner.last_step:.2e}",
                    flush=True,
                )
            if args.v_continuous:
                h_source = torch.as_tensor(world.pool[t.source])
                channel = t.channel if args.channel == "rr" else None
                with torch.no_grad():
                    true_ll = -world.network.true_counterfactual_log_probs(
                        h_source, torch.as_tensor(t.y), channel
                    ).numpy()[sched.start :]
            else:
                true_ll = np.where(
                    t.y == t.y_clean, -np.log(1 - args.eps / 2), -np.log(args.eps / 2)
                )[sched.start :]
            ends, _ = code.at_boundaries()
            curves.append(np.cumsum(code.item_losses - true_ll)[ends - sched.start - 1])
        mean = np.mean(curves, 0)
        fit = fit_log_slope(ends, mean, n_min=args.n_min, growth=args.growth)
        per_seed = [
            fit_log_slope(ends, c, n_min=args.n_min, growth=args.growth).corrected_slope
            for c in curves
        ]
        se = float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed)))
        rows.append(
            {
                "k": k,
                "learner": args.learner,
                "channel": args.channel,
                "beta_scale": args.beta_scale,
                "v_continuous": args.v_continuous,
                "v_margin": args.v_margin,
                "lambda_effective": world.lam,
                "lambda_bayes": world.lam_bayes,
                "lambda_nominal": 0.5 * (args.d - 1),
                "corrected_slope": fit.corrected_slope,
                "seed_se": se,
                "final_regret_nats": float(mean[-1]),
                "per_seed_slopes": [float(x) for x in per_seed],
                "per_seed_final_regret": [float(c[-1]) for c in curves],
                "ends": ends.tolist(),
                "mean_regret": mean.tolist(),
            }
        )
        print(
            f"d={args.d} k={k:2d}: lambda_eff={world.lam:5.2f} lambda_bayes={world.lam_bayes:5.2f} "
            f"nominal={0.5 * (args.d - 1):5.2f} "
            f"corrected slope {fit.corrected_slope:6.2f} +- {se:4.2f}; regret at n={ends[-1]}: "
            f"{mean[-1]:.1f} nats",
            flush=True,
        )
        print(
            f"   per seed: slopes {np.round(per_seed, 2).tolist()}, final regret "
            f"{np.round([c[-1] for c in curves], 1).tolist()}",
            flush=True,
        )
        (args.out / "sweep.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
