"""Three estimates of the learning coefficient of reduced-rank regression against theory.

For each shape ``(M, N, H, r)`` the data ``y = A* x + e`` (rank ``r``) are coded
prequentially on a geometric schedule by the least-squares plug-in and by the Bayes code
(Hamiltonian Monte Carlo); the regret of each against the true law is fitted to ``ln n``
and compared with Aoyagi–Watanabe's ``lambda`` and the plug-in's ``M N / 2``. The local
learning coefficient is also estimated at the MAP by Langevin sampling. Example:

  bitsev-rrr-codes --shapes 2,2,2,0 3,3,3,0 3,3,3,1 4,4,4,0 4,4,4,2 --seeds 8 --n 16384
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bitsev.codes.gain import fit_log_slope
from bitsev.codes.prequential import prequential_code
from bitsev.codes.schedule import batching_factor, geometric
from bitsev.exact.rrr import aoyagi_watanabe, ols_kappa
from bitsev.testbeds import rrr


def regret_at_boundaries(code, true_log_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative regret against the true law at every block end."""
    start = code.schedule.start
    excess = code.item_losses + true_log_probs[start:]
    cum = np.cumsum(excess)
    ends = code.schedule.boundaries[1:]
    return ends, cum[ends - start - 1]


def run_shape(shape, args, seed) -> dict:
    M, N, H, r = shape
    task = rrr.RrrTask(M, N, r, seed=seed)
    x, y = task.sample(args.n, seed=10_000 + seed)
    truth = task.true_log_prob(x, y)
    schedule = geometric(args.n, args.growth, start=args.start)
    settings = rrr.HmcSettings(
        chains=args.chains, warmup=args.warmup, samples=args.samples, thin=args.thin
    )
    learners = {
        "ols": rrr.OlsLearner(M, N),
        "bayes": rrr.RrrBayesLearner(M, N, H, settings=settings, seed=seed),
    }
    out: dict = {"shape": list(shape), "seed": seed}
    for key, learner in learners.items():
        t0 = time.time()
        code = prequential_code(learner, x, y, schedule)
        ends, reg = regret_at_boundaries(code, truth)
        fit = fit_log_slope(ends, reg, n_min=args.n_min, growth=args.growth)
        out[key] = {
            "ends": ends.tolist(),
            "regret": reg.tolist(),
            "slope": fit.corrected_slope,
            "fit_se": fit.slope_stderr / batching_factor(args.growth),
            "seconds": time.time() - t0,
        }
        if key == "bayes":
            out[key]["accept"] = learner.last_accept
    rng = np.random.default_rng(seed)
    stats = rrr.Stats.of(x, y)
    w_star = rrr.map_point(learners["bayes"], stats, rng)
    lam, se = rrr.local_learning_coefficient(
        learners["bayes"],
        stats,
        w_star,
        rng,
        steps=args.llc_steps,
        step=args.llc_step,
        gamma=args.llc_gamma,
    )
    out["llc"] = {"lam": lam, "se": se}
    return out


def summarise(rows: list[dict]) -> str:
    lines = [
        f"{'(M,N,H,r)':<12}{'lambda':>8}{'m':>3}{'MN/2':>7}  "
        f"{'OLS slope':>14}  {'Bayes slope':>14}  {'LLC at MAP':>14}  {'HMC acc':>8}"
    ]
    by_shape: dict[tuple, list[dict]] = {}
    for row in rows:
        by_shape.setdefault(tuple(row["shape"]), []).append(row)
    for shape, group in by_shape.items():
        M, N, H, r = shape
        lam, mult = aoyagi_watanabe(M, N, H, r)
        cells = []
        for key in ("ols", "bayes"):
            v = np.array([g[key]["slope"] for g in group])
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            cells.append(f"{v.mean():7.2f} ± {se:4.2f}")
        v = np.array([g["llc"]["lam"] for g in group])
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        cells.append(f"{v.mean():7.2f} ± {se:4.2f}")
        acc = np.mean([g["bayes"]["accept"] for g in group])
        lines.append(
            f"{str(shape):<12}{float(lam):8.2f}{mult:>3}{float(ols_kappa(M, N)):7.2f}  "
            + "  ".join(f"{c:>14}" for c in cells)
            + f"  {acc:8.2f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", nargs="+", default=["2,2,2,0", "3,3,3,0", "3,3,3,1", "4,4,4,0"])
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--n-min", type=int, default=256)
    p.add_argument("--growth", type=float, default=2.0)
    p.add_argument("--start", type=int, default=8)
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--llc-steps", type=int, default=4000)
    p.add_argument("--llc-step", type=float, default=1e-4)
    p.add_argument("--llc-gamma", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("results/rrr_codes"))
    args = p.parse_args(argv)
    shapes = [tuple(int(v) for v in s.split(",")) for s in args.shapes]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape in shapes:
        for seed in range(args.seeds):
            row = run_shape(shape, args, seed)
            rows.append(row)
            print(
                f"{shape} seed {seed}: OLS slope {row['ols']['slope']:.2f}, Bayes slope "
                f"{row['bayes']['slope']:.2f} (accept {row['bayes']['accept']:.2f}, "
                f"{row['bayes']['seconds']:.0f}s), LLC {row['llc']['lam']:.2f} ± "
                f"{row['llc']['se']:.2f}",
                flush=True,
            )
            (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    table = summarise(rows)
    print(table)
    (args.out / "summary.txt").write_text(
        f"n = {args.n}, growth {args.growth}, start {args.start}, slopes on n >= {args.n_min}, "
        f"{args.seeds} seeds; HMC {args.chains} chains x {args.samples} samples (thin "
        f"{args.thin}, warmup {args.warmup}); LLC {args.llc_steps} steps, step {args.llc_step}, "
        f"gamma {args.llc_gamma}\n{table}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
