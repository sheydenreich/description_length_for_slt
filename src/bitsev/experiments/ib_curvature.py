"""Hessian and Fisher spectra of the subspace map's loss at its MAP on InterpBench.

Same analysis as :mod:`bitsev.experiments.ioi_curvature` (dense Hessian from one product per
coordinate, empirical Fisher from per-example gradients, plug-in and Laplace predictions), on
an InterpBench case where the frame is small enough for CPU. Example:
  bitsev-ib-curvature --case 18 --variable freq --loci resid_2 resid_1 --out results/ib_curvature
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bitsev.alignment.targets import make_targets
from bitsev.experiments.ioi_curvature import analyse_locus
from bitsev.testbeds import interpbench as IB


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="18")
    p.add_argument("--variable", default="freq")
    p.add_argument("--loci", nargs="+", default=["resid_2", "resid_1"])
    p.add_argument("--n-targets", type=int, default=1024)
    p.add_argument("--n-pool", type=int, default=3000)
    p.add_argument("--d-v", type=int, default=4)
    p.add_argument("--prior-precision", type=float, default=0.01)
    p.add_argument("--map-steps", type=int, default=100)
    p.add_argument("--fisher-split", choices=["train", "iid"], default="train")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("results/ib_curvature"))
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    ib = IB.load_case(args.case)
    alg, tok = ib.algorithm, ib.algorithm.tokens
    pool = tok.encode(tok.sample(args.n_pool, np.random.default_rng(20261015)))
    targets = make_targets(alg, pool, args.n_targets, args.variable, 0.05, seed=1)
    fresh = make_targets(alg, pool, 1000, args.variable, 0.05, seed=2)
    args.out.mkdir(parents=True, exist_ok=True)
    inner = SimpleNamespace(
        d_v=args.d_v,
        prior_precision=args.prior_precision,
        map_steps=args.map_steps,
        top=200,
        full=True,
    )
    report = {}
    for locus in args.loci:
        report[locus] = analyse_locus(
            ib.model,
            pool,
            targets,
            locus,
            inner,
            "cpu",
            fisher_targets=fresh if args.fisher_split == "iid" else None,
        )
        (args.out / "curvature.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
