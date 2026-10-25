"""H-A10: the Jacobian lens as a parameter-free rival on IOI.

For every locus the J-vectors of the name tokens are computed on the trained network and
on its untrained twin, the fixed J-lens alignment is coded prequentially on the Stage 1
targets (its code is the loss of one predictor, learning coefficient zero) and its
faithfulness is measured on the i.i.d. and shifted splits, next to the label-only code and
the condition number of the J-vector matrix (the appendix's distortion). With ``--jc`` the
Jacobian-consistency statistic is computed on triples for the J-lens itself, a freshly
fitted subspace map on the trained network and a manufactured coupling alignment on the
twin. Run on a GPU:
  sbatch slurm/gpu.sbatch python -m bitsev.experiments.ioi_jlens --jc --out results/ioi_jlens_gpt2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bitsev.alignment import jlens as J
from bitsev.alignment import maps as M
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import GradientAlignmentLearner, TrainingBudget
from bitsev.alignment.score import faithfulness, label_only_code
from bitsev.codes import schedule as S
from bitsev.codes.prequential import prequential_code
from bitsev.experiments import run_stage1_ioi as R
from bitsev.experiments.run_stage1_toy import NATS_TO_BITS


def _triples(targets, n: int, seed: int) -> np.ndarray:
    """(base, source, source') with two different sources per base."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(targets.base), size=n, replace=True)
    other = rng.permutation(len(targets.base))[:n]
    s2 = targets.source[other]
    return np.column_stack([targets.base[idx], targets.source[idx], s2])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 6, 9, 10, 11, 12])
    p.add_argument("--n-jac-prompts", type=int, default=1000)
    p.add_argument("--n-targets", type=int, default=1024)
    p.add_argument("--twin-seed", type=int, default=7)
    p.add_argument("--jc", action="store_true", help="Jacobian consistency of three alignments")
    p.add_argument("--jc-layers", type=int, nargs="+", default=[10, 11])
    p.add_argument("--jc-triples", type=int, default=500)
    p.add_argument("--d-v", type=int, default=16)
    p.add_argument("--map-steps", type=int, default=400)
    p.add_argument("--adversary-steps", type=int, default=4000)
    p.add_argument("--out", type=Path, default=Path("results/ioi_jlens"))
    args = p.parse_args(argv)
    ioi_args = R.parse(
        [
            "--model", args.model, "--dtype", args.dtype, "--layers", *map(str, args.layers),
            "--n-targets", str(args.n_targets), "--twin-seed", str(args.twin_seed),
        ]
    )  # fmt: skip
    net, alg, pool, targets, evals, _ = R.build(ioi_args)
    twin = net.reinitialised(args.twin_seed)
    device = ioi_args.device
    args.out.mkdir(parents=True, exist_ok=True)
    sched = S.geometric(len(targets), 2.0, start=8)
    label_only = label_only_code(targets, sched)
    x_jac = net.prepare_input(pool[: args.n_jac_prompts])
    rows = []
    vectors: dict[tuple[str, str], J.JVectors] = {}
    for locus in net.loci:
        for which, network in (("trained", net), ("untrained", twin)):
            t0 = time.time()
            vec_all = J.jacobian_vectors(network, x_jac, locus)
            vectors[(which, locus)] = vec_all
            cache = HiddenCache.build(network, locus, pool, device)
            # "all": J-vectors of the 32 names of the output alphabet; "train": only the 16
            # names the learned maps see, the values of V on the training distribution
            variants = {
                "all": vec_all,
                "train": J.JVectors(vec_all.raw[: ioi_args.n_names], locus, vec_all.n_prompts),
            }
            for names, vec in variants.items():
                t0 = time.time()
                phi = J.jlens_map(vec)
                learner = J.FixedMapLearner(cache, phi, f"jlens[{locus}]", targets.channel)
                code = prequential_code(learner, targets.pairs, targets.y, sched)
                pred = learner.fit(targets.pairs, targets.y)
                f = {k: faithfulness(pred, t) for k, t in evals.items()}
                row = {
                    "network": which,
                    "names": names,
                    "locus": locus,
                    "rank": phi.d_v,
                    "condition_number": vec.condition_number,
                    "effective_rank": vec.effective_rank,
                    "code_bits": code.total * NATS_TO_BITS,
                    "gain_over_label_only_bits": (label_only.code.total - code.total)
                    * NATS_TO_BITS,
                    "iid_iia": f["iid"].iia,
                    "iid_log_loss": f["iid"].log_loss,
                    "shifted_iia": f["shifted"].iia,
                    "shifted_log_loss": f["shifted"].log_loss,
                    "seconds": time.time() - t0,
                }
                rows.append(row)
                print(
                    f"{which:<9} {names:<5} {locus:<8} rank {phi.d_v:<3} "
                    f"cond {vec.condition_number:8.1f} "
                    f"code {row['code_bits']:8.1f} bits (gain over label-only "
                    f"{row['gain_over_label_only_bits']:+8.1f})  IIA iid {f['iid'].iia:.2f} "
                    f"shifted {f['shifted'].iia:.2f}  ({row['seconds']:.0f}s)",
                    flush=True,
                )
                (args.out / "jlens.json").write_text(json.dumps(rows, indent=1))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"label-only code: {label_only.code.total * NATS_TO_BITS:.1f} bits")
    if not args.jc:
        return 0
    jc_rows = []
    triples = _triples(evals["iid"], args.jc_triples, seed=5)
    for layer in args.jc_layers:
        locus = f"layer{layer}"
        cache_t = HiddenCache.build(net, locus, pool, device)
        cache_u = HiddenCache.build(twin, locus, pool, device)
        aligns = {"jlens": (cache_t, J.jlens_map(vectors[("trained", locus)]), "trained")}
        t0 = time.time()
        sub = GradientAlignmentLearner(
            cache_t,
            lambda d=cache_t.width: M.SubspaceMap(d, args.d_v),
            TrainingBudget(args.map_steps, 1e-2),
            seed=0,
            family_name="subspace",
            channel=targets.channel,
        ).fit(targets.pairs, targets.y)
        aligns["subspace"] = (cache_t, sub.phi, "trained")
        print(f"{locus}: subspace fitted ({time.time() - t0:.0f}s)", flush=True)
        t0 = time.time()
        adv = GradientAlignmentLearner(
            cache_u,
            lambda d=cache_u.width: M.AdditiveCoupling(d, args.d_v, n_blocks=4, width=512),
            TrainingBudget(args.adversary_steps, 1e-3),
            seed=0,
            family_name="coupling",
            channel=targets.channel,
        ).fit(targets.pairs, targets.y)
        aligns["manufactured"] = (cache_u, adv.phi, "untrained")
        print(f"{locus}: manufactured alignment fitted ({time.time() - t0:.0f}s)", flush=True)
        for name, (cache, phi, which) in aligns.items():
            jc = J.jacobian_consistency(cache, phi, vectors[(which, locus)], triples)
            pred = J.FixedMapLearner(cache, phi, name, targets.channel).fit(
                targets.pairs, targets.y
            )
            f = faithfulness(pred, evals["iid"])
            jc_rows.append(
                {
                    "locus": locus,
                    "alignment": name,
                    "network": which,
                    "iid_iia": f.iia,
                    "jc_median": float(np.median(jc)),
                    "jc_mean": float(np.mean(jc)),
                    "jc_q90": float(np.quantile(jc, 0.9)),
                }
            )
            print(
                f"{locus} {name:<13} ({which}): IIA {f.iia:.2f}  JC median {np.median(jc):.3f} "
                f"mean {np.mean(jc):.3f} q90 {np.quantile(jc, 0.9):.3f}",
                flush=True,
            )
            (args.out / "jacobian_consistency.json").write_text(json.dumps(jc_rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
