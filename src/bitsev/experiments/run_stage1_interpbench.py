"""Stage 1 on an InterpBench case: a semi-synthetic transformer with a known program.

Same protocol as the hierarchical-equality runner, on a transformer: the hypothesis is a
rotation at each residual-stream locus, scored against the reinitialised twin, the input
locus and the coupling map; a coupling map on the twin is the manufactured alignment. The
targets are counterfactual outputs of the re-implemented program for one named variable.

Usage: ``bitsev-stage1-interpbench --case 52 --variable value`` (only cases in
``interpbench.VERIFIED`` are safe to interpret).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from bitsev.alignment import controls as C
from bitsev.alignment import maps as M
from bitsev.alignment.hmc import HmcBudget
from bitsev.alignment.learners import LaplaceBudget, TrainingBudget
from bitsev.alignment.network import reinitialised_twin
from bitsev.alignment.score import label_only_code, score_configuration
from bitsev.alignment.targets import make_targets
from bitsev.codes import schedule as S
from bitsev.experiments.run_stage1_toy import NATS_TO_BITS, summarise
from bitsev.report import markdown as md
from bitsev.testbeds import interpbench as IB


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="52")
    p.add_argument("--variable", default=None, help="default: the case's first variable")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--learner", choices=["mixture", "laplace", "hmc"], default="mixture")
    p.add_argument("--map", choices=["rotation", "subspace"], default="rotation")
    p.add_argument("--prior-precision", type=float, default=0.01, help="Laplace: Gaussian prior")
    p.add_argument("--sampler", choices=["laplace", "sgld"], default="sgld")
    p.add_argument(
        "--warm-start",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=True,
        help="Laplace/MAP learner: resume each block from the previous MAP (default True)",
    )
    p.add_argument("--sgld-lr", type=float, default=1e-5, help="Langevin step size (a hedge knob)")
    p.add_argument("--sgld-steps", type=int, default=200)
    p.add_argument("--sgld-scale", choices=["map", "init"], default="map")
    p.add_argument("--laplace-samples", type=int, default=16, help="Laplace: posterior samples")
    p.add_argument(
        "--laplace-steps", type=int, default=100, help="Laplace: L-BFGS iterations per block"
    )
    p.add_argument(
        "--adversary-only",
        action="store_true",
        help="score only the manufactured alignment, on the untrained network and on a second twin",
    )
    p.add_argument(
        "--second-twin-seed",
        type=int,
        default=8,
        help="initialisation of the second untrained twin",
    )
    p.add_argument(
        "--summary-only", action="store_true", help="rebuild summary.md from existing records"
    )
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-pool", type=int, default=3000)
    p.add_argument("--n-targets", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--noise-rate", type=float, default=0.05)
    p.add_argument("--growth", type=float, default=2.0)
    p.add_argument("--start", type=int, default=8)
    p.add_argument("--d-v", type=int, default=4)
    p.add_argument(
        "--members", type=int, default=4, help="maps per block; >1 is the mixture learner"
    )
    p.add_argument("--map-steps", type=int, default=400)
    p.add_argument("--map-lr", type=float, default=1e-2)
    p.add_argument("--map-weight-decay", type=float, default=0.0)
    p.add_argument("--coupling-blocks", type=int, default=2)
    p.add_argument("--coupling-width", type=int, default=64)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--hmc-chains", type=int, default=4)
    p.add_argument("--hmc-warmup", type=int, default=200)
    p.add_argument("--hmc-samples", type=int, default=100)
    p.add_argument("--hmc-leapfrog", type=int, default=20)
    p.add_argument("--hmc-step", type=float, default=1e-2)
    p.add_argument("--prior-var", type=float, default=1.0, help="hmc: Gaussian prior variance")
    p.add_argument(
        "--only",
        default=None,
        help="score only configurations whose name fully matches this regular expression",
    )
    p.add_argument("--seed-offset", type=int, default=0, help="first seed (one job per seed)")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = Path(f"results/stage1_ib{args.case}" + ("_smoke" if args.smoke else ""))
    if args.smoke:
        args.seeds, args.n_targets, args.n_eval, args.map_steps, args.n_pool = 1, 48, 100, 20, 300
        args.start = 4
    return args


def main(argv=None) -> int:
    args = parse(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.case not in IB.VERIFIED:
        print(f"warning: case {args.case} is not verified against its program", file=sys.stderr)
    ib = IB.load_case(args.case)
    alg, tok = ib.algorithm, ib.algorithm.tokens
    variable = args.variable or alg.variables[0]
    rng = np.random.default_rng(20261015)
    pool = tok.encode(tok.sample(args.n_pool, rng))
    targets = make_targets(alg, pool, args.n_targets, variable, args.noise_rate, seed=1)
    evals = {"iid": make_targets(alg, pool, args.n_eval, variable, args.noise_rate, seed=2)}
    trained, untrained = ib.model, reinitialised_twin(ib.model, seed=7)
    sched = S.geometric(len(targets), args.growth, start=args.start)
    budget = TrainingBudget(args.map_steps, args.map_lr, weight_decay=args.map_weight_decay)

    def rotation(d):
        return M.SubspaceMap(d, args.d_v) if args.map == "subspace" else M.Rotation(d, args.d_v)

    laplace_budget = LaplaceBudget(
        steps=args.laplace_steps,
        prior_precision=args.prior_precision,
        n_samples=args.laplace_samples,
        optimizer="lbfgs",
        sampler=args.sampler,
        sgld_lr=args.sgld_lr,
        sgld_steps=args.sgld_steps,
        sgld_scale=args.sgld_scale,
    )
    hmc_budget = HmcBudget(
        chains=args.hmc_chains,
        warmup=args.hmc_warmup,
        samples=args.hmc_samples,
        leapfrog=args.hmc_leapfrog,
        step=args.hmc_step,
    )
    # L-BFGS overflows on the coupling family; its MAP uses the plug-in's Adam budget
    coupling_laplace_budget = LaplaceBudget(
        steps=args.map_steps,
        lr=args.map_lr,
        prior_precision=args.prior_precision,
        n_samples=args.laplace_samples,
        optimizer="adam",
        batch_size=256,
        sampler=args.sampler,
        sgld_lr=args.sgld_lr,
        sgld_steps=args.sgld_steps,
        sgld_scale=args.sgld_scale,
    )

    def coupling(d):
        return M.AdditiveCoupling(
            d, args.d_v, n_blocks=args.coupling_blocks, width=args.coupling_width
        )

    values = alg.variable_values(pool[targets.base])[variable].astype(int)
    _, probe_acc = C.least_decodable_locus(trained, pool[targets.base], values)
    wrong_locus, attainable = C.least_intervenable_locus(
        trained, rotation, targets, evals["iid"], pool, budget=budget
    )
    print(
        f"probe accuracies per locus {probe_acc}; attainable IIA per locus {attainable}; "
        f"wrong-locus control (least intervenable): {wrong_locus}"
    )
    configs = []
    for locus in trained.loci[1:]:
        hyp = C.Configuration(f"trained/{locus}/{args.map}", trained, locus, args.map, rotation)
        configs += [
            hyp,
            C.untrained_control(hyp, untrained),
            *([C.wrong_locus_control(hyp, wrong_locus)] if wrong_locus != locus else []),
            C.other_map_control(hyp, "coupling", coupling),
        ]
    last = trained.loci[-1]
    configs.append(
        C.Configuration(
            f"untrained/{last}/coupling",
            untrained,
            last,
            "coupling",
            coupling,
            notes={"manufactured": True},
        )
    )
    if args.adversary_only:
        adversary = configs[-1]
        assert adversary.notes.get("manufactured")
        twin2 = reinitialised_twin(ib.model, seed=args.second_twin_seed)
        configs = [
            adversary,
            C.Configuration(
                f"{adversary.name}/twin={args.second_twin_seed}",
                twin2,
                adversary.locus,
                adversary.map_family,
                adversary.map_factory,
                notes={"manufactured": True, "twin_seed": args.second_twin_seed},
            ),
        ]
    out = args.out
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(
        json.dumps(
            vars(args)
            | {
                "targets_hash": targets.content_hash(),
                "task": ib.description,
                "task_accuracy": IB.task_accuracy(ib),
                "edges": ib.edges,
                "wrong_locus": wrong_locus,
                "probe_accuracy": probe_acc,
                "attainable_iia": attainable,
            },
            default=str,
            indent=1,
        )
    )
    label_only = label_only_code(targets, sched)
    first = args.seed_offset
    for seed in range(first, first + (0 if args.summary_only else args.seeds)):
        for cfg in configs:
            path = out / "records" / f"{cfg.name.replace('/', '_')}__seed{seed}.json"
            if path.exists() or (args.only and not re.fullmatch(args.only, cfg.name)):
                continue
            res = score_configuration(
                cfg,
                targets,
                pool,
                sched,
                budget,
                seed=seed,
                eval_targets=evals,
                n_members=args.members,
                learner_kind=args.learner,
                warm_start=args.warm_start if args.learner == "laplace" else False,
                laplace_budget=coupling_laplace_budget
                if cfg.map_family == "coupling"
                else laplace_budget,
                hmc_budget=hmc_budget,
                prior_var=args.prior_var,
            )
            rec = res.record() | {
                "seed": seed,
                "notes": cfg.notes,
                "differs_in": cfg.differs_from_hypothesis_in,
            }
            # the summary expects a 'shifted' split; InterpBench has none, so report iid twice
            rec["faithfulness"]["shifted"] = rec["faithfulness"]["iid"]
            path.write_text(json.dumps(rec))
            print(
                f"seed {seed} {cfg.name:<32} DL={rec['total_nats'] * NATS_TO_BITS:8.1f} bits "
                f"IIA={rec['faithfulness']['iid']['iia']:.2f}"
            )
    md.write(
        out / "summary.md",
        summarise(out, label_only, title=f"Stage 1 on InterpBench case {args.case}"),
    )
    print(f"summary -> {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
