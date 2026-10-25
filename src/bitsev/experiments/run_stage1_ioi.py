"""Stage 1 on IOI in a Hugging Face language model, with a Sutter-style adversary.

Hypotheses: a subspace map for the variable ``io`` at every selected residual locus of
the trained model (last position). Controls: the reinitialised twin, the least decodable
locus, the coupling map. Adversary: an expressive coupling map on the twin with a large
training budget (the non-linear representation dilemma). Faithfulness is reported on an
i.i.d. split and on a disjoint-name split. Runs on a GPU (``slurm/stage1_ioi.sbatch``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bitsev.alignment import controls as C
from bitsev.alignment import maps as M
from bitsev.alignment.hf_network import HFNetwork
from bitsev.alignment.learners import LaplaceBudget, TrainingBudget
from bitsev.alignment.score import label_only_code, score_configuration
from bitsev.alignment.targets import TargetSet, make_targets
from bitsev.codes import schedule as S
from bitsev.experiments.run_stage1_toy import NATS_TO_BITS, summarise
from bitsev.report import markdown as md
from bitsev.testbeds import ioi, mib


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument(
        "--task",
        choices=["ioi", "arith"],
        default="ioi",
        help="ioi: indirect-object identification; arith: MIB two-digit addition (carry)",
    )
    p.add_argument("--revision", default=None, help="checkpoint revision (Pythia: stepN)")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 6, 9, 10, 11, 12],
        help="residual loci at the final position (sweep: the subspace works from layer 10)",
    )
    p.add_argument("--variable", default=None, help="default: io (ioi) or carry (arith)")
    p.add_argument(
        "--n-names", type=int, default=16, help="names per split (train / disjoint test)"
    )
    p.add_argument(
        "--n-prompts",
        type=int,
        default=4000,
        help="prompts sampled per split before length filtering",
    )
    p.add_argument("--n-targets", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--noise-rate", type=float, default=0.05)
    p.add_argument("--growth", type=float, default=2.0)
    p.add_argument("--start", type=int, default=8)
    p.add_argument("--d-v", type=int, default=16)
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--map-steps", type=int, default=400)
    p.add_argument("--map-lr", type=float, default=1e-2, help="subspace maps (sweep-chosen)")
    p.add_argument("--map-batch", type=int, default=256, help="training batch of the maps")
    p.add_argument(
        "--forward-chunk", type=int, default=1024, help="prompts per intervened forward pass"
    )
    p.add_argument("--coupling-lr", type=float, default=1e-3, help="coupling maps diverge at 1e-2")
    p.add_argument(
        "--adversary-steps", type=int, default=4000, help="budget of the manufactured alignment"
    )
    p.add_argument("--coupling-blocks", type=int, default=4)
    p.add_argument("--coupling-width", type=int, default=512)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-offset", type=int, default=0, help="first seed (one job per seed)")
    p.add_argument(
        "--adversary-layer",
        type=int,
        nargs="+",
        default=[6],
        help="loci of the manufactured alignment(s)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--learner", choices=["mixture", "laplace"], default="mixture")
    p.add_argument(
        "--min-epochs",
        type=float,
        default=0.0,
        help="Laplace/MAP learner: Adam steps per block at least this many passes over the prefix",
    )
    p.add_argument(
        "--temperature",
        action="store_true",
        help="every map family gets one learned inverse temperature on the readout's logits",
    )
    p.add_argument("--prior-precision", type=float, default=0.01, help="Laplace: Gaussian prior")
    p.add_argument("--sampler", choices=["laplace", "sgld"], default="sgld")
    p.add_argument(
        "--warm-start",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=False,
        help="Laplace/MAP learner: resume each block from the previous MAP (default False)",
    )
    p.add_argument("--sgld-lr", type=float, default=1e-5, help="Langevin step size (a hedge knob)")
    p.add_argument("--sgld-steps", type=int, default=200)
    p.add_argument("--sgld-scale", choices=["map", "init"], default="map")
    p.add_argument("--laplace-samples", type=int, default=8, help="Laplace: posterior samples")
    p.add_argument(
        "--fisher-examples", type=int, default=256, help="Laplace: per-example gradients used"
    )
    p.add_argument("--twin-seed", type=int, default=7, help="initialisation of the untrained twin")
    p.add_argument(
        "--adversary-only",
        action="store_true",
        help="score only the manufactured alignments (named with their budget)",
    )
    p.add_argument(
        "--summary-only", action="store_true", help="rebuild summary.md from existing records"
    )
    args = p.parse_args(argv)
    if args.variable is None:
        args.variable = "io" if args.task == "ioi" else "carry"
    if args.out is None:
        args.out = Path(
            f"results/stage1_{args.task}_{args.model.replace('/', '_')}"
            + (f"_{args.revision}" if args.revision else "")
            + ("_smoke" if args.smoke else "")
        )
    if args.smoke:
        args.seeds, args.n_targets, args.n_eval, args.map_steps, args.adversary_steps = (
            1,
            48,
            64,
            10,
            20,
        )
        args.n_prompts, args.start, args.layers = 400, 4, args.layers or [0, 6]
    return args


def build(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype)
    )
    if args.task == "arith":
        return _build_arith(args, tok, model)
    alg, pool, merged, n_train = ioi_task(args, tok)
    net = HFNetwork(
        model,
        merged.name_token_ids,
        layers=args.layers,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    targets, evals = ioi_targets(args, alg, pool, n_train)
    return net, alg, pool, targets, evals, merged


def ioi_task(args, tok):
    """Algorithm and prompt pool of the IOI task from the tokenizer alone (no model)."""
    names = ioi.single_token_names(tok)
    if len(names) < 2 * args.n_names:
        raise ValueError(f"only {len(names)} single-token names; need {2 * args.n_names}")
    rng = np.random.default_rng(20261016)
    train = ioi.build_prompts(tok, tuple(names[: args.n_names]), args.n_prompts, rng)
    test = ioi.build_prompts(
        tok, tuple(names[args.n_names : 2 * args.n_names]), args.n_prompts, rng
    )
    if train.tokens.shape[1] != test.tokens.shape[1]:
        raise ValueError("train and test prompts differ in token length; adjust templates")
    # one pool: train prompts first, disjoint-name prompts after; names of both splits share
    # the output alphabet (32 names), so the algorithm sees a single name list
    all_names = tuple(names[: 2 * args.n_names])
    pool = np.concatenate([train.tokens, test.tokens])
    merged = ioi.IOIPrompts(
        train.text + test.text,
        pool,
        np.concatenate([train.io, test.io + args.n_names]),
        np.concatenate([train.s, test.s + args.n_names]),
        np.concatenate([train.order, test.order]),
        all_names,
        np.array([tok.encode(" " + nm, add_special_tokens=False)[0] for nm in all_names]),
    )
    return ioi.IOIAlgorithm(merged), pool, merged, len(train)


def ioi_targets(args, alg, pool, n_train):
    """Coded targets and the two evaluation splits (i.i.d. pairs, disjoint-name pairs)."""
    targets = _targets_within(
        alg, pool, np.arange(n_train), args.n_targets, args.variable, args.noise_rate, 1
    )
    evals = {
        "iid": _targets_within(
            alg, pool, np.arange(n_train), args.n_eval, args.variable, args.noise_rate, 2
        ),
        "shifted": _targets_within(
            alg, pool, np.arange(n_train, len(pool)), args.n_eval, args.variable, args.noise_rate, 3
        ),
    }
    return targets, evals


def _build_arith(args, tok, model):
    """MIB two-digit addition: training-split prompts first, validation-split prompts after
    (the 'shifted' evaluation), one output alphabet (first answer tokens)."""
    rng = np.random.default_rng(20261021)
    train = mib.build_arithmetic(tok, "train", args.n_prompts, rng)
    test = mib.build_arithmetic(tok, "validation", args.n_prompts, rng)
    merged = mib.ArithmeticPrompts.concat(train, test)
    alg = mib.ArithmeticAlgorithm(merged)
    net = HFNetwork(
        model,
        merged.output_token_ids,
        layers=args.layers,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    pool = merged.tokens
    n_train = len(train)
    targets = _targets_within(
        alg, pool, np.arange(n_train), args.n_targets, args.variable, args.noise_rate, 1
    )
    evals = {
        "iid": _targets_within(
            alg, pool, np.arange(n_train), args.n_eval, args.variable, args.noise_rate, 2
        ),
        "shifted": _targets_within(
            alg, pool, np.arange(n_train, len(pool)), args.n_eval, args.variable, args.noise_rate, 3
        ),
    }
    merged.names = tuple(str(t) for t in merged.output_token_ids)  # for the settings record
    return net, alg, pool, targets, evals, merged


def _targets_within(alg, pool, idx, n, variable, noise, seed) -> TargetSet:
    """Targets whose base and source both come from the given pool indices."""
    sub = make_targets(alg, pool[idx], n, variable, noise, seed)
    return TargetSet(
        idx[sub.base], idx[sub.source], variable, sub.y_clean, sub.y, noise, alg.n_outputs, seed
    )


def main(argv=None) -> int:
    args = parse(argv)
    net, alg, pool, targets, evals, prompts = build(args)
    net.forward_chunk = args.forward_chunk
    twin = net.reinitialised(args.twin_seed)
    twin.forward_chunk = args.forward_chunk
    x = net.prepare_input(pool)
    with torch.no_grad():
        acc = float(
            (
                net.logits(x[: len(targets)]).argmax(1).cpu().numpy()
                == alg.output(pool[: len(targets)])
            ).mean()
        )
    values = alg.variable_values(pool[targets.base])[args.variable].astype(int)
    _, probe_acc = C.least_decodable_locus(net, pool[targets.base], values)
    sched = S.geometric(len(targets), args.growth, start=args.start)
    budget = TrainingBudget(args.map_steps, args.map_lr, batch_size=args.map_batch)
    coupling_budget = TrainingBudget(args.map_steps, args.coupling_lr, batch_size=args.map_batch)
    adversary_budget = TrainingBudget(
        args.adversary_steps, args.coupling_lr, batch_size=args.map_batch
    )

    def tempered(phi):
        return M.Tempered(phi) if args.temperature else phi

    def subspace(d):
        return tempered(M.SubspaceMap(d, args.d_v))

    def coupling(d):
        width = args.coupling_width
        return tempered(M.AdditiveCoupling(d, args.d_v, n_blocks=args.coupling_blocks, width=width))

    wrong_locus, attainable = C.least_intervenable_locus(
        net, subspace, targets, evals["iid"], pool, budget=budget, device=args.device
    )

    configs = []
    for locus in net.loci:
        hyp = C.Configuration(f"trained/{locus}/subspace", net, locus, "subspace", subspace)
        configs += [hyp, C.untrained_control(hyp, twin)]
        if wrong_locus != locus:
            configs.append(C.wrong_locus_control(hyp, wrong_locus))
        configs.append(C.other_map_control(hyp, "coupling", coupling))
    adversaries = [
        C.Configuration(
            f"untrained/layer{layer}/coupling"
            + (f"/steps={args.adversary_steps}" if args.adversary_only else "")
            + (f"/twin={args.twin_seed}" if args.twin_seed != 7 else ""),
            twin,
            f"layer{layer}",
            "coupling",
            coupling,
            notes={"manufactured": True, "budget": args.adversary_steps},
        )
        for layer in args.adversary_layer
    ]
    if args.adversary_only:
        configs = []
    out = args.out
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(
        json.dumps(
            vars(args)
            | {
                "targets_hash": targets.content_hash(),
                "model_ioi_accuracy_among_names": acc,
                "wrong_locus": wrong_locus,
                "probe_accuracy": probe_acc,
                "attainable_iia": attainable,
                "n_prompts": len(pool),
                "token_length": int(pool.shape[1]),
                "names": prompts.names,
            },
            default=str,
            indent=1,
        )
    )
    print(
        f"model accuracy among names {acc:.3f}; probe accuracies {probe_acc}; "
        f"attainable IIA {attainable}"
    )
    print(f"wrong locus {wrong_locus}")
    label_only = label_only_code(targets, sched)
    seeds = [] if args.summary_only else range(args.seed_offset, args.seed_offset + args.seeds)
    for seed in seeds:
        for cfg in [*configs, *adversaries]:
            path = out / "records" / f"{cfg.name.replace('/', '_')}__seed{seed}.json"
            if path.exists():
                continue
            if cfg.notes.get("manufactured"):
                b = adversary_budget
            elif cfg.map_family == "coupling":
                b = coupling_budget
            else:
                b = budget
            laplace_budget = LaplaceBudget(
                steps=b.steps,
                lr=b.lr,
                prior_precision=args.prior_precision,
                n_samples=args.laplace_samples,
                batch_size=b.batch_size,
                min_epochs=args.min_epochs,
                optimizer="adam",
                fisher_examples=args.fisher_examples,
                per_example="loop",
                sampler=args.sampler,
                sgld_lr=args.sgld_lr,
                sgld_steps=args.sgld_steps,
                sgld_scale=args.sgld_scale,
            )
            res = score_configuration(
                cfg,
                targets,
                pool,
                sched,
                b,
                seed=seed,
                eval_targets=evals,
                device=args.device,
                n_members=args.members,
                learner_kind=args.learner,
                warm_start=args.warm_start if args.learner == "laplace" else False,
                laplace_budget=laplace_budget,
            )
            rec = res.record() | {
                "seed": seed,
                "notes": cfg.notes,
                "differs_in": cfg.differs_from_hypothesis_in,
            }
            path.write_text(json.dumps(rec))
            f = rec["faithfulness"]
            print(
                f"seed {seed} {cfg.name:<32} DL={rec['total_nats'] * NATS_TO_BITS:8.1f} bits "
                f"IIA iid={f['iid']['iia']:.2f} shifted={f['shifted']['iia']:.2f}",
                flush=True,
            )
    md.write(
        out / "summary.md", summarise(out, label_only, title=f"Stage 1 on {args.task} in {args.model}")
    )
    print(f"summary -> {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
