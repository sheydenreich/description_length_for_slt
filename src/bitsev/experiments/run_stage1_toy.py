"""Stage 1 on hierarchical equality: trained against untrained networks, loci and map families.

Week 3 of the plan. One hypothesis per hidden locus (rotation, d_V = 1) is scored against
the three standard controls and the label-only code on frozen targets, for several seeds.
The untrained-network control is the step-0 checkpoint of the same training run. A
manufactured alignment (expressive coupling map on the untrained network) is scored too.
With ``--trajectory`` every checkpoint is scored for the H-A2 curve.

Every (configuration, seed) record is written once to ``<out>/records/<name>.json`` and
skipped if present, so an interrupted array job resumes. ``summary.md`` is regenerated
from the records on every run.

Smoke test (login node): ``bitsev-stage1-toy --smoke``. Full run: see ``slurm/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from bitsev.alignment import controls as C
from bitsev.alignment import maps as M
from bitsev.alignment.algorithms import HierarchicalEquality
from bitsev.alignment.hmc import HmcBudget
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import TrainingBudget
from bitsev.alignment.network import LayeredNetwork, mlp, reinitialised_twin
from bitsev.alignment.score import ScoreResult, label_only_code, score_configuration
from bitsev.alignment.targets import TargetSet, make_targets
from bitsev.baselines.cae import causal_abstraction_error
from bitsev.baselines.probing_mdl import online_probe_code
from bitsev.codes import schedule as S
from bitsev.diagnostics import separation as D1
from bitsev.diagnostics import stretch as D2
from bitsev.report import markdown as md
from bitsev.testbeds import hierarchical_equality as he
from bitsev.testbeds.training import train_classifier

NATS_TO_BITS = 1.0 / np.log(2.0)


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("results/stage1_toy"))
    p.add_argument("--smoke", action="store_true", help="tiny settings for a login-node check")
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
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=12)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    p.add_argument("--train-steps", type=int, default=4000)
    p.add_argument("--n-targets", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--noise-rate", type=float, default=0.05)
    p.add_argument("--growth", type=float, default=2.0)
    p.add_argument("--start", type=int, default=8, help="first coded target index")
    p.add_argument("--map-steps", type=int, default=400)
    p.add_argument("--map-lr", type=float, default=1e-2)
    p.add_argument("--d-v", type=int, default=4, help="dimension of the aligned subspace")
    p.add_argument("--map-weight-decay", type=float, default=0.0)
    p.add_argument(
        "--members", type=int, default=4, help="maps per block; >1 is the mixture learner"
    )
    p.add_argument("--coupling-blocks", type=int, default=2)
    p.add_argument("--coupling-width", type=int, default=64)
    p.add_argument("--variable", default="V1")
    p.add_argument(
        "--trajectory", action="store_true", help="score every training checkpoint (H-A2)"
    )
    p.add_argument("--no-diagnostics", action="store_true", help="skip CAE and D1/D2 per record")
    p.add_argument("--n-triples", type=int, default=100, help="triples for the D2 stretch budget")
    p.add_argument(
        "--learner",
        choices=["mixture", "hmc"],
        default="mixture",
        help="mixture: plug-in maps (members > 1); hmc: Bayes code by HMC",
    )
    p.add_argument("--hmc-chains", type=int, default=4)
    p.add_argument("--hmc-warmup", type=int, default=200)
    p.add_argument("--hmc-samples", type=int, default=100)
    p.add_argument("--hmc-leapfrog", type=int, default=20)
    p.add_argument("--hmc-step", type=float, default=1e-2)
    p.add_argument("--prior-var", type=float, default=1.0, help="hmc: Gaussian prior variance")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args(argv)
    if args.smoke:
        args.out = Path("results/stage1_toy_smoke")
        args.seeds, args.train_steps, args.n_targets, args.n_eval = 1, 300, 48, 100
        args.map_steps, args.hidden, args.start = 20, [16, 16], 4
    return args


def build_world(args: argparse.Namespace):
    """Vocabularies, training data, trained network with checkpoints, frozen targets."""
    rng = np.random.default_rng(20261014)
    torch.manual_seed(20261014)
    vocab_a, vocab_b = he.make_disjoint_vocabularies(args.vocab_size, args.dim, rng)
    train = he.sample(20_000, vocab_a, rng)
    net = mlp(4 * args.dim, args.hidden, n_outputs=2)
    run = train_classifier(
        net, train.x, train.y, steps=args.train_steps, seed=0, device=args.device
    )
    alg = HierarchicalEquality(args.dim)
    pool_a = he.sample(2000, vocab_a, rng).x
    pool_b = he.sample(2000, vocab_b, rng).x
    targets = make_targets(alg, pool_a, args.n_targets, args.variable, args.noise_rate, seed=1)
    eval_iid = make_targets(alg, pool_a, args.n_eval, args.variable, args.noise_rate, seed=2)
    # the shifted split uses pool B; indices are offsets into the concatenated pool
    eval_shift = make_targets(alg, pool_b, args.n_eval, args.variable, args.noise_rate, seed=3)
    pool = np.concatenate([pool_a, pool_b])
    eval_shift = TargetSet(
        eval_shift.base + len(pool_a),
        eval_shift.source + len(pool_a),
        eval_shift.variable,
        eval_shift.y_clean,
        eval_shift.y,
        eval_shift.noise_rate,
        eval_shift.n_outputs,
        eval_shift.seed,
    )
    return net, run, pool, targets, {"iid": eval_iid, "shifted": eval_shift}, alg


def network_at(template: LayeredNetwork, run, step: int) -> LayeredNetwork:
    import copy

    return run.load(copy.deepcopy(template), step)


def configurations(
    args, trained: LayeredNetwork, untrained: LayeredNetwork, wrong_locus: str
) -> list[C.Configuration]:
    def rotation(d: int) -> M.AlignmentMap:
        return M.Rotation(d, args.d_v)

    def coupling(d: int) -> M.AlignmentMap:
        return M.AdditiveCoupling(
            d, args.d_v, n_blocks=args.coupling_blocks, width=args.coupling_width
        )

    configs: list[C.Configuration] = []
    for locus in trained.loci[1:]:
        hyp = C.Configuration(f"trained/{locus}/rotation", trained, locus, "rotation", rotation)
        configs += [
            hyp,
            C.untrained_control(hyp, untrained),
            *([C.wrong_locus_control(hyp, wrong_locus)] if wrong_locus != locus else []),
            C.other_map_control(hyp, "coupling", coupling),
        ]
    # manufactured alignment: expressive map on the untrained network at the last locus
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
    return configs


def rivals_and_diagnostics(args, cfg, result, targets, evals, pool, alg) -> dict:
    """Causal abstraction error on the i.i.d. evaluation targets, D1 at the locus, D2 budgets.

    D1 is a property of the network and locus; it is read against the untrained twin's.
    """
    cache = HiddenCache.build(cfg.network, cfg.locus, pool, args.device)
    phi = result.final_map
    out = {"cae": causal_abstraction_error(cache, phi, alg, pool, evals["iid"]).record()}
    distinct = np.unique(targets.base)
    states = cache.states[distinct].cpu().numpy()
    values = alg.variable_values(pool[distinct])[args.variable].astype(int)
    out["d1"] = D1.margin_statistics(states, values).record()
    triples = D2.label_triples(targets, max_triples=args.n_triples)
    budgets = D2.stretch_budgets(cache, phi, targets, triples, n_points=3)
    out["d2"] = D2.summarise_budgets(budgets)
    return out


def run_one(args, cfg: C.Configuration, seed: int, targets, pool, evals, alg, out: Path) -> dict:
    path = out / "records" / f"{cfg.name.replace('/', '_')}__seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    sched = S.geometric(len(targets), args.growth, start=args.start)
    budget = TrainingBudget(
        steps=args.map_steps, lr=args.map_lr, weight_decay=args.map_weight_decay
    )
    result = score_configuration(
        cfg,
        targets,
        pool,
        sched,
        budget,
        seed=seed,
        eval_targets=evals,
        device=args.device,
        n_members=args.members,
        learner_kind=args.learner,
        hmc_budget=HmcBudget(
            chains=args.hmc_chains,
            warmup=args.hmc_warmup,
            samples=args.hmc_samples,
            leapfrog=args.hmc_leapfrog,
            step=args.hmc_step,
        ),
        prior_var=args.prior_var,
    )
    rec = result.record() | {
        "seed": seed,
        "notes": cfg.notes,
        "differs_in": cfg.differs_from_hypothesis_in,
    }
    if not args.no_diagnostics:
        rec |= rivals_and_diagnostics(args, cfg, result, targets, evals, pool, alg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def probe_records(args, trained, untrained, targets, pool, alg, out: Path) -> None:
    """Online probing codes of V on the base inputs of the targets, same schedule as Stage 1."""
    sched = S.geometric(len(targets), args.growth, start=args.start)
    values = alg.variable_values(pool[targets.base])[args.variable].astype(int)
    for net_name, net in (("trained", trained), ("untrained", untrained)):
        for locus in net.loci[1:]:
            path = out / "records" / f"probe_{net_name}_{locus}.json"
            if path.exists():
                continue
            states = HiddenCache.build(net, locus, pool[targets.base], args.device).states.numpy()
            code = online_probe_code(states, values, sched)
            rec = {
                "configuration": f"probe/{net_name}/{locus}",
                "learner": code.learner,
                "sees": ["locus", "network", "variable"],
                "locus": locus,
                "map_family": "logistic-probe",
                "targets_hash": targets.content_hash(),
                "schedule": sched.label(),
                "seed": 0,
                "boundaries": sched.boundaries.tolist(),
                "growth": sched.growth,
                "item_losses": code.item_losses.tolist(),
                "total_nats": code.total,
                "faithfulness": {},
                "map_params": None,
                "distortion_bound": None,
                "notes": {"baseline": "online-probe"},
                "differs_in": "baseline",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec))


def summarise(
    out: Path, label_only: ScoreResult, title: str = "Stage 1 on hierarchical equality"
) -> str:
    records = [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]
    by_name: dict[str, list[dict]] = {}
    for r in records:
        by_name.setdefault(r["configuration"], []).append(r)
    base = label_only.code.total

    def stats(values):
        v = np.asarray(values, float)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        return f"{v.mean():.1f} ± {se:.1f}"

    probes = {n: r for n, r in by_name.items() if r[0].get("notes", {}).get("baseline")}
    rows = []
    for name, recs in sorted(by_name.items()):
        if name in probes:
            continue
        totals = [r["total_nats"] for r in recs]
        rows.append(
            [
                name,
                len(recs),
                stats([(base - t) * NATS_TO_BITS for t in totals]),
                stats([r["faithfulness"]["iid"]["iia"] for r in recs])
                if recs[0]["faithfulness"]
                else "-",
                stats([r["faithfulness"]["shifted"]["iia"] for r in recs])
                if recs[0]["faithfulness"]
                else "-",
                stats([r["faithfulness"]["iid"]["log_loss"] for r in recs])
                if recs[0]["faithfulness"]
                else "-",
                f"{recs[0]['distortion_bound']:.2g}" if recs[0].get("distortion_bound") else "-",
                stats([r["cae"]["error"] for r in recs]) if recs[0].get("cae") else "-",
                stats([r["cae"]["complement"]["disagreement"] for r in recs])
                if recs[0].get("cae", {}).get("complement")
                else "-",
                stats([r["cae"]["random_direction"]["disagreement"] for r in recs])
                if recs[0].get("cae")
                else "-",
                f"{recs[0]['d1']['nn_ratio']:.2f} / {recs[0]['d1']['sep_ratio']:.2f}"
                if recs[0].get("d1", {}).get("nn_ratio")
                else "-",
                stats([r["d2"]["share_network_term_nonpositive"] for r in recs])
                if recs[0].get("d2") and recs[0]["d2"].get("n")
                else "-",
            ]
        )
    header = [
        "configuration",
        "seeds",
        "gain over label-only (bits)",
        "IIA iid",
        "IIA shifted",
        "log loss iid",
        "Dist bound",
        "CAE",
        "complement err",
        "random-dir err",
        "D1 NN / sep ratio",
        "D2 share≤0",
    ]
    text = (
        f"# {title}\n\nTargets hash `{label_only.targets_hash}`, "
        f"schedule {label_only.schedule}, label-only code {base * NATS_TO_BITS:.1f} bits on "
        f"{label_only.code.schedule.n_coded} coded targets.\n\n"
    )
    text += md.table(header, rows)
    if probes:
        text += (
            "\n## Online probe codes\n\nA logistic probe codes the *variable* V from the locus "
            "(noise-free, no interventions, no readout), so these lengths are a different "
            "quantity from the Stage 1 codes above and are not gains over the label-only code.\n\n"
        )
        text += md.table(
            ["probe", "code (bits)", "coded items"],
            [
                [n, f"{r[0]['total_nats'] * NATS_TO_BITS:.1f}", len(r[0]["item_losses"])]
                for n, r in sorted(probes.items())
            ],
        )
    # gains against the matched controls
    gain_rows = []
    for name, recs in sorted(by_name.items()):
        if recs[0]["differs_in"] is not None:
            continue
        for ctrl_name, ctrl_recs in by_name.items():
            if not ctrl_name.startswith(name + "/"):
                continue
            paired = [(h, c) for h in recs for c in ctrl_recs if h["seed"] == c["seed"]]
            if paired:
                # anytime evidence: the running log Bayes factor of the control's code against
                # the hypothesis's is an e-process under "the control's predictions are the
                # law"; it rejects at level alpha once it exceeds log(1/alpha) (Section e-values)
                firsts, peaks = [], []
                for h, c in paired:
                    running = np.cumsum(np.asarray(c["item_losses"]) - np.asarray(h["item_losses"]))
                    start = h["boundaries"][0]
                    peaks.append(running.max() * NATS_TO_BITS)
                    hit = np.nonzero(running >= np.log(100.0))[0]
                    firsts.append(int(hit[0]) + start + 1 if len(hit) else None)
                first = (
                    f"{max(f for f in firsts)}" if all(f is not None for f in firsts) else "never"
                )
                gain_rows.append(
                    [
                        name,
                        ctrl_name.split("/", 2)[-1],
                        len(paired),
                        stats(
                            [(c["total_nats"] - h["total_nats"]) * NATS_TO_BITS for h, c in paired]
                        ),
                        stats(peaks),
                        first,
                    ]
                )
    text += "\n\n## Gains against matched controls (bits, mean ± s.e. over seeds)\n\n"
    text += (
        "`peak evidence` is the maximum of the running log Bayes factor (an e-process under "
        "the control); `rejects α=0.01 by n` is the coded index by which every seed's running "
        "evidence first exceeds log(100) ≈ 6.6 bits, or `never`.\n\n"
    )
    text += md.table(
        ["hypothesis", "control", "seeds", "gain G(n)", "peak evidence", "rejects α=0.01 by n"],
        gain_rows,
    )
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    net, run, pool, targets, evals, alg = build_world(args)
    untrained = network_at(net, run, 0)
    trained = network_at(net, run, run.steps()[-1])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(
        json.dumps(vars(args) | {"targets_hash": targets.content_hash()}, default=str, indent=1)
    )
    (out / "training.json").write_text(
        json.dumps([asdict(c) | {"state_dict": None} for c in run.checkpoints], indent=1)
    )
    sched = S.geometric(len(targets), args.growth, start=args.start)
    label_only = label_only_code(targets, sched)
    values = alg.variable_values(pool[targets.base])[args.variable].astype(int)
    _, probe_acc = C.least_decodable_locus(trained, pool[targets.base], values)
    wrong_locus, attainable = C.least_intervenable_locus(
        trained,
        lambda d: M.Rotation(d, args.d_v),
        targets,
        evals["iid"],
        pool,
        budget=TrainingBudget(args.map_steps, args.map_lr, weight_decay=args.map_weight_decay),
    )
    print(
        f"probe accuracies per locus {probe_acc}; attainable IIA per locus {attainable}; "
        f"wrong-locus control (least intervenable): {wrong_locus}"
    )
    (out / "wrong_locus.json").write_text(
        json.dumps(
            {"wrong_locus": wrong_locus, "attainable_iia": attainable, "probe_accuracy": probe_acc}
        )
    )
    configs = configurations(args, trained, untrained, wrong_locus)
    if args.adversary_only:
        adversary = next(c for c in configs if c.notes.get("manufactured"))
        twin2 = reinitialised_twin(untrained, seed=args.second_twin_seed)
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
    if args.trajectory and not args.adversary_only:
        for locus in trained.loci[1:]:
            for step in run.steps()[1:-1]:
                configs.append(
                    C.Configuration(
                        f"trained@{step}/{locus}/rotation",
                        network_at(net, run, step),
                        locus,
                        "rotation",
                        lambda d: M.Rotation(d, args.d_v),
                        notes={"step": step},
                    )
                )
    n_seeds = 0 if args.summary_only else args.seeds
    for seed in range(args.seed_offset, args.seed_offset + n_seeds):
        for cfg in configs:
            rec = run_one(args, cfg, seed, targets, pool, evals, alg, out)
            faith = rec["faithfulness"]
            print(
                f"seed {seed}  {cfg.name:<40} DL={rec['total_nats'] * NATS_TO_BITS:8.1f} bits  "
                f"IIA iid={faith['iid']['iia']:.2f} shifted={faith['shifted']['iia']:.2f}"
            )
    if not args.adversary_only:
        probe_records(args, trained, untrained, targets, pool, alg, out)
    md.write(out / "summary.md", summarise(out, label_only))
    print(f"summary -> {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
