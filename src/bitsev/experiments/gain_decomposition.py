"""Which term carries a gain: the fit of the best-in-family predictors, or the cost of learning?

For a hypothesis H and a control C coded on the same targets, the gain splits exactly as

    G(n) = DL_C(n) − DL_H(n) = [Fit_C(n) − Fit_H(n)] + [Reg_C(n) − Reg_H(n)],

where Fit_X(n) is the loss of X's final fit (``extra.final_item_nats``) on the first n targets and
Reg_X = DL_X − Fit_X is the regret of the prequential code against it (as in ``bitsev-h-a3``).
The fit term grows like n·Δℓ (falsification when the control's family cannot reach the
hypothesis's loss, negative when it fits better); the regret term carries the learner
coefficients (Δκ ln n) and the O(1) remainder. The label-only control is rebuilt from the run's
settings (IOI runs only; the targets hash is checked); its final fit is the empirical label
frequencies. Per-block code rates (nats per target in each block, coded before it is fitted)
give the same comparison without the final fit. Example:
  python -m bitsev.experiments.gain_decomposition results/stage1_ioi_gpt2_map_n32768 \\
      --hypothesis trained/layer11/subspace --out reports/stage1_ioi/gain_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bitsev.alignment.score import label_only_code
from bitsev.codes import schedule as S

LABEL_ONLY = "label-only"


def channel_floor(noise_rate: float, n_outputs: int) -> float:
    """Entropy (nats) of the observed label law when the clean label is known."""
    top = 1 - noise_rate + noise_rate / n_outputs
    rest = noise_rate / n_outputs
    return float(-(top * np.log(top) + (n_outputs - 1) * rest * np.log(rest)))


def label_only_items(run_dir: Path, settings: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Prequential and final-fit per-item losses of the label-only code, from rebuilt targets."""
    from transformers import AutoTokenizer

    from bitsev.experiments import run_stage1_ioi as R

    args = R.parse(
        [
            "--model", settings["model"], "--n-targets", str(settings["n_targets"]),
            "--n-names", str(settings["n_names"]), "--noise-rate", str(settings["noise_rate"]),
            "--variable", settings["variable"], "--n-eval", str(settings["n_eval"]),
        ]
    )  # fmt: skip
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    alg, pool, _, n_train = R.ioi_task(args, tok)
    targets, _ = R.ioi_targets(args, alg, pool, n_train)
    if targets.content_hash() != settings["targets_hash"]:
        raise ValueError(f"rebuilt targets differ from {run_dir} ({targets.content_hash()})")
    sched = S.geometric(len(targets), settings["growth"], start=settings["start"])
    code = label_only_code(targets, sched).code
    freq = np.bincount(targets.y, minlength=targets.n_outputs) / len(targets)
    final = -np.log(freq[targets.y])[sched.start :]
    return code.item_losses, final, targets.n_outputs


def cumulative(losses: np.ndarray, final: np.ndarray, ends: np.ndarray, start: int):
    """Prequential and final-fit cumulative losses at the block ends."""
    idx = ends - start - 1
    return np.cumsum(losses)[idx], np.cumsum(final)[idx]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--controls", nargs="*", default=None, help="default: label-only and all")
    p.add_argument("--no-label-only", action="store_true")
    p.add_argument(
        "--out", type=Path, default=None, help="default: <run_dir>/gain_decomposition.json"
    )
    args = p.parse_args(argv)
    settings = json.loads((args.run_dir / "settings.json").read_text())
    by_name: dict[str, dict[int, dict]] = {}
    for path in sorted((args.run_dir / "records").glob("*.json")):
        r = json.loads(path.read_text())
        if "final_item_nats" in r.get("extra", {}):
            by_name.setdefault(r["configuration"], {})[r["seed"]] = r
    hyp = by_name[args.hypothesis]
    first = next(iter(hyp.values()))
    start, ends = first["boundaries"][0], np.asarray(first["boundaries"][1:], int)
    seeds = sorted(hyp)
    controls = args.controls or [c for c in by_name if c != args.hypothesis]
    curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    held_out: dict[str, list[float]] = {}
    for name in [args.hypothesis, *controls]:
        curves[name] = {}
        for s in seeds:
            r = by_name[name][s]
            if r["targets_hash"] != first["targets_hash"] or r["boundaries"] != first["boundaries"]:
                raise ValueError(f"{name} seed {s} was coded on other targets or blocks")
            final = np.asarray(r["extra"]["final_item_nats"], float)[start:]
            curves[name][s] = cumulative(np.asarray(r["item_losses"], float), final, ends, start)
        held_out[name] = [by_name[name][s]["faithfulness"]["iid"]["log_loss"] for s in seeds]
    n_outputs = settings.get("n_outputs", 2 * settings["n_names"])
    if not args.no_label_only:
        lo, lo_final, n_outputs = label_only_items(args.run_dir, settings)
        curves[LABEL_ONLY] = dict.fromkeys(seeds, cumulative(lo, lo_final, ends, start))
        controls = [LABEL_ONLY, *controls]

    def rows(name):
        dl = np.array([curves[name][s][0] for s in seeds])
        fit = np.array([curves[name][s][1] for s in seeds])
        width = np.diff(np.concatenate([[start], ends]))
        rate = np.diff(np.concatenate([np.zeros((len(seeds), 1)), dl], 1), axis=1) / width
        return dl, fit, rate

    out = {
        "run": str(args.run_dir),
        "targets_hash": first["targets_hash"],
        "hypothesis": args.hypothesis,
        "seeds": seeds,
        "block_ends": ends.tolist(),
        "growth": first["growth"],
        "channel_floor_nats": channel_floor(settings["noise_rate"], n_outputs),
        "units": "nats; per-seed arrays over block ends",
        "codes": {},
        "gains": {},
    }
    for name in [args.hypothesis, *controls]:
        dl, fit, rate = rows(name)
        out["codes"][name] = {
            "code": dl.tolist(),
            "final_fit": fit.tolist(),
            "block_rate": rate.tolist(),
            "held_out_final_fit": held_out.get(name),
        }
    dl_h, fit_h, _ = rows(args.hypothesis)
    print(f"hypothesis {args.hypothesis}; bits at n = {ends[-1]} (mean over {len(seeds)} seeds)")
    for c in controls:
        dl_c, fit_c, _ = rows(c)
        gain, fit_term = dl_c - dl_h, fit_c - fit_h
        out["gains"][c] = {
            "gain": gain.tolist(),
            "fit_term": fit_term.tolist(),
            "regret_term": (gain - fit_term).tolist(),
        }
        g, f = gain[:, -1].mean() / np.log(2), fit_term[:, -1].mean() / np.log(2)
        print(f"  vs {c:<40} gain {g:+10.0f} = fit {f:+10.0f} + regret {g - f:+8.0f}")
    dest = args.out or args.run_dir / "gain_decomposition.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
