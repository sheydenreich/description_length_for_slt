"""Figures for Stage 1 results, generated from the per-(configuration, seed) records.

Three figures per run directory:

* ``gain_curves``: cumulative gain ``G(m)`` in bits of each hypothesis over each of its
  matched controls, mean over seeds, against the number of coded targets (log axis);
* ``trajectory``: for the H-A2 check, the gain against the step-0 network and the held-out
  IIA of the rotation at the last hidden locus as functions of the training step, in two
  stacked panels sharing the x axis (one measure per axis);
* ``block_losses``: mean code length per target in each block against the prefix size the
  block's predictor was fitted on, one line per configuration.

Colours follow a fixed categorical order (one slot per configuration role) so that the same
role keeps its colour across figures and runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NATS_TO_BITS = 1.0 / np.log(2.0)

# Reference categorical palette (light surface), assigned in fixed order by role.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ROLE_ORDER = [
    "hypothesis",
    "untrained",
    "locus=input",
    "map=coupling",
    "manufactured",
    "label-only",
]
ROLE_STYLE = {
    "hypothesis": "-",
    "untrained": "--",
    "locus=input": (0, (1, 1)),
    "map=coupling": "-.",
}
TEXT = "#333333"
GRID = "#dddddd"


def _style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def role_of(record: dict) -> str:
    if record.get("notes", {}).get("manufactured"):
        return "manufactured"
    d = record.get("differs_in")
    if d is None:
        return "hypothesis"
    if d == "network":
        return "untrained"
    return record["configuration"].rsplit("/", 1)[-1]


def colour(role: str) -> str:
    return PALETTE[ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER)]


def load_records(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run_dir / "records").glob("*.json"))]


def _cumulative_bits(rec: dict) -> np.ndarray:
    return np.cumsum(rec["item_losses"]) * NATS_TO_BITS


def _sizes(rec: dict) -> np.ndarray:
    start = rec["boundaries"][0]
    return np.arange(start + 1, rec["boundaries"][-1] + 1)


def gain_curves(records: list[dict], out: Path) -> Path | None:
    hyps = [r for r in records if r["differs_in"] is None and "@" not in r["configuration"]]
    all_names = {r["configuration"] for r in records}
    names = sorted(
        n
        for n in {r["configuration"] for r in hyps}
        if any(c.startswith(n + "/") for c in all_names)
    )
    if not names:
        return None
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.8 * len(names), 3.6), squeeze=False, sharey=True
    )
    for ax, name in zip(axes[0], names, strict=True):
        by_seed = {r["seed"]: r for r in hyps if r["configuration"] == name}
        controls = sorted(
            {r["configuration"] for r in records if r["configuration"].startswith(name + "/")}
        )
        for ctrl in controls:
            curves = []
            for r in records:
                if r["configuration"] == ctrl and r["seed"] in by_seed:
                    curves.append(_cumulative_bits(r) - _cumulative_bits(by_seed[r["seed"]]))
            if not curves:
                continue
            mean = np.mean(curves, axis=0)
            role = ctrl.rsplit("/", 1)[-1] if not ctrl.endswith("/untrained") else "untrained"
            ax.plot(
                _sizes(by_seed[next(iter(by_seed))]),
                mean,
                color=colour(role),
                linewidth=2,
                label=f"vs {role} (n={len(curves)})",
            )
        ax.axhline(0, color=TEXT, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("coded targets m")
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
        _style(ax)
    axes[0][0].set_ylabel("gain G(m), bits")
    fig.tight_layout()
    path = out / "gain_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def trajectory(records: list[dict], out: Path) -> Path | None:
    traj = [r for r in records if "@" in r["configuration"]]
    if not traj:
        return None
    loci = sorted({r["configuration"].split("/")[1] for r in traj})
    fig, axes = plt.subplots(
        2, len(loci), figsize=(5.2 * len(loci), 5.4), sharex=True, squeeze=False
    )
    for col, locus in enumerate(loci):
        by_step: dict[int, list[dict]] = {}
        for r in traj:
            if r["configuration"].split("/")[1] == locus:
                by_step.setdefault(int(r["notes"]["step"]), []).append(r)
        final = [r for r in records if r["configuration"] == f"trained/{locus}/rotation"]
        untrained = {
            r["seed"]: r
            for r in records
            if r["configuration"] == f"trained/{locus}/rotation/untrained"
        }
        if final:
            by_step[max(by_step) * 2] = final  # the final network, after the last checkpoint
        steps = sorted(by_step)
        gain_mean, gain_se, iia_iid, iia_shift = [], [], [], []
        for s in steps:
            g = [
                (untrained[r["seed"]]["total_nats"] - r["total_nats"]) * NATS_TO_BITS
                for r in by_step[s]
                if r["seed"] in untrained
            ]
            gain_mean.append(np.mean(g) if g else np.nan)
            gain_se.append(np.std(g, ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0)
            iia_iid.append(np.mean([r["faithfulness"]["iid"]["iia"] for r in by_step[s]]))
            iia_shift.append(np.mean([r["faithfulness"]["shifted"]["iia"] for r in by_step[s]]))
        ax1, ax2 = axes[0][col], axes[1][col]
        x = np.array(steps, dtype=float)
        ax1.errorbar(
            x,
            gain_mean,
            yerr=gain_se,
            color=colour("hypothesis"),
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax1.axhline(0, color=TEXT, linewidth=0.8)
        ax1.set_title(f"Rotation at {locus} along training (H-A2)", fontsize=10)
        ax2.plot(
            x,
            iia_iid,
            color=colour("hypothesis"),
            linewidth=2,
            marker="o",
            markersize=4,
            label="IIA, i.i.d. split",
        )
        ax2.plot(
            x,
            iia_shift,
            color=colour("untrained"),
            linewidth=2,
            marker="s",
            markersize=4,
            label="IIA, disjoint vocabulary",
        )
        ax2.set_ylim(0.4, 1.02)
        ax2.set_xlabel("training step (last point: final network)")
        ax2.set_xscale("log")
        for ax in (ax1, ax2):
            _style(ax)
    axes[0][0].set_ylabel("gain vs step-0 network, bits")
    axes[1][0].set_ylabel("held-out IIA")
    axes[1][0].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    path = out / "trajectory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def block_losses(records: list[dict], out: Path) -> Path | None:
    main = [r for r in records if "@" not in r["configuration"]]
    if not main:
        return None
    names = sorted({r["configuration"] for r in main})
    groups: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name in names:
        recs = [r for r in main if r["configuration"] == name]
        b = np.array(recs[0]["boundaries"])
        per_block = []
        for r in recs:
            losses = np.array(r["item_losses"]) * NATS_TO_BITS
            edges = b - b[0]
            per_block.append(
                [losses[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
            )
        role = role_of(recs[0])
        group = "/".join(name.split("/")[:2])  # e.g. trained/h1: one colour per network+locus
        if group not in groups:
            groups[group] = PALETTE[len(groups) % len(PALETTE)]
        ax.plot(
            b[:-1],
            np.mean(per_block, axis=0),
            linestyle=ROLE_STYLE.get(role, ":"),
            color=groups[group],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=name,
        )
    ax.axhline(1.0, color=TEXT, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prefix size the block's predictor was fitted on")
    ax.set_ylabel("mean code length per target, bits")
    ax.set_title("Per-block code lengths (1 bit = uninformative)", fontsize=10)
    _style(ax)
    ax.legend(
        frameon=False,
        fontsize=7,
        labelcolor=TEXT,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.tight_layout()
    path = out / "block_losses.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all(run_dir: Path) -> list[Path]:
    records = load_records(run_dir)
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    return [
        p
        for p in (gain_curves(records, out), trajectory(records, out), block_losses(records, out))
        if p
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Figures for a Stage 1 run directory")
    p.add_argument("run_dir", type=Path)
    args = p.parse_args(argv)
    for path in make_all(args.run_dir):
        print(path)
    return 0
