"""(a) The complexity ladder on GPT-2 IOI, four explanations of the indirect-object name side by side:
fitted parameters, IIA, plug-in learner coefficient kappa, and gain over the label-only code.

Sources:
  reports/stage1_ioi/regret_slopes_map_code.txt      kappa of the MAP code (d_V 16, n <= 1024, 3 seeds)
  reports/stage1_ioi/gpt2_summary_single_map_map0.md gains and IIA of the same MAP-code run
  reports/stage1_ioi/adversary_by_learner.txt        IIA of the manufactured alignment under the MAP code (2 decimals)
  reports/stage1_ioi/jlens_gpt2.json                 Jacobian lens, layer 11, 16 training names
Parameter counts: subspace frame d * d_V; coupling map = 4 additive blocks of width 512 on halves of 384
plus a rotation of R^768 (paper, section "Map families and learners": 1,869,440 parameters).
"""
import re

import numpy as np
import matplotlib.pyplot as plt

from common import (style, save, read, load_json, grab, NUM, gpt2_map_slopes, summary_rows,
                    JLENS, VALID, COUPLING, MANUF, INK, INK2, MUTED)


def data():
    k = gpt2_map_slopes()
    s = summary_rows("stage1_ioi/gpt2_summary_single_map_map0.md")
    adv = read("stage1_ioi/adversary_by_learner.txt")
    iia_manuf = float(grab(r"MAP only \(AdamW, cold start\).*?IIA (" + NUM + ")", adv))
    jl = [r for r in load_json("stage1_ioi/jlens_gpt2.json")
          if r["network"] == "trained" and r["names"] == "train" and r["locus"] == "layer11"][0]
    t = read("stage1_ioi/regret_slopes_map_code.txt")
    d = int(grab(r"width (\d+), d_V 16", t))
    dv = 16
    half, width, blocks = d // 2, 512, 4
    coupling_params = blocks * (half * width + width + width * half) + d * (d - 1) // 2
    rows = [
        ("Jacobian lens\n(no fitted map)", JLENS, 0, jl["iid_iia"], 0.0, 0.0, jl["gain_over_label_only_bits"]),
        ("learned subspace\n(trained GPT-2)", VALID, d * dv, s["trained/layer11/subspace"][2], *k["valid"],
         s["trained/layer11/subspace"][0]),
        ("coupling map\n(trained GPT-2)", COUPLING, coupling_params, s["trained/layer11/subspace/map=coupling"][2],
         *k["coupling"], s["trained/layer11/subspace/map=coupling"][0]),
        ("manufactured\n(coupling, untrained)", MANUF, coupling_params, iia_manuf, *k["manufactured"],
         s["untrained/layer10/coupling"][0]),
    ]
    return rows


def main():
    style()
    rows = data()
    n = len(rows)
    y = np.arange(n)[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2), sharey=True,
                             gridspec_kw={"wspace": 0.12, "width_ratios": [1, 0.8, 1, 1.15]})
    labels = [r[0] for r in rows]
    cols = [r[1] for r in rows]

    # parameters
    ax = axes[0]
    for yi, r in zip(y, rows):
        if r[2] > 0:
            ax.barh(yi, r[2], left=1, color=r[1], height=0.6)
            ax.text(r[2] * 1.3, yi, f"{r[2]:,}", va="center", fontsize=14, color=INK)
        else:
            ax.text(1.4, yi, "0", va="center", fontsize=14, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e8)
    ax.set_title("fitted\nparameters", loc="left", fontsize=16)
    ax.set_yticks(y, labels)

    # IIA
    ax = axes[1]
    for yi, r in zip(y, rows):
        ax.barh(yi, r[3], color=r[1], height=0.6)
        ax.text(r[3] + 0.02, yi, f"{r[3]:.2f}", va="center", fontsize=14, color=INK)
    ax.axvline(0.8, color=MUTED, ls="--", lw=1.5)
    ax.text(0.78, -0.62, "accept at 0.8", ha="right", va="center", fontsize=12, color=INK2)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.9, n - 0.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_title("interchange\naccuracy (IIA)", loc="left", fontsize=16)

    # kappa
    ax = axes[2]
    for yi, r in zip(y, rows):
        if r[4] > 0:
            ax.barh(yi, r[4], left=1, color=r[1], height=0.6, xerr=r[5], ecolor=INK2, capsize=3)
            ax.text((r[4] + r[5]) * 1.35, yi, f"{r[4]:.0f}", va="center", fontsize=14, color=INK)
        else:
            ax.text(1.3, yi, "0 (exact)", va="center", fontsize=14, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 2e4)
    ax.set_title("plug-in κ\n(regret per ln n)", loc="left", fontsize=16)

    # gain
    ax = axes[3]
    for yi, r in zip(y, rows):
        ax.barh(yi, r[6], color=r[1], height=0.6)
        ha = "left" if r[6] > 0 else "right"
        ax.text(r[6] + (120 if r[6] > 0 else -120), yi, f"{r[6]:+.0f}", va="center", ha=ha, fontsize=14,
                color=INK)
    ax.axvline(0, color=INK2, lw=1.2)
    ax.set_xlim(-3900, 5200)
    ax.set_title("gain over\nlabel-only (bits)", loc="left", fontsize=16)

    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.text(0.0, -0.06, "GPT-2 IOI, indirect-object name; single-map MAP code, 1016 coded targets; "
             "κ at n ≤ 1024, 3 seeds", fontsize=13, color=INK2)
    save(fig, "ladder_panels")


if __name__ == "__main__":
    main()
