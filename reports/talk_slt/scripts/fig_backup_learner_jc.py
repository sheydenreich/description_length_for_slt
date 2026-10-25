"""Backup figures.
learner_decides_verdict: the GPT-2 layer-10 manufactured alignment under four learners (gain over label-only, IIA),
  with the valid layer-11 subspace under the same learners. Source: reports/stage1_ioi/adversary_by_learner.txt.
jacobian_consistency: median and 90th percentile of JC = ||delta - W_U J Delta|| / ||delta|| over 500 triples for the
  Jacobian lens, a learned subspace and the manufactured alignment at GPT-2 layers 10 and 11.
  Source: reports/stage1_ioi/jlens_gpt2_jacobian_consistency.json.
"""
import re

import numpy as np
import matplotlib.pyplot as plt

from common import style, save, read, load_json, NUM, JLENS, VALID, MANUF, INK, INK2, MUTED


def learner():
    t = read("stage1_ioi/adversary_by_learner.txt")
    rows = []
    for m in re.finditer(r"^(\S.*?)\s{2,}(" + NUM + r") \±\s+(" + NUM + r") bits, per seed \[.*?\], IIA (" + NUM + ")",
                         t, re.M):
        rows.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    ref = re.search(r"mixture \+(\d+) ± \d+ \(5 seeds\); MAP only \+(\d+) ± \d+; MAP \+ Langevin \(init-scale\) \+(\d+)", t)
    return rows, [float(x) for x in ref.groups()]


def main():
    style()
    rows, ref = learner()
    fig, ax = plt.subplots(figsize=(12, 5.6))
    y = np.arange(len(rows))[::-1]
    for yi, (name, g, se, iia) in zip(y, rows):
        ax.barh(yi, g, xerr=se, color=MANUF, height=0.55, ecolor=INK2, capsize=4)
        ax.text(g + (se + 150 if g > 0 else -se - 150), yi, f"{g:+.0f}  (IIA {iia:.2f})", va="center",
                ha="left" if g > 0 else "right", fontsize=14, color=INK)
    ax.set_yticks(y, [r[0].replace(" (canonical learner)", "\n(canonical)") for r in rows])
    ax.axvline(0, color=INK, lw=1.3)
    ax.set_xlim(-5600, 4200)
    ax.set_xlabel("gain over label-only (bits), GPT-2 manufactured alignment, layer 10, 3 seeds")
    ax.text(4100, y[-1] - 0.55, f"valid layer-11 subspace: +{ref[0]:.0f} (mixture), +{ref[1]:.0f} (MAP), "
            f"+{ref[2]:.0f} (MAP + Langevin)", ha="right", fontsize=13, color=VALID)
    ax.set_ylim(-0.9, len(rows) - 0.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, "learner_decides_verdict")

    jc = load_json("stage1_ioi/jlens_gpt2_jacobian_consistency.json")
    fig, ax = plt.subplots(figsize=(11, 5.6))
    names = [("jlens", "Jacobian lens", JLENS), ("subspace", "learned subspace", VALID),
             ("manufactured", "manufactured (untrained twin)", MANUF)]
    for i, locus in enumerate(("layer10", "layer11")):
        for j, (key, lbl, c) in enumerate(names):
            r = [x for x in jc if x["locus"] == locus and x["alignment"] == key][0]
            xi = i + (j - 1) * 0.26
            ax.bar(xi, r["jc_median"], width=0.24, color=c, label=lbl if i == 0 else None)
            ax.plot([xi, xi], [r["jc_median"], r["jc_q90"]], color=INK2, lw=1.5)
            ax.plot(xi, r["jc_q90"], "_", color=INK2, ms=14, mew=1.5)
            ax.text(xi, r["jc_median"] / 2, f"{r['jc_median']:.2f}", ha="center", va="center", fontsize=13,
                    color="white")
            ax.text(xi, -0.08, f"IIA {r['iid_iia']:.2f}", ha="center", va="top", fontsize=11, color=INK2)
    ax.set_xticks([0, 1], ["GPT-2 layer 10", "GPT-2 layer 11"])
    ax.tick_params(axis="x", pad=26, length=0)
    ax.set_ylim(0, 5.3)
    ax.set_ylabel("Jacobian consistency\n(median; whisker: 90th pct.)")
    ax.legend(loc="upper left", fontsize=13)
    ax.grid(axis="x", visible=False)
    save(fig, "jacobian_consistency")


if __name__ == "__main__":
    main()
