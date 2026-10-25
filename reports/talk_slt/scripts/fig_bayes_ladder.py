"""(d) The n-versus-lambda ladder: code length of the valid alignment relative to the label-only code, for the
genuine Bayes code (HMC) and the plug-in code (mixture of four, the canonical learner), against the nominal
Bayes coefficient of the alignment family, at 1024 targets.

Sources:
  reports/stage1_toy/bayes_code_hmc.txt                hierarchical equality (rotation, d_V 4, nominal 56)
  reports/stage1_interpbench/case18_bayes_code_hmc.txt InterpBench 18 (subspace d_V 4, nominal 512), 3 seeds;
                                                       8192-target seed (label-only "~10,700" bits, approximate)
  reports/stage1_ioi/regret_slopes_map_code.txt        nominal 512 and 6016
  reports/stage1_ioi/gpt2_summary_v2.md                GPT-2 plug-in (mixture, 5 seeds) and label-only
  GPT-2 Bayes code (not run): drawn as its ceiling. Every predictor codes the observed law, so a target costs
  at most -ln(eps/K) nats (eps 0.05, K 32 names: 6.46); on the 1016 coded targets that is 2.2x the label-only
  code. The Laplace count 6016 ln n (curvature_gpt2.txt) exceeds the ceiling until n ~ 1e4, so it gives no
  number at n = 1024 (corrected 2026-10-22; the earlier "about 65,000 nats, 22x" ignored the ceiling).
"""
import math
import re

import numpy as np
import matplotlib.pyplot as plt

# settings of every GPT-2 IOI run (results/stage1_ioi_gpt2*/settings.json): noise rate, output names
NOISE, N_NAMES = 0.05, 32

from common import style, save, read, grab, NUM, summary_rows, label_only_bits, PLUGIN, BAYES, MUTED, INK, INK2


def data():
    toy = read("stage1_toy/bayes_code_hmc.txt")
    L_toy = float(grab(r"label-only\s+(" + NUM + r") bits", toy))
    m = re.search(r"valid h1 rotation\s+(" + NUM + r") \+- " + NUM + r" \(\S+\)\s+(" + NUM + ")", toy)
    bayes_toy, plug_toy = float(m.group(1)), float(m.group(2))
    nom_toy = float(grab(r"coefficient \d+ x \d+ / 2 = (\d+)", toy))

    ib = read("stage1_interpbench/case18_bayes_code_hmc.txt")
    L_ib = float(grab(r"Label-only code (" + NUM + r") bits", ib))
    seeds = [float(x) for x in re.search(r"valid resid_2 subspace\s+(\d+) / (\d+) / (\d+)", ib).groups()]
    plug_ib = float(grab(r"valid resid_2 subspace.*?([+-]\d+) bits /", ib))
    b8192 = float(grab(r"8192 targets: valid (\d+) bits", ib))
    L8192 = float(grab(r"label-only code of ~(\d+),(\d+)", ib, 1) + grab(r"label-only code of ~(\d+),(\d+)", ib, 2))
    slopes = read("stage1_ioi/regret_slopes_map_code.txt")
    nom_ib = float(grab(r"nominal lambda (\d+), frame 1040", slopes))
    nom_gpt = float(grab(r"nominal Grassmannian lambda 16\*752/2 = (\d+)", slopes))

    L_gpt = label_only_bits("stage1_ioi/gpt2_summary_v2.md")
    plug_gpt = summary_rows("stage1_ioi/gpt2_summary_v2.md")["trained/layer11/subspace"][0]
    n_coded = int(grab(r"label-only code " + NUM + r" bits on (\d+) coded targets", read("stage1_ioi/gpt2_summary_v2.md")))
    ceiling_nats = n_coded * -math.log(NOISE / N_NAMES)
    return dict(
        toy=(nom_toy, (L_toy - plug_toy) / L_toy, [(L_toy - bayes_toy) / L_toy]),
        ib=(nom_ib, (L_ib - plug_ib) / L_ib, [s / L_ib for s in seeds]),
        ib8192=(nom_ib, b8192 / L8192),
        gpt=(nom_gpt, (L_gpt - plug_gpt) / L_gpt, ceiling_nats / math.log(2) / L_gpt),
    )


def main():
    style()
    d = data()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axhspan(1, 60, color="#f6f5f1", zorder=0)
    ax.axhline(1, color=INK2, lw=1.5)
    ax.text(40, 1.12, "label-only code", fontsize=14, color=INK2)
    ax.text(40, 30, "longer than label-only", fontsize=14, color=INK2)

    xs = [d["toy"][0], d["ib"][0], d["gpt"][0]]
    ax.plot(xs, [d["toy"][1], d["ib"][1], d["gpt"][1]], "-o", color=PLUGIN, ms=11, lw=2.5,
            label="plug-in code (mixture of 4)")
    bm = [np.mean(d["toy"][2]), np.mean(d["ib"][2])]
    ax.plot(xs[:2], bm, "-s", color=BAYES, ms=11, lw=2.5, label="Bayes code by HMC, 1024 targets")
    ax.plot([d["ib"][0]] * 3, d["ib"][2], "s", mfc="none", mec=BAYES, ms=8, mew=1.5)
    ax.plot([xs[1], xs[2]], [bm[1], d["gpt"][2]], ls=":", color=BAYES, lw=2)
    ax.plot(xs[2], d["gpt"][2], "v", mfc="white", mec=BAYES, ms=13, mew=2.5,
            label="Bayes code, not run: its ceiling (−ln(ε/K) per target)")
    ax.plot(d["ib8192"][0] * 1.12, d["ib8192"][1], "^", mfc="white", mec=BAYES, ms=11, mew=2)
    ax.text(d["ib8192"][0] * 1.28, d["ib8192"][1], "Bayes at 8192\ntargets (1 seed)", fontsize=12, color=INK2,
            va="center")

    names = [("hierarchical\nequality", d["toy"]), ("InterpBench\ncase 18", d["ib"]), ("GPT-2 IOI", d["gpt"])]
    for (lbl, row) in names:
        ax.text(row[0], 0.07, lbl + f"\nλ = {row[0]:,.0f}", ha="center", va="bottom", fontsize=13, color=INK)
    ax.text(xs[2], d["gpt"][2] * 1.4, f"≤ {d['gpt'][2]:.1f}×", ha="center", va="bottom", fontsize=15, color=INK2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(25, 20000)
    ax.set_ylim(0.06, 60)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_xlabel("nominal Bayes coefficient λ of the alignment family")
    ax.set_ylabel("code length / label-only code")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.88))
    save(fig, "bayes_vs_plugin_ladder")


if __name__ == "__main__":
    main()
