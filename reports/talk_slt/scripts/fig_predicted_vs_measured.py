"""(b) Predicted against measured plug-in coefficient: (1/2) tr(F H^-1) from the Hessian and the empirical
Fisher at the MAP, against the measured regret slope kappa of the MAP code, on the oracle (k sweep) and on
GPT-2; the Bayes coefficient of each family is drawn at the same measured kappa to show the gap (oracle:
lambda = k/2 + (11 - k)/4, the unreadable directions being quartic; GPT-2: the nominal d_V (d - d_V)/2).

Sources:
  reports/oracle/hessian_vs_fisher.txt                   oracle predictions (1/2) tr(F H^+) per k (text table)
  reports/oracle/hmc_n16384/ksweep_map_k*.json            measured MAP-code slope, n in [256, 16384], 4 seeds;
                                                          Bayes lambda of the oracle ("lambda_bayes")
  reports/stage1_ioi/curvature_gpt2_layer{10,11,12}_*heldout_fisher.json   GPT-2 predictions
  reports/stage1_ioi/curvature_gpt2_layer11_dv64_full_heldout_fisher.json   GPT-2 d_V 64 prediction (job 1820081)
  reports/stage1_ioi/regret_slopes_map_code.txt           measured GPT-2 kappa (d_V 4 / 16 / 64, layer 11)
  reports/stage1_ioi/h_a3_gpt2_map_l10_l12.json           measured GPT-2 kappa at layers 10 and 12 (2 seeds)
"""
import re

import numpy as np
import matplotlib.pyplot as plt

from common import (style, save, read, load_json, grab, NUM, gpt2_dv_table, PLUGIN, MUTED, INK, INK2)


def oracle():
    t = read("oracle/hessian_vs_fisher.txt")
    pred = {}
    for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s+(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + r")\s*$",
                         t, re.M):
        pred[int(m.group(1))] = float(m.group(4))
    meas = {}
    for k in pred:
        r = load_json(f"oracle/hmc_n16384/ksweep_map_k{k}.json")[0]
        meas[k] = (r["corrected_slope"], r["seed_se"], r["lambda_bayes"])
    return pred, meas


def gpt2():
    def p(f, key):
        return load_json(f"stage1_ioi/{f}")[key]["predictions"]["1024"]["plugin_slope_tr_FHinv"]
    tab = gpt2_dv_table()
    loc = {r["configuration"]: (r["corrected_slope"], r["seed_se"])
           for r in load_json("stage1_ioi/h_a3_gpt2_map_l10_l12.json")["rows"]}
    d = int(grab(r"width (\d+), d_V 16", read("stage1_ioi/regret_slopes_map_code.txt")))
    nom = lambda dv: dv * (d - dv) / 2
    return [
        # label, predicted, measured (value, se) or None, nominal
        ("L11, d_V 4", p("curvature_gpt2_layer11_dv4_full_heldout_fisher.json", "layer11"), tab["valid"][0], nom(4)),
        ("L11, d_V 16", p("curvature_gpt2_layer11_full_heldout_fisher.json", "layer11"), tab["valid"][1], nom(16)),
        ("L11, d_V 64", p("curvature_gpt2_layer11_dv64_full_heldout_fisher.json", "layer11"), tab["valid"][2], nom(64)),
        ("L10, d_V 16", p("curvature_gpt2_layer10_full_heldout_fisher.json", "layer10"), loc["trained/layer10/subspace"],
         nom(16)),
        ("L12, d_V 16", p("curvature_gpt2_layer12_full_heldout_fisher.json", "layer12"), loc["trained/layer12/subspace"],
         nom(16)),
    ]


def panel(ax, compact=False):
    """compact: one label for the GPT-2 cluster and the legend at upper left (slide composite)."""
    lim = (0.3, 6e4)
    ax.plot(lim, lim, color=MUTED, lw=1.2)
    if not compact:
        ax.text(1.6, 0.72, "measured = predicted", color=INK2, fontsize=13)

    pred, meas = oracle()
    for i, k in enumerate(sorted(pred)):
        y, e, nom = meas[k]
        ax.plot([pred[k], nom], [y, y], color=MUTED, lw=1, ls=":", zorder=1)
        ax.plot(nom, y, "o", mfc="white", mec=MUTED, mew=1.5, ms=8, zorder=2)
        ax.errorbar(pred[k], y, yerr=e, fmt="o", color=PLUGIN, capsize=3, ms=9, zorder=3,
                    label="plug-in prediction ½ tr(F H⁻¹)" if i == 0 else None)
    if compact:
        ax.text(1.5, 0.6, "oracle, k = 1 … 11", fontsize=13, color=INK2, va="center")
    else:
        ax.text(0.7, 7.5, "oracle, k = 1 … 11 readable\ndirections (d = 12, n ≤ 16384)", fontsize=13, color=INK2)

    rows = gpt2()
    # label positions (data coordinates) with leader lines, so the five clustered GPT-2 points stay legible
    at = {"L11, d_V 4": (40, 170), "L12, d_V 16": (200, 140), "L10, d_V 16": (9, 95),
          "L11, d_V 64": (330, 40), "L11, d_V 16": (12, 40)}
    for i, (label, p, m, nom) in enumerate(rows):
        ax.plot([p, nom], [m[0], m[0]], color=MUTED, lw=1, ls=":", zorder=1)
        ax.plot(nom, m[0], "D", mfc="white", mec=MUTED, mew=1.5, ms=9, zorder=2,
                label="Bayes λ at the same measured κ" if i == 0 else None)
        ax.errorbar(p, m[0], yerr=m[1], fmt="D", color=PLUGIN, capsize=3, ms=10, zorder=3)
        if not compact:
            ax.annotate("GPT-2 " + label, (p, m[0]), xytext=at[label], fontsize=12, color=INK, ha="center",
                        va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=6))
    if compact:
        ax.text(2.2, 65, "GPT-2, layers 10–12,\nd_V 4–64", fontsize=13, color=INK, ha="center", va="center")
    _, p16, m16, n16 = rows[1]
    ax.annotate("", xy=(n16, m16[0] * 0.62), xytext=(p16, m16[0] * 0.62),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.5))
    ax.text((p16 * n16) ** 0.5, m16[0] * 0.5, f"{n16 / p16:.0f}× at d_V 16", ha="center", va="top", fontsize=15,
            color=INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(lim[0], 400)
    ax.set_xlabel("predicted coefficient per ln n")
    ax.set_ylabel("measured κ (MAP code)")
    ax.legend(loc="upper left" if compact else "lower right")


def main():
    style()
    fig, ax = plt.subplots(figsize=(11, 6.4))
    panel(ax)
    save(fig, "predicted_vs_measured")


if __name__ == "__main__":
    main()
