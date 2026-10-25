"""(a, variant) GPT-2 layer 11: the learner coefficient kappa of the MAP code against the subspace dimension
d_V, next to the nominal Bayes coefficient d_V (d - d_V) / 2 and the plug-in prediction (1/2) tr(F H^-1).

Sources:
  reports/stage1_ioi/regret_slopes_map_code.txt  (d_V 4 / 16 / 64 table; 2 seeds at 4 and 64, 3 at 16)
  reports/stage1_ioi/curvature_gpt2_layer11_dv4_full_heldout_fisher.json, ..._layer11_full_heldout_fisher.json
      ..._layer11_dv64_full_heldout_fisher.json (plug-in predictions; d_V 64 from job 1820081)
"""
import numpy as np
import matplotlib.pyplot as plt

from common import (style, save, load_json, read, grab, gpt2_dv_table, VALID, COUPLING, TWIN, MANUF, MUTED,
                    INK2, PLUGIN)


def panel(ax, compact=False):
    """compact: short series labels spaced for a low panel, legend at lower right (slide composite)."""
    tab = gpt2_dv_table()
    d = int(grab(r"width (\d+), d_V 16", read("stage1_ioi/regret_slopes_map_code.txt")))
    dvs = np.array([4, 16, 64])
    nominal = dvs * (d - dvs) / 2
    pred = {
        4: load_json("stage1_ioi/curvature_gpt2_layer11_dv4_full_heldout_fisher.json")["layer11"]["predictions"]["1024"]["plugin_slope_tr_FHinv"],
        16: load_json("stage1_ioi/curvature_gpt2_layer11_full_heldout_fisher.json")["layer11"]["predictions"]["1024"]["plugin_slope_tr_FHinv"],
        64: load_json("stage1_ioi/curvature_gpt2_layer11_dv64_full_heldout_fisher.json")["layer11"]["predictions"]["1024"]["plugin_slope_tr_FHinv"],
    }
    ax.plot(dvs, nominal, color=MUTED, ls="--", marker="o", mfc="white", lw=2)
    for x, v in zip(dvs, nominal):
        ax.text(x, v * 1.35, f"{v:,.0f}", ha="center", color=INK2, fontsize=14)
    ax.text(17, 2300, "nominal Bayes λ = d_V(d − d_V)/2", color=INK2, fontsize=15)
    series = [("manufactured" if compact else "manufactured (untrained twin)", "manufactured", MANUF),
              ("coupling (trained)", "coupling", COUPLING),
              ("valid subspace", "valid", VALID),
              ("untrained twin" if compact else "untrained twin, subspace", "twin", TWIN)]
    offsets = {"coupling": 1.25, "valid": 1.5, "twin": 0.62} if compact else {"valid": 1.3, "twin": 0.78}
    for label, key, c in series:
        v = np.array([t[0] for t in tab[key]])
        e = np.array([t[1] for t in tab[key]])
        ax.errorbar(dvs, v, yerr=e, color=c, marker="o", capsize=3, lw=2.2)
        off = offsets.get(key, 1.0)
        ax.text(dvs[-1] * 1.12, v[-1] * off, label, color=c if key != "twin" else "#a87200", va="center", fontsize=14)
    px = np.array(sorted(pred))
    ax.plot(px, [pred[k] for k in px], ls="none", marker="D", mfc="white", mec=PLUGIN, mew=2, ms=11,
            label="plug-in prediction ½ tr(F H⁻¹)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dvs, [str(v) for v in dvs])
    ax.set_xlim(3, 400 if compact else 260)
    ax.set_ylim(10, 1e5)
    ax.set_xlabel("subspace dimension d_V  (GPT-2 layer 11, d = 768)")
    ax.set_ylabel("per ln n: κ measured, λ nominal")
    ax.legend(loc="lower right" if compact else "lower left")


def main():
    style()
    fig, ax = plt.subplots(figsize=(10, 6))
    panel(ax)
    save(fig, "kappa_vs_dv")


if __name__ == "__main__":
    main()
