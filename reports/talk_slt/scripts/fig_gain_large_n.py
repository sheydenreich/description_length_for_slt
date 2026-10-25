"""(backup) GPT-2 IOI, MAP code to 131072 targets (3 seeds, Adam steps per block raised to 10 passes over the
prefix): gain of the valid layer-11 subspace over the manufactured alignment and over the coupling map on the
trained network, and of the trained coupling over the manufactured alignment, with the readout as is and with
one learned inverse temperature per predictor. Gains only: at this size the final fits are noisy endpoints of
constant-rate Adam, so the fit/regret split is not shown (see stage1_ioi/large_n_gains.txt).

Colour roles: readout as is = ink, tempered = plug-in blue (validated pair, as in fig_gain_decomposition).

Sources: reports/stage1_ioi/gain_decomposition_gpt2_n131072_ep10.json, gain_decomposition_gpt2_n131072_ep10_temp.json
"""
import numpy as np
import matplotlib.pyplot as plt

from common import style, save, load_json, INK, INK2, MUTED, PLUGIN

LN2 = np.log(2)
RUNS = [("stage1_ioi/gain_decomposition_gpt2_n131072_ep10.json", "readout as is", INK, "o"),
        ("stage1_ioi/gain_decomposition_gpt2_n131072_ep10_temp.json", "one learned temperature per predictor",
         PLUGIN, "s")]
MANUF, COUPLING = "untrained/layer10/coupling", "trained/layer11/subspace/map=coupling"
PANELS = [("valid subspace vs manufactured", lambda g: g[MANUF]),
          ("valid subspace vs coupling (trained)", lambda g: g[COUPLING]),
          ("coupling (trained) vs manufactured", lambda g: np.asarray(g[MANUF]) - np.asarray(g[COUPLING]))]


def main():
    style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for src, label, color, marker in RUNS:
        d = load_json(src)
        n = np.array(d["block_ends"])
        gains = {k: v["gain"] for k, v in d["gains"].items()}
        for ax, (title, pick) in zip(axes, PANELS):
            g = np.asarray(pick(gains)) / LN2 / 1e3
            m, se = g.mean(0), g.std(0, ddof=1) / np.sqrt(len(g))
            ax.fill_between(n, m - se, m + se, color=color, alpha=0.15, lw=0)
            ax.plot(n, m, color=color, marker=marker, ms=6, lw=2.2, label=label)
            ax.annotate(f"{m[-1]:+.1f}k", (n[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=13)
    for ax, (title, _) in zip(axes, PANELS):
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(32768, color=MUTED, lw=1, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 256, 4096, 131072], ["16", "256", "4096", "131072"])  # dotted: 32768
        ax.set_xlim(12, 131072 * 3)
        ax.set_title(title, fontsize=16, loc="left")
        ax.set_xlabel("targets n")
    axes[0].set_ylabel("gain, thousands of bits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07), fontsize=15)
    fig.tight_layout()
    save(fig, "gain_large_n")


if __name__ == "__main__":
    main()
