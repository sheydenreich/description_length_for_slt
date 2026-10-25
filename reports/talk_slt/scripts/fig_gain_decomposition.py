"""GPT-2 IOI, MAP code to 32768 targets (3 seeds): which term carries the gain of the valid layer-11
subspace over a control?

  gain G(n) = fit term + regret term
  fit term     = loss of the control's final fit − loss of the hypothesis's final fit on the first n
                 targets (≈ n·Δℓ: falsification if > 0; the control fits better if < 0)
  regret term  = difference of the two codes' regrets against their final fits (Δκ ln n + O(1))

Figure 1 (gain_decomposition): the split against the manufactured alignment and the coupling map on the
trained network, with the readout as is (top) and with one learned inverse temperature on every predictor's
logits (bottom, --temperature; against label-only the gain is all fit term in both). Figure 2
(code_rate_per_block): the per-target code rate in each doubling block minus the channel floor, the same
comparison without the final fits (each block is coded before it is fitted). Figure 3
(gain_coupling_pair): the same split between the two coupling maps, on the trained network (layer 11) and on
its untrained twin (layer 10, the manufactured alignment): same family, same parameter count. Its curves are
the difference of two archived splits against the valid subspace (exact: every term is a difference of
per-alignment sums over the same targets and seeds). Figure 4 (gain_decomposition_slide, for the 7-minute deck):
the readout-as-is split against the manufactured alignment and against the trained coupling map, and the
same-family pair of figure 3, side by side.

Colour roles: total = ink, regret term = plug-in blue (it carries κ), fit term = magenta, dashed
(validated as a pair with the dataviz checker); configurations in figure 2 use the common colours.

Sources: reports/stage1_ioi/gain_decomposition_gpt2_n32768.json, gain_decomposition_gpt2_n32768_temp.json
  (python -m bitsev.experiments.gain_decomposition results/stage1_ioi_gpt2_map_n32768 ...).
"""
import numpy as np
import matplotlib.pyplot as plt

from common import style, save, load_json, INK, INK2, MUTED, PLUGIN, VALID, COUPLING, MANUF

FIT = "#d55181"
LN2 = np.log(2)
SRC = "stage1_ioi/gain_decomposition_gpt2_n32768.json"
SRC_TEMP = "stage1_ioi/gain_decomposition_gpt2_n32768_temp.json"
MANUFACTURED = "untrained/layer10/coupling"
COUPLING_TRAINED = "trained/layer11/subspace/map=coupling"
PANELS = [
    (MANUFACTURED, "vs manufactured alignment"),
    (COUPLING_TRAINED, "vs coupling map, trained network"),
]
ROWS = [(SRC, "readout as is"), (SRC_TEMP, "one learned temperature per predictor")]


def band(ax, n, arr, color, **kw):
    """Seed mean with a ±1 s.e. band, in thousands of bits."""
    arr = np.asarray(arr) / LN2 / 1e3
    m = arr.mean(0)
    se = arr.std(0, ddof=1) / np.sqrt(len(arr))
    ax.fill_between(n, m - se, m + se, color=color, alpha=0.18, lw=0)
    ax.plot(n, m, color=color, **kw)
    return m


def split_panel(ax, n, g, title):
    """Gain, regret term and fit term of one comparison against n; returns their values at the last n."""
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(1024, color=MUTED, lw=1, ls=":")
    tot = band(ax, n, g["gain"], INK, lw=2.6, marker="o", ms=6, label="gain G(n)")
    reg = band(ax, n, g["regret_term"], PLUGIN, lw=2.2, marker="s", ms=6, label="regret term (Δκ ln n + O(1))")
    fit = band(ax, n, g["fit_term"], FIT, lw=2.2, ls="--", marker="D", ms=6, label="fit term (≈ n·Δℓ)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 128, 1024, 32768], ["16", "128", "1024", "32768"])
    ax.set_title(title, fontsize=15, loc="left")
    ax.set_xlim(12, 32768 * 2.6)
    return tot[-1], reg[-1], fit[-1]


def end_labels(ax, x, values):
    """Label the curves at the last n, gain first; a term that would sit on an earlier label is left out.
    Call after the y-range is final."""
    placed = []
    for v in values:
        if all(abs(v - p) > 0.06 * np.ptp(ax.get_ylim()) for p in placed):
            placed.append(v)
            txt = f"{v:+.1f}k" if abs(v) >= 0.05 else "0.0k"
            ax.annotate(txt, (x, v), xytext=(6, 0), textcoords="offset points", va="center", color=INK2,
                        fontsize=13)


def decomposition(runs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4), sharex=True)
    ends = {}
    for row, (d, row_title) in zip(axes, runs):
        n = np.array(d["block_ends"])
        for ax, (key, title) in zip(row, PANELS):
            ends[ax] = (n[-1], split_panel(ax, n, d["gains"][key], f"{title}\n{row_title}"))
        row[0].set_ylabel("thousands of bits")
    for ax in axes[1]:
        ax.set_xlabel("targets n")
    for col in range(2):  # same y-range within a column, so the rows compare directly
        lo = min(axes[r][col].get_ylim()[0] for r in range(2))
        hi = max(axes[r][col].get_ylim()[1] for r in range(2))
        for r in range(2):
            axes[r][col].set_ylim(lo, hi)
    for ax, (x, values) in ends.items():  # end labels after the shared y-range is set
        end_labels(ax, x, values)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=15)
    fig.tight_layout()
    save(fig, "gain_decomposition")


def pair_split(d, hyp, ctrl):
    """Split of the gain of control-hypothesis ``hyp`` over ``ctrl``, from two splits against the same
    reference: G(hyp -> ctrl) = G(ref -> ctrl) - G(ref -> hyp), term by term."""
    a, b = d["gains"][ctrl], d["gains"][hyp]
    return {k: (np.asarray(a[k]) - np.asarray(b[k])).tolist() for k in ("gain", "fit_term", "regret_term")}


def coupling_pair(runs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, (d, title) in zip(axes, runs):
        n = np.array(d["block_ends"])
        g = pair_split(d, COUPLING_TRAINED, MANUFACTURED)
        tot, _, fit = split_panel(ax, n, g, title)
        ax.set_xlabel("targets n")
        for v in (tot, fit):
            txt = f"{v:+.1f}k" if abs(v) >= 0.05 else "0.0k"
            ax.annotate(txt, (n[-1], v), xytext=(6, 0), textcoords="offset points", va="center",
                        color=INK2, fontsize=13)
    axes[0].set_ylabel("thousands of bits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=15)
    fig.tight_layout()
    save(fig, "gain_coupling_pair")


def slide(d):
    """Readout as is: the valid subspace against the manufactured alignment and against the coupling map on the
    trained network, and the fair comparison of the two equally expressive coupling maps; own y-range each."""
    n = np.array(d["block_ends"])
    panels = [(d["gains"][MANUFACTURED], "valid subspace vs\nmanufactured alignment"),
              (d["gains"][COUPLING_TRAINED], "valid subspace vs\ncoupling map, trained network"),
              (pair_split(d, COUPLING_TRAINED, MANUFACTURED),
               "coupling map, trained vs untrained\nnetwork (same family, same size)")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    ends = [split_panel(ax, n, g, title) for ax, (g, title) in zip(axes, panels)]
    axes[0].set_ylabel("thousands of bits")
    for ax, values in zip(axes, ends):
        ax.set_xlabel("targets n")
        end_labels(ax, n[-1], values)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=15)
    fig.tight_layout()
    save(fig, "gain_decomposition_slide")


def rates(d):
    n = np.array(d["block_ends"])
    floor = d["channel_floor_nats"]
    series = [("trained/layer11/subspace", "valid subspace (layer 11)", VALID, "o"),
              ("trained/layer11/subspace/map=coupling", "coupling map, trained network", COUPLING, "s"),
              ("untrained/layer10/coupling", "manufactured alignment", MANUF, "D")]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, c, mk in series:
        r = np.asarray(d["codes"][key]["block_rate"]) - floor
        m = r.mean(0)
        se = r.std(0, ddof=1) / np.sqrt(len(r))
        ax.errorbar(n, m, yerr=se, color=c, marker=mk, capsize=3, lw=2.2, label=label)
        ax.annotate(f"{m[-1]:.3f}", (n[-1], m[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 128, 1024, 8192, 32768], ["16", "128", "1024", "8192", "32768"])
    ax.set_xlim(12, 32768 * 2)
    ax.set_xlabel("targets n (end of doubling block)")
    ax.set_ylabel("code rate in block − channel floor\n(nats per target)")
    ax.set_title(f"channel floor {floor:.3f} nats per target (ε = 0.05, 32 names)", fontsize=15,
                 loc="left", color=INK2)
    ax.legend(loc="upper right")
    save(fig, "code_rate_per_block")


def main():
    style()
    d = load_json(SRC)
    decomposition([(d, ROWS[0][1]), (load_json(SRC_TEMP), ROWS[1][1])])
    rates(d)
    coupling_pair([(d, ROWS[0][1]), (load_json(SRC_TEMP), ROWS[1][1])])
    slide(d)


if __name__ == "__main__":
    main()
