"""(h) The oracle: MAP code (kappa) against Bayes code by HMC (lambda), and how both move from n <= 2048 to
n <= 16384. Left: k readable directions of d - 1 = 11 (d = 12, no channel). Right: readout sharpness at k = 5
(d = 6, continuous V, channel 0.1). Arrows go from the n <= 2048 slope to the n <= 16384 slope.

Sources:
  reports/oracle/bayes_code_hmc.txt          slopes on n in [256, 2048] (both learners, both sweeps; text tables)
  reports/oracle/hmc_n16384/ksweep_{hmc,map}_k*.json, reports/oracle/sharp_n16384/x*_{hmc,map}.json  (n <= 16384)
  reports/oracle/hessian_vs_fisher.txt       plug-in prediction (1/2) tr(F H^+) per k
The Bayes reference line is lambda = k/2 + (11 - k)/4 ("lambda_bayes" in the records): the 11 - k unreadable
directions are quartic, not flat, so the nominal (d - 1)/2 = 5.5 holds only at k = 11.
"""
import re

import numpy as np
import matplotlib.pyplot as plt

from common import style, save, read, load_json, NUM, PLUGIN, BAYES, MUTED, INK, INK2

PM = r"(" + NUM + r") \+- (" + NUM + r")"


def small_n():
    t = read("oracle/bayes_code_hmc.txt")
    ks = t.split("k sweep, d = 12")[1].split("Reading.")[0]
    k = {int(m.group(1)): (float(m.group(3)), float(m.group(5)))
         for m in re.finditer(r"^\s*(\d+)\s+(" + NUM + r")\s+" + PM + r"\s+" + PM, ks, re.M)}
    k = {kk: (v[0], v[1]) for kk, v in k.items()}
    sh = t.split("Sharpness sweep, d = 6")[1].split("k sweep")[0]
    s = {}
    for m in re.finditer(r"^\s*x(\d+)( \(none\))?\s+" + PM + r"\s+" + PM, sh, re.M):
        if m.group(2):
            continue  # the matched-sharpness row without the learner's channel
        s[int(m.group(1))] = (float(m.group(3)), float(m.group(5)))
    return k, s


def large_n():
    k = {}
    for kk in (1, 2, 4, 7, 11):
        m = load_json(f"oracle/hmc_n16384/ksweep_map_k{kk}.json")[0]
        b = load_json(f"oracle/hmc_n16384/ksweep_hmc_k{kk}.json")[0]
        k[kk] = ((m["corrected_slope"], m["seed_se"]), (b["corrected_slope"], b["seed_se"]), b["lambda_bayes"])
    s = {}
    for x in (1, 10, 30, 100):
        m = load_json(f"oracle/sharp_n16384/x{x}_map.json")[0]
        b = load_json(f"oracle/sharp_n16384/x{x}_hmc.json")[0]
        s[x] = ((m["corrected_slope"], m["seed_se"]), (b["corrected_slope"], b["seed_se"]))
    return k, s


def trace_pred():
    t = read("oracle/hessian_vs_fisher.txt")
    return {int(m.group(1)): float(m.group(4)) for m in re.finditer(
        r"^\s*(\d+)\s+(\d+)\s+(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + r")\s*$", t, re.M)}


def arrows(ax, xs, y0, y1, e1, color, marker, label, dx=1.0):
    xs = np.asarray(xs, float) * dx
    ax.plot(xs, y0, marker, mfc="white", mec=color, mew=2, ms=10, ls="none")
    ax.errorbar(xs, y1, yerr=e1, fmt=marker, color=color, ms=10, capsize=3, label=label)
    for x, a, b in zip(xs, y0, y1):
        ax.annotate("", xy=(x, b), xytext=(x, a), arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                                                   shrinkA=6, shrinkB=6))


def draw_k(a1, ks, kl, tp):
    K = sorted(kl)
    a1.plot(K, [kl[k][2] for k in K], color=MUTED, ls="--", lw=1.5)
    a1.text(1.3, 3.95, "Bayes λ = k/2 + (11 − k)/4", color=INK2, fontsize=13, rotation=9)
    a1.plot([0, 12], [0, 6], color=MUTED, ls=":", lw=1.5)
    a1.text(9.2, 4.0, "k/2", color=INK2, fontsize=13)
    a1.plot(K, [tp[k] for k in K], "x", color=INK, ms=11, mew=2.2, label="½ tr(F H⁺) at the MAP")
    arrows(a1, K, [ks[k][0] for k in K], [kl[k][0][0] for k in K], [kl[k][0][1] for k in K], PLUGIN, "o",
           "MAP code κ", dx=0.97)
    arrows(a1, K, [ks[k][1] for k in K], [kl[k][1][0] for k in K], [kl[k][1][1] for k in K], BAYES, "s",
           "Bayes code λ (HMC)", dx=1.03)
    a1.set_xlim(0, 12)
    a1.set_ylim(0, 8.4)
    a1.set_xticks(K)
    a1.set_xlabel("readable directions k (of 11)")
    a1.set_ylabel("regret slope per ln n")
    a1.legend(loc="upper left", fontsize=13, ncol=1)
    a1.text(0.98, 0.03, "open: n ≤ 2048   filled: n ≤ 16384", transform=a1.transAxes, ha="right",
            fontsize=12, color=INK2)



def draw_s(a2, ss, sl):
    X = sorted(sl)
    a2.axhline(2.5, color=MUTED, ls=":", lw=1.5)
    a2.text(3.2, 2.0, "k/2 = 2.5", color=INK2, fontsize=13)
    a2.axhline(5, color=MUTED, ls="--", lw=1.5)
    a2.text(2.0, 5.2, "k = 5 (one per direction)", color=INK2, fontsize=13)
    arrows(a2, X, [ss[x][0] for x in X], [sl[x][0][0] for x in X], [sl[x][0][1] for x in X], PLUGIN, "o",
           "MAP code κ", dx=0.93)
    arrows(a2, X, [ss[x][1] for x in X], [sl[x][1][0] for x in X], [sl[x][1][1] for x in X], BAYES, "s",
           "Bayes code λ (HMC)", dx=1.07)
    a2.set_xscale("log")
    a2.set_xlim(0.7, 150)
    a2.set_ylim(0, 16)
    a2.set_xticks(X, [f"×{x}" for x in X])
    a2.minorticks_off()
    a2.set_xlabel("readout sharpness (k = 5, d = 6)")
    a2.set_ylabel("regret slope per ln n")
    a2.legend(loc="upper left", fontsize=13)


def main():
    style()
    ks, ss = small_n()
    kl, sl = large_n()
    tp = trace_pred()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"wspace": 0.22})
    draw_k(a1, ks, kl, tp)
    draw_s(a2, ss, sl)
    save(fig, "oracle_kappa_vs_lambda")
    fig, ax = plt.subplots(figsize=(10, 6.4))
    draw_k(ax, ks, kl, tp)
    save(fig, "oracle_ksweep_slide")
    fig, ax = plt.subplots(figsize=(10, 6.4))
    draw_s(ax, ss, sl)
    save(fig, "oracle_sharpness_slide")


if __name__ == "__main__":
    main()
