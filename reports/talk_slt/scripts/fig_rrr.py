"""(g) Reduced-rank regression, the exact check: the Bayes code's regret slope is the Aoyagi-Watanabe lambda,
the least-squares plug-in's is the regular MN/2 whatever the true rank; gains between two families on the same
data grow like Delta-lambda ln n.

Sources (text tables, n = 65536, 16 seeds, slopes on n >= 256, batching-corrected):
  reports/rrr/codes_summary.txt, reports/rrr/gain_summary.txt     per-shape slopes
  reports/rrr/gain_between_families.txt                           between-family gain slopes
"""
import re

import numpy as np
import matplotlib.pyplot as plt

from common import style, save, read, NUM, PLUGIN, BAYES, MUTED, INK, INK2

SHAPE = r"\((\d+), (\d+), (\d+), (\d+)\)"
PM = r"(" + NUM + r") ± (" + NUM + r")"


def shapes():
    rows = {}
    for f in ("rrr/codes_summary.txt", "rrr/gain_summary.txt"):
        for m in re.finditer(r"^" + SHAPE + r"\s+(" + NUM + r")\s+(\d+)\s+(" + NUM + r")\s+" + PM + r"\s+" + PM,
                             read(f), re.M):
            g = m.groups()
            key = tuple(int(x) for x in g[:4])
            rows[key] = dict(lam=float(g[4]), m=int(g[5]), mn2=float(g[6]), ols=(float(g[7]), float(g[8])),
                             bayes=(float(g[9]), float(g[10])))
    return rows


def gains():
    out = []
    for m in re.finditer(SHAPE + r" vs " + SHAPE + r"\s+" + NUM + r" / " + NUM + r"\s+(" + NUM + r")\s+" + PM,
                         read("rrr/gain_between_families.txt")):
        g = m.groups()
        out.append((float(g[8]), float(g[9]), float(g[10])))
    return out


def main():
    style()
    rows = shapes()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.25})
    lam = np.array([r["lam"] for r in rows.values()])
    mn2 = np.array([r["mn2"] for r in rows.values()])
    b = np.array([r["bayes"] for r in rows.values()])
    o = np.array([r["ols"] for r in rows.values()])
    a1.plot([0, 9], [0, 9], color=MUTED, lw=1.2)
    a1.text(7.6, 7.0, "slope = λ", color=INK2, fontsize=13, rotation=0)
    a1.plot(lam, mn2, "_", color=INK, ms=26, mew=2.5, label="regular count MN/2", zorder=2)
    a1.errorbar(lam, o[:, 0], yerr=o[:, 1], fmt="o", color=PLUGIN, ms=9, capsize=3, label="plug-in (least squares)",
                zorder=3)
    a1.errorbar(lam, b[:, 0], yerr=b[:, 1], fmt="s", color=BAYES, ms=9, capsize=3, label="Bayes code (HMC)", zorder=4)
    a1.set_xlim(1, 8.2)
    a1.set_ylim(0.8, 9.2)
    a1.set_xlabel("Aoyagi–Watanabe λ of the shape (M, N, H, r)")
    a1.set_ylabel("regret slope per ln n")
    a1.legend(loc="lower right")

    g = np.array(gains())
    a2.plot([0, 5], [0, 5], color=MUTED, lw=1.2)
    a2.errorbar(g[:, 0], g[:, 1], yerr=g[:, 2], fmt="s", color=BAYES, ms=10, capsize=3)
    a2.set_xlim(0.5, 4.8)
    a2.set_ylim(0.5, 4.8)
    a2.set_xlabel("Δλ between the two families")
    a2.set_ylabel("Bayes gain slope per ln n")
    a2.text(0.6, 4.4, "rank-restricted vs full family,\nsame data, Bayes codes", fontsize=13, color=INK2, va="top")
    save(fig, "rrr_exact_check")


if __name__ == "__main__":
    main()
