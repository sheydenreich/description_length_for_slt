"""(i) MIB two-digit addition, variable "carry": gain over the trivial floor (layer 0, where the interchange does
nothing and the model answers from the base) of the subspace map and the coupling map on the trained network,
per layer, for Llama-3.1-8B and Gemma-2-2b; IIA printed at each point.

Source: reports/stage1_mib/arithmetic.txt (text tables). Learner: single-map MAP code (results/<run>/settings.json:
learner "laplace", 0 posterior samples; the file header says mixture-of-4).
"""
import re

import numpy as np
import matplotlib.pyplot as plt

from common import style, save, read, NUM, VALID, COUPLING, MANUF, INK, INK2, MUTED


def table(model_header):
    t = read("stage1_mib/arithmetic.txt")
    b = t[t.index(model_header):]
    b = b[:b.index("manufactured")]
    rows = []
    cell = r"([+-]\d+) \((" + NUM + r")\)"
    for m in re.finditer(r"^\s*L(\d+)(?: \(floor\))?\s+" + cell + r"\s+" + cell + r"\s+" + cell, b, re.M):
        g = m.groups()
        rows.append((int(g[0]), float(g[1]), float(g[2]), float(g[3]), float(g[4])))
    man = re.search(r"manufactured \(coupling, twin, L(\d+)\): ([+-]\d+) bits, IIA (" + NUM + ")",
                    t[t.index(model_header):])
    return rows, (int(man.group(1)), float(man.group(2)), float(man.group(3)))


def main():
    style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (hdr, name) in zip(axes, [("Llama-3.1-8B (label-only", "Llama-3.1-8B (whole-number answer)"),
                                       ("Gemma-2-2b (label-only", "Gemma-2-2b (first answer digit)")]):
        rows, man = table(hdr)
        L = np.array([r[0] for r in rows])
        floor = rows[0][1]
        sub = np.array([r[1] for r in rows]) - floor
        cou = np.array([r[3] for r in rows]) - floor
        ax.axhline(0, color=INK, lw=1.3)
        ax.plot(L, sub, "-o", color=VALID, lw=2.4, label="16-dim subspace")
        ax.plot(L, cou, "-o", color=COUPLING, lw=2.4, label="coupling map (trained network)")
        for x, ys, yc, i_s, i_c in zip(L, sub, cou, [r[2] for r in rows], [r[4] for r in rows]):
            up = 1 if yc >= ys else -1  # coupling label on the outside of the pair, subspace label opposite
            ax.text(x, yc + up * 140, f"{i_c:.2f}", ha="center", va="bottom" if up > 0 else "top", fontsize=12,
                    color="#127a55")
            ax.text(x, ys - up * 140, f"{i_s:.2f}", ha="center", va="top" if up > 0 else "bottom", fontsize=12,
                    color=VALID)
        ax.text(0.02, 0.03, f"manufactured (untrained twin, L{man[0]}): {man[1] - floor:+.0f} vs floor, IIA {man[2]:.2f}",
                transform=ax.transAxes, fontsize=12, color=MANUF)
        ax.set_title(name, loc="left", fontsize=16)
        ax.set_xlabel("layer")
        ax.set_xticks(L)
    axes[0].set_ylabel("gain over the trivial floor (bits)")
    fig.text(0.5, -0.04, "numbers: IIA; floor = layer 0, the model's own answer on the base input", ha="center", fontsize=13, color=INK2)
    axes[0].set_ylim(-3400, 3000)
    axes[1].legend(loc="upper left", fontsize=13)
    save(fig, "mib_arithmetic")


if __name__ == "__main__":
    main()
