"""(e) Competence onset in Pythia-1.4b: task accuracy (top) and the code of the valid alignment and its
controls (bottom) against the training checkpoint. Single-map MAP code, one seed, 1024 targets.

Source: reports/stage1_ioi/pythia14_checkpoint_series.txt (text table; accuracy at step 3000 not measured;
label-only "~4325 bits" from the header).
"""
import re

import numpy as np
import matplotlib.pyplot as plt

from common import style, save, read, grab, NUM, VALID, COUPLING, TWIN, MANUF, INK, INK2, MUTED


def data():
    t = read("stage1_ioi/pythia14_checkpoint_series.txt")
    L = float(grab(r"label-only ~(\d+) bits", t))
    rows = []
    cell = r"(\d+) \((" + NUM + r")\)"
    pat = re.compile(r"^\s*(\d+)\s+(" + NUM + r"|-)\s+" + r"\s+".join([cell] * 5) + r"\s*$", re.M)
    for m in pat.finditer(t):
        g = m.groups()
        acc = float(g[1]) if g[1] != "-" else np.nan
        codes = [float(g[i]) for i in (2, 4, 6, 8, 10)]
        iias = [float(g[i]) for i in (3, 5, 7, 9, 11)]
        rows.append((int(g[0]), acc, codes, iias))
    return L, rows


def main():
    style()
    L, rows = data()
    steps = np.array([r[0] for r in rows])
    acc = np.array([r[1] for r in rows])
    codes = np.array([r[2] for r in rows])
    iias = np.array([r[3] for r in rows])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.08})
    ok = ~np.isnan(acc)
    a1.plot(steps[ok], acc[ok], "-o", color=INK, lw=2.2)
    a1.set_ylim(0, 1.1)
    a1.set_yticks([0, 0.5, 1])
    a1.set_ylabel("IOI accuracy")

    series = [("valid subspace, layer 20", 1, VALID, "-o"), ("valid subspace, layer 16", 0, VALID, "--o"),
              ("coupling, trained, layer 20", 3, COUPLING, "-o"), ("untrained twin subspace, layer 20", 2, TWIN, "-o"),
              ("manufactured, untrained twin (IIA 0.78)", 4, MANUF, "-o")]
    for label, j, c, fmt in series:
        if j in (2, 4):  # untrained twin: does not depend on the checkpoint, one value
            a2.axhline(codes[0, j], color=c, lw=2.2, ls=(0, (6, 3)), label=label)
            continue
        a2.plot(steps, codes[:, j], fmt, color=c, lw=2.2, label=label, mfc=c if fmt == "-o" else "white")
    a2.axhline(L, color=INK2, lw=1.5)
    a2.text(steps[-1] * 1.08, L, "label-only", va="center", fontsize=13, color=INK2)
    for s, c, i in zip(steps, codes[:, 1], iias[:, 1]):
        a2.text(s, c - 300, f"{i:.2f}", ha="center", va="top", fontsize=12, color=VALID,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    a2.text(steps[0] * 0.9, 80, "numbers: IIA of the layer-20 subspace", fontsize=12, color=VALID, va="bottom")
    a2.set_xscale("log")
    a2.set_ylim(0, 7600)
    a2.set_ylabel("code length (bits)")
    a2.set_xlabel("Pythia-1.4b training step")
    a2.set_xticks(steps, [f"{s // 1000}k" for s in steps])
    a2.minorticks_off()
    a2.legend(loc="upper center", fontsize=13, ncol=2)
    save(fig, "pythia_onset")


if __name__ == "__main__":
    main()
