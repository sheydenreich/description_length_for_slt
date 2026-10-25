"""(f) The same verdicts on four language models: gain over the label-only code (4324.5 bits, identical
targets) of the valid subspace at its best locus, the untrained twin, the coupling family on the trained
network, and the manufactured alignment (coupling on the untrained twin), with the manufactured alignment's IIA.

Sources:
  reports/stage1_ioi/gpt2_summary_single_map_map0.md   GPT-2 (layer 11; adversary at layer 10), 3 seeds
  reports/stage1_ioi/adversary_by_learner.txt          GPT-2 adversary IIA under the MAP code (2 decimals)
  reports/stage1_ioi/tier2_models.txt                  Llama-3.1-8B (L30), Qwen2.5-3B (L34), Qwen3-1.7B (L28)
Learners (checked against results/<run>/settings.json): GPT-2, Llama and Qwen2.5 runs used the single-map MAP
code (learner "laplace", 0 posterior samples); the Qwen3 run used the mixture of four. The header of
tier2_models.txt says "mixture-of-4" for all three; that is correct for Qwen3 only.
Qwen3 manufactured arm (mixture of 4, layer 26) archived 2026-10-22.
"""
import re

import numpy as np
import matplotlib.pyplot as plt

from common import (style, save, read, grab, NUM, summary_rows, VALID, COUPLING, TWIN, MANUF, INK, INK2)


def tier2(model_header: str, locus: int):
    t = read("stage1_ioi/tier2_models.txt")
    start = t.index("\n" + model_header + " (")
    end = t.find("\n\n", start)
    b = t[start:end]

    def row(name):
        line = [ln for ln in b.splitlines() if ln.strip().startswith(name)][0]
        m = re.search(r"L" + str(locus) + r"\s+([+-]\d+)(?: \+- \d+)? \((" + NUM + r")\)", line)
        return float(m.group(1)), float(m.group(2))
    man = re.search(r"manufactured \(coupling, twin, L\d+\) ([+-]?\d+)(?: \+- \d+)? \(IIA (" + NUM + r")", b)
    return dict(valid=row("valid subspace"), twin=row("twin"), coupling=row("coupling"),
                manufactured=(float(man.group(1)), float(man.group(2))) if man else None)


def data():
    s = summary_rows("stage1_ioi/gpt2_summary_single_map_map0.md")
    adv = read("stage1_ioi/adversary_by_learner.txt")
    iia = float(grab(r"MAP only \(AdamW, cold start\).*?IIA (" + NUM + ")", adv))
    gpt = dict(valid=(s["trained/layer11/subspace"][0], s["trained/layer11/subspace"][2]),
               twin=(s["trained/layer11/subspace/untrained"][0], s["trained/layer11/subspace/untrained"][2]),
               coupling=(s["trained/layer11/subspace/map=coupling"][0], s["trained/layer11/subspace/map=coupling"][2]),
               manufactured=(s["untrained/layer10/coupling"][0], iia))
    return [
        ("GPT-2 small\nlayer 11, MAP", gpt),
        ("Qwen3-1.7B\nlayer 28, mixture", tier2("Qwen3-1.7B-Base", 28)),
        ("Qwen2.5-3B\nlayer 34, MAP", tier2("Qwen2.5-3B", 34)),
        ("Llama-3.1-8B\nlayer 30, MAP", tier2("Llama-3.1-8B", 30)),
    ]


def main():
    style()
    rows = data()
    keys = [("valid", "valid subspace", VALID), ("coupling", "coupling, trained net", COUPLING),
            ("twin", "subspace, untrained twin", TWIN), ("manufactured", "manufactured (coupling, untrained twin)", MANUF)]
    fig, ax = plt.subplots(figsize=(14, 6.4))
    w = 0.2
    x = np.arange(len(rows))
    for j, (k, lbl, c) in enumerate(keys):
        for i, (_, r) in enumerate(rows):
            xi = x[i] + (j - 1.5) * w
            if r[k] is None:
                ax.text(xi, -150, "pending", rotation=90, ha="center", va="top", fontsize=12, color=INK2)
                continue
            g, iia = r[k]
            ax.bar(xi, g, width=w * 0.92, color=c, label=lbl if i == 0 else None)
            if k == "manufactured":
                ax.text(xi, g - 80, f"IIA\n{iia:.2f}", ha="center", va="top", fontsize=13, color=INK)
    ax.axhline(0, color=INK, lw=1.3)
    ax.text(x[0] - 0.5, -60, "label-only", fontsize=13, color=INK2, ha="left", va="top")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-2900, 4400)
    ax.set_ylabel("gain over label-only (bits),\n1024 targets")
    ax.legend(loc="upper center", ncol=4, fontsize=13, bbox_to_anchor=(0.5, 1.1))
    save(fig, "models_verdicts")


if __name__ == "__main__":
    main()
