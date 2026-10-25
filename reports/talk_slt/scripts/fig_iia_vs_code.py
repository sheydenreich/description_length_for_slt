"""Opening figure: interchange-intervention accuracy against code gain over label-only, every configuration of
the IOI runs on four language models. IIA does not separate the manufactured alignments from faithful ones;
the code does.

Sources:
  reports/stage1_ioi/tier2_models.txt                 Llama-3.1-8B, Qwen2.5-3B (MAP code), Qwen3-1.7B (mixture)
  reports/stage1_ioi/gpt2_summary_single_map_map0.md  GPT-2 (MAP code; IIA rounded to one decimal in the summary)
  reports/stage1_ioi/adversary_by_learner.txt         GPT-2 manufactured alignment, IIA 0.83 under the MAP code
"""
import re

import matplotlib.pyplot as plt

from common import (style, save, read, grab, NUM, summary_rows, VALID, COUPLING, TWIN, MANUF, MUTED, INK, INK2)

KIND = {"valid subspace": ("valid subspace", VALID), "twin": ("subspace on untrained twin", TWIN),
        "coupling": ("coupling, trained net", COUPLING)}


def tier2_points():
    t = read("stage1_ioi/tier2_models.txt")
    pts = []
    for model in ("Llama-3.1-8B", "Qwen2.5-3B", "Qwen3-1.7B-Base"):
        s = t.index("\n" + model + " (")
        b = t[s:t.find("\n\n", s + 1)]
        for ln in b.splitlines():
            st = ln.strip()
            for key in KIND:
                if st.startswith(key):
                    for m in re.finditer(r"L(\d+)\s+([+-]\d+)(?: \+- \d+)? \((" + NUM + r")\)", st):
                        pts.append((model, key, float(m.group(3)), float(m.group(2))))
        m = re.search(r"wrong locus\s+([+-]\d+) \((" + NUM + r")\)", b)
        pts.append((model, "wrong locus", float(m.group(2)), float(m.group(1))))
        m = re.search(r"manufactured \(coupling, twin, L\d+\) ([+-]?\d+)(?: \+- \d+)? \(IIA (" + NUM + r")", b)
        if m:
            pts.append((model, "manufactured", float(m.group(2)), float(m.group(1))))
    return pts


def gpt2_points():
    s = summary_rows("stage1_ioi/gpt2_summary_single_map_map0.md")
    iia = float(grab(r"MAP only \(AdamW, cold start\).*?IIA (" + NUM + ")", read("stage1_ioi/adversary_by_learner.txt")))
    return [("GPT-2", "valid subspace", s["trained/layer11/subspace"][2], s["trained/layer11/subspace"][0]),
            ("GPT-2", "twin", s["trained/layer11/subspace/untrained"][2], s["trained/layer11/subspace/untrained"][0]),
            ("GPT-2", "coupling", s["trained/layer11/subspace/map=coupling"][2],
             s["trained/layer11/subspace/map=coupling"][0]),
            ("GPT-2", "wrong locus", s["trained/layer0/subspace"][2], s["trained/layer0/subspace"][0]),
            ("GPT-2", "manufactured", iia, s["untrained/layer10/coupling"][0])]


def main():
    style()
    pts = gpt2_points() + tier2_points()
    colors = {"valid subspace": VALID, "coupling": COUPLING, "twin": TWIN, "wrong locus": MUTED,
              "manufactured": MANUF}
    labels = {"valid subspace": "learned subspace, trained net", "coupling": "coupling map, trained net",
              "twin": "subspace, untrained twin", "wrong locus": "wrong locus (embedding)",
              "manufactured": "manufactured (coupling, untrained twin)"}
    markers = {"GPT-2": "o", "Qwen3-1.7B-Base": "D", "Qwen2.5-3B": "s", "Llama-3.1-8B": "^"}
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.axvspan(0.8, 1.02, color="#f6f5f1", zorder=0)
    ax.text(0.81, -3900, "accepted by an\nIIA ≥ 0.8 rule", fontsize=13, color=INK2, va="bottom")
    ax.axhline(0, color=INK, lw=1.2)
    ax.text(0.01, 80, "label-only code", fontsize=13, color=INK2)
    seen = set()
    for model, kind, iia, gain in pts:
        ax.scatter(iia, gain, marker=markers[model], s=95, color=colors[kind], edgecolor="white", linewidth=1.5,
                   zorder=3, label=labels[kind] if kind not in seen else None)
        seen.add(kind)
    for model, kind, iia, gain in pts:
        if kind == "manufactured":
            off = (8, 8) if model == "Qwen2.5-3B" else (8, -16)
            ax.annotate(model.replace("-Base", ""), (iia, gain), xytext=off, textcoords="offset points",
                        fontsize=12, color=MANUF)
    h, l = ax.get_legend_handles_labels()
    leg1 = ax.legend(h, l, loc="upper left", fontsize=13)
    ax.add_artist(leg1)
    from matplotlib.lines import Line2D
    mh = [Line2D([], [], marker=m, ls="none", color=INK2, ms=9, label=k.replace("-Base", "")) for k, m in markers.items()]
    ax.legend(handles=mh, loc="lower left", bbox_to_anchor=(0.0, 0.12), fontsize=12, title="model", title_fontsize=12)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-4000, 4300)
    ax.set_xlabel("interchange-intervention accuracy (i.i.d. pairs)")
    ax.set_ylabel("gain over label-only (bits),\n1024 targets")
    save(fig, "iia_vs_code")


if __name__ == "__main__":
    main()
