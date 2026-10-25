"""Shared style, paths and parsers for the SLT-talk figures.

Every number plotted by the scripts in this directory is read from a file under `reports/`.
Where the only source is a text table, it is parsed here with a regular expression and the
file is named in the docstring of the parser.

Colour roles (validated with the dataviz skill's palette checker, adjacent pairs, light mode):
  learners:        plug-in / MAP code (kappa) = blue, Bayes code (lambda) = orange
  configurations:  Jacobian lens = green, valid subspace = blue, coupling on the trained
                   network = aqua, untrained twin = yellow, manufactured alignment = red
  reference ink:   nominal / theoretical values = grey, dashed
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path(__file__).resolve().parents[2]
OUT = REPORTS / "talk_slt"

# ink and chrome
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# learners
PLUGIN = "#2a78d6"
BAYES = "#eb6834"

# configurations
JLENS = "#008300"
VALID = "#2a78d6"
COUPLING = "#1baf7a"
TWIN = "#eda100"
MANUF = "#e34948"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 17,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 9,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    })


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def read(rel: str) -> str:
    return (REPORTS / rel).read_text()


def load_json(rel: str):
    return json.loads((REPORTS / rel).read_text())


NUM = r"[+-]?\d+(?:\.\d+)?"


def grab(pattern: str, text: str, group: int = 1, flags=re.S) -> str:
    m = re.search(pattern, text, flags)
    if m is None:
        raise ValueError(f"pattern not found: {pattern!r}")
    return m.group(group)


# ---------------------------------------------------------------- GPT-2 learner coefficients
def gpt2_map_slopes() -> dict:
    """Learner coefficients of the single-map MAP code on GPT-2 IOI at d_V 16, n <= 1024.

    Source: reports/stage1_ioi/regret_slopes_map_code.txt (first table).
    Returns {name: (slope, se)}.
    """
    t = read("stage1_ioi/regret_slopes_map_code.txt")
    block = t.split("InterpBench 18")[0]
    rows = {
        "valid": r"valid subspace, layer 11\s+(" + NUM + r") \+- (" + NUM + ")",
        "coupling": r"same locus, coupling map \(trained network\)\s+(" + NUM + r") \+- (" + NUM + ")",
        "twin": r"untrained twin, subspace, layer 11\s+(" + NUM + r") \+- (" + NUM + ")",
        "manufactured": r"manufactured alignment \(coupling, twin 7, L10\)\s+(" + NUM + r") \+- (" + NUM + ")",
        "manufactured_twin8": r"manufactured alignment \(coupling, twin 8, L10\)\s+(" + NUM + r") \+- (" + NUM + ")",
    }
    return {k: (float(grab(p, block, 1)), float(grab(p, block, 2))) for k, p in rows.items()}


def gpt2_dv_table() -> dict:
    """Learner coefficients across subspace dimensions d_V = 4, 16, 64 on GPT-2 layer 11.

    Source: reports/stage1_ioi/regret_slopes_map_code.txt (table "subspace dimension 4 and 64").
    Returns {row: [(slope, se) for d_V 4, 16, 64]}.
    """
    t = read("stage1_ioi/regret_slopes_map_code.txt")
    block = t.split("GPT-2 IOI, subspace dimension 4 and 64")[1].split("Reading:")[0]
    names = {
        "valid": "valid subspace, layer 11",
        "coupling": "same locus, coupling map (trained network)",
        "twin": "untrained twin, subspace, layer 11",
        "manufactured": "manufactured alignment (coupling, twin, L10)",
    }
    out = {}
    for key, label in names.items():
        m = re.search(re.escape(label) + r"\s+" + r"\s+".join([r"(" + NUM + r") \+- (" + NUM + r")"] * 3), block)
        vals = [float(x) for x in m.groups()]
        out[key] = [(vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5])]
    return out


def summary_rows(rel: str) -> dict:
    """Rows of a Stage 1 summary table (markdown): {configuration: (gain, se, iia_iid)}."""
    out = {}
    for line in read(rel).splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(" + NUM + r") ± (\S+)\s*\|\s*(" + NUM + r") ±", line)
        if m:
            se = float(m.group(4)) if m.group(4) != "nan" else float("nan")
            out[m.group(1)] = (float(m.group(3)), se, float(m.group(5)))
    return out


def label_only_bits(rel: str) -> float:
    return float(grab(r"label-only code (" + NUM + r") bits", read(rel)))
