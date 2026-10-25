"""System 09: segment polarity gene regulatory network (Wg -> Fz paracrine signalling).

Original: ``systems/09_grn/grn.py`` of MelouxM/CAE (``parse_ginml``, ``SegmentPolarityModel``,
``SegmentPolarityLowLevelModel``), network ``regulatoryGraph.ginml`` (Sanchez, Chaouiya &
Thieffry 2008): 6 cells x 12 multi-valued logical variables = 72 nodes, 182 regulatory
edges, GINsim semantics.  The parsed network is embedded below as plain data
(:data:`NODE_ORDER`, :data:`NODES`, :data:`EDGES`), so no XML is needed at run time;
:meth:`SegmentPolarityNetwork.from_ginml` re-parses the file for the equivalence test.

Update rule (one synchronous step, all nodes at once): an incoming edge is *active* when
its source value lies in ``[minvalue, maxvalue]`` (``maxvalue`` absent = no upper bound);
the node takes the value of the first parameter whose active-edge set equals the current
set exactly, else its ``basevalue``.

Variables (numpy order): ``wg_c2`` (input, Wg level in cell 2, (batch, 1), in {0, 1, 2}),
``fz_c1``, ``fz_c3``, ``fz_c4``, ``fz_c5`` (Fz in cells 1, 3, 4, 5 after one step, (batch, 1)).
The low-level model starts every sample from the all-zero state with ``Wg2 = wg_c2``
(rounded half-to-even and clamped to [0, 2]), applies one synchronous step and reports the
rounded input plus the four Fz values.  As in the original, ``wg_c2`` is taken from the
interventions first, then the inputs (default 0); interventions on the Fz outputs are
*ignored* (the original never reads them).

Non-differentiable: every step is a threshold comparison or a set equality on integer
levels; no gradient path exists (the outputs are piecewise constant in ``wg_c2``).  No
smooth relaxation is natural for the exact-set-match semantics, so none is offered.

High-level abstractions (original ``grn.py``):

* Valid coarse-graining ``wg_src <- [wg_c2]``, ``fz_tgt <- [fz_c1, fz_c3]`` (cells 1 and 3
  are the true neighbours of cell 2); wrong map ``fz_tgt <- [fz_c4, fz_c5]``.
* Value map: ``wg_src = int(wg_c2 >= 1.9)`` (0: Wg below paracrine level, rect (-0.1, 1.9);
  1: Wg = 2, rect (1.9, 2.1)); ``fz_tgt = int(any(fz > 0.5))`` (grounding rects
  (-0.1, 0.1)^2 and (0.9, 1.1)^2).
* Valid high-level rule ``fz_tgt = wg_src``; reversed rule ``fz_tgt = 1 - wg_src``;
  ``wg_src`` prior {0: 0.5, 1: 0.5}; noise condition: Gaussian noise of std 0.4 on the Fz
  outputs of the valid pair.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Mapping, Sequence

import torch
from torch import Tensor

from cae_torch.base import TorchSystem

F64 = torch.float64

# High-level abstraction data (plain Python) ------------------------------------------
VALID_COARSE_GRAINING = {"wg_src": ["wg_c2"], "fz_tgt": ["fz_c1", "fz_c3"]}
WRONG_COARSE_GRAINING = {"wg_src": ["wg_c2"], "fz_tgt": ["fz_c4", "fz_c5"]}
WG_SPECS = {0: ((-0.1, 1.9),), 1: ((1.9, 2.1),)}                       # RectSubspace bounds
FZ_PAIR_SPECS = {0: ((-0.1, 0.1), (-0.1, 0.1)), 1: ((0.9, 1.1), (0.9, 1.1))}
WG_SRC_DISTRIBUTION = {0: 0.5, 1: 0.5}
NOISE_STD = 0.4


def abstract_wg_src(values) -> int:
    return int(float(torch.as_tensor(values, dtype=F64).reshape(-1)[0]) >= 1.9)


def abstract_fz_tgt(values) -> int:
    return int(bool((torch.as_tensor(values, dtype=F64).reshape(-1) > 0.5).any()))


def ground_wg_src(label: int) -> list[float]:
    return [0.0] if int(label) == 0 else [2.0]


def ground_fz_tgt(label: int) -> list[float]:
    return [0.0, 0.0] if int(label) == 0 else [1.0, 1.0]


def valid_rule(wg_src: int) -> int:
    return int(wg_src)


def reversed_rule(wg_src: int) -> int:
    return 1 - int(wg_src)


# Embedded network (generated from regulatoryGraph.ginml) ---------------------------------
NODE_ORDER = ['Wg1', 'Fz1', 'Dsh1', 'Slp1', 'Nkd1', 'En1', 'Hh1', 'Ci1', 'Ciact1', 'Cirep1', 'Pka1', 'Ptc1', 'Wg2', 'Fz2', 'Dsh2', 'Slp2', 'Nkd2', 'En2', 'Hh2', 'Ci2', 'Ciact2', 'Cirep2', 'Pka2', 'Ptc2', 'Wg3', 'Fz3', 'Dsh3', 'Slp3', 'Nkd3', 'En3', 'Hh3', 'Ci3', 'Ciact3', 'Cirep3', 'Pka3', 'Ptc3', 'Wg4', 'Fz4', 'Dsh4', 'Slp4', 'Nkd4', 'En4', 'Hh4', 'Ci4', 'Ciact4', 'Cirep4', 'Pka4', 'Ptc4', 'Wg5', 'Fz5', 'Dsh5', 'Slp5', 'Nkd5', 'En5', 'Hh5', 'Ci5', 'Ciact5', 'Cirep5', 'Pka5', 'Ptc5', 'Wg6', 'Fz6', 'Dsh6', 'Slp6', 'Nkd6', 'En6', 'Hh6', 'Ci6', 'Ciact6', 'Cirep6', 'Pka6', 'Ptc6']
# node id -> (name, basevalue, maxvalue, [(sorted active edge ids, value), ...])
NODES = {
    'Wg1': ('Wg', 0, 2, [(('Ciact1_Wg1_0', 'Slp1_Wg1_0'), 1), (('Ciact1_Wg1_1', 'Slp1_Wg1_0'), 2), (('Ciact1_Wg1_1', 'Nkd1_Wg1_0', 'Slp1_Wg1_0'), 2)]),
    'Fz1': ('Fz', 0, 1, [(('Wg1_Fz1_0',), 1), (('Wg2_Fz1_0',), 1), (('Wg1_Fz1_0', 'Wg2_Fz1_0'), 1)]),
    'Dsh1': ('Dsh', 0, 1, [(('Fz1_Dsh1_0',), 1)]),
    'Slp1': ('Slp', 0, 1, [(('Dsh1_Slp1_0',), 1)]),
    'Nkd1': ('Nkd', 1, 2, [(('Dsh1_Nkd1_0',), 2)]),
    'En1': ('En', 0, 1, [(('Dsh1_En1_0',), 1), (('Dsh1_En1_0', 'En1_En1_0'), 1), (('Dsh1_En1_0', 'Nkd1_En1_0'), 1), (('Dsh1_En1_0', 'En1_En1_0', 'Nkd1_En1_0'), 1)]),
    'Hh1': ('Hh', 0, 1, [(('En1_Hh1_0',), 1)]),
    'Ci1': ('Ci', 1, 1, []),
    'Ciact1': ('Ciact', 0, 2, [(('Ci1_Ciact1_0',), 1), (('Ci1_Ciact1_0', 'Dsh1_Ciact1_0'), 2), (('Ci1_Ciact1_0', 'Dsh1_Ciact1_0', 'Pka1_Ciact1_0'), 1)]),
    'Cirep1': ('Cirep', 0, 1, [(('Ci1_Cirep1_0', 'Pka1_Cirep1_0'), 1)]),
    'Pka1': ('Pka', 0, 2, [(('Ptc1_Pka1_0',), 2)]),
    'Ptc1': ('Ptc', 1, 2, [(('Ciact1_Ptc1_0',), 2)]),
    'Wg2': ('Wg', 0, 2, [(('Ciact2_Wg2_0', 'Slp2_Wg2_0'), 1), (('Ciact2_Wg2_1', 'Slp2_Wg2_0'), 2), (('Ciact2_Wg2_1', 'Nkd2_Wg2_0', 'Slp2_Wg2_0'), 2)]),
    'Fz2': ('Fz', 0, 1, [(('Wg2_Fz2_0',), 1), (('Wg3_Fz2_0',), 1), (('Wg1_Fz2_0',), 1), (('Wg1_Fz2_0', 'Wg3_Fz2_0'), 1), (('Wg2_Fz2_0', 'Wg3_Fz2_0'), 1), (('Wg1_Fz2_0', 'Wg2_Fz2_0'), 1), (('Wg1_Fz2_0', 'Wg2_Fz2_0', 'Wg3_Fz2_0'), 1)]),
    'Dsh2': ('Dsh', 0, 1, [(('Fz2_Dsh2_0',), 1)]),
    'Slp2': ('Slp', 0, 1, [(('Dsh2_Slp2_0',), 1)]),
    'Nkd2': ('Nkd', 1, 2, [(('Dsh2_Nkd2_0',), 2)]),
    'En2': ('En', 0, 1, [(('Dsh2_En2_0',), 1), (('Dsh2_En2_0', 'En2_En2_0'), 1), (('Dsh2_En2_0', 'Nkd2_En2_0'), 1), (('Dsh2_En2_0', 'En2_En2_0', 'Nkd2_En2_0'), 1)]),
    'Hh2': ('Hh', 0, 1, [(('En2_Hh2_0',), 1)]),
    'Ci2': ('Ci', 1, 1, []),
    'Ciact2': ('Ciact', 0, 2, [(('Ci2_Ciact2_0',), 1), (('Ci2_Ciact2_0', 'Dsh2_Ciact2_0'), 2), (('Ci2_Ciact2_0', 'Dsh2_Ciact2_0', 'Pka2_Ciact2_0'), 1)]),
    'Cirep2': ('Cirep', 0, 1, [(('Ci2_Cirep2_0', 'Pka2_Cirep2_0'), 1)]),
    'Pka2': ('Pka', 0, 2, [(('Ptc2_Pka2_0',), 2)]),
    'Ptc2': ('Ptc', 1, 2, [(('Ciact2_Ptc2_0',), 2)]),
    'Wg3': ('Wg', 0, 2, [(('Ciact3_Wg3_0', 'Slp3_Wg3_0'), 1), (('Ciact3_Wg3_1', 'Slp3_Wg3_0'), 2), (('Ciact3_Wg3_1', 'Nkd3_Wg3_0', 'Slp3_Wg3_0'), 2)]),
    'Fz3': ('Fz', 0, 1, [(('Wg3_Fz3_0',), 1), (('Wg2_Fz3_0',), 1), (('Wg4_Fz3_0',), 1), (('Wg2_Fz3_0', 'Wg4_Fz3_0'), 1), (('Wg3_Fz3_0', 'Wg4_Fz3_0'), 1), (('Wg2_Fz3_0', 'Wg3_Fz3_0'), 1), (('Wg2_Fz3_0', 'Wg3_Fz3_0', 'Wg4_Fz3_0'), 1)]),
    'Dsh3': ('Dsh', 0, 1, [(('Fz3_Dsh3_0',), 1)]),
    'Slp3': ('Slp', 0, 1, [(('Dsh3_Slp3_0',), 1)]),
    'Nkd3': ('Nkd', 1, 2, [(('Dsh3_Nkd3_0',), 2)]),
    'En3': ('En', 0, 1, [(('Dsh3_En3_0',), 1), (('Dsh3_En3_0', 'En3_En3_0'), 1), (('Dsh3_En3_0', 'Nkd3_En3_0'), 1), (('Dsh3_En3_0', 'En3_En3_0', 'Nkd3_En3_0'), 1)]),
    'Hh3': ('Hh', 0, 1, [(('En3_Hh3_0',), 1)]),
    'Ci3': ('Ci', 1, 1, []),
    'Ciact3': ('Ciact', 0, 2, [(('Ci3_Ciact3_0',), 1), (('Ci3_Ciact3_0', 'Dsh3_Ciact3_0'), 2), (('Ci3_Ciact3_0', 'Dsh3_Ciact3_0', 'Pka3_Ciact3_0'), 1)]),
    'Cirep3': ('Cirep', 0, 1, [(('Ci3_Cirep3_0', 'Pka3_Cirep3_0'), 1)]),
    'Pka3': ('Pka', 0, 2, [(('Ptc3_Pka3_0',), 2)]),
    'Ptc3': ('Ptc', 1, 2, [(('Ciact3_Ptc3_0',), 2)]),
    'Wg4': ('Wg', 0, 2, [(('Ciact4_Wg4_0', 'Slp4_Wg4_0'), 1), (('Ciact4_Wg4_1', 'Slp4_Wg4_0'), 2), (('Ciact4_Wg4_1', 'Nkd4_Wg4_0', 'Slp4_Wg4_0'), 2)]),
    'Fz4': ('Fz', 0, 1, [(('Wg4_Fz4_0',), 1), (('Wg5_Fz4_0',), 1), (('Wg3_Fz4_0',), 1), (('Wg3_Fz4_0', 'Wg5_Fz4_0'), 1), (('Wg3_Fz4_0', 'Wg4_Fz4_0'), 1), (('Wg4_Fz4_0', 'Wg5_Fz4_0'), 1), (('Wg3_Fz4_0', 'Wg4_Fz4_0', 'Wg5_Fz4_0'), 1)]),
    'Dsh4': ('Dsh', 0, 1, [(('Fz4_Dsh4_0',), 1)]),
    'Slp4': ('Slp', 0, 1, [(('Dsh4_Slp4_0',), 1)]),
    'Nkd4': ('Nkd', 1, 2, [(('Dsh4_Nkd4_0',), 2)]),
    'En4': ('En', 0, 1, [(('Dsh4_En4_0',), 1), (('Dsh4_En4_0', 'En4_En4_0'), 1), (('Dsh4_En4_0', 'Nkd4_En4_0'), 1), (('Dsh4_En4_0', 'En4_En4_0', 'Nkd4_En4_0'), 1)]),
    'Hh4': ('Hh', 0, 1, [(('En4_Hh4_0',), 1)]),
    'Ci4': ('Ci', 1, 1, []),
    'Ciact4': ('Ciact', 0, 2, [(('Ci4_Ciact4_0',), 1), (('Ci4_Ciact4_0', 'Dsh4_Ciact4_0'), 2), (('Ci4_Ciact4_0', 'Dsh4_Ciact4_0', 'Pka4_Ciact4_0'), 1)]),
    'Cirep4': ('Cirep', 0, 1, [(('Ci4_Cirep4_0', 'Pka4_Cirep4_0'), 1)]),
    'Pka4': ('Pka', 0, 2, [(('Ptc4_Pka4_0',), 2)]),
    'Ptc4': ('Ptc', 1, 2, [(('Ciact4_Ptc4_0',), 2)]),
    'Wg5': ('Wg', 0, 2, [(('Ciact5_Wg5_0', 'Slp5_Wg5_0'), 1), (('Ciact5_Wg5_1', 'Slp5_Wg5_0'), 2), (('Ciact5_Wg5_1', 'Nkd5_Wg5_0', 'Slp5_Wg5_0'), 2)]),
    'Fz5': ('Fz', 0, 1, [(('Wg5_Fz5_0',), 1), (('Wg4_Fz5_0',), 1), (('Wg6_Fz5_0',), 1), (('Wg4_Fz5_0', 'Wg6_Fz5_0'), 1), (('Wg4_Fz5_0', 'Wg5_Fz5_0'), 1), (('Wg5_Fz5_0', 'Wg6_Fz5_0'), 1), (('Wg4_Fz5_0', 'Wg5_Fz5_0', 'Wg6_Fz5_0'), 1)]),
    'Dsh5': ('Dsh', 0, 1, [(('Fz5_Dsh5_0',), 1)]),
    'Slp5': ('Slp', 0, 1, [(('Dsh5_Slp5_0',), 1)]),
    'Nkd5': ('Nkd', 1, 2, [(('Dsh5_Nkd5_0',), 2)]),
    'En5': ('En', 0, 1, [(('Dsh5_En5_0',), 1), (('Dsh5_En5_0', 'En5_En5_0'), 1), (('Dsh5_En5_0', 'Nkd5_En5_0'), 1), (('Dsh5_En5_0', 'En5_En5_0', 'Nkd5_En5_0'), 1)]),
    'Hh5': ('Hh', 0, 1, [(('En5_Hh5_0',), 1)]),
    'Ci5': ('Ci', 1, 1, []),
    'Ciact5': ('Ciact', 0, 2, [(('Ci5_Ciact5_0',), 1), (('Ci5_Ciact5_0', 'Dsh5_Ciact5_0'), 2), (('Ci5_Ciact5_0', 'Dsh5_Ciact5_0', 'Pka5_Ciact5_0'), 1)]),
    'Cirep5': ('Cirep', 0, 1, [(('Ci5_Cirep5_0', 'Pka5_Cirep5_0'), 1)]),
    'Pka5': ('Pka', 0, 2, [(('Ptc5_Pka5_0',), 2)]),
    'Ptc5': ('Ptc', 1, 2, [(('Ciact5_Ptc5_0',), 2)]),
    'Wg6': ('Wg', 0, 2, [(('Ciact6_Wg6_0', 'Slp6_Wg6_0'), 1), (('Ciact6_Wg6_1', 'Slp6_Wg6_0'), 2), (('Ciact6_Wg6_1', 'Nkd6_Wg6_0', 'Slp6_Wg6_0'), 2)]),
    'Fz6': ('Fz', 0, 1, [(('Wg6_Fz6_0',), 1), (('Wg5_Fz6_0',), 1), (('Wg5_Fz6_0', 'Wg6_Fz6_0'), 1)]),
    'Dsh6': ('Dsh', 0, 1, [(('Fz6_Dsh6_0',), 1)]),
    'Slp6': ('Slp', 0, 1, [(('Dsh6_Slp6_0',), 1)]),
    'Nkd6': ('Nkd', 1, 2, [(('Dsh6_Nkd6_0',), 2)]),
    'En6': ('En', 0, 1, [(('Dsh6_En6_0',), 1), (('Dsh6_En6_0', 'En6_En6_0'), 1), (('Dsh6_En6_0', 'Nkd6_En6_0'), 1), (('Dsh6_En6_0', 'En6_En6_0', 'Nkd6_En6_0'), 1)]),
    'Hh6': ('Hh', 0, 1, [(('En6_Hh6_0',), 1)]),
    'Ci6': ('Ci', 1, 1, []),
    'Ciact6': ('Ciact', 0, 2, [(('Ci6_Ciact6_0',), 1), (('Ci6_Ciact6_0', 'Dsh6_Ciact6_0'), 2), (('Ci6_Ciact6_0', 'Dsh6_Ciact6_0', 'Pka6_Ciact6_0'), 1)]),
    'Cirep6': ('Cirep', 0, 1, [(('Ci6_Cirep6_0', 'Pka6_Cirep6_0'), 1)]),
    'Pka6': ('Pka', 0, 2, [(('Ptc6_Pka6_0',), 2)]),
    'Ptc6': ('Ptc', 1, 2, [(('Ciact6_Ptc6_0',), 2)]),
}
# edge id -> (source, target, minvalue, maxvalue or None, sign)
EDGES = {
    'Wg1_Fz1_0': ('Wg1', 'Fz1', 1, None, 'positive'),
    'Wg1_Fz2_0': ('Wg1', 'Fz2', 2, None, 'positive'),
    'Nkd1_En1_0': ('Nkd1', 'En1', 1, None, 'negative'),
    'Nkd1_Wg1_0': ('Nkd1', 'Wg1', 2, None, 'negative'),
    'Slp4_En4_0': ('Slp4', 'En4', 1, None, 'negative'),
    'Slp4_Wg4_0': ('Slp4', 'Wg4', 1, None, 'positive'),
    'Ptc2_Pka2_0': ('Ptc2', 'Pka2', 1, None, 'positive'),
    'Fz2_Dsh2_0': ('Fz2', 'Dsh2', 1, None, 'positive'),
    'Fz5_Dsh5_0': ('Fz5', 'Dsh5', 1, None, 'positive'),
    'Ciact5_Ptc5_0': ('Ciact5', 'Ptc5', 1, None, 'positive'),
    'Ciact5_Wg5_0': ('Ciact5', 'Wg5', 1, 1, 'positive'),
    'Ciact5_Wg5_1': ('Ciact5', 'Wg5', 2, None, 'positive'),
    'Ptc6_Pka6_0': ('Ptc6', 'Pka6', 1, None, 'positive'),
    'Ci1_Ciact1_0': ('Ci1', 'Ciact1', 1, None, 'positive'),
    'Ci1_Cirep1_0': ('Ci1', 'Cirep1', 1, None, 'positive'),
    'Ci3_Ciact3_0': ('Ci3', 'Ciact3', 1, None, 'positive'),
    'Ci3_Cirep3_0': ('Ci3', 'Cirep3', 1, None, 'positive'),
    'En1_Ptc1_0': ('En1', 'Ptc1', 1, None, 'negative'),
    'En1_Ci1_0': ('En1', 'Ci1', 1, None, 'negative'),
    'En1_Nkd1_0': ('En1', 'Nkd1', 1, None, 'negative'),
    'En1_En1_0': ('En1', 'En1', 1, None, 'positive'),
    'En1_Slp1_0': ('En1', 'Slp1', 1, None, 'negative'),
    'En1_Wg1_0': ('En1', 'Wg1', 1, None, 'negative'),
    'En1_Hh1_0': ('En1', 'Hh1', 1, None, 'positive'),
    'Hh3_Ptc2_0': ('Hh3', 'Ptc2', 1, None, 'negative'),
    'Hh3_Ptc4_0': ('Hh3', 'Ptc4', 1, None, 'negative'),
    'Cirep1_Hh1_0': ('Cirep1', 'Hh1', 1, None, 'negative'),
    'Cirep3_Hh3_0': ('Cirep3', 'Hh3', 1, None, 'negative'),
    'Ciact4_Ptc4_0': ('Ciact4', 'Ptc4', 1, None, 'positive'),
    'Ciact4_Wg4_0': ('Ciact4', 'Wg4', 1, 1, 'positive'),
    'Ciact4_Wg4_1': ('Ciact4', 'Wg4', 2, None, 'positive'),
    'En3_Ptc3_0': ('En3', 'Ptc3', 1, None, 'negative'),
    'En3_Ci3_0': ('En3', 'Ci3', 1, None, 'negative'),
    'En3_Nkd3_0': ('En3', 'Nkd3', 1, None, 'negative'),
    'En3_En3_0': ('En3', 'En3', 1, None, 'positive'),
    'En3_Slp3_0': ('En3', 'Slp3', 1, None, 'negative'),
    'En3_Wg3_0': ('En3', 'Wg3', 1, None, 'negative'),
    'En3_Hh3_0': ('En3', 'Hh3', 1, None, 'positive'),
    'Hh1_Ptc2_0': ('Hh1', 'Ptc2', 1, None, 'negative'),
    'Wg3_Fz3_0': ('Wg3', 'Fz3', 1, None, 'positive'),
    'Wg3_Fz2_0': ('Wg3', 'Fz2', 2, None, 'positive'),
    'Wg3_Fz4_0': ('Wg3', 'Fz4', 2, None, 'positive'),
    'Dsh5_Ciact5_0': ('Dsh5', 'Ciact5', 1, None, 'positive'),
    'Dsh5_Nkd5_0': ('Dsh5', 'Nkd5', 1, None, 'positive'),
    'Dsh5_En5_0': ('Dsh5', 'En5', 1, None, 'positive'),
    'Dsh5_Slp5_0': ('Dsh5', 'Slp5', 1, None, 'positive'),
    'Dsh5_Cirep5_0': ('Dsh5', 'Cirep5', 1, None, 'negative'),
    'Hh5_Ptc6_0': ('Hh5', 'Ptc6', 1, None, 'negative'),
    'Hh5_Ptc4_0': ('Hh5', 'Ptc4', 1, None, 'negative'),
    'Pka3_Ciact3_0': ('Pka3', 'Ciact3', 1, None, 'negative'),
    'Pka3_Cirep3_0': ('Pka3', 'Cirep3', 2, None, 'positive'),
    'Ci6_Ciact6_0': ('Ci6', 'Ciact6', 1, None, 'positive'),
    'Ci6_Cirep6_0': ('Ci6', 'Cirep6', 1, None, 'positive'),
    'Cirep5_Hh5_0': ('Cirep5', 'Hh5', 1, None, 'negative'),
    'Slp3_En3_0': ('Slp3', 'En3', 1, None, 'negative'),
    'Slp3_Wg3_0': ('Slp3', 'Wg3', 1, None, 'positive'),
    'Nkd5_En5_0': ('Nkd5', 'En5', 1, None, 'negative'),
    'Nkd5_Wg5_0': ('Nkd5', 'Wg5', 2, None, 'negative'),
    'Hh2_Ptc1_0': ('Hh2', 'Ptc1', 1, None, 'negative'),
    'Hh2_Ptc3_0': ('Hh2', 'Ptc3', 1, None, 'negative'),
    'Pka6_Ciact6_0': ('Pka6', 'Ciact6', 1, None, 'negative'),
    'Pka6_Cirep6_0': ('Pka6', 'Cirep6', 2, None, 'positive'),
    'Ptc3_Pka3_0': ('Ptc3', 'Pka3', 1, None, 'positive'),
    'Ci2_Ciact2_0': ('Ci2', 'Ciact2', 1, None, 'positive'),
    'Ci2_Cirep2_0': ('Ci2', 'Cirep2', 1, None, 'positive'),
    'Slp1_En1_0': ('Slp1', 'En1', 1, None, 'negative'),
    'Slp1_Wg1_0': ('Slp1', 'Wg1', 1, None, 'positive'),
    'Dsh2_Ciact2_0': ('Dsh2', 'Ciact2', 1, None, 'positive'),
    'Dsh2_Nkd2_0': ('Dsh2', 'Nkd2', 1, None, 'positive'),
    'Dsh2_En2_0': ('Dsh2', 'En2', 1, None, 'positive'),
    'Dsh2_Slp2_0': ('Dsh2', 'Slp2', 1, None, 'positive'),
    'Dsh2_Cirep2_0': ('Dsh2', 'Cirep2', 1, None, 'negative'),
    'En5_Ptc5_0': ('En5', 'Ptc5', 1, None, 'negative'),
    'En5_Ci5_0': ('En5', 'Ci5', 1, None, 'negative'),
    'En5_Nkd5_0': ('En5', 'Nkd5', 1, None, 'negative'),
    'En5_En5_0': ('En5', 'En5', 1, None, 'positive'),
    'En5_Slp5_0': ('En5', 'Slp5', 1, None, 'negative'),
    'En5_Wg5_0': ('En5', 'Wg5', 1, None, 'negative'),
    'En5_Hh5_0': ('En5', 'Hh5', 1, None, 'positive'),
    'Wg5_Fz5_0': ('Wg5', 'Fz5', 1, None, 'positive'),
    'Wg5_Fz6_0': ('Wg5', 'Fz6', 2, None, 'positive'),
    'Wg5_Fz4_0': ('Wg5', 'Fz4', 2, None, 'positive'),
    'Ptc5_Pka5_0': ('Ptc5', 'Pka5', 1, None, 'positive'),
    'Dsh6_Ciact6_0': ('Dsh6', 'Ciact6', 1, None, 'positive'),
    'Dsh6_Nkd6_0': ('Dsh6', 'Nkd6', 1, None, 'positive'),
    'Dsh6_En6_0': ('Dsh6', 'En6', 1, None, 'positive'),
    'Dsh6_Slp6_0': ('Dsh6', 'Slp6', 1, None, 'positive'),
    'Dsh6_Cirep6_0': ('Dsh6', 'Cirep6', 1, None, 'negative'),
    'Dsh1_Ciact1_0': ('Dsh1', 'Ciact1', 1, None, 'positive'),
    'Dsh1_Nkd1_0': ('Dsh1', 'Nkd1', 1, None, 'positive'),
    'Dsh1_En1_0': ('Dsh1', 'En1', 1, None, 'positive'),
    'Dsh1_Slp1_0': ('Dsh1', 'Slp1', 1, None, 'positive'),
    'Dsh1_Cirep1_0': ('Dsh1', 'Cirep1', 1, None, 'negative'),
    'Cirep4_Hh4_0': ('Cirep4', 'Hh4', 1, None, 'negative'),
    'Cirep6_Hh6_0': ('Cirep6', 'Hh6', 1, None, 'negative'),
    'Pka2_Ciact2_0': ('Pka2', 'Ciact2', 1, None, 'negative'),
    'Pka2_Cirep2_0': ('Pka2', 'Cirep2', 2, None, 'positive'),
    'Fz4_Dsh4_0': ('Fz4', 'Dsh4', 1, None, 'positive'),
    'Fz1_Dsh1_0': ('Fz1', 'Dsh1', 1, None, 'positive'),
    'Pka1_Ciact1_0': ('Pka1', 'Ciact1', 1, None, 'negative'),
    'Pka1_Cirep1_0': ('Pka1', 'Cirep1', 2, None, 'positive'),
    'Pka5_Ciact5_0': ('Pka5', 'Ciact5', 1, None, 'negative'),
    'Pka5_Cirep5_0': ('Pka5', 'Cirep5', 2, None, 'positive'),
    'Slp5_En5_0': ('Slp5', 'En5', 1, None, 'negative'),
    'Slp5_Wg5_0': ('Slp5', 'Wg5', 1, None, 'positive'),
    'Fz3_Dsh3_0': ('Fz3', 'Dsh3', 1, None, 'positive'),
    'Hh6_Ptc5_0': ('Hh6', 'Ptc5', 1, None, 'negative'),
    'Slp6_En6_0': ('Slp6', 'En6', 1, None, 'negative'),
    'Slp6_Wg6_0': ('Slp6', 'Wg6', 1, None, 'positive'),
    'Fz6_Dsh6_0': ('Fz6', 'Dsh6', 1, None, 'positive'),
    'Wg2_Fz2_0': ('Wg2', 'Fz2', 1, None, 'positive'),
    'Wg2_Fz1_0': ('Wg2', 'Fz1', 2, None, 'positive'),
    'Wg2_Fz3_0': ('Wg2', 'Fz3', 2, None, 'positive'),
    'Nkd3_En3_0': ('Nkd3', 'En3', 1, None, 'negative'),
    'Nkd3_Wg3_0': ('Nkd3', 'Wg3', 2, None, 'negative'),
    'Slp2_En2_0': ('Slp2', 'En2', 1, None, 'negative'),
    'Slp2_Wg2_0': ('Slp2', 'Wg2', 1, None, 'positive'),
    'En4_Ptc4_0': ('En4', 'Ptc4', 1, None, 'negative'),
    'En4_Ci4_0': ('En4', 'Ci4', 1, None, 'negative'),
    'En4_Nkd4_0': ('En4', 'Nkd4', 1, None, 'negative'),
    'En4_En4_0': ('En4', 'En4', 1, None, 'positive'),
    'En4_Slp4_0': ('En4', 'Slp4', 1, None, 'negative'),
    'En4_Wg4_0': ('En4', 'Wg4', 1, None, 'negative'),
    'En4_Hh4_0': ('En4', 'Hh4', 1, None, 'positive'),
    'Cirep2_Hh2_0': ('Cirep2', 'Hh2', 1, None, 'negative'),
    'Ci4_Ciact4_0': ('Ci4', 'Ciact4', 1, None, 'positive'),
    'Ci4_Cirep4_0': ('Ci4', 'Cirep4', 1, None, 'positive'),
    'Nkd2_En2_0': ('Nkd2', 'En2', 1, None, 'negative'),
    'Nkd2_Wg2_0': ('Nkd2', 'Wg2', 2, None, 'negative'),
    'Ciact3_Ptc3_0': ('Ciact3', 'Ptc3', 1, None, 'positive'),
    'Ciact3_Wg3_0': ('Ciact3', 'Wg3', 1, 1, 'positive'),
    'Ciact3_Wg3_1': ('Ciact3', 'Wg3', 2, None, 'positive'),
    'Pka4_Ciact4_0': ('Pka4', 'Ciact4', 1, None, 'negative'),
    'Pka4_Cirep4_0': ('Pka4', 'Cirep4', 2, None, 'positive'),
    'Ciact1_Ptc1_0': ('Ciact1', 'Ptc1', 1, None, 'positive'),
    'Ciact1_Wg1_0': ('Ciact1', 'Wg1', 1, 1, 'positive'),
    'Ciact1_Wg1_1': ('Ciact1', 'Wg1', 2, None, 'positive'),
    'Ptc4_Pka4_0': ('Ptc4', 'Pka4', 1, None, 'positive'),
    'Ciact6_Ptc6_0': ('Ciact6', 'Ptc6', 1, None, 'positive'),
    'Ciact6_Wg6_0': ('Ciact6', 'Wg6', 1, 1, 'positive'),
    'Ciact6_Wg6_1': ('Ciact6', 'Wg6', 2, None, 'positive'),
    'Nkd4_En4_0': ('Nkd4', 'En4', 1, None, 'negative'),
    'Nkd4_Wg4_0': ('Nkd4', 'Wg4', 2, None, 'negative'),
    'Ptc1_Pka1_0': ('Ptc1', 'Pka1', 1, None, 'positive'),
    'Dsh4_Ciact4_0': ('Dsh4', 'Ciact4', 1, None, 'positive'),
    'Dsh4_Nkd4_0': ('Dsh4', 'Nkd4', 1, None, 'positive'),
    'Dsh4_En4_0': ('Dsh4', 'En4', 1, None, 'positive'),
    'Dsh4_Slp4_0': ('Dsh4', 'Slp4', 1, None, 'positive'),
    'Dsh4_Cirep4_0': ('Dsh4', 'Cirep4', 1, None, 'negative'),
    'Dsh3_Ciact3_0': ('Dsh3', 'Ciact3', 1, None, 'positive'),
    'Dsh3_Nkd3_0': ('Dsh3', 'Nkd3', 1, None, 'positive'),
    'Dsh3_En3_0': ('Dsh3', 'En3', 1, None, 'positive'),
    'Dsh3_Slp3_0': ('Dsh3', 'Slp3', 1, None, 'positive'),
    'Dsh3_Cirep3_0': ('Dsh3', 'Cirep3', 1, None, 'negative'),
    'En2_Ptc2_0': ('En2', 'Ptc2', 1, None, 'negative'),
    'En2_Ci2_0': ('En2', 'Ci2', 1, None, 'negative'),
    'En2_Nkd2_0': ('En2', 'Nkd2', 1, None, 'negative'),
    'En2_En2_0': ('En2', 'En2', 1, None, 'positive'),
    'En2_Slp2_0': ('En2', 'Slp2', 1, None, 'negative'),
    'En2_Wg2_0': ('En2', 'Wg2', 1, None, 'negative'),
    'En2_Hh2_0': ('En2', 'Hh2', 1, None, 'positive'),
    'Wg6_Fz6_0': ('Wg6', 'Fz6', 1, None, 'positive'),
    'Wg6_Fz5_0': ('Wg6', 'Fz5', 2, None, 'positive'),
    'En6_Ptc6_0': ('En6', 'Ptc6', 1, None, 'negative'),
    'En6_Ci6_0': ('En6', 'Ci6', 1, None, 'negative'),
    'En6_Nkd6_0': ('En6', 'Nkd6', 1, None, 'negative'),
    'En6_En6_0': ('En6', 'En6', 1, None, 'positive'),
    'En6_Slp6_0': ('En6', 'Slp6', 1, None, 'negative'),
    'En6_Wg6_0': ('En6', 'Wg6', 1, None, 'negative'),
    'En6_Hh6_0': ('En6', 'Hh6', 1, None, 'positive'),
    'Nkd6_En6_0': ('Nkd6', 'En6', 1, None, 'negative'),
    'Nkd6_Wg6_0': ('Nkd6', 'Wg6', 2, None, 'negative'),
    'Ci5_Ciact5_0': ('Ci5', 'Ciact5', 1, None, 'positive'),
    'Ci5_Cirep5_0': ('Ci5', 'Cirep5', 1, None, 'positive'),
    'Ciact2_Ptc2_0': ('Ciact2', 'Ptc2', 1, None, 'positive'),
    'Ciact2_Wg2_0': ('Ciact2', 'Wg2', 1, 1, 'positive'),
    'Ciact2_Wg2_1': ('Ciact2', 'Wg2', 2, None, 'positive'),
    'Hh4_Ptc3_0': ('Hh4', 'Ptc3', 1, None, 'negative'),
    'Hh4_Ptc5_0': ('Hh4', 'Ptc5', 1, None, 'negative'),
    'Wg4_Fz4_0': ('Wg4', 'Fz4', 1, None, 'positive'),
    'Wg4_Fz3_0': ('Wg4', 'Fz3', 2, None, 'positive'),
    'Wg4_Fz5_0': ('Wg4', 'Fz5', 2, None, 'positive'),
}


# Network simulator ---------------------------------------------------------------------

def parse_ginml(filepath: str) -> tuple[list[str], dict, dict]:
    """Parse a GINsim ``.ginml`` file into the plain-data format of NODE_ORDER / NODES / EDGES."""
    graph = ET.parse(filepath).getroot().find("graph")
    node_order = graph.attrib["nodeorder"].split()
    edges = {}
    for e in graph.findall("edge"):
        mx = e.attrib.get("maxvalue")
        edges[e.attrib["id"]] = (e.attrib["from"], e.attrib["to"], int(e.attrib["minvalue"]),
                                 int(mx) if mx is not None else None, e.attrib["sign"])
    nodes = {}
    for n in graph.findall("node"):
        params = [(tuple(sorted(p.attrib["idActiveInteractions"].split())), int(p.attrib["val"]))
                  for p in n.findall("parameter")]
        nodes[n.attrib["id"]] = (n.attrib.get("name", n.attrib["id"]), int(n.attrib.get("basevalue", "0")),
                                 int(n.attrib.get("maxvalue", "1")), params)
    return node_order, nodes, edges


class SegmentPolarityNetwork:
    """Batched synchronous multi-valued logical network (GINsim semantics).

    ``synchronous_step(state)`` maps a ``(batch, n_vars)`` tensor of levels to the next
    levels.  Built from the embedded data by default or from a ``.ginml`` file.
    """

    def __init__(self, node_order: Sequence[str] = NODE_ORDER, nodes: Mapping = NODES, edges: Mapping = EDGES):
        self.node_order = list(node_order)
        self.nodes = dict(nodes)
        self.edges = dict(edges)
        self.n_vars = len(self.node_order)
        self.var_to_idx = {nid: i for i, nid in enumerate(self.node_order)}
        incoming: dict[str, list[str]] = {nid: [] for nid in self.node_order}
        for eid, (src, tgt, _mn, _mx, _sign) in self.edges.items():
            incoming[tgt].append(eid)
        self.incoming_edges = incoming
        # Per node: source indices, bounds, parameter signatures over the incoming edges.
        self._rules = []
        big = 10 ** 6
        for nid in self.node_order:
            eids = incoming[nid]
            _name, base, _maxv, params = self.nodes[nid]
            src = torch.tensor([self.var_to_idx[self.edges[e][0]] for e in eids], dtype=torch.long)
            mn = torch.tensor([self.edges[e][2] for e in eids], dtype=F64)
            mx = torch.tensor([self.edges[e][3] if self.edges[e][3] is not None else big for e in eids], dtype=F64)
            sig = torch.tensor([[e in set(active) for e in eids] for active, _v in params], dtype=torch.bool)
            vals = torch.tensor([float(v) for _a, v in params], dtype=F64)
            self._rules.append((src, mn, mx, sig.reshape(len(params), len(eids)), vals, float(base)))

    @classmethod
    def from_ginml(cls, ginml_path: str) -> "SegmentPolarityNetwork":
        return cls(*parse_ginml(ginml_path))

    def state_from_dict(self, values: Mapping[str, float], batch: int = 1) -> Tensor:
        state = torch.zeros((batch, self.n_vars), dtype=F64)
        for nid, val in values.items():
            if nid in self.var_to_idx:
                state[:, self.var_to_idx[nid]] = torch.as_tensor(val, dtype=F64)
        return state

    def active_edges(self, state: Tensor, node_id: str) -> Tensor:
        """Boolean (batch, k) activity of the incoming edges of ``node_id`` (order of ``incoming_edges``)."""
        src, mn, mx, _sig, _vals, _base = self._rules[self.var_to_idx[node_id]]
        s = state[:, src]
        return (s >= mn) & (s <= mx)

    def synchronous_step(self, state: Tensor) -> Tensor:
        state = torch.as_tensor(state, dtype=F64)
        squeeze = state.ndim == 1
        if squeeze:
            state = state.unsqueeze(0)
        cols = []
        for src, mn, mx, sig, vals, base in self._rules:
            s = state[:, src]
            active = (s >= mn) & (s <= mx)                                  # (B, k)
            if sig.shape[0] == 0:
                cols.append(torch.full((state.shape[0],), base, dtype=F64))
                continue
            match = (active.unsqueeze(1) == sig.unsqueeze(0)).all(-1)        # (B, P)
            first = torch.argmax(match.to(torch.int64), dim=1)               # first matching parameter
            cols.append(torch.where(match.any(1), vals[first], torch.full_like(vals[first], base)))
        out = torch.stack(cols, dim=1)
        return out.squeeze(0) if squeeze else out


# Low-level model ----------------------------------------------------------------------

class SegmentPolarityLowLevelModel(TorchSystem):
    """Torch translation of ``SegmentPolarityLowLevelModel(model)``; ``model`` defaults to the embedded network."""

    variables = ("wg_c2", "fz_c1", "fz_c3", "fz_c4", "fz_c5")
    input_variables = ("wg_c2",)

    def __init__(self, model: SegmentPolarityNetwork | None = None):
        self.model = model if model is not None else SegmentPolarityNetwork()

    def forward_with_interventions(self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> dict[str, Tensor]:
        raw = interventions.get("wg_c2", inputs.get("wg_c2"))
        if raw is None:
            raw = torch.zeros((max(1, self.batch_size(inputs, interventions)), 1), dtype=F64)
        wg = torch.as_tensor(raw, dtype=F64).reshape(-1, 1)[:, 0]
        wg_val = torch.clamp(torch.round(wg), 0.0, 2.0)                     # int(round(.)), clamp to [0, 2]
        state = torch.zeros((wg_val.shape[0], self.model.n_vars), dtype=F64)
        state[:, self.model.var_to_idx["Wg2"]] = wg_val
        nxt = self.model.synchronous_step(state)

        def get(node):
            return nxt[:, self.model.var_to_idx[node]].reshape(-1, 1)

        return {"wg_c2": wg_val.reshape(-1, 1), "fz_c1": get("Fz1"), "fz_c3": get("Fz3"),
                "fz_c4": get("Fz4"), "fz_c5": get("Fz5")}
