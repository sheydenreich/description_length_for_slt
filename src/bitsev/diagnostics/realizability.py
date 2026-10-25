"""D3: do counterfactual states stay on the network's natural manifold? (plan, Appendix D.5)

Causal abstraction constrains outputs only. If the representation factorises through the
algorithm's variables, the intervened state ``h'(b, s)`` must lie close to the natural
state of the *recombined* input ``x*`` (the input with ``V`` taken from the source and the
rest from the base), within ``(sqrt(2) Dist(phi) + 1) eps``. Three statistics follow:

1. the counterfactual error ``||h'(b, s) - h(x*)||`` where recombined inputs exist,
   normalised by the nearest-neighbour spacing of the natural cloud;
2. the one-sided Hausdorff distance from the intervened cloud ``I`` to the natural cloud
   ``H`` where they do not;
3. the persistence of the inclusion ``H -> H u I`` in degrees zero and one: new degree-zero
   bars are components made of intervened states only, new degree-one bars are loops the
   intervened states fill in. By Rips stability the bottleneck distance between the
   diagrams of ``H`` and ``H u I`` is at most twice the Hausdorff distance.

Persistence needs ``gudhi`` (optional extra ``[topo]``); the first two statistics do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def nearest_neighbour_spacing(states: np.ndarray) -> np.ndarray:
    """Distance from every state to its nearest other state."""
    d, _ = cKDTree(states).query(states, k=2)
    return d[:, 1]


def counterfactual_errors(
    h_intervened: np.ndarray, h_recombined: np.ndarray, spacing: float
) -> np.ndarray:
    """``||h'(b, s) - h(x*)|| / spacing`` for aligned rows."""
    return np.linalg.norm(h_intervened - h_recombined, axis=1) / spacing


def one_sided_hausdorff(intervened: np.ndarray, natural: np.ndarray) -> float:
    """``max_{i in I} min_{h in H} ||i - h||``."""
    d, _ = cKDTree(natural).query(intervened, k=1)
    return float(d.max())


def realizability_bound(distortion: float, eps: float) -> float:
    """``(sqrt(2) Dist(phi) + 1) eps`` from the plan's realizability bound."""
    return float((np.sqrt(2.0) * distortion + 1.0) * eps)


@dataclass
class InclusionPersistence:
    """Bottleneck distances between diagrams of ``H`` and ``H u I`` per degree, and the
    number of intervened-only components at the median nearest-neighbour scale."""

    bottleneck: dict[int, float]
    hausdorff: float
    stability_bound: float  # 2 * hausdorff
    n_natural: int
    n_intervened: int

    def within_bound(self) -> bool:
        return all(v <= self.stability_bound + 1e-9 for v in self.bottleneck.values())


def _rips_diagrams(points: np.ndarray, max_dim: int, max_edge: float) -> dict[int, np.ndarray]:
    import gudhi

    st = gudhi.RipsComplex(points=points, max_edge_length=max_edge).create_simplex_tree(
        max_dimension=max_dim + 1
    )
    st.compute_persistence()
    out = {}
    for dim in range(max_dim + 1):
        dgm = st.persistence_intervals_in_dimension(dim)
        out[dim] = dgm if len(dgm) else np.empty((0, 2))
    return out


def inclusion_persistence(
    natural: np.ndarray,
    intervened: np.ndarray,
    max_points: int = 600,
    max_dim: int = 1,
    max_edge: float | None = None,
    seed: int = 0,
) -> InclusionPersistence:
    """Compare Rips diagrams of ``H`` and ``H u I`` on subsamples (gudhi)."""
    import gudhi

    rng = np.random.default_rng(seed)
    if len(natural) > max_points:
        natural = natural[rng.choice(len(natural), max_points, replace=False)]
    if len(intervened) > max_points:
        intervened = intervened[rng.choice(len(intervened), max_points, replace=False)]
    hausdorff = one_sided_hausdorff(intervened, natural)
    if max_edge is None:
        max_edge = float(3.0 * np.median(nearest_neighbour_spacing(natural)) + 2.0 * hausdorff)
    d_h = _rips_diagrams(natural, max_dim, max_edge)
    d_hi = _rips_diagrams(np.vstack([natural, intervened]), max_dim, max_edge)
    bottleneck = {}
    for dim in range(max_dim + 1):
        a = np.where(np.isinf(d_h[dim]), max_edge, d_h[dim])
        b = np.where(np.isinf(d_hi[dim]), max_edge, d_hi[dim])
        bottleneck[dim] = float(gudhi.bottleneck_distance(a, b))
    return InclusionPersistence(
        bottleneck, hausdorff, 2.0 * hausdorff, len(natural), len(intervened)
    )
