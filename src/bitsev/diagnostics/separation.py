"""D1: how far apart does a network keep inputs with different values of ``V``?

``sep_V`` is the smallest distance between states of inputs whose values of ``V`` differ. It
is also the first birth of the degree-zero kernel of the inclusion of the monochromatic
Vietoris-Rips complexes into the full one: for ``t < sep_V`` every path is monochromatic,
and at ``t = sep_V`` the closest bichromatic edge joins two classes. With the output margin
``gamma`` and a reference sensitivity ``Lambda_nat`` of the downstream computation, the
separation ratio ``rho_V = Lambda_nat sep_V / gamma`` says whether a rotation can satisfy
the stretch budget (``rho_V >= 1``) or an alignment must supply ``log2(1 / rho_V)`` bits of
stretch.

For states that are blind to ``V`` (positions independent of the colouring) the number of
bichromatic pairs within distance ``r`` is approximately Poisson, so the median of
``sep_V`` scales like ``n^{-2/k}`` in intrinsic dimension ``k``; a network that computes
``V`` with a margin keeps ``sep_V`` bounded below for every ``n``.

Everything here is numpy and scipy; the states are plain arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


def bichromatic_nn_distances(
    states: np.ndarray, labels: np.ndarray, boxsize: float | None = None
) -> np.ndarray:
    """For every state, the distance to the nearest state with a different label.

    ``boxsize`` makes the metric periodic (flat torus), used by the Poisson check.
    """
    states = np.asarray(states, dtype=float)
    labels = np.asarray(labels)
    out = np.full(len(states), np.inf)
    classes = np.unique(labels)
    trees = {c: cKDTree(states[labels == c], boxsize=boxsize) for c in classes}
    for c in classes:
        mask = labels == c
        if mask.sum() == 0:
            continue
        for other in classes:
            if other == c:
                continue
            d, _ = trees[other].query(states[mask], k=1)
            out[mask] = np.minimum(out[mask], d)
    return out


def separation(states: np.ndarray, labels: np.ndarray, boxsize: float | None = None) -> float:
    """``sep_V``: the smallest bichromatic distance."""
    return float(bichromatic_nn_distances(states, labels, boxsize).min())


def same_label_nn_distances(states: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """For every state, the distance to the nearest *other* state with the same label
    (``inf`` for a singleton class)."""
    states = np.asarray(states, dtype=float)
    labels = np.asarray(labels)
    out = np.full(len(states), np.inf)
    for c in np.unique(labels):
        mask = labels == c
        if mask.sum() < 2:
            continue
        d, _ = cKDTree(states[mask]).query(states[mask], k=2)
        out[mask] = d[:, 1]
    return out


@dataclass
class MarginStatistics:
    """Separation of the classes of ``V`` in units of the within-class spacing.

    ``sep_ratio = sep_V / median within-class NN distance`` is the worst-case margin and
    ``nn_ratio = median between-class NN / median within-class NN`` the typical one. Both
    are scale-free, so a network and its untrained twin at the same locus can be compared
    directly; on its own neither is a test, because ``V`` is a function of the input and
    inputs that share the slots ``V`` depends on are near each other in any continuous
    representation.
    """

    sep_v: float
    within_nn: float  # median distance to the nearest other state with the same value
    between_nn: float  # median distance to the nearest state with a different value
    n_states: int

    @property
    def sep_ratio(self) -> float:
        return self.sep_v / self.within_nn

    @property
    def nn_ratio(self) -> float:
        return self.between_nn / self.within_nn

    def record(self) -> dict:
        return {
            "sep_v": self.sep_v,
            "within_nn": self.within_nn,
            "between_nn": self.between_nn,
            "sep_ratio": self.sep_ratio,
            "nn_ratio": self.nn_ratio,
            "n_states": self.n_states,
        }


def margin_statistics(states: np.ndarray, labels: np.ndarray) -> MarginStatistics:
    """``MarginStatistics`` of distinct states (duplicates would put zero distances in the
    within-class spacing, so pass one state per distinct input)."""
    between = bichromatic_nn_distances(states, labels)
    within = same_label_nn_distances(states, labels)
    return MarginStatistics(
        float(between.min()),
        float(np.median(within[np.isfinite(within)])),
        float(np.median(between)),
        len(states),
    )


def separation_ratio(sep: float, margin: float, sensitivity: float) -> float:
    """``rho_V = Lambda_nat sep_V / gamma``; below one an alignment must stretch."""
    return sensitivity * sep / margin


def stretch_bits_required(rho: float) -> float:
    """``log2(1 / rho_V)`` when ``rho_V < 1``, else zero."""
    return max(0.0, -np.log2(rho))


@dataclass
class ShuffleNull:
    observed: float
    null_samples: np.ndarray
    p_value: float  # fraction of shuffles with separation at least as large as observed

    @property
    def log_e_value(self) -> float:
        """A calibrated e-value from the exact permutation p-value: ``E = 1 / (2 sqrt p)``
        is a valid calibrator (Vovk and Wang 2021)."""
        return float(-np.log(2.0 * np.sqrt(max(self.p_value, 1e-300))))


def shuffle_null(
    states: np.ndarray,
    labels: np.ndarray,
    n_shuffles: int = 200,
    seed: int = 0,
    boxsize: float | None = None,
) -> ShuffleNull:
    """Permutation null for ``sep_V``: exact under exchangeability of labels given states.

    Rejecting shows only that the labels are not exchangeable given the states. When ``V``
    is a function of the input this holds for any injective representation, untrained
    networks included (inputs that differ only in coordinates ``V`` ignores share its value
    and are each other's neighbours), and repeated inputs make the null's separation zero.
    The Stage 1 runners therefore report ``margin_statistics`` against the untrained twin.
    """
    rng = np.random.default_rng(seed)
    observed = separation(states, labels, boxsize)
    null = np.array(
        [separation(states, rng.permutation(labels), boxsize) for _ in range(n_shuffles)]
    )
    p = (1.0 + np.sum(null >= observed)) / (n_shuffles + 1.0)
    return ShuffleNull(observed, null, float(p))


# ----------------------------------------------------------------------------------------
# Degree-zero kernel of the coloured inclusion, from minimum spanning trees


def _mst_edge_lengths(states: np.ndarray) -> np.ndarray:
    if len(states) < 2:
        return np.empty(0)
    dense = squareform(pdist(states))
    tree = minimum_spanning_tree(dense)
    return np.sort(tree.data)


@dataclass
class KernelRank:
    """Rank function of the degree-zero kernel module at every event scale.

    ``rank(t) = #{global MST edges <= t} - #{within-class MST edges <= t}``, which equals
    ``sum_v beta_0(VR_t(H_v)) - beta_0(VR_t(H))``.
    """

    scales: np.ndarray
    rank: np.ndarray
    global_edges: np.ndarray
    class_edges: np.ndarray

    @property
    def first_birth(self) -> float:
        positive = self.scales[self.rank > 0]
        return float(positive[0]) if len(positive) else float("inf")

    @property
    def bichromatic_merge_scales(self) -> np.ndarray:
        """Global MST edge lengths that raise the kernel rank (scales at which classes mingle)."""
        return np.setdiff1d(self.global_edges, self.class_edges)


def kernel_rank(
    states: np.ndarray, labels: np.ndarray, max_points: int = 4000, seed: int = 0
) -> KernelRank:
    """Compute the kernel rank function from a global and per-class minimum spanning trees.

    Uses dense distance matrices, so large clouds are subsampled to ``max_points``.
    """
    states = np.asarray(states, dtype=float)
    labels = np.asarray(labels)
    if len(states) > max_points:
        idx = np.random.default_rng(seed).choice(len(states), max_points, replace=False)
        states, labels = states[idx], labels[idx]
    global_edges = _mst_edge_lengths(states)
    class_edges = np.sort(
        np.concatenate([_mst_edge_lengths(states[labels == c]) for c in np.unique(labels)])
    )
    scales = np.unique(np.concatenate([global_edges, class_edges]))
    rank = np.searchsorted(global_edges, scales, side="right") - np.searchsorted(
        class_edges, scales, side="right"
    )
    return KernelRank(scales, rank, global_edges, class_edges)


# ----------------------------------------------------------------------------------------
# The Poisson prediction for geometry blind to V


def unit_ball_volume(k: int) -> float:
    from scipy.special import gamma

    return float(np.pi ** (k / 2) / gamma(k / 2 + 1))


def poisson_median_separation(
    n: int, k: int, class_probs: np.ndarray, density_l2: float = 1.0
) -> float:
    """Median of ``sep_V`` if positions are blind to the labels.

    The number of bichromatic pairs within ``r`` is approximately Poisson with mean
    ``C(n, 2) (1 - sum p_v^2) omega_k r^k integral f^2``; the median solves mean = ln 2.
    """
    p = np.asarray(class_probs, dtype=float)
    mean_coefficient = n * (n - 1) / 2 * (1.0 - np.sum(p**2)) * unit_ball_volume(k) * density_l2
    return float((np.log(2.0) / mean_coefficient) ** (1.0 / k))
