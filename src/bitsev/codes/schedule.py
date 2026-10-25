"""Prequential schedules: which prefix is used to predict which items.

A schedule is a strictly increasing sequence of boundaries ``b_0 < b_1 < ... < b_K = n``.
Block ``j`` codes the items with indices in ``[b_j, b_{j+1})`` using a predictor fitted
on the first ``b_j`` items. Items with index below ``b_0`` are not coded (a plug-in
learner cannot predict from an empty prefix; a Bayes learner can, so ``b_0 = 0`` is
allowed).

Two families matter in the plan:

* per-sample updating, ``b_j = b_0 + j``;
* geometric blocks, ``b_j = ceil(b_0 * r**j)``, which multiply logarithmic coefficients
  of the cumulative excess by the batching factor ``(r - 1) / ln r``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


def batching_factor(growth: float) -> float:
    """Return ``(r - 1) / ln r``, the factor by which geometric blocks of growth ``r``
    inflate a logarithmic coefficient. Per-sample updating (``r == 1``) gives 1."""
    if growth < 1:
        raise ValueError("growth factor must be >= 1")
    if growth == 1:
        return 1.0
    return (growth - 1.0) / np.log(growth)


@dataclass(frozen=True)
class Schedule:
    """Block boundaries of a prequential code.

    Attributes
    ----------
    boundaries:
        Strictly increasing integer array ``b_0 < ... < b_K``; ``b_K`` is the number of
        items ``n``. ``b_0`` is the index of the first coded item.
    growth:
        The growth factor ``r`` for geometric schedules, 1 for per-sample updating, or
        ``None`` for an irregular schedule. Only used for labelling and for the batching
        correction.
    """

    boundaries: np.ndarray
    growth: float | None = None

    def __post_init__(self) -> None:
        b = np.asarray(self.boundaries, dtype=np.int64)
        if b.ndim != 1 or len(b) < 2:
            raise ValueError("a schedule needs at least two boundaries")
        if np.any(np.diff(b) <= 0):
            raise ValueError("boundaries must be strictly increasing")
        if b[0] < 0:
            raise ValueError("boundaries must be non-negative")
        object.__setattr__(self, "boundaries", b)

    @property
    def n(self) -> int:
        """Total number of items (coded or not)."""
        return int(self.boundaries[-1])

    @property
    def start(self) -> int:
        """Index of the first coded item."""
        return int(self.boundaries[0])

    @property
    def n_coded(self) -> int:
        return self.n - self.start

    @property
    def n_blocks(self) -> int:
        return len(self.boundaries) - 1

    def blocks(self) -> Iterator[tuple[int, int]]:
        """Yield ``(fit_size, block_end)``: block ``j`` codes items ``[fit_size, block_end)``."""
        for lo, hi in zip(self.boundaries[:-1], self.boundaries[1:], strict=True):
            yield int(lo), int(hi)

    def fit_sizes(self) -> np.ndarray:
        """For every coded item, the number of items the predictor was fitted on.

        Returns an array of length ``n_coded`` aligned with items ``start, ..., n - 1``.
        """
        lengths = np.diff(self.boundaries)
        return np.repeat(self.boundaries[:-1], lengths)

    def label(self) -> str:
        if self.growth is None:
            return f"irregular({self.n_blocks} blocks)"
        if self.growth == 1:
            return "per-sample"
        return f"geometric(r={self.growth:g})"


def per_sample(n: int, start: int = 1) -> Schedule:
    """Update after every item: boundaries ``start, start + 1, ..., n``."""
    if not 0 <= start < n:
        raise ValueError("need 0 <= start < n")
    return Schedule(np.arange(start, n + 1), growth=1.0)


def geometric(n: int, growth: float, start: int = 1) -> Schedule:
    """Boundaries ``ceil(start * r**j)`` (deduplicated), clipped to ``n``.

    With ``start = 1`` this is the plan's ``n_j = ceil(r**j)``. ``growth == 1`` returns the
    per-sample schedule.
    """
    if growth == 1:
        return per_sample(n, start)
    if growth < 1:
        raise ValueError("growth factor must be >= 1")
    if not 0 < start < n:
        raise ValueError("need 0 < start < n for a geometric schedule")
    n_steps = int(np.ceil(np.log(n / start) / np.log(growth))) + 2
    raw = np.ceil(start * growth ** np.arange(n_steps)).astype(np.int64)
    b = np.unique(np.minimum(raw, n))
    if b[-1] != n:
        b = np.append(b, n)
    return Schedule(b, growth=float(growth))


def from_boundaries(boundaries: np.ndarray, growth: float | None = None) -> Schedule:
    """Wrap explicit boundaries, for example checkpoint sizes of a trained learner."""
    return Schedule(np.asarray(boundaries), growth=growth)
