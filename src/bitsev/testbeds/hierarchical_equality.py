"""Hierarchical equality (Geiger et al.): the first ground-truth task of the plan.

An input is four vectors ``(a, b, c, d)`` drawn from a vocabulary. The algorithm computes
two intermediate equality bits ``V1 = [a == b]`` and ``V2 = [c == d]`` and outputs
``y = [V1 == V2]``. Both variables are *computed*, not copied, which is the case the plan
expects stretch budgets and code lengths to separate.

Distribution shift for the strongest baseline: a second, disjoint vocabulary. A map
trained on inputs from vocabulary A and evaluated on vocabulary B has to have aligned the
equality computation rather than the identities of particular vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vocabulary:
    """``size`` distinct vectors of dimension ``dim`` that the four slots are filled from."""

    vectors: np.ndarray  # (size, dim)

    @property
    def size(self) -> int:
        return self.vectors.shape[0]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]


def make_vocabulary(size: int, dim: int, rng: np.random.Generator) -> Vocabulary:
    return Vocabulary(rng.normal(size=(size, dim)))


def make_disjoint_vocabularies(
    size: int, dim: int, rng: np.random.Generator
) -> tuple[Vocabulary, Vocabulary]:
    """Two vocabularies without shared vectors, for the distribution-shifted split."""
    both = rng.normal(size=(2 * size, dim))
    return Vocabulary(both[:size]), Vocabulary(both[size:])


@dataclass
class HierarchicalEqualityData:
    """Inputs with their slot identities and the algorithm's variables."""

    x: np.ndarray  # (n, 4 dim)
    slots: np.ndarray  # (n, 4) vocabulary indices
    v1: np.ndarray  # bool, a == b
    v2: np.ndarray  # bool, c == d
    y: np.ndarray  # int, [v1 == v2]

    def __len__(self) -> int:
        return len(self.y)


def sample(n: int, vocab: Vocabulary, rng: np.random.Generator) -> HierarchicalEqualityData:
    """Balanced sampling: the two equality bits are independent fair coins."""
    v1 = rng.random(n) < 0.5
    v2 = rng.random(n) < 0.5
    slots = np.empty((n, 4), dtype=np.int64)
    for pair, equal in ((slice(0, 2), v1), (slice(2, 4), v2)):
        first = rng.integers(0, vocab.size, n)
        # for unequal pairs draw a second index different from the first
        other = (first + rng.integers(1, vocab.size, n)) % vocab.size
        slots[:, pair] = np.stack([first, np.where(equal, first, other)], axis=1)
    x = vocab.vectors[slots].reshape(n, 4 * vocab.dim)
    return HierarchicalEqualityData(x, slots, v1, v2, (v1 == v2).astype(np.int64))


def variables_from_input(
    x: np.ndarray, dim: int, atol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the equality bits from the raw input (used by the algorithm object)."""
    blocks = x.reshape(len(x), 4, dim)
    v1 = np.all(np.abs(blocks[:, 0] - blocks[:, 1]) < atol, axis=1)
    v2 = np.all(np.abs(blocks[:, 2] - blocks[:, 3]) < atol, axis=1)
    return v1, v2
