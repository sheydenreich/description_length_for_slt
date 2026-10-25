"""Frozen counterfactual target sets with randomized response.

A target set fixes the (base, source) pairs, the intervened variable and the labels. The
labels pass through randomized response: with probability ``noise_rate`` a label is
replaced by a uniform draw from the output alphabet, so that a configuration which fits
the targets exactly does so at finite logits (the plan's realizability argument). The
content hash is stored with every result so that comparisons are known to share targets.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from bitsev.alignment.algorithms import Algorithm


@dataclass(frozen=True)
class NoiseChannel:
    """Randomized response: each label is replaced with probability ``rate`` by a uniform draw.

    The channel is declared with the targets, so every predictor codes the *observed* label
    law ``(1 - rate) p(y) + rate / K`` rather than the clean law ``p(y)``. This is the
    conditional law the plan calls realizable at finite logits; a frozen readout that is
    sharper than the noise cannot reach it on its own, and without the channel each flipped
    label costs tens of bits against a correct alignment.
    """

    rate: float
    n_outputs: int

    def log_prob(self, log_p: Tensor) -> Tensor:
        """Log probability of the observed label from the clean log probability of it."""
        if self.rate == 0:
            return log_p
        floor = torch.full_like(log_p, math.log(self.rate / self.n_outputs))
        return torch.logaddexp(math.log1p(-self.rate) + log_p, floor)

    def probs(self, p: np.ndarray) -> np.ndarray:
        """Observed-label probabilities from clean ones (rows sum to one)."""
        return (1 - self.rate) * p + self.rate / self.n_outputs

    @property
    def label(self) -> str:
        return f"rr={self.rate:g}"


@dataclass(frozen=True)
class TargetSet:
    base: np.ndarray  # (n,) indices into an input pool
    source: np.ndarray  # (n,) indices, source != base
    variable: str
    y_clean: np.ndarray  # algorithm's counterfactual output
    y: np.ndarray  # after randomized response; what is coded
    noise_rate: float
    n_outputs: int
    seed: int

    @property
    def channel(self) -> NoiseChannel:
        return NoiseChannel(self.noise_rate, self.n_outputs)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def pairs(self) -> np.ndarray:
        """``(n, 2)`` array of (base, source) indices: the ``x`` of the prequential coder."""
        return np.stack([self.base, self.source], axis=1)

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.base, self.source, self.y_clean, self.y):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(f"{self.variable}|{self.noise_rate}|{self.n_outputs}|{self.seed}".encode())
        return h.hexdigest()[:16]

    def target_law(self) -> np.ndarray:
        """Probability the clean label receives after randomized response."""
        return np.full(len(self), 1.0 - self.noise_rate * (1.0 - 1.0 / self.n_outputs))

    def subset(self, idx: np.ndarray) -> TargetSet:
        return TargetSet(
            self.base[idx],
            self.source[idx],
            self.variable,
            self.y_clean[idx],
            self.y[idx],
            self.noise_rate,
            self.n_outputs,
            self.seed,
        )


def randomized_response(
    y: np.ndarray, n_outputs: int, noise_rate: float, rng: np.random.Generator
) -> np.ndarray:
    flip = rng.random(len(y)) < noise_rate
    uniform = rng.integers(0, n_outputs, len(y))
    return np.where(flip, uniform, y)


def make_targets(
    algorithm: Algorithm,
    x_pool: np.ndarray,
    n_pairs: int,
    variable: str,
    noise_rate: float,
    seed: int,
) -> TargetSet:
    """Draw ``n_pairs`` (base, source) pairs from the pool and label them by the algorithm."""
    rng = np.random.default_rng(seed)
    n_pool = len(x_pool)
    base = rng.integers(0, n_pool, n_pairs)
    source = (base + rng.integers(1, n_pool, n_pairs)) % n_pool  # never equal to base
    y_clean = algorithm.counterfactual(x_pool[base], x_pool[source], variable)
    y = randomized_response(y_clean, algorithm.n_outputs, noise_rate, rng)
    return TargetSet(base, source, variable, y_clean, y, noise_rate, algorithm.n_outputs, seed)
