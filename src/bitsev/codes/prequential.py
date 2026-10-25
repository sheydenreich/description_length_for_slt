"""The prequential coder: a learner, a schedule and data give a code length.

``prequential_code`` returns a :class:`CodeLength` holding the per-item losses of the coded
items. Nothing is aggregated or corrected here: block corrections and gains live in
:mod:`bitsev.codes.gain`, so stored results stay raw and reinterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bitsev.codes.learner import Learner, SequentialLearner
from bitsev.codes.schedule import Schedule

NATS_PER_BIT = float(np.log(2.0))


@dataclass
class CodeLength:
    """Per-item code lengths (nats) of the items a schedule codes.

    ``item_losses[i]`` is the loss of item ``schedule.start + i``.
    """

    item_losses: np.ndarray
    schedule: Schedule
    learner: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.item_losses = np.asarray(self.item_losses, dtype=float)
        if len(self.item_losses) != self.schedule.n_coded:
            raise ValueError("item_losses must have one entry per coded item")

    @property
    def total(self) -> float:
        """DL(n) in nats."""
        return float(self.item_losses.sum())

    @property
    def total_bits(self) -> float:
        return self.total / NATS_PER_BIT

    def cumulative(self) -> np.ndarray:
        """DL(m) for m = start + 1, ..., n (nats)."""
        return np.cumsum(self.item_losses)

    def at_boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample sizes at block ends and the cumulative code length there."""
        ends = self.schedule.boundaries[1:]
        return ends, self.cumulative()[ends - self.schedule.start - 1]

    def block_losses(self) -> np.ndarray:
        """Summed loss per block."""
        return np.add.reduceat(
            self.item_losses, self.schedule.boundaries[:-1] - self.schedule.start
        )

    def coded_sizes(self) -> np.ndarray:
        """Sample sizes m = start + 1, ..., n aligned with :meth:`cumulative`."""
        return np.arange(self.schedule.start + 1, self.schedule.n + 1)


def prequential_code(
    learner: Learner, x: np.ndarray | None, y: np.ndarray, schedule: Schedule
) -> CodeLength:
    """Code ``y`` (given ``x``) prequentially under ``schedule``.

    Sequential learners are evaluated in one pass; any other learner is refitted once per
    block on the prefix before it.
    """
    y = np.asarray(y)
    if schedule.n != len(y):
        raise ValueError(f"schedule codes {schedule.n} items but {len(y)} were given")
    if schedule.start < learner.min_fit_size:
        raise ValueError(
            f"{learner.name} needs a prefix of at least {learner.min_fit_size} items, "
            f"schedule starts at {schedule.start}"
        )
    if isinstance(learner, SequentialLearner):
        log_probs = learner.prequential_log_probs(x, y, schedule)
    else:
        log_probs = np.empty(schedule.n_coded)
        for fit_size, block_end in schedule.blocks():
            predictor = learner.fit(_prefix(x, fit_size), y[:fit_size])
            block = slice(fit_size, block_end)
            log_probs[fit_size - schedule.start : block_end - schedule.start] = predictor.log_prob(
                _prefix(x, block_end, fit_size), y[block]
            )
    return CodeLength(-np.asarray(log_probs, dtype=float), schedule, learner.name)


def _prefix(x: np.ndarray | None, end: int, start: int = 0):
    return None if x is None else x[start:end]
