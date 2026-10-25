"""Learner and predictor protocols shared by every prequential code.

A :class:`Learner` turns a prefix of the data into a :class:`Predictor`; a predictor
assigns log probabilities to labels. This is all the prequential coder in
:mod:`bitsev.codes.prequential` needs, so gradient-trained alignment maps, SGLD
posteriors and conjugate families all enter through the same interface.

Learners that can evaluate a whole schedule at once (conjugate families keep running
sufficient statistics) implement :class:`SequentialLearner` and are used directly; the
generic coder falls back to refitting once per block.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from bitsev.codes.schedule import Schedule


@runtime_checkable
class Predictor(Protocol):
    def log_prob(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-item log probability (nats) of labels ``y`` given inputs ``x``."""
        ...


@runtime_checkable
class Learner(Protocol):
    """Fit on a prefix, return a predictor for the next items."""

    name: str
    min_fit_size: int
    """Smallest prefix the learner can be fitted on (0 if it has a prior predictive)."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> Predictor: ...


@runtime_checkable
class SequentialLearner(Learner, Protocol):
    """A learner that evaluates a full schedule in one vectorised pass."""

    def prequential_log_probs(self, x: np.ndarray, y: np.ndarray, schedule: Schedule) -> np.ndarray:
        """Log probabilities of the coded items ``schedule.start, ..., n - 1``, each
        predicted from the prefix the schedule assigns to it."""
        ...
