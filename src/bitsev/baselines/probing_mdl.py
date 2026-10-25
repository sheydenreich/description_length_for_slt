"""Online probing codes (Voita and Titov 2020) without controls.

A probe predicts the variable ``V`` itself from the hidden state at a locus, and its
prequential code length over the inputs is the rival "online code". Unlike the Stage 1
score it does not use the network's readout or interventions, has no untrained-network
control built in, and measures what a *restricted family* can read off a representation,
not whether the network uses it. It is included as a baseline, with the same schedules
and the same learner protocol as the Stage 1 codes so that the numbers are comparable.

The probe is a multinomial logistic readout on the cached states, trained by L-BFGS with
an L2 penalty per prefix (cold start), which makes it a plug-in learner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

from bitsev.codes.prequential import CodeLength, prequential_code
from bitsev.codes.schedule import Schedule


@dataclass
class LogisticProbe:
    weights: np.ndarray  # (d + 1, K)

    def log_prob(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        logits = _augment(x) @ self.weights
        return log_softmax(logits, axis=1)[np.arange(len(y)), np.asarray(y)]


def _augment(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(x, dtype=float), np.ones((len(x), 1))], axis=1)


class OnlineProbeLearner:
    """Plug-in multinomial logistic probe with ridge penalty ``l2``."""

    min_fit_size = 1

    def __init__(self, n_classes: int, l2: float = 1e-2, max_iter: int = 200):
        self.n_classes, self.l2, self.max_iter = n_classes, l2, max_iter
        self.name = f"online-probe[logistic,l2={l2:g}]"

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticProbe:
        xa = _augment(x)
        y = np.asarray(y, dtype=int)
        d1, k = xa.shape[1], self.n_classes
        onehot = np.eye(k)[y]

        def objective(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
            w = w_flat.reshape(d1, k)
            logits = xa @ w
            lp = log_softmax(logits, axis=1)
            nll = -np.sum(lp * onehot)
            p = np.exp(lp)
            grad = xa.T @ (p - onehot) + self.l2 * w
            return nll + 0.5 * self.l2 * np.sum(w * w), grad.ravel()

        res = minimize(
            objective,
            np.zeros(d1 * k),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        return LogisticProbe(res.x.reshape(d1, k))


def online_probe_code(
    states: np.ndarray, variable_values: np.ndarray, schedule: Schedule, l2: float = 1e-2
) -> CodeLength:
    """Prequential code of ``V`` from the states under the given schedule."""
    values = np.unique(variable_values, return_inverse=True)[1]
    learner = OnlineProbeLearner(int(values.max()) + 1, l2=l2)
    return prequential_code(learner, np.asarray(states, dtype=float), values, schedule)


def uniform_code_nats(n_coded: int, n_classes: int) -> float:
    return float(n_coded * np.log(n_classes))
