"""E-values from codes.

Two constructions from the plan (Appendix "E-values from codes"):

1. **Bayes factor.** If ``q_H`` and ``q_C`` are mixture codes, ``q_H(y^n) / q_C(y^n)`` has
   expectation one under ``q_C``: it is an e-value against the simple null "the data follow
   the control's prior predictive", and evaluated sequentially it is a test martingale, so
   Ville's inequality gives anytime-valid monitoring. In code lengths,
   ``log E_n = DL_C(n) - DL_H(n)``: the gain is the evidence.

2. **Universal inference.** Against a composite null ``{p_theta}``,
   ``E = q^preq_H(y^n) / sup_theta p_theta(y^n)`` has expectation at most one under every
   ``p_theta`` (Wasserman, Ramdas and Balakrishnan 2020). The supremum must be bounded from
   *above*; an optimizer's best value is a lower bound and would inflate ``E``. For convex
   null families on a ball, the Frank-Wolfe gap certifies the bound.

E-values from several rounds are multiplied only when each is valid conditionally on the
past; otherwise a prespecified weighted average is used, which is always valid.
"""

from __future__ import annotations

import numpy as np

from bitsev.codes.prequential import CodeLength

LOG2 = float(np.log(2.0))


# ----------------------------------------------------------------------------------------
# Bayes-factor e-processes


def log_bayes_factor(hypothesis: CodeLength, control: CodeLength) -> np.ndarray:
    """``log E_m = DL_control(m) - DL_hypothesis(m)`` for every coded sample size.

    Valid as an e-process only if ``control`` is a mixture (Bayes) code and the null is
    that the data follow its prior predictive; the function does not check the learner.
    """
    if not np.array_equal(hypothesis.schedule.boundaries, control.schedule.boundaries):
        raise ValueError("both codes must share the schedule")
    return control.cumulative() - hypothesis.cumulative()


def evidence_bits(log_e: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(log_e) / LOG2


def rejects(log_e: float, alpha: float) -> bool:
    """Reject at level ``alpha`` when ``E >= 1 / alpha``."""
    return bool(log_e >= -np.log(alpha))


def ever_rejects(log_e_sequence: np.ndarray, alpha: float) -> bool:
    """Anytime-valid rejection: the running e-process ever reaches ``1 / alpha``."""
    return bool(np.max(log_e_sequence) >= -np.log(alpha))


def weighted_average(log_evalues: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Log of a convex combination of e-values; valid whenever each e-value is valid."""
    log_e = np.asarray(log_evalues, dtype=float)
    w = np.full(len(log_e), 1.0 / len(log_e)) if weights is None else np.asarray(weights, float)
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to one")
    shift = log_e.max()
    return float(shift + np.log(np.sum(w * np.exp(log_e - shift))))


# ----------------------------------------------------------------------------------------
# Universal inference with certified bounds for convex null families


def certified_min_on_ball(value: float, grad: np.ndarray, w: np.ndarray, radius: float) -> float:
    """Lower bound on ``min_{||u|| <= radius} f(u)`` for convex ``f`` from one evaluation.

    By convexity ``f(u) >= f(w) + grad . (u - w)`` for all ``u``, and the right-hand side is
    minimised over the ball at ``u = -radius grad / ||grad||``, giving
    ``f(w) - grad . w - radius ||grad||``. This is the Frank-Wolfe duality gap: it is tight
    when ``w`` is optimal and it never exceeds the true minimum.
    """
    grad = np.asarray(grad, dtype=float)
    return float(value - grad @ w - radius * np.linalg.norm(grad))


def logistic_nll(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Summed negative log likelihood of labels ``y in {0, 1}`` and its gradient."""
    z = x @ w
    # log(1 + e^z) computed stably
    log1pexp = np.logaddexp(0.0, z)
    nll = float(np.sum(log1pexp - y * z))
    p = 1.0 / (1.0 + np.exp(-z))
    return nll, x.T @ (p - y)


def fit_logistic_in_ball(
    x: np.ndarray, y: np.ndarray, radius: float, n_iter: int = 2000, tol: float = 1e-10
) -> np.ndarray:
    """Projected gradient descent for the logistic MLE constrained to ``||w|| <= radius``.

    The step is ``1 / L`` with ``L = ||X||_2^2 / 4`` the Lipschitz constant of the gradient.
    Stops when the Frank-Wolfe gap falls below ``tol``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    step = 4.0 / np.linalg.norm(x, 2) ** 2
    w = np.zeros(x.shape[1])
    for _ in range(n_iter):
        value, grad = logistic_nll(w, x, y)
        if value - certified_min_on_ball(value, grad, w, radius) < tol:
            break
        w = w - step * grad
        norm = np.linalg.norm(w)
        if norm > radius:
            w *= radius / norm
    return w


def certified_logistic_min_nll(
    x: np.ndarray, y: np.ndarray, radius: float, **kw
) -> tuple[float, float]:
    """Return ``(optimizer value, certified lower bound)`` of the constrained minimum NLL."""
    w = fit_logistic_in_ball(x, y, radius, **kw)
    value, grad = logistic_nll(w, x, y)
    return value, certified_min_on_ball(value, grad, w, radius)


def universal_log_evalue(log_lik_hypothesis: float, min_null_nll_lower_bound: float) -> float:
    """``log E = log q_H(y) - sup log p_null(y) >= log q_H(y) + LB(min NLL_null)``.

    ``log_lik_hypothesis`` must come from a prequential code or a model fitted on
    separate data; otherwise the ratio is not an e-value.
    """
    return float(log_lik_hypothesis + min_null_nll_lower_bound)


def universal_log_evalues(
    hypothesis: CodeLength, null_min_nll: tuple[float, float]
) -> tuple[float, float]:
    """Certified and (invalid, for comparison) optimizer-based log e-values.

    ``null_min_nll`` is ``(optimizer value, certified lower bound)`` as returned by
    :func:`certified_logistic_min_nll`. The second output uses the optimizer value and is
    reported only to show the inflation it causes.
    """
    opt_value, lower_bound = null_min_nll
    log_lik = -hypothesis.total
    return universal_log_evalue(log_lik, lower_bound), universal_log_evalue(log_lik, opt_value)
