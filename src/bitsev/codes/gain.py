"""Gains between codes, logarithmic-slope fits and the batching correction.

The gain of a configuration over a control on the same ordered targets is
``G(n) = DL_control(n) - DL(n)``. In the Occam regime it grows at most logarithmically,
and the plan's decomposition (eq. gain) reads its coefficient as a difference of
learning coefficients only after two corrections handled here:

* geometric blocks of growth ``r`` inflate the coefficient by ``(r - 1) / ln r``;
* the regressors ``log n`` and a constant are well separated, but ``(log n) / n`` and
  ``1 / n`` are nearly collinear over a few decades, so multiplicity terms are not fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bitsev.codes.prequential import CodeLength
from bitsev.codes.schedule import batching_factor


@dataclass
class GainCurve:
    """``G(m)`` for the sample sizes a shared schedule codes."""

    sizes: np.ndarray
    values: np.ndarray  # nats
    growth: float | None
    hypothesis: str
    control: str

    @property
    def total(self) -> float:
        return float(self.values[-1])

    @property
    def total_bits(self) -> float:
        return self.total / np.log(2.0)

    def at_boundaries(self, boundaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(self.sizes, boundaries)
        return self.sizes[idx], self.values[idx]


def gain(hypothesis: CodeLength, control: CodeLength) -> GainCurve:
    """Cumulative gain of ``hypothesis`` over ``control`` on the same schedule."""
    if not np.array_equal(hypothesis.schedule.boundaries, control.schedule.boundaries):
        raise ValueError("gain requires both codes to use the same schedule on the same targets")
    return GainCurve(
        sizes=hypothesis.coded_sizes(),
        values=control.cumulative() - hypothesis.cumulative(),
        growth=hypothesis.schedule.growth,
        hypothesis=hypothesis.learner,
        control=control.learner,
    )


@dataclass
class LogFit:
    """Least-squares fit ``values ~ intercept + slope * ln n``.

    ``corrected_slope`` divides out the batching factor of the schedule, so for a Bayes
    code it estimates the learning coefficient (difference) and for other learners the
    learner's own coefficient ``kappa``.
    """

    slope: float
    intercept: float
    slope_stderr: float
    growth: float | None
    n_points: int
    n_range: tuple[int, int]

    @property
    def corrected_slope(self) -> float:
        return self.slope / batching_factor(self.growth if self.growth is not None else 1.0)


def fit_log_slope(
    sizes: np.ndarray,
    values: np.ndarray,
    n_min: int | None = None,
    n_max: int | None = None,
    growth: float | None = None,
) -> LogFit:
    """Fit ``values`` against ``ln n`` on ``n_min <= n <= n_max`` (inclusive)."""
    sizes = np.asarray(sizes)
    values = np.asarray(values, dtype=float)
    mask = np.ones(len(sizes), dtype=bool)
    if n_min is not None:
        mask &= sizes >= n_min
    if n_max is not None:
        mask &= sizes <= n_max
    if mask.sum() < 3:
        raise ValueError("need at least three points to fit a slope with a standard error")
    log_n, v = np.log(sizes[mask]), values[mask]
    design = np.column_stack([np.ones_like(log_n), log_n])
    coef, residuals, _, _ = np.linalg.lstsq(design, v, rcond=None)
    dof = len(v) - 2
    sigma2 = (
        float(residuals[0]) / dof
        if residuals.size
        else float(np.sum((v - design @ coef) ** 2)) / dof
    )
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return LogFit(
        slope=float(coef[1]),
        intercept=float(coef[0]),
        slope_stderr=float(np.sqrt(cov[1, 1])),
        growth=growth,
        n_points=int(mask.sum()),
        n_range=(int(sizes[mask].min()), int(sizes[mask].max())),
    )


def variance_inflation(regressors: np.ndarray) -> np.ndarray:
    """Variance inflation factor of each column after removing a constant.

    For two regressors this is ``1 / (1 - rho^2)``; the standard error of each
    coefficient is inflated by its square root.
    """
    x = np.asarray(regressors, dtype=float)
    x = x - x.mean(axis=0)
    x = x / np.linalg.norm(x, axis=0)
    inv = np.linalg.inv(x.T @ x)
    return np.diag(inv)


def log_term_vif(sizes: np.ndarray) -> float:
    """VIF of ``(log n) / n`` against ``1 / n`` on the given sample sizes.

    Large values mean a ``(log n) / n`` signature cannot be told from an ``O(1)``
    offset plus ``1 / n``; the plan therefore does not fit multiplicity terms.
    """
    n = np.asarray(sizes, dtype=float)
    return float(variance_inflation(np.column_stack([np.log(n) / n, 1.0 / n]))[0])
