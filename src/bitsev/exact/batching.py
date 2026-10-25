"""Batching tables: fitted logarithmic coefficients of block codes for known learning curves.

For a learner whose expected excess after ``m`` items is ``e(m)``, the expected cumulative
excess under geometric blocks of growth ``r`` grows like ``kappa (r - 1) / ln r * ln n``.
This module sums the exact curves of :mod:`bitsev.exact.gaussian_mean` and
:mod:`bitsev.exact.germs` over schedules and fits the coefficient, with and without the
batching correction, so that criterion 3 of ``preregistration/G-A0.md`` can be checked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bitsev.codes import schedule as S
from bitsev.codes.gain import fit_log_slope
from bitsev.codes.schedule import batching_factor
from bitsev.exact.learning_curve import cumulative_excess


@dataclass
class BatchingRow:
    model: str
    learner: str
    growth: float
    kappa: float  # the coefficient the learner should have (lambda for Bayes codes)
    fitted: float
    fitted_stderr: float

    @property
    def predicted(self) -> float:
        return self.kappa * batching_factor(self.growth)

    @property
    def corrected(self) -> float:
        return self.fitted / batching_factor(self.growth)

    @property
    def error(self) -> float:
        return self.fitted - self.predicted


def batching_rows(
    model: str,
    learner: str,
    excess: Callable[[np.ndarray], np.ndarray],
    kappa: float,
    growths: tuple[float, ...],
    n: int,
    fit_range: tuple[int, int],
    start: int = 1,
) -> list[BatchingRow]:
    """One row per growth factor for a learner with known expected excess curve."""
    rows = []
    for r in growths:
        ends, cum = cumulative_excess(excess, S.geometric(n, r, start=start))
        fit = fit_log_slope(ends, cum, *fit_range)
        rows.append(BatchingRow(model, learner, r, kappa, fit.slope, fit.slope_stderr))
    return rows


def format_batching_rows(rows: list[BatchingRow]) -> str:
    header = (
        f"{'model':<12}{'learner':<10}{'r':>5}{'kappa':>7}{'fitted':>9}"
        f"{'predicted':>11}{'corrected':>11}{'error':>10}"
    )
    lines = [header]
    for w in rows:
        lines.append(
            f"{w.model:<12}{w.learner:<10}{w.growth:>5g}{w.kappa:>7.3g}{w.fitted:>9.4f}"
            f"{w.predicted:>11.4f}{w.corrected:>11.4f}{w.error:>10.1e}"
        )
    return "\n".join(lines)
