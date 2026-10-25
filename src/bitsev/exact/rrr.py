"""Reduced-rank regression: an exact example where a plug-in coefficient differs from lambda.

Model: ``y = U V x + noise`` with ``U`` of shape ``(M, H)``, ``V`` of shape ``(H, N)``,
unit Gaussian noise and inputs, true rank ``r``. Aoyagi and Watanabe (2005) give the
learning coefficient ``lambda`` and multiplicity ``m`` of the Bayes code in closed form.

When ``H >= min(M, N)`` the products ``U V`` fill every ``M x N`` matrix, so the
maximum-likelihood fit is ordinary least squares whatever the parametrisation. Its expected
excess log loss after ``m`` samples with Gaussian design is exactly
``M N / (2 (m - N - 1))`` for ``m > N + 1``, hence ``kappa = M N / 2``: the plug-in learner
has the *regular* coefficient while the Bayes code of the same family has the singular
``lambda <= M N / 2``, with strict inequality whenever the true rank is below ``min(M, N)``.
This is the plan's statement T4 in its simplest exact form.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


def aoyagi_watanabe(M: int, N: int, H: int, r: int) -> tuple[Fraction, int]:
    """``(lambda, m)`` of reduced-rank regression (Aoyagi and Watanabe 2005).

    ``M`` output dimension, ``N`` input dimension, ``H`` inner width, ``r`` true rank.
    Ported from the author's ``slt_validation_hpc`` package, where it was cross-checked
    against an independent implementation.
    """
    if not 0 <= r <= min(M, N, H):
        raise ValueError("need 0 <= r <= min(M, N, H)")
    if N + H < M + r:
        return Fraction(H * N + r * (M - H), 2), 1
    if M + H < N + r:
        return Fraction(H * M + r * (N - H), 2), 1
    if M + N < H + r:
        return Fraction(M * N, 2), 1
    base = 2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2
    if (M + N + H + r) % 2 == 0:
        return Fraction(base, 8), 1
    return Fraction(base + 1, 8), 2


def ols_excess(m: np.ndarray, M: int, N: int) -> np.ndarray:
    """Exact expected excess KL of least squares after ``m`` Gaussian-design samples.

    ``E[KL] = (1/2) E tr[(B_hat - B)(B_hat - B)^T] = M N / (2 (m - N - 1))`` because each row
    of ``B_hat - B`` is ``N(0, (X^T X)^{-1})`` given ``X`` and ``E (X^T X)^{-1} = I / (m - N - 1)``.
    """
    m = np.asarray(m, dtype=float)
    return np.where(m > N + 1, M * N / (2.0 * (m - N - 1)), np.inf)


def ols_kappa(M: int, N: int) -> Fraction:
    """Asymptotic coefficient of the least-squares learner: the regular value ``M N / 2``."""
    return Fraction(M * N, 2)


@dataclass
class RrrRow:
    M: int
    N: int
    H: int
    r: int
    lam: Fraction
    mult: int
    kappa_plugin: Fraction

    @property
    def gap(self) -> Fraction:
        """``kappa_plugin - lambda``; positive means the plug-in code pays more."""
        return self.kappa_plugin - self.lam


def comparison_table(
    shapes: tuple[tuple[int, int, int], ...] = (
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
        (3, 2, 3),
        (4, 2, 4),
    ),
) -> list[RrrRow]:
    """``lambda`` against the least-squares ``kappa`` for ``H >= min(M, N)`` and every true rank."""
    rows = []
    for M, N, H in shapes:
        if H < min(M, N):
            raise ValueError("the least-squares comparison needs H >= min(M, N)")
        for r in range(min(M, N) + 1):
            lam, mult = aoyagi_watanabe(M, N, H, r)
            rows.append(RrrRow(M, N, H, r, lam, mult, ols_kappa(M, N)))
    return rows


def format_comparison(rows: list[RrrRow]) -> str:
    lines = [f"{'(M,N,H)':<10}{'r':>3}{'lambda':>9}{'m':>3}{'kappa_OLS':>11}{'gap':>7}"]
    for w in rows:
        lines.append(
            f"{str((w.M, w.N, w.H)):<10}{w.r:>3}{str(w.lam):>9}{w.mult:>3}"
            f"{str(w.kappa_plugin):>11}{str(w.gap):>7}"
        )
    return "\n".join(lines)
