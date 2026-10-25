"""Collinearity of logarithmic complexity terms with offsets (Appendix C.2 of the plan).

After removing a constant, ``(log n) / n`` and ``1 / n`` are almost collinear over a few
decades of ``n``, so the coefficient of ``(log n) / n`` has a standard error inflated by
the square root of a large variance inflation factor. The plan concludes that
multiplicity terms are not fitted from code-length curves.

The plan quotes VIFs of 568, 348 and 401 over one, two and three decades. The grid of the
original script is not recorded; log-spaced grids here give values between about 100 and
650 depending on the range and spacing, which supports the same conclusion. The tests
therefore assert the qualitative claim (VIF above 100, standard error inflated more than
tenfold) rather than the printed numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bitsev.codes.gain import log_term_vif


@dataclass
class VifRow:
    decades: float
    n_min: float
    n_max: float
    points: int
    vif: float

    @property
    def stderr_inflation(self) -> float:
        return float(np.sqrt(self.vif))


def vif_over_decades(
    decades: tuple[float, ...] = (1, 2, 3),
    n_min: float = 1e2,
    points_per_decade: int = 20,
) -> list[VifRow]:
    """VIF of ``(log n)/n`` against ``1/n`` on log-spaced grids spanning ``decades``."""
    rows = []
    for d in decades:
        n_max = n_min * 10**d
        n = np.logspace(np.log10(n_min), np.log10(n_max), int(points_per_decade * d) + 1)
        rows.append(VifRow(d, n_min, n_max, len(n), log_term_vif(n)))
    return rows


def format_vif_table(rows: list[VifRow]) -> str:
    lines = [f"{'decades':>8}{'n range':>22}{'points':>8}{'VIF':>10}{'SE x':>8}"]
    for r in rows:
        rng = f"[{r.n_min:.0e}, {r.n_max:.0e}]"
        lines.append(
            f"{r.decades:>8g}{rng:>22}{r.points:>8}{r.vif:>10.0f}{r.stderr_inflation:>8.1f}"
        )
    return "\n".join(lines)
