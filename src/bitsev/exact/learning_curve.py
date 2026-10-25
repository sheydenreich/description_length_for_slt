"""Expected cumulative excess of a learner with a known learning curve under a schedule.

If a learner's expected excess log loss after ``m`` examples is ``e(m)``, coding the items
of block ``j`` with the predictor fitted on ``b_j`` items costs ``(b_{j+1} - b_j) e(b_j)``
in expectation. Summing over blocks gives the expected cumulative excess at every block
end, which is what the batching factor describes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bitsev.codes.schedule import Schedule


def cumulative_excess(
    excess: Callable[[np.ndarray], np.ndarray], schedule: Schedule
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(block_ends, expected cumulative excess at those ends)``.

    ``excess(m)`` must be vectorised over an integer array of fit sizes ``m >= start``.
    """
    fit_sizes = schedule.boundaries[:-1]
    lengths = np.diff(schedule.boundaries)
    per_block = lengths * excess(fit_sizes.astype(float))
    return schedule.boundaries[1:], np.cumsum(per_block)
