"""Exact learning curves for the Gaussian mean with known variance.

With ``y ~ N(theta, sigma^2)``:

* the plug-in (sample-mean) predictor has expected excess ``1 / (2 m)`` after ``m``
  samples, whatever ``sigma`` and ``theta``;
* the Bayes predictive under ``theta ~ N(prior_mean, prior_var)`` has an exact
  Kullback-Leibler excess given below, which tends to ``1 / (2 m)`` as well: both learners
  have coefficient ``kappa = lambda = 1/2``.

Summing either curve over a block schedule and fitting against ``ln n`` reproduces the
batching table of the plan (Appendix "The batching factor").
"""

from __future__ import annotations

import numpy as np

from bitsev.exact.batching import BatchingRow, batching_rows


def plugin_excess(m: np.ndarray) -> np.ndarray:
    """Expected KL from ``N(theta, s^2)`` to ``N(sample mean of m draws, s^2)``."""
    return 0.5 / np.asarray(m, dtype=float)


def bayes_excess(
    m: np.ndarray,
    theta: float,
    sigma: float = 1.0,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
) -> np.ndarray:
    """Expected KL from ``N(theta, s^2)`` to the Bayes predictive after ``m`` draws.

    The posterior of the mean is ``N(mu_m, v_m)`` with ``v_m = 1 / (1/tau^2 + m/s^2)`` and
    ``mu_m = v_m (mu_0 / tau^2 + sum y / s^2)``; the predictive is ``N(mu_m, s^2 + v_m)``.
    Averaging ``KL(N(theta, s^2) || N(mu_m, s^2 + v_m))`` over the data uses
    ``E[(theta - mu_m)^2] = (theta - E mu_m)^2 + Var(mu_m)`` with ``Var(mu_m) = v_m^2 m / s^2``.
    """
    m = np.asarray(m, dtype=float)
    s2, tau2 = sigma**2, prior_var
    v = 1.0 / (1.0 / tau2 + m / s2)
    mean_bias = theta - v * (prior_mean / tau2 + m * theta / s2)
    sq_error = mean_bias**2 + v**2 * m / s2
    w = s2 + v
    return 0.5 * (np.log(w / s2) + (s2 + sq_error) / w - 1.0)


def batching_table(
    n: int = 2**22,
    fit_range: tuple[int, int] = (2**8, 2**22),
    theta: float = 0.3,
    growths_plugin: tuple[float, ...] = (1.0, 1.5, 2.0, 4.0),
    growths_bayes: tuple[float, ...] = (1.0, 2.0),
) -> list[BatchingRow]:
    """Fitted logarithmic coefficients of the expected cumulative excess under block
    schemes, as in the plan's table (plug-in at r = 1, 1.5, 2, 4; Bayes at r = 1, 2).

    The Bayes case uses prior ``N(0, 1)`` and true mean ``theta = 0.3``. The fit is a
    least-squares line in ``ln n`` through the block ends inside ``fit_range``.
    """
    rows = batching_rows("gauss_mean", "plug-in", plugin_excess, 0.5, growths_plugin, n, fit_range)
    rows += batching_rows(
        "gauss_mean", "bayes", lambda m: bayes_excess(m, theta), 0.5, growths_bayes, n, fit_range
    )
    return rows
