"""Regression against the plan's batching table (Appendix B, 'The batching factor')."""

import numpy as np
import pytest

from bitsev.codes.conjugate import BayesLearner, GaussianMean
from bitsev.exact import gaussian_mean as GM

PLAN_TABLE = {
    ("plug-in", 1.0): 0.5001,
    ("plug-in", 1.5): 0.6166,
    ("plug-in", 2.0): 0.7213,
    ("plug-in", 4.0): 1.0820,
    ("bayes", 1.0): 0.4998,
    ("bayes", 2.0): 0.7208,
}


@pytest.fixture(scope="module")
def table():
    return {(r.learner, r.growth): r for r in GM.batching_table()}


def test_batching_table_matches_plan(table):
    for key, planned in PLAN_TABLE.items():
        assert abs(table[key].fitted - planned) < 1.5e-3, (key, table[key].fitted, planned)


def test_batching_table_matches_theory(table):
    for key, row in table.items():
        assert abs(row.fitted - row.predicted) < 2e-3, (key, row.fitted, row.predicted)


def test_bayes_excess_special_case_formula():
    """Prior N(0,1), sigma = 1: the plan's closed form in terms of m and theta."""
    m = np.arange(0, 50, dtype=float)
    theta = 0.3
    expected = (
        0.5 * np.log((m + 2) / (m + 1))
        + (1 + (theta**2 + m) / (m + 1) ** 2) * (m + 1) / (2 * (m + 2))
        - 0.5
    )
    np.testing.assert_allclose(GM.bayes_excess(m, theta), expected, rtol=1e-12)


def test_bayes_excess_is_positive_and_decays_like_half_over_m():
    m = np.array([1e2, 1e3, 1e4, 1e5])
    e = GM.bayes_excess(m, theta=0.3)
    assert np.all(e > 0)
    np.testing.assert_allclose(e * m, 0.5, rtol=0.05)


def test_bayes_excess_agrees_with_monte_carlo(rng):
    """Empirical excess of the Bayes code over the true-density code at fixed m."""
    theta, m, reps = 0.3, 5, 40_000
    y = rng.normal(theta, 1.0, size=(reps, m + 1))
    fam = GaussianMean()
    learner = BayesLearner(fam)
    # posterior predictive after m samples, evaluated on the (m+1)-th
    mean, var = fam.posterior(y[:, :m].sum(axis=1), np.full(reps, m))
    pred = -0.5 * (np.log(2 * np.pi * (var + 1)) + (y[:, m] - mean) ** 2 / (var + 1))
    truth = -0.5 * (np.log(2 * np.pi) + (y[:, m] - theta) ** 2)
    empirical = np.mean(truth - pred)
    stderr = np.std(truth - pred) / np.sqrt(reps)
    assert abs(empirical - GM.bayes_excess(np.array([m]), theta)[0]) < 4 * stderr
    assert learner.min_fit_size == 0
