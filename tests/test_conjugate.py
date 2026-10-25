"""Chain rule: the prequential Bayes code equals the mixture code, for every schedule."""

import numpy as np
import pytest

from bitsev.codes import schedule as S
from bitsev.codes.conjugate import (
    BayesLearner,
    BayesLinearRegression,
    BetaBernoulli,
    DirichletCategorical,
    GaussianMean,
    PlugInLearner,
)
from bitsev.codes.prequential import prequential_code


@pytest.mark.parametrize(
    "family, sampler",
    [
        (
            GaussianMean(sigma=0.7, prior_mean=0.2, prior_var=2.0),
            lambda rng, n: rng.normal(0.5, 0.7, n),
        ),
        (BetaBernoulli(a=0.5, b=2.0), lambda rng, n: rng.integers(0, 2, n)),
        (DirichletCategorical(4, alpha=[0.5, 1, 2, 3]), lambda rng, n: rng.integers(0, 4, n)),
    ],
)
def test_per_sample_bayes_code_is_marginal_likelihood(family, sampler, rng):
    y = sampler(rng, 200)
    code = prequential_code(BayesLearner(family), None, y, S.per_sample(len(y), start=0))
    assert np.isclose(code.total, -family.log_marginal_likelihood(y), rtol=0, atol=1e-9)


def test_block_schedule_equals_refit_per_block(rng):
    """The vectorised sequential path agrees with the generic refit-per-block path."""
    family = GaussianMean()
    y = rng.normal(0.3, 1.0, 300)
    learner = BayesLearner(family)
    sched = S.geometric(len(y), growth=2.0, start=1)
    fast = prequential_code(learner, None, y, sched)

    class Slow:  # same learner without the sequential fast path
        name, min_fit_size = learner.name, learner.min_fit_size
        fit = staticmethod(learner.fit)

    slow = prequential_code(Slow(), None, y, sched)
    np.testing.assert_allclose(fast.item_losses, slow.item_losses, atol=1e-12)


def test_plugin_gaussian_mean_matches_formula(rng):
    y = rng.normal(0.0, 1.0, 50)
    code = prequential_code(PlugInLearner(GaussianMean()), None, y, S.per_sample(50, start=1))
    means = np.cumsum(y)[:-1] / np.arange(1, 50)
    expected = 0.5 * np.log(2 * np.pi) + 0.5 * (y[1:] - means) ** 2
    np.testing.assert_allclose(code.item_losses, expected)


def test_plugin_requires_nonempty_prefix():
    with pytest.raises(ValueError):
        prequential_code(PlugInLearner(GaussianMean()), None, np.zeros(5), S.per_sample(5, start=0))


def test_linear_regression_chain_rule_against_direct_marginal(rng):
    d, n, sigma, tau2 = 3, 60, 0.5, 1.5
    x = rng.normal(size=(n, d))
    y = x @ rng.normal(size=d) + rng.normal(0, sigma, n)
    family = BayesLinearRegression(d, sigma=sigma, prior_var=tau2)
    code = prequential_code(BayesLearner(family), x, y, S.per_sample(n, start=0))
    cov = sigma**2 * np.eye(n) + tau2 * x @ x.T
    sign, logdet = np.linalg.slogdet(cov)
    log_ml = -0.5 * (n * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(cov, y))
    assert np.isclose(code.total, -log_ml, atol=1e-8)


def test_linear_regression_plugin_with_block_schedule_runs(rng):
    d, n = 2, 40
    x = rng.normal(size=(n, d))
    y = x @ np.array([1.0, -2.0]) + rng.normal(0, 1, n)
    code = prequential_code(
        PlugInLearner(BayesLinearRegression(d)), x, y, S.geometric(n, 2.0, start=d)
    )
    assert np.all(np.isfinite(code.item_losses)) and code.schedule.start == d


def test_code_length_helpers(rng):
    y = rng.normal(size=16)
    code = prequential_code(BayesLearner(GaussianMean()), None, y, S.geometric(16, 2.0, start=1))
    ends, dl = code.at_boundaries()
    assert ends.tolist() == [2, 4, 8, 16]
    assert np.isclose(dl[-1], code.total)
    assert np.isclose(code.block_losses().sum(), code.total)
    assert np.isclose(code.total_bits, code.total / np.log(2))


@pytest.mark.parametrize(
    "family, x, y",
    [
        (GaussianMean(prior_mean=0.4, prior_var=2.0), None, np.array([0.1, -0.3])),
        (DirichletCategorical(3), None, np.array([0, 2])),
        (BayesLinearRegression(2), np.array([[1.0, 0.0], [0.5, 0.5]]), np.array([1.0, 2.0])),
    ],
)
def test_bayes_fit_on_empty_prefix_is_prior_predictive(family, x, y):
    predictor = BayesLearner(family).fit(None if x is None else x[:0], y[:0])
    from_prior = predictor.log_prob(x, y)
    # The prior predictive equals the per-sample code of the first item.
    seq = BayesLearner(family).prequential_log_probs(x, y, S.per_sample(len(y), start=0))
    assert np.isclose(from_prior[0], seq[0])
