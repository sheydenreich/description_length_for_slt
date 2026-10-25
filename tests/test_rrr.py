"""Reduced-rank regression testbed: gradients, plug-in code and the Bayes sampler."""

import numpy as np

from bitsev.codes.prequential import prequential_code
from bitsev.codes.schedule import geometric
from bitsev.testbeds import rrr


def test_energy_gradient_matches_finite_differences():
    task = rrr.RrrTask(3, 2, 1, seed=0)
    x, y = task.sample(50, seed=1)
    stats = rrr.Stats.of(x, y)
    learner = rrr.RrrBayesLearner(3, 2, 2, prior_var=0.5)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3 * 2 + 2 * 2))
    energy, grad = learner.energy_grad(w, stats)
    eps = 1e-6
    for j in range(w.shape[1]):
        dw = np.zeros_like(w)
        dw[:, j] = eps
        e_plus, _ = learner.energy_grad(w + dw, stats)
        e_minus, _ = learner.energy_grad(w - dw, stats)
        np.testing.assert_allclose(grad[:, j], (e_plus - e_minus) / (2 * eps), rtol=1e-4, atol=1e-5)


def test_sum_sq_residual_matches_direct_computation():
    task = rrr.RrrTask(2, 3, 2, seed=1)
    x, y = task.sample(40, seed=2)
    stats = rrr.Stats.of(x, y)
    a = np.random.default_rng(3).standard_normal((5, 2, 3))
    direct = np.array([np.sum((y - x @ ai.T) ** 2) for ai in a])
    np.testing.assert_allclose(stats.sum_sq_residual(a), direct, rtol=1e-10)


def test_plugin_code_regret_is_finite_and_ols_recovers_truth():
    task = rrr.RrrTask(2, 2, 2, seed=0)
    x, y = task.sample(4096, seed=5)
    code = prequential_code(rrr.OlsLearner(2, 2), x, y, geometric(4096, 2.0, start=8))
    assert np.isfinite(code.total)
    a_hat = rrr.Stats.of(x, y).ols()
    assert np.abs(a_hat - task.truth).max() < 0.1


def test_bayes_posterior_concentrates_on_truth_in_the_regular_case():
    task = rrr.RrrTask(2, 2, 2, seed=0)
    x, y = task.sample(2000, seed=6)
    settings = rrr.HmcSettings(chains=16, warmup=100, samples=50, thin=1, leapfrog=10)
    learner = rrr.RrrBayesLearner(2, 2, 2, settings=settings, seed=0)
    samples = learner.sample_posterior(rrr.Stats.of(x, y), np.random.default_rng(0))
    assert learner.last_accept > 0.3
    assert np.abs(samples.mean(axis=0) - task.truth).max() < 0.15
    # the predictive of the posterior is at least as good as the truth minus a small regret
    pred = rrr.MixturePredictor(samples)
    x2, y2 = task.sample(500, seed=7)
    assert pred.log_prob(x2, y2).mean() > task.true_log_prob(x2, y2).mean() - 0.05


def test_llc_of_regular_full_rank_model_is_near_half_the_dimension():
    task = rrr.RrrTask(2, 2, 2, seed=0)
    x, y = task.sample(4096, seed=8)
    stats = rrr.Stats.of(x, y)
    learner = rrr.RrrBayesLearner(2, 2, 2, prior_var=1.0)
    rng = np.random.default_rng(0)
    w_star = rrr.map_point(learner, stats, rng)
    lam, se = rrr.local_learning_coefficient(learner, stats, w_star, rng, steps=2000)
    # lambda = 2 (Aoyagi-Watanabe, full rank); the localised estimate is of that order
    assert 1.0 < lam < 3.5
