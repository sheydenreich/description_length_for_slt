import numpy as np
import pytest

from bitsev.codes import evalues as E
from bitsev.codes import schedule as S
from bitsev.codes.conjugate import BayesLearner, DirichletCategorical, GaussianMean
from bitsev.codes.prequential import prequential_code


def _bayes_factor_sequences(rng, reps, n, null, alt):
    """log E_m sequences under the null's prior predictive."""
    theta = rng.normal(null.prior_mean, np.sqrt(null.prior_var), size=reps)
    y = rng.normal(theta[:, None], null.sigma, size=(reps, n))
    out = np.empty((reps, n))
    sched = S.per_sample(n, start=0)
    for i in range(reps):
        out[i] = E.log_bayes_factor(
            prequential_code(BayesLearner(alt), None, y[i], sched),
            prequential_code(BayesLearner(null), None, y[i], sched),
        )
    return out


def test_bayes_factor_has_expectation_at_most_one_under_null(rng):
    null = GaussianMean(prior_mean=0.0, prior_var=1.0)
    alt = GaussianMean(prior_mean=0.7, prior_var=0.3)
    log_e = _bayes_factor_sequences(rng, reps=4000, n=30, null=null, alt=alt)
    for m in (1, 5, 10, 30):
        e = np.exp(log_e[:, m - 1])
        assert e.mean() < 1.0 + 3 * e.std() / np.sqrt(len(e))
        assert abs(e.mean() - 1.0) < 4 * e.std() / np.sqrt(len(e))  # exactly one in expectation


def test_bayes_factor_satisfies_ville(rng):
    null = GaussianMean(prior_mean=0.0, prior_var=1.0)
    alt = GaussianMean(prior_mean=0.7, prior_var=0.3)
    log_e = _bayes_factor_sequences(rng, reps=4000, n=40, null=null, alt=alt)
    alpha = 0.05
    crossings = np.mean([E.ever_rejects(seq, alpha) for seq in log_e])
    assert crossings <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / len(log_e))


def test_bayes_factor_grows_under_the_alternative(rng):
    null = GaussianMean(prior_mean=0.0, prior_var=0.01)  # tight null around 0
    alt = GaussianMean(prior_mean=1.0, prior_var=0.01)
    y = rng.normal(1.0, 1.0, 200)
    sched = S.per_sample(200, start=0)
    log_e = E.log_bayes_factor(
        prequential_code(BayesLearner(alt), None, y, sched),
        prequential_code(BayesLearner(null), None, y, sched),
    )
    assert E.rejects(log_e[-1], 1e-3) and E.evidence_bits(log_e[-1]) > 10


def test_weighted_average_of_evalues():
    log_e = np.log([4.0, 0.25])
    assert np.isclose(np.exp(E.weighted_average(log_e)), 2.125)
    with pytest.raises(ValueError):
        E.weighted_average(log_e, weights=np.array([0.7, 0.7]))


def test_certified_bound_never_exceeds_true_minimum(rng):
    x = rng.normal(size=(60, 3))
    y = (rng.random(60) < 1 / (1 + np.exp(-x @ np.array([1.0, -0.5, 0.2])))).astype(float)
    radius = 1.5
    # Loose starting point: the bound must hold from any w, and is tight after optimisation.
    w0 = np.array([0.3, 0.3, 0.3])
    v0, g0 = E.logistic_nll(w0, x, y)
    loose = E.certified_min_on_ball(v0, g0, w0, radius)
    opt_value, tight = E.certified_logistic_min_nll(x, y, radius)
    assert loose <= opt_value and tight <= opt_value
    assert opt_value - tight < 1e-6
    assert loose <= tight


def test_universal_inference_is_valid_and_optimizer_bound_inflates(rng):
    """Null: logistic readouts with ||w|| <= R on fixed features; hypothesis: label-only
    Dirichlet code. Under the null E[E] <= 1 with the certified bound."""
    d, n, radius, reps = 2, 40, 2.0, 400
    log_e_cert, log_e_opt = np.empty(reps), np.empty(reps)
    sched = S.per_sample(n, start=0)
    for i in range(reps):
        x = rng.normal(size=(n, d))
        w_true = rng.normal(size=d)
        w_true *= 0.8 * radius / np.linalg.norm(w_true)
        y = (rng.random(n) < 1 / (1 + np.exp(-x @ w_true))).astype(int)
        hyp = prequential_code(BayesLearner(DirichletCategorical(2)), None, y, sched)
        bounds = E.certified_logistic_min_nll(x, y, radius, n_iter=200, tol=1e-3)
        log_e_cert[i], log_e_opt[i] = E.universal_log_evalues(hyp, bounds)
    e = np.exp(log_e_cert)
    assert e.mean() <= 1.0 + 3 * e.std() / np.sqrt(reps)
    assert np.all(log_e_opt >= log_e_cert)
