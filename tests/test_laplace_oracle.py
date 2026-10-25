import numpy as np
import pytest
import torch

from bitsev.alignment import maps as M
from bitsev.alignment.intervene import HiddenCache, intervened_log_probs
from bitsev.alignment.learners import LaplaceAlignmentLearner, LaplaceBudget
from bitsev.alignment.targets import make_targets
from bitsev.codes import schedule as S
from bitsev.codes.prequential import prequential_code
from bitsev.testbeds.oracle import make_oracle


@pytest.fixture(scope="module")
def oracle():
    return make_oracle(d=5, n_pool=800, noise_rate=0.1, seed=0)


def test_oracle_target_law_is_realizable_by_the_true_rotation(oracle):
    """A rotation selecting Q0's first coordinate reproduces P(y = V) = 1 - eps/2 exactly."""
    t = make_targets(oracle.algorithm, oracle.pool, 300, "V", 0.1, seed=1)
    cache = HiddenCache.build(oracle.network, "h1", oracle.pool)
    phi = M.Rotation(5, 1)
    q0 = oracle.network.q0
    # make Q = Q0 (a proper rotation) from its real matrix logarithm
    import scipy.linalg

    log_q = torch.as_tensor(np.real(scipy.linalg.logm(q0.numpy())), dtype=torch.float32)
    with torch.no_grad():
        phi.theta.copy_(log_q[phi.rows, phi.cols])
    torch.testing.assert_close(phi.matrix(), q0, atol=1e-4, rtol=1e-4)
    lp = intervened_log_probs(cache, phi, t.pairs, t.y_clean).detach().numpy()
    np.testing.assert_allclose(np.exp(lp), 1 - 0.1 / 2, atol=1e-5)
    assert oracle.lam == 2.0  # d_v (d - d_v) / 2 with d = 5


def test_laplace_learner_runs_and_hedges(oracle):
    t = make_targets(oracle.algorithm, oracle.pool, 96, "V", 0.1, seed=2)
    cache = HiddenCache.build(oracle.network, "h1", oracle.pool)
    budget = LaplaceBudget(steps=40, lr=3e-2, prior_precision=1.0, n_samples=8)
    learner = LaplaceAlignmentLearner(cache, lambda: M.Rotation(5, 1), budget, warm_start=True)
    code = prequential_code(learner, t.pairs, t.y, S.geometric(96, 2.0, start=8))
    assert np.all(np.isfinite(code.item_losses)) and code.learner.startswith("laplace[")
    pred = learner.fit(t.pairs, t.y)
    probs = pred.predicted_probs(t.pairs)
    assert probs.shape == (96, 2) and np.allclose(probs.sum(1), 1.0)
    assert pred.samples.shape == (8, 10)  # d (d - 1) / 2 = 10 parameters


def test_diagonal_fisher_path_runs(oracle):
    t = make_targets(oracle.algorithm, oracle.pool, 40, "V", 0.1, seed=3)
    cache = HiddenCache.build(oracle.network, "h1", oracle.pool)
    budget = LaplaceBudget(steps=10, n_samples=4, full_hessian_max_params=1)  # force the diagonal
    pred = LaplaceAlignmentLearner(cache, lambda: M.Rotation(5, 1), budget).fit(t.pairs, t.y)
    assert np.all(np.isfinite(pred.log_prob(t.pairs, t.y)))


def test_subspace_map_is_globally_gauge_invariant():
    """W -> W A for invertible A leaves interchange interventions unchanged exactly."""
    phi = M.SubspaceMap(6, 2, seed=1)
    hb, hs = torch.randn(7, 6), torch.randn(7, 6)
    before = phi.interchange(hb, hs).detach()
    with torch.no_grad():
        a = torch.tensor([[2.0, 0.3], [-0.5, 1.5]])
        phi.frame.copy_(phi.frame @ a)
    torch.testing.assert_close(phi.interchange(hb, hs), before, atol=1e-5, rtol=1e-5)


def test_h_a3_on_the_oracle_subspace_family():
    """The Bayes/MAP code of the gauge-free subspace family has log-slope lambda = (d-1)/2
    on the oracle after the batching correction (five informative Hessian directions, one
    flat radial direction; full check in PLAN.md, slope 2.75 +- 0.13 on n in [256, 4096])."""
    from bitsev.codes import schedule as S
    from bitsev.codes.gain import fit_log_slope

    d, eps, n, r = 6, 0.1, 2048, 1.25
    world = make_oracle(d, 6000, eps, seed=0)
    cache = HiddenCache.build(world.network, "h1", world.pool)
    sched = S.geometric(n, r, start=16)
    curves = []
    for seed in range(4):
        t = make_targets(world.algorithm, world.pool, n, "V", eps, seed=300 + seed)
        learner = LaplaceAlignmentLearner(
            cache,
            lambda: M.SubspaceMap(d, 1, seed=0),
            LaplaceBudget(steps=100, prior_precision=1.0, n_samples=0),
            warm_start=True,
            seed=seed,
        )
        code = prequential_code(learner, t.pairs, t.y, sched)
        true_ll = np.where(t.y == t.y_clean, -np.log(1 - eps / 2), -np.log(eps / 2))[sched.start :]
        ends, _ = code.at_boundaries()
        curves.append(np.cumsum(code.item_losses - true_ll)[ends - sched.start - 1])
    fit = fit_log_slope(ends, np.mean(curves, 0), n_min=256, growth=r)
    assert world.lam == 2.5
    assert 1.6 < fit.corrected_slope < 3.6, fit


def test_sgld_sampler_keeps_the_map_quality_on_the_oracle():
    """Langevin samples from the MAP must not wreck the predictive the way a prior-wide
    Gaussian does on a saturated readout: held-out loss within 0.05 nats of the MAP's."""
    oracle = make_oracle(d=5, n_pool=800, noise_rate=0.1, seed=0)
    cache = HiddenCache.build(oracle.network, "h", oracle.pool)
    t = make_targets(oracle.algorithm, oracle.pool, 300, "V", 0.1, seed=1)
    ev = make_targets(oracle.algorithm, oracle.pool, 300, "V", 0.1, seed=4)
    from bitsev.alignment.score import faithfulness

    def fit(n_samples):
        learner = LaplaceAlignmentLearner(
            cache,
            lambda: M.SubspaceMap(5, 1, seed=0),
            LaplaceBudget(steps=100, prior_precision=0.1, n_samples=n_samples, sampler="sgld"),
            warm_start=False,
            seed=0,
            channel=t.channel,
        )
        return faithfulness(learner.fit(t.pairs, t.y), ev)

    map_only, sgld = fit(0), fit(8)
    assert sgld.log_loss < map_only.log_loss + 0.05, (map_only, sgld)
    assert sgld.iia > map_only.iia - 0.05
