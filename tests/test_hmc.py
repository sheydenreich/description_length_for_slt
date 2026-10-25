"""HMC Bayes code of a subspace map on the oracle."""

import torch

from bitsev.alignment import maps as M
from bitsev.alignment.hmc import HmcAlignmentLearner, HmcBudget
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.targets import make_targets
from bitsev.testbeds.oracle import make_oracle


def _world():
    torch.manual_seed(0)
    world = make_oracle(4, 600, 0.1, seed=0, k_relevant=3, v_continuous=True)
    cache = HiddenCache.build(world.network, "h1", world.pool)
    targets = make_targets(world.algorithm, world.pool, 300, "V", 0.1, seed=1)
    return world, cache, targets


def test_energy_gradient_matches_finite_differences():
    _, cache, t = _world()
    learner = HmcAlignmentLearner(cache, lambda: M.SubspaceMap(4, 1, seed=0), prior_var=0.7)
    v = torch.randn(4, generator=torch.Generator().manual_seed(3))
    e, g = learner.energy_grad(v, t.pairs[:50], t.y[:50])
    for j in range(4):
        dv = torch.zeros(4)
        dv[j] = 1e-2
        e_plus, _ = learner.energy_grad(v + dv, t.pairs[:50], t.y[:50])
        e_minus, _ = learner.energy_grad(v - dv, t.pairs[:50], t.y[:50])
        assert abs(float(g[j]) - (e_plus - e_minus) / 2e-2) < 2e-2 * (1 + abs(float(g[j])))


def test_posterior_predictive_is_close_to_the_true_law():
    world, cache, t = _world()
    budget = HmcBudget(chains=2, warmup=100, samples=40, thin=1, leapfrog=10)
    learner = HmcAlignmentLearner(cache, lambda: M.SubspaceMap(4, 1, seed=0), budget, seed=0)
    pred = learner.fit(t.pairs[:200], t.y[:200])
    assert 0.3 < learner.last_accept <= 1.0
    fresh = make_targets(world.algorithm, world.pool, 200, "V", 0.1, seed=2)
    with torch.no_grad():
        true_lp = world.network.true_counterfactual_log_probs(
            torch.as_tensor(world.pool[fresh.source]), torch.as_tensor(fresh.y)
        ).numpy()
    model_lp = pred.log_prob(fresh.pairs, fresh.y)
    assert model_lp.mean() > true_lp.mean() - 0.1
