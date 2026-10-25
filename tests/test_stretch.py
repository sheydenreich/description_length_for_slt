import numpy as np
import torch
from torch import nn

from bitsev.alignment import maps as M
from bitsev.alignment.algorithms import HierarchicalEquality
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.network import LayeredNetwork, mlp
from bitsev.alignment.targets import make_targets
from bitsev.diagnostics import stretch as D
from bitsev.testbeds import hierarchical_equality as he


def _linear_network(d: int, k: int, seed: int = 0) -> LayeredNetwork:
    torch.manual_seed(seed)
    first = nn.Linear(d, d)
    readout = nn.Linear(d, k, bias=False)
    return LayeredNetwork([first, readout], [d, d], k)


def test_jacobian_norm_of_linear_readout_is_spectral_norm():
    net = _linear_network(6, 3)
    cache = HiddenCache.build(net, "h1", np.random.default_rng(0).normal(size=(10, 6)))
    w = net.blocks[1].weight.detach()
    expected = float(torch.linalg.matrix_norm(w, 2))
    assert abs(D.jacobian_norm(cache, cache.states[0], n_iter=60) - expected) < 1e-4
    assert abs(D.segment_sensitivity(cache, cache.states[0], cache.states[1]) - expected) < 1e-4
    assert abs(D.natural_sensitivity(cache, n_samples=5, n_iter=60) - expected) < 1e-4


def test_rotation_has_unit_inverse_stretch_and_forward_stretch_at_most_one():
    rng = np.random.default_rng(1)
    vocab = he.make_vocabulary(5, 2, rng)
    pool = he.sample(200, vocab, rng).x
    net = mlp(8, [12], n_outputs=2)
    alg = HierarchicalEquality(2)
    targets = make_targets(alg, pool, 400, "V1", 0.05, seed=0)
    triples = D.label_triples(targets, max_triples=20)
    assert len(triples) > 0
    cache = HiddenCache.build(net, "h1", pool)
    phi = M.Rotation(12, 2, init_scale=0.4)
    budgets = D.stretch_budgets(cache, phi, targets, triples, natural_sens=1.0, n_points=2)
    for b in budgets:
        assert b.forward_stretch <= 1.0 + 1e-5
        assert abs(b.inverse_stretch - 1.0) < 1e-4
    summary = D.summarise_budgets(budgets)
    assert summary["n"] == len(budgets) and 0 <= summary["share_both_correct"] <= 1


def test_chain_inequality_holds_for_linear_downstream():
    """gamma_achieved <= Lambda_seg ||h'(b,s) - h'(b,s')|| when both interventions are correct
    and the sensitivity is exact (linear downstream computation)."""
    rng = np.random.default_rng(2)
    vocab = he.make_vocabulary(5, 2, rng)
    pool = he.sample(300, vocab, rng).x
    net = _linear_network(8, 2, seed=3)
    alg = HierarchicalEquality(2)
    targets = make_targets(alg, pool, 600, "V1", 0.05, seed=1)
    triples = D.label_triples(targets, max_triples=40)
    cache = HiddenCache.build(net, "h1", pool)
    phi = M.AdditiveCoupling(8, 2, n_blocks=2, width=8, init_scale=0.5)
    for b in D.stretch_budgets(cache, phi, targets, triples, natural_sens=1.0, n_points=2):
        if b.both_correct:
            d_int = b.inverse_stretch * b.forward_stretch * b.natural_distance
            assert np.sqrt(2) * b.achieved_margin <= b.segment_sensitivity * d_int + 1e-4


def test_output_margin_matches_plan_formula():
    assert np.isclose(D.output_margin(2, 0.05), np.log(1 + 2 * 0.95 / 0.05))
    assert np.isclose(D.gamma_from_margin(1.0), np.sqrt(2))
