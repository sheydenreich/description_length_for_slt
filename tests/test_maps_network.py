import numpy as np
import pytest
import torch

from bitsev.alignment import maps as M
from bitsev.alignment.intervene import HiddenCache, intervened_log_probs, intervened_logits
from bitsev.alignment.network import mlp, reinitialised_twin

torch.manual_seed(0)


@pytest.fixture
def net():
    return mlp(6, [8, 5], n_outputs=2)


def test_layered_network_loci_compose(net):
    x = torch.randn(7, 6)
    assert net.loci == ("input", "h1", "h2")
    assert net.width("h1") == 8 and net.width("h2") == 5
    for locus in net.loci:
        torch.testing.assert_close(
            net.logits_from_hidden(net.hidden(x, locus), locus), net.logits(x)
        )
    with pytest.raises(KeyError):
        net.hidden(x, "h9")


def test_reinitialised_twin_differs_but_matches_shapes(net):
    twin = reinitialised_twin(net, seed=1)
    for p, q in zip(net.parameters(), twin.parameters(), strict=True):
        assert p.shape == q.shape and not torch.allclose(p, q)
    assert torch.equal(
        torch.cat([p.flatten() for p in reinitialised_twin(net, 1).parameters()]),
        torch.cat([p.flatten() for p in twin.parameters()]),
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: M.Rotation(6, 2, init_scale=0.5),
        lambda: M.AdditiveCoupling(6, 2, n_blocks=3, width=10, init_scale=0.5, rotate=True),
        lambda: M.AdditiveCoupling(5, 1, n_blocks=2, width=4, init_scale=1.0, rotate=False),
    ],
)
def test_maps_are_invertible_and_bounded(make):
    phi = make()
    h = torch.randn(50, phi.d)
    torch.testing.assert_close(phi.inverse(phi(h)), h, atol=1e-5, rtol=1e-5)
    lip, lip_inv = phi.lipschitz_bounds()
    assert M.sample_distortion(phi, h) <= lip * lip_inv + 1e-6
    assert phi.n_params > 0


def test_rotation_is_orthogonal():
    q = M.Rotation(5, 2, init_scale=1.0).matrix()
    torch.testing.assert_close(q @ q.T, torch.eye(5), atol=1e-5, rtol=0)


def test_interchange_with_full_subspace_copies_the_source(net):
    phi = M.Rotation(8, 8, init_scale=0.3)  # d_v = d: everything comes from the source
    x = torch.randn(20, 6)
    cache = HiddenCache.build(net, "h1", x.numpy())
    pairs = np.stack([np.arange(10), np.arange(10, 20)], axis=1)
    torch.testing.assert_close(
        intervened_logits(cache, phi, pairs), net.logits(x[10:]), atol=1e-5, rtol=1e-5
    )


def test_interchange_with_zero_width_change_keeps_the_base(net):
    """d_v = 1 with a source equal to the base leaves the logits unchanged."""
    phi = M.AdditiveCoupling(8, 1)
    x = torch.randn(10, 6)
    cache = HiddenCache.build(net, "h1", x.numpy())
    pairs = np.stack([np.arange(10), np.arange(10)], axis=1)
    torch.testing.assert_close(
        intervened_logits(cache, phi, pairs), net.logits(x), atol=1e-5, rtol=1e-5
    )
    lp = intervened_log_probs(cache, phi, pairs, np.zeros(10, dtype=int))
    assert lp.shape == (10,) and torch.all(lp <= 0)


def test_tempered_map_scales_only_the_logits(net):
    torch.manual_seed(1)
    inner = M.SubspaceMap(5, 2, seed=0)
    phi = M.Tempered(inner)
    cache = HiddenCache.build(net, "h2", np.random.default_rng(0).normal(size=(20, 6)))
    pairs = np.array([[0, 1], [2, 3], [4, 5]])
    y = np.array([0, 1, 0])
    torch.testing.assert_close(
        intervened_logits(cache, phi, pairs), intervened_logits(cache, inner, pairs)
    )
    assert phi.n_params == inner.n_params + 1 and phi.distortion_bound() == inner.distortion_bound()
    with torch.no_grad():
        phi.log_scale.fill_(np.log(3.0))
    torch.testing.assert_close(
        intervened_logits(cache, phi, pairs), 3.0 * intervened_logits(cache, inner, pairs)
    )
    assert not torch.allclose(
        intervened_log_probs(cache, phi, pairs, y), intervened_log_probs(cache, inner, pairs, y)
    )
    phi.retract()  # delegates to the subspace frame
    torch.testing.assert_close(inner.frame.T @ inner.frame, torch.eye(2), atol=1e-5, rtol=1e-5)


def test_map_learner_fits_the_temperature(net):
    from bitsev.alignment.learners import LaplaceAlignmentLearner, LaplaceBudget
    from bitsev.alignment.targets import NoiseChannel

    rng = np.random.default_rng(0)
    cache = HiddenCache.build(net, "h2", rng.normal(size=(40, 6)))
    pairs = rng.integers(0, 40, size=(200, 2))
    with torch.no_grad():
        y = intervened_logits(cache, M.SubspaceMap(5, 2, seed=0), pairs).argmax(1).numpy()
    channel = NoiseChannel(0.05, 2)
    budget = LaplaceBudget(steps=200, lr=5e-2, prior_precision=0.01, n_samples=0, optimizer="adam")
    learner = LaplaceAlignmentLearner(
        cache, lambda: M.Tempered(M.SubspaceMap(5, 2)), budget, channel=channel
    )
    pred = learner.fit(pairs, y)
    # deterministic labels behind the channel: the fitted readout sharpens
    assert pred.phi.log_scale.item() > 0.1
    assert np.isfinite(pred.log_prob(pairs, y)).all()
    sampled = LaplaceAlignmentLearner(
        cache, lambda: M.Tempered(M.SubspaceMap(5, 2)), LaplaceBudget(n_samples=4), channel=channel
    )
    with pytest.raises(NotImplementedError):
        sampled.fit(pairs, y)


def test_map_learner_steps_grow_with_the_prefix(net):
    from bitsev.alignment.learners import LaplaceAlignmentLearner, LaplaceBudget

    cache = HiddenCache.build(net, "h2", np.random.default_rng(0).normal(size=(10, 6)))
    budget = LaplaceBudget(steps=400, batch_size=256, min_epochs=10, optimizer="adam", n_samples=0)
    learner = LaplaceAlignmentLearner(cache, lambda: M.SubspaceMap(5, 2), budget)
    assert learner._adam_steps(1024) == 400  # the fixed budget is the floor
    assert learner._adam_steps(131072) == 5120  # ten passes of 512 minibatches
    assert learner._adam_steps(100) == 400 and "epochs>=10" in learner.name
    plain = LaplaceAlignmentLearner(cache, lambda: M.SubspaceMap(5, 2), LaplaceBudget(steps=400))
    assert plain._adam_steps(131072) == 400 and "epochs" not in plain.name
