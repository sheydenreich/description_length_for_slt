import numpy as np
import torch

from bitsev.alignment import maps as M
from bitsev.alignment.algorithms import HierarchicalEquality
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.network import mlp
from bitsev.alignment.targets import TargetSet, make_targets
from bitsev.baselines.cae import causal_abstraction_error
from bitsev.testbeds import hierarchical_equality as he


def test_complement_interchange_is_the_other_half():
    phi = M.AdditiveCoupling(6, 2, n_blocks=2, width=4)
    hb, hs = torch.randn(5, 6), torch.randn(5, 6)
    zb, zs = phi(hb), phi(hs)
    z = phi(phi.interchange_complement(hb, hs))
    torch.testing.assert_close(z[:, :2], zb[:, :2], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(z[:, 2:], zs[:, 2:], atol=1e-5, rtol=1e-5)


def test_cae_runs_and_complement_swap_is_the_reversed_mapped_interchange(rng):
    vocab = he.make_vocabulary(5, 2, rng)
    pool = he.sample(200, vocab, rng).x
    net = mlp(8, [10], n_outputs=2)
    alg = HierarchicalEquality(2)
    targets = make_targets(alg, pool, 150, "V1", 0.0, seed=0)
    cache = HiddenCache.build(net, "h1", pool)
    phi = M.AdditiveCoupling(10, 3, n_blocks=2, width=4)
    cae = causal_abstraction_error(cache, phi, alg, pool, targets)
    rec = cae.record()
    for kind in ("mapped", "complement", "random_direction"):
        assert 0 <= rec[kind]["disagreement"] <= 1 and rec[kind]["n"] == 150
    assert np.isclose(
        rec["error"], 0.5 * (rec["mapped"]["disagreement"] + rec["complement"]["disagreement"])
    )
    # the complement swap on (b, s) is the mapped interchange on (s, b), scored against
    # A(s; V1 <- V1(b)): its error is the mapped error on the reversed targets
    y_rev = alg.counterfactual(pool[targets.source], pool[targets.base], "V1")
    reversed_targets = TargetSet(targets.source, targets.base, "V1", y_rev, y_rev, 0.0, 2, 0)
    rev = causal_abstraction_error(cache, phi, alg, pool, reversed_targets)
    assert np.isclose(cae.complement.disagreement, rev.mapped.disagreement, atol=1e-6)


def test_faithful_two_variable_network_has_zero_complement_error():
    """A hand-built network whose state is (V1, V2) coordinates: aligning V1 to its own
    coordinate is a perfect abstraction, and the complement swap must then be scored as
    a V2 swap (the old prediction A(b) gave 0.5 here)."""
    rng = np.random.default_rng(0)
    vocab = he.make_vocabulary(6, 2, rng)
    pool = he.sample(300, vocab, rng).x
    alg = HierarchicalEquality(2)

    class TwoBits(torch.nn.Module):
        loci = ("h1",)
        n_outputs = 2

        def width(self, locus):
            return 2

        def hidden(self, x, locus):
            v1, v2 = alg.variable_values(x.numpy())["V1"], alg.variable_values(x.numpy())["V2"]
            return torch.as_tensor(np.stack([v1, v2], 1), dtype=torch.float32) * 2 - 1

        def logits_from_hidden(self, h, locus):
            same = (h[:, 0] * h[:, 1]) * 10.0
            return torch.stack([-same, same], 1)

    net = TwoBits()
    targets = make_targets(alg, pool, 200, "V1", 0.0, seed=0)
    cache = HiddenCache.build(net, "h1", pool)
    cae = causal_abstraction_error(cache, M.Rotation(2, 1), alg, pool, targets)
    assert cae.mapped.disagreement == 0.0
    assert cae.complement.disagreement == 0.0
