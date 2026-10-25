import numpy as np
import torch

from bitsev.alignment.maps import SubspaceMap
from bitsev.testbeds.oracle import make_oracle


def _kl_along(world, j: int, phis, n_pairs: int = 8000) -> np.ndarray:
    """Mean KL from the true counterfactual law to the interchange along the V axis tilted by
    ``phi`` towards ``Q0`` coordinate ``j``."""
    net, h, q0 = world.network, torch.as_tensor(world.pool), world.network.q0
    rng = np.random.default_rng(1)
    b = rng.integers(0, len(h), n_pairs)
    s = (b + rng.integers(1, len(h), n_pairs)) % len(h)
    out = []
    for phi in phis:
        phi_map = SubspaceMap(net.d, 1)
        with torch.no_grad():
            phi_map.frame.copy_((np.cos(phi) * q0[:, 0] + np.sin(phi) * q0[:, j])[:, None])
            lp = torch.log_softmax(net.logits_from_hidden(phi_map.interchange(h[b], h[s]), "h1"), 1)
            lt = torch.log_softmax(net.logits_from_hidden(h[s], "h1"), 1)
        out.append(float((lt.exp() * (lt - lp)).sum(1).mean()))
    return np.array(out)


def test_unreadable_directions_are_quartic_and_readable_ones_quadratic():
    world = make_oracle(12, 4000, 0.1, seed=0, k_relevant=1)
    phis = [0.04, 0.08, 0.16]
    readable = _kl_along(world, 1, phis)  # Q0 coordinate 1 carries data (k = 1)
    unreadable = _kl_along(world, 5, phis)  # no data along it
    assert np.allclose(readable[1:] / readable[:-1], 4.0, rtol=0.15)
    assert np.allclose(unreadable[1:] / unreadable[:-1], 16.0, rtol=0.15)
    assert world.lam == 0.5 and world.lam_bayes == 0.5 + 10 / 4
