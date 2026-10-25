"""An oracle network whose counterfactual target law is exactly realizable by a rotation.

The hidden state at the single locus is ``h = Q0 [v ; w]`` with ``v`` the ``d_v`` coordinates
that encode the variable ``V`` (``+-1`` codes, one per bit) and ``w`` independent noise
coordinates; ``Q0`` is a fixed random rotation. The readout is ``logit = beta * v_1``
(for ``d_v = 1``) or a fixed linear readout of ``v``, with ``beta`` chosen so that
``P(y = V)`` equals the randomized-response law ``1 - eps / 2``. A rotation whose aligned
subspace is ``Q0``'s first ``d_v`` coordinates therefore reproduces the target law exactly
(the Occam regime), and the set of such rotations is a Grassmannian. When the data span
every direction (``k_relevant = d - 1``) the family is regular on it with
``lambda = d_v (d - d_v) / 2``; see ``make_oracle`` for fewer readable directions.
This is the known-``lambda`` ground for hypothesis H-A3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


class OracleNetwork(nn.Module):
    loci = ("input", "h1")
    n_outputs = 2

    def true_counterfactual_log_probs(self, h_source: Tensor, y: Tensor, channel=None) -> Tensor:
        """Log probability of the observed label under the oracle's own counterfactual law:
        the output after the true V direction is taken from the source, optionally through
        the learner's channel. This is the reference for regret when the law is not the
        fixed randomized-response formula (continuous V, sharpened readout)."""
        logits = self.logits_from_hidden(h_source, "h1")
        lp = torch.log_softmax(logits, dim=1).gather(1, y[:, None])[:, 0]
        return channel.log_prob(lp) if channel is not None else lp

    def __init__(self, d: int, d_v: int, noise_rate: float, seed: int = 0, beta_scale: float = 1.0):
        super().__init__()
        if d_v != 1:
            raise NotImplementedError("the oracle readout is defined for d_v = 1")
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        if torch.linalg.det(q) < 0:  # a proper rotation, reachable by exp of a skew matrix
            q[:, 0] = -q[:, 0]
        self.register_buffer("q0", q)
        self.d, self.d_v = d, d_v
        p_correct = 1.0 - noise_rate / 2.0  # randomized response with two classes
        # beta_scale > 1 sharpens the readout beyond the matched law (saturated regime)
        self.beta = beta_scale * 0.5 * float(np.log(p_correct / (1.0 - p_correct)))

    def width(self, locus: str) -> int:
        return self.d

    def hidden(self, x: Tensor, locus: str) -> Tensor:
        """``x`` already is the hidden state (the oracle has one locus after the input)."""
        return x

    def logits_from_hidden(self, h: Tensor, locus: str) -> Tensor:
        v1 = (h @ self.q0)[:, 0]
        return torch.stack([-self.beta * v1, self.beta * v1], dim=1)

    def logits(self, x: Tensor) -> Tensor:
        return self.logits_from_hidden(x, "h1")

    def forward(self, x):
        return self.logits(x)


class OracleAlgorithm:
    """``V`` is the sign of the first ``Q0`` coordinate; the output is ``V`` itself."""

    name = "oracle"
    variables = ("V",)
    n_outputs = 2

    def __init__(self, network: OracleNetwork):
        self.q0 = network.q0.numpy()

    def variable_values(self, x):
        return {"V": (np.asarray(x) @ self.q0)[:, 0] > 0}

    def output(self, x):
        return self.variable_values(x)["V"].astype(int)

    def counterfactual(self, base, source, variable):
        return self.variable_values(source)["V"].astype(int)


@dataclass
class OracleWorld:
    network: OracleNetwork
    algorithm: OracleAlgorithm
    pool: np.ndarray
    lam: float  # plug-in coefficient (1/2) tr(F H^+) = k / 2
    lam_bayes: float  # Bayes learning coefficient k / 2 + (d - 1 - k) / 4 (quartic directions)


def make_oracle(
    d: int,
    n_pool: int,
    noise_rate: float,
    seed: int = 0,
    k_relevant: int | None = None,
    beta_scale: float = 1.0,
    v_continuous: bool = False,
    v_margin: float = 0.0,
) -> OracleWorld:
    """``k_relevant`` (default ``d - 1``): the hidden state varies only along the V direction
    and ``k_relevant`` of the other ``Q0`` coordinates.

    Tilting the aligned direction ``u`` by an angle ``phi`` towards a readable coordinate
    changes the swapped amount at first order (``u . dh`` gains ``sin(phi) dw``), so the KL
    is quadratic there. Towards one of the ``d - 1 - k`` unreadable coordinates the data have
    no component, ``u . dh = cos(phi) dv``, and the readout sees ``v_b + cos^2(phi) dv``: the
    prediction moves at order ``phi^2`` and the KL at order ``phi^4``. Those directions are
    quartic, not flat: Fisher and population Hessian vanish there, but the Bayes code still
    pays for them. Hence two coefficients (``OracleWorld``): the plug-in's ``(1/2) tr(F H^+)
    = k / 2`` and the Bayes ``lambda = k / 2 + (d - 1 - k) / 4``, which equals the nominal
    ``(d - 1) / 2`` only when every direction is readable."""
    rng = np.random.default_rng(seed)
    net = OracleNetwork(d, 1, noise_rate, seed, beta_scale=beta_scale)
    k = d - 1 if k_relevant is None else k_relevant
    if not 1 <= k <= d - 1:
        raise ValueError("k_relevant must lie in [1, d - 1]")
    noise = np.zeros((n_pool, d - 1))
    noise[:, :k] = rng.normal(size=(n_pool, k))
    # with v_continuous the V coordinate is N(0, 1) instead of +-1, so points near the decision
    # boundary exist and a sharpened readout (beta_scale > 1) is genuinely saturated there
    if v_continuous:
        v = rng.normal(size=(n_pool, 1))
        while v_margin > 0 and (np.abs(v) < v_margin).any():  # exclude a band at the boundary
            inside = np.abs(v[:, 0]) < v_margin
            v[inside, 0] = rng.normal(size=int(inside.sum()))
    else:
        v = rng.choice([-1.0, 1.0], size=(n_pool, 1))
    codes = np.concatenate([v, noise], axis=1)
    pool = codes @ net.q0.numpy().T  # h = Q0 [v; w]
    return OracleWorld(
        net, OracleAlgorithm(net), pool.astype(np.float32), 0.5 * k, 0.5 * k + 0.25 * (d - 1 - k)
    )
