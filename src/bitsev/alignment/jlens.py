"""The Jacobian lens as a parameter-free alignment (plan appendix, version 7.3).

For a locus ``layer l`` the J-vector of output token ``w`` is the average over prompts of
the gradient of the final logit of ``w`` with respect to the layer-``l`` state at the read
position: the row ``(W_U J_l)_w`` of the averaged Jacobian. The J-lens alignment of a
token-valued variable swaps the coordinates along the span of the J-vectors of its values
and leaves the orthogonal complement of the base untouched. As an intervention this is the
subspace interchange with a *fixed*, analytically given frame, so it has no fitted
parameters: its prequential code is the loss of one predictor on every target and its
learning coefficient is zero, the floor of the complexity ladder. At the final locus
(post-norm state, unembedding only) the J-vectors are the unembedding rows: the logit lens.

The distortion of the bijection ``phi_J(h) = (V_J^+ h, Q^T h)`` is the condition number of
the matrix of unit J-vectors (appendix identity); the interchange itself depends only on the
span and is an isometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from bitsev.alignment.hf_network import HFNetwork
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import MapPredictor
from bitsev.alignment.maps import AlignmentMap
from bitsev.alignment.targets import NoiseChannel


@dataclass
class JVectors:
    """Raw J-vectors ``(m, d)`` (rows of ``W_U J_l`` for the output tokens) and the unit
    versions used for the frame."""

    raw: Tensor
    locus: str
    n_prompts: int

    @property
    def unit(self) -> Tensor:
        return self.raw / self.raw.norm(dim=1, keepdim=True).clamp_min(1e-12)

    @property
    def condition_number(self) -> float:
        s = torch.linalg.svdvals(self.unit.T)
        return float(s[0] / s[-1].clamp_min(1e-12))

    @property
    def effective_rank(self) -> float:
        s = torch.linalg.svdvals(self.unit.T)
        p = s / s.sum()
        return float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))


def jacobian_vectors(
    network: HFNetwork, tokens: Tensor, locus: str, batch_size: int = 64
) -> JVectors:
    """Average over prompts of ``d logit_w / d h_l`` at the read position, one row per output
    token. The final locus needs no backward pass: the Jacobian is the unembedding."""
    layer = network._layer(locus)
    out_ids = network.output_token_ids
    if layer >= len(network.decoder_layers):
        w_u = network.model.get_output_embeddings().weight[out_ids].detach().float()
        return JVectors(w_u.cpu(), locus, 0)
    target = network.decoder_layers[layer]
    pos = network.position
    captured: list[Tensor] = []

    def pre_hook(module, args, kwargs):
        hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        hs = hs.detach().clone().requires_grad_(True)
        captured.append(hs)
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs
            return args, kwargs
        return (hs, *args[1:]), kwargs

    total = torch.zeros(len(out_ids), network.d_model, dtype=torch.float64)
    n = 0
    handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        for i in range(0, len(tokens), batch_size):
            captured.clear()
            batch = tokens[i : i + batch_size].to(network.device)
            logits = network.model(batch).logits[:, pos, :][:, out_ids].float()
            hs = captured[0]
            for j in range(len(out_ids)):
                (g,) = torch.autograd.grad(
                    logits[:, j].sum(), hs, retain_graph=j < len(out_ids) - 1
                )
                total[j] += g[:, pos, :].double().sum(0).cpu()
            n += len(batch)
    finally:
        handle.remove()
    return JVectors((total / n).float(), locus, n)


class FixedSubspaceMap(AlignmentMap):
    """Subspace interchange along a fixed orthonormal frame; no parameters."""

    def __init__(self, frame: Tensor, seed: int = 0):
        d, d_v = frame.shape
        super().__init__(d, d_v)
        g = torch.Generator().manual_seed(seed)
        q, r = torch.linalg.qr(
            torch.cat([frame.cpu(), torch.randn(d, d, generator=g)], dim=1)[:, :d]
        )
        q = q * torch.sign(torch.diagonal(r))[None, :]
        self.register_buffer("basis", q)

    def forward(self, h: Tensor) -> Tensor:
        return h @ self.basis.to(h.device, h.dtype)

    def inverse(self, z: Tensor) -> Tensor:
        return z @ self.basis.to(z.device, z.dtype).T

    def lipschitz_bounds(self) -> tuple[float, float]:
        return 1.0, 1.0


def jlens_map(vectors: JVectors, rank_tol: float = 1e-6) -> FixedSubspaceMap:
    """Frame of the span of the unit J-vectors (orthonormalised, numerically rank-truncated)."""
    u, s, _ = torch.linalg.svd(vectors.unit.T, full_matrices=False)
    keep = int((s > rank_tol * s[0]).sum())
    return FixedSubspaceMap(u[:, :keep])


class FixedMapLearner:
    """A learner that returns the same fixed map whatever the prefix (zero parameters)."""

    min_fit_size = 0

    def __init__(
        self, cache: HiddenCache, phi: AlignmentMap, name: str, channel: NoiseChannel | None
    ):
        self.cache, self.phi, self.name, self.channel = (
            cache,
            phi.to(cache.states.device),
            name,
            channel,
        )

    def fit(self, pairs: np.ndarray, y: np.ndarray) -> MapPredictor:
        return MapPredictor(self.cache, self.phi, self.channel)


def jacobian_consistency(
    cache: HiddenCache,
    phi: AlignmentMap,
    vectors: JVectors,
    triples: np.ndarray,
) -> np.ndarray:
    """``JC = ||delta - (W_U J) Delta|| / ||delta||`` for triples ``(b, s, s')``: the actual
    change of the output logits between two interventions on the same base against the change
    predicted by the averaged Jacobian from the change of the intervened state."""
    with torch.no_grad():
        h_b, h_s = cache.pair_states(triples[:, :2])
        _, h_s2 = cache.pair_states(triples[:, [0, 2]])
        new1, new2 = phi.interchange(h_b, h_s), phi.interchange(h_b, h_s2)
        logit1 = cache.downstream(new1, triples[:, :2])
        logit2 = cache.downstream(new2, triples[:, [0, 2]])
        delta = (logit1 - logit2).float()
        predicted = (new1 - new2).float() @ vectors.raw.to(new1.device).T
        return ((delta - predicted).norm(dim=1) / delta.norm(dim=1).clamp_min(1e-12)).cpu().numpy()
