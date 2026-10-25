"""D2: the stretch budget of an alignment map (plan, Appendix D.3).

For a base ``b`` and two sources ``s, s'`` whose counterfactual labels differ, interchange
interventions that are both correct with output margin ``mu`` force

    gamma <= Lambda_seg * ||h'(b,s) - h'(b,s')||,    gamma = sqrt(2) mu,

where ``Lambda_seg`` is the largest Jacobian norm of the downstream computation ``g`` along
the segment between the two intervened states. Written with the forward stretch
``tau_f = ||z_V(s) - z_V(s')|| / ||h(s) - h(s')||`` and the inverse stretch
``tau_i = ||h'(b,s) - h'(b,s')|| / ||z_V(s) - z_V(s')||`` this is the budget

    log(gamma / (Lambda_nat ||h(s) - h(s')||))
        <= log tau_f + log tau_i + log(Lambda_seg / Lambda_nat).

The left side is set by the network and the targets; the right side says where an
alignment pays: forward stretch, inverse stretch, or sending intervened states to regions
where ``g`` is more sensitive than on natural states. Jacobian norms are estimated by power
iteration on Jacobian-vector products at a few points per segment; sampling underestimates
the supremum, so an unexplained budget points to sensitivity spikes between sample points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.func import jvp, vjp

from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.maps import AlignmentMap
from bitsev.alignment.targets import TargetSet


def output_margin(n_outputs: int, noise_rate: float) -> float:
    """Logit margin ``mu`` of a prediction that reproduces the randomized-response law exactly."""
    return float(np.log(1.0 + n_outputs * (1.0 - noise_rate) / noise_rate))


def gamma_from_margin(mu: float) -> float:
    return float(np.sqrt(2.0) * mu)


def jacobian_norm(cache: HiddenCache, h: Tensor, n_iter: int = 30, seed: int = 0) -> float:
    """Spectral norm of ``Dg(h)`` for one state ``h`` by power iteration on ``J^T J``."""

    def g(state: Tensor) -> Tensor:
        return cache.network.logits_from_hidden(state[None], cache.locus)[0]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(h.shape, generator=gen).to(h.device)
    v = v / torch.linalg.norm(v)
    sigma = 0.0
    for _ in range(n_iter):
        _, jv = jvp(g, (h,), (v,))
        _, pullback = vjp(g, h)
        (jtjv,) = pullback(jv)
        sigma = float(torch.linalg.norm(jtjv).sqrt().detach())
        norm = torch.linalg.norm(jtjv)
        if norm == 0:
            return 0.0
        v = jtjv / norm
    return sigma


def segment_sensitivity(
    cache: HiddenCache, h1: Tensor, h2: Tensor, n_points: int = 5, **kw
) -> float:
    """Largest sampled Jacobian norm on the segment from ``h1`` to ``h2``."""
    ts = torch.linspace(0.0, 1.0, n_points, device=h1.device)
    return max(jacobian_norm(cache, (1 - t) * h1 + t * h2, **kw) for t in ts)


def natural_sensitivity(cache: HiddenCache, n_samples: int = 64, seed: int = 0, **kw) -> float:
    """``Lambda_nat``: median Jacobian norm over natural states at the locus."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(cache.states), generator=gen)[:n_samples]
    return float(np.median([jacobian_norm(cache, cache.states[i], **kw) for i in idx]))


@dataclass
class TripleBudget:
    base: int
    source: int
    source_other: int
    natural_distance: float  # ||h(s) - h(s')||
    forward_stretch: float
    inverse_stretch: float
    segment_sensitivity: float
    natural_sensitivity: float
    gamma: float
    both_correct: bool
    achieved_margin: float  # min over the two interventions of the target-logit gap

    @property
    def network_term(self) -> float:
        """``log(gamma / (Lambda_nat ||h(s) - h(s')||))``: nonpositive means a rotation suffices."""
        return float(np.log(self.gamma / (self.natural_sensitivity * self.natural_distance)))

    @property
    def stretch_terms(self) -> tuple[float, float, float]:
        return (
            float(np.log(self.forward_stretch)),
            float(np.log(self.inverse_stretch)),
            float(np.log(self.segment_sensitivity / self.natural_sensitivity)),
        )

    @property
    def unexplained(self) -> float:
        """Budget the measured terms do not account for (positive points to unsampled spikes)."""
        return self.network_term - sum(self.stretch_terms)


def label_triples(targets: TargetSet, max_triples: int = 200, seed: int = 0) -> np.ndarray:
    """``(base, source, source')`` triples sharing a base with different clean labels."""
    rng = np.random.default_rng(seed)
    out = []
    by_base: dict[int, list[int]] = {}
    for i, b in enumerate(targets.base):
        by_base.setdefault(int(b), []).append(i)
    for b, idx in by_base.items():
        for i in idx:
            for j in idx:
                if i < j and targets.y_clean[i] != targets.y_clean[j]:
                    out.append((b, int(targets.source[i]), int(targets.source[j]), i, j))
    if not out:
        return np.empty((0, 5), dtype=int)
    arr = np.array(out, dtype=int)
    if len(arr) > max_triples:
        arr = arr[rng.choice(len(arr), max_triples, replace=False)]
    return arr


def stretch_budgets(
    cache: HiddenCache,
    phi: AlignmentMap,
    targets: TargetSet,
    triples: np.ndarray,
    natural_sens: float | None = None,
    n_points: int = 5,
) -> list[TripleBudget]:
    """Evaluate the three terms of the budget on each triple ``(b, s, s', i, j)``."""
    if natural_sens is None:
        natural_sens = natural_sensitivity(cache)
    gamma = gamma_from_margin(output_margin(targets.n_outputs, targets.noise_rate))
    out = []
    with torch.no_grad():
        for b, s, s2, i, j in triples:
            hb, hs, hs2 = cache.states[b], cache.states[s], cache.states[s2]
            z_s, z_s2 = phi(hs[None])[0], phi(hs2[None])[0]
            d_nat = float(torch.linalg.norm(hs - hs2))
            d_z = float(torch.linalg.norm(z_s[: phi.d_v] - z_s2[: phi.d_v]))
            h1 = phi.interchange(hb[None], hs[None])[0]
            h2 = phi.interchange(hb[None], hs2[None])[0]
            d_int = float(torch.linalg.norm(h1 - h2))
            logits = cache.network.logits_from_hidden(torch.stack([h1, h2]), cache.locus)
            margins = []
            for row, y in zip(logits, (targets.y_clean[i], targets.y_clean[j]), strict=True):
                others = torch.cat([row[:y], row[y + 1 :]])
                margins.append(float(row[y] - others.max()))
            # sensitivity needs autograd: outside no_grad
            seg = segment_sensitivity(cache, h1.detach(), h2.detach(), n_points=n_points)
            out.append(
                TripleBudget(
                    int(b),
                    int(s),
                    int(s2),
                    d_nat,
                    forward_stretch=d_z / d_nat if d_nat > 0 else float("inf"),
                    inverse_stretch=d_int / d_z if d_z > 0 else float("inf"),
                    segment_sensitivity=seg,
                    natural_sensitivity=natural_sens,
                    gamma=gamma,
                    both_correct=min(margins) > 0,
                    achieved_margin=min(margins),
                )
            )
    return out


def summarise_budgets(budgets: list[TripleBudget]) -> dict:
    if not budgets:
        return {"n": 0}
    net = np.array([b.network_term for b in budgets])
    terms = np.array([b.stretch_terms for b in budgets])
    return {
        "n": len(budgets),
        "share_network_term_nonpositive": float(np.mean(net <= 0)),
        "median_network_term": float(np.median(net)),
        "median_log_forward_stretch": float(np.median(terms[:, 0])),
        "median_log_inverse_stretch": float(np.median(terms[:, 1])),
        "median_log_sensitivity_ratio": float(np.median(terms[:, 2])),
        "share_both_correct": float(np.mean([b.both_correct for b in budgets])),
        "median_unexplained": float(np.median([b.unexplained for b in budgets])),
    }
