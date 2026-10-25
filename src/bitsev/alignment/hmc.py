"""Bayes code of an alignment map by Hamiltonian Monte Carlo (small networks only).

The posterior over the map's parameters given the coded prefix is sampled with a Gaussian
prior; the predictor for the next block is the average of the intervened network's
predictive over the samples (:class:`~bitsev.alignment.learners.LaplacePredictor`). Unlike
the Laplace and Langevin learners this is a genuine Bayes code up to Monte Carlo error, so
for a Bayes predictive its regret slope estimates the learning coefficient itself rather
than a learner coefficient. Full-batch gradients through the network make it affordable on
the oracle and on toy networks, not on language models.

For the gauge-free :class:`~bitsev.alignment.maps.SubspaceMap` the likelihood depends on
the frame only through its span, so a Gaussian prior on the raw frame is a proper prior
whose marginal on the Grassmannian is uniform; no retraction is needed or used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.func import functional_call

from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import LaplacePredictor, _Interchange, _unflatten
from bitsev.alignment.maps import AlignmentMap
from bitsev.alignment.targets import NoiseChannel


@dataclass
class HmcBudget:
    chains: int = 4
    warmup: int = 200
    samples: int = 100
    thin: int = 2
    leapfrog: int = 10
    step: float = 1e-2
    target_accept: float = 0.7
    init_steps: int = 200  # Adam steps towards the mode before the first chain starts
    init_lr: float = 1e-2


class HmcAlignmentLearner:
    """Prequential Bayes code of a map family, posterior sampled by HMC.

    ``warm_start`` keeps each chain's last state as the start of its next block, which is
    legitimate (the chains simply continue on the larger prefix) and shortens the burn-in.
    """

    min_fit_size = 0

    def __init__(
        self,
        cache: HiddenCache,
        map_factory: Callable[[], AlignmentMap],
        budget: HmcBudget | None = None,
        prior_var: float = 1.0,
        warm_start: bool = True,
        seed: int = 0,
        family_name: str = "subspace",
        channel: NoiseChannel | None = None,
    ):
        self.cache = cache
        self.map_factory = map_factory
        self.budget = budget or HmcBudget()
        self.prior_var = prior_var
        self.warm_start = warm_start
        self.seed = seed
        self.channel = channel
        self.name = f"hmc[{family_name},prior_var={prior_var:g},chains={self.budget.chains}]"
        self.phi = map_factory().to(cache.states.device)
        self.wrapper = _Interchange(self.phi)
        self._states: torch.Tensor | None = None  # (chains, p) last chain states
        self.last_accept = float("nan")
        self.last_step = self.budget.step

    # energy ------------------------------------------------------------------------------

    def _log_lik(self, vector: torch.Tensor, pairs: np.ndarray, y: np.ndarray) -> torch.Tensor:
        h_base, h_source = self.cache.pair_states(pairs)
        h_new = functional_call(self.wrapper, _unflatten(self.phi, vector), (h_base, h_source))
        logits = self.cache.downstream(h_new, pairs)
        labels = torch.as_tensor(y, device=logits.device)
        log_p = torch.log_softmax(logits, dim=1).gather(1, labels[:, None])[:, 0]
        if self.channel is not None:
            log_p = self.channel.log_prob(log_p)
        return log_p.sum()

    def energy_grad(
        self, vector: torch.Tensor, pairs: np.ndarray, y: np.ndarray
    ) -> tuple[float, torch.Tensor]:
        v = vector.detach().requires_grad_(True)
        energy = 0.5 * (v * v).sum() / self.prior_var
        if len(y) > 0:
            energy = energy - self._log_lik(v, pairs, y)
        (grad,) = torch.autograd.grad(energy, v)
        return float(energy.detach()), grad.detach()

    # sampling ----------------------------------------------------------------------------

    def _initial_states(self, pairs, y, gen: torch.Generator) -> torch.Tensor:
        p = sum(q.numel() for q in self.phi.parameters() if q.requires_grad)
        device = self.cache.states.device
        if self.warm_start and self._states is not None:
            return self._states.clone()
        states = torch.randn(self.budget.chains, p, generator=gen).to(device) * np.sqrt(
            self.prior_var
        )
        if len(y) == 0 or self.budget.init_steps == 0:
            return states
        # a few Adam steps on the energy so the chains start near the mode
        out = []
        for s in states:
            v = s.clone().requires_grad_(True)
            opt = torch.optim.Adam([v], lr=self.budget.init_lr)
            for _ in range(self.budget.init_steps):
                opt.zero_grad()
                e = 0.5 * (v * v).sum() / self.prior_var - self._log_lik(v, pairs, y)
                e.backward()
                opt.step()
            out.append(v.detach())
        return torch.stack(out)

    def sample_posterior(self, pairs: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """Posterior samples ``(S, p)``; sets ``last_accept`` and ``last_step``."""
        b = self.budget
        gen = torch.Generator().manual_seed(self.seed * 7919 + len(y))
        device = self.cache.states.device
        states = self._initial_states(pairs, y, gen)
        step = self.last_step if self.warm_start else b.step
        kept, accepts, final = [], [], []
        for c in range(b.chains):
            q = states[c].clone()
            energy, grad = self.energy_grad(q, pairs, y)
            for it in range(b.warmup + b.samples * b.thin):
                p = torch.randn(q.shape, generator=gen).to(device)
                h0 = energy + 0.5 * float((p * p).sum())
                q_new, p_new = q.clone(), p - 0.5 * step * grad
                for leap in range(b.leapfrog):
                    q_new = q_new + step * p_new
                    e_new, g_new = self.energy_grad(q_new, pairs, y)
                    p_new = p_new - (step if leap < b.leapfrog - 1 else 0.5 * step) * g_new
                h1 = e_new + 0.5 * float((p_new * p_new).sum())
                accept = np.isfinite(h1) and np.log(torch.rand(1, generator=gen).item()) < h0 - h1
                if accept:
                    q, energy, grad = q_new, e_new, g_new
                if it < b.warmup:
                    step *= 1.05 if accept else 0.9
                else:
                    accepts.append(float(accept))
                    if (it - b.warmup) % b.thin == 0:
                        kept.append(q.clone())
            final.append(q.clone())
        self._states = torch.stack(final)
        self.last_accept = float(np.mean(accepts))
        self.last_step = step
        return torch.stack(kept)

    def fit(self, pairs: np.ndarray, y: np.ndarray) -> LaplacePredictor:
        samples = self.sample_posterior(pairs, y)
        return LaplacePredictor(self.cache, self.phi, samples, self.channel)
