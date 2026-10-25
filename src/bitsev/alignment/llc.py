"""Local learning coefficient of an alignment map from the loss landscape (SGLD estimator).

Lau, Murfet, Wei et al. (2023): at a fitted parameter ``theta*``, sample the localised
tempered posterior ``exp(-(n beta L_n(theta) + gamma/2 |theta - theta*|^2))`` with
``beta = beta0 / ln n`` by Langevin dynamics and estimate

    lambda_hat = n beta (E[L_n(theta)] - L_n(theta*)).

For a regular model this is half the number of directions the data constrain; flat
directions (gauge, unread by the network's readout) contribute nothing. It is the same
quantity whose ``lambda ln n`` growth the prequential code's regret measures, so the two
estimators can be cross-checked. Steps are taken in units of each tensor's scale at
``theta*`` (as in the learner's Langevin sampler); maps with a ``retract`` method are
retracted after every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from bitsev.alignment.intervene import HiddenCache, intervened_log_probs
from bitsev.alignment.maps import AlignmentMap
from bitsev.alignment.targets import NoiseChannel


@dataclass
class LLCEstimate:
    lam: float
    lam_se: float  # over chains
    beta: float
    n: int
    loss_star: float
    mean_loss: float
    chains: list[float]
    traces: list[list[tuple[int, float]]] | None = None  # (step, full-set loss) per chain


def _retract(phi):
    r = getattr(phi, "retract", None)
    if r is not None:
        r()


def local_learning_coefficient(
    cache: HiddenCache,
    phi: AlignmentMap,
    pairs: np.ndarray,
    y: np.ndarray,
    channel: NoiseChannel | None = None,
    n_chains: int = 4,
    steps: int = 1000,
    lr: float = 1e-5,
    beta0: float = 1.0,
    gamma: float = 100.0,
    batch_size: int = 512,
    eval_every: int = 10,
    seed: int = 0,
) -> LLCEstimate:
    """Estimate the local learning coefficient of ``phi`` (fitted) on the given targets.

    ``gamma`` localises the chain around ``theta*`` (in units of each tensor's scale); the
    loss is evaluated on the full target set every ``eval_every`` steps after a burn-in of
    half the chain. The estimate is the average over chains, with its standard error.
    """
    n = len(y)
    beta = beta0 / np.log(n)
    params = [p for p in phi.parameters() if p.requires_grad]
    star = [p.detach().clone() for p in params]
    scales = [
        torch.tensor(
            max(float(p.detach().std()) if p.numel() > 1 else float(p.detach().abs().mean()), 1e-3),
            device=p.device,
            dtype=p.dtype,
        )
        for p in params
    ]
    with torch.no_grad():
        loss_star = float(-intervened_log_probs(cache, phi, pairs, y, channel).mean())
    chain_means = []
    traces: list[list[tuple[int, float]]] = []
    for c in range(n_chains):
        with torch.no_grad():
            for p, s in zip(params, star, strict=True):
                p.copy_(s)
        g = torch.Generator().manual_seed(seed * 1000 + c)
        noise_gen = torch.Generator(device=params[0].device).manual_seed(seed * 1000 + c + 1)
        losses = []
        trace: list[tuple[int, float]] = []
        for step in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),), generator=g).numpy()
            mean_nll = -intervened_log_probs(cache, phi, pairs[idx], y[idx], channel).mean()
            grads = torch.autograd.grad(mean_nll, params, allow_unused=True)
            with torch.no_grad():
                for p, s, sc, gr in zip(params, star, scales, grads, strict=True):
                    drift = torch.zeros_like(p) if gr is None else n * beta * gr
                    drift = drift + gamma * (p - s) / sc**2
                    eta = lr * sc**2
                    noise = torch.randn(
                        p.shape, generator=noise_gen, device=p.device, dtype=p.dtype
                    )
                    p.add_(-eta * drift + (2 * eta).sqrt() * noise)
                _retract(phi)
                if step % eval_every == 0:
                    full = float(-intervened_log_probs(cache, phi, pairs, y, channel).mean())
                    trace.append((step, full))
                    if step >= steps // 2:
                        losses.append(full)
        chain_means.append(float(np.mean(losses)))
        traces.append(trace)
    with torch.no_grad():  # restore theta*
        for p, s in zip(params, star, strict=True):
            p.copy_(s)
    lams = [n * beta * (m - loss_star) for m in chain_means]
    se = float(np.std(lams, ddof=1) / np.sqrt(len(lams))) if len(lams) > 1 else float("nan")
    return LLCEstimate(
        float(np.mean(lams)),
        se,
        float(beta),
        n,
        loss_star,
        float(np.mean(chain_means)),
        lams,
        traces,
    )
