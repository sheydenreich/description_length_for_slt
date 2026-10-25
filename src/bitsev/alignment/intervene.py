"""Interchange interventions on cached hidden states.

For a fixed input pool the hidden states at a locus are computed once; an intervention on
a (base, source) pair then costs only the downstream computation from the locus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from bitsev.alignment.maps import AlignmentMap
from bitsev.alignment.network import Network
from bitsev.alignment.targets import NoiseChannel


@dataclass
class HiddenCache:
    """Hidden states of every pool input at one locus of one network.

    Networks whose downstream computation needs the rest of the input (a transformer with
    a position-specific locus) set ``needs_base_input = True``; the cache then keeps the
    prepared inputs and passes the base inputs of each pair to ``logits_from_hidden``.
    """

    network: Network
    locus: str
    states: Tensor  # (n_pool, d)
    inputs: Tensor | None = None

    @classmethod
    def build(
        cls, network: Network, locus: str, x_pool: np.ndarray, device: str | torch.device = "cpu"
    ) -> HiddenCache:
        prepare = getattr(network, "prepare_input", None)
        x = prepare(x_pool) if prepare else torch.as_tensor(x_pool, dtype=torch.float32)
        x = x.to(device)
        with torch.no_grad():
            states = network.hidden(x, locus)
        keep = x if getattr(network, "needs_base_input", False) else None
        return cls(network, locus, states, keep)

    @property
    def width(self) -> int:
        return self.states.shape[1]

    def pair_states(self, pairs: np.ndarray) -> tuple[Tensor, Tensor]:
        idx = torch.as_tensor(pairs, device=self.states.device)
        return self.states[idx[:, 0]], self.states[idx[:, 1]]

    def downstream(self, h_new: Tensor, pairs: np.ndarray) -> Tensor:
        """Logits from intervened states, giving the base inputs to networks that need them."""
        if self.inputs is not None:
            base = self.inputs[torch.as_tensor(pairs[:, 0], device=self.inputs.device)]
            return self.network.logits_from_hidden(h_new, self.locus, base)
        return self.network.logits_from_hidden(h_new, self.locus)


def intervened_logits(cache: HiddenCache, phi: AlignmentMap, pairs: np.ndarray) -> Tensor:
    """Logits of the network on each base input with the aligned subspace taken from the source."""
    h_base, h_source = cache.pair_states(pairs)
    return phi.scale_logits(cache.downstream(phi.interchange(h_base, h_source), pairs))


def intervened_log_probs(
    cache: HiddenCache,
    phi: AlignmentMap,
    pairs: np.ndarray,
    y: np.ndarray,
    channel: NoiseChannel | None = None,
) -> Tensor:
    """Log probability the intervened network assigns to each target label.

    With a ``channel`` the label law is passed through the declared randomized response.
    """
    logits = intervened_logits(cache, phi, pairs)
    labels = torch.as_tensor(y, device=logits.device)
    log_p = torch.log_softmax(logits, dim=1).gather(1, labels[:, None])[:, 0]
    return channel.log_prob(log_p) if channel is not None else log_p
