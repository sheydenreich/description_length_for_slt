"""Networks seen through their loci.

The Stage 1 score needs two operations from a network: read the hidden state at a locus
and run the rest of the computation from a (possibly modified) hidden state. Networks
whose computation is a chain of blocks expose both directly (:class:`LayeredNetwork`);
hook-based adapters for transformers will implement the same protocol later.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class Network(Protocol):
    loci: tuple[str, ...]
    n_outputs: int

    def hidden(self, x: Tensor, locus: str) -> Tensor:
        """Hidden state at ``locus`` for inputs ``x`` (shape ``(n, d_locus)``)."""
        ...

    def logits_from_hidden(self, h: Tensor, locus: str) -> Tensor:
        """Output logits when the state at ``locus`` is replaced by ``h``."""
        ...

    def logits(self, x: Tensor) -> Tensor: ...

    def width(self, locus: str) -> int: ...


class LayeredNetwork(nn.Module):
    """A chain of blocks; locus ``k`` is the output of block ``k`` (``"input"`` is ``k = 0``).

    ``blocks[i]`` maps the state at locus ``i`` to the state at locus ``i + 1``; the last
    block produces the logits. Loci are named ``"input", "h1", ..., "h{L-1}"``.
    """

    def __init__(self, blocks: list[nn.Module], widths: list[int], n_outputs: int):
        super().__init__()
        if len(widths) != len(blocks):
            raise ValueError("one width per locus: widths[i] is the input width of blocks[i]")
        self.blocks = nn.ModuleList(blocks)
        self.loci = ("input",) + tuple(f"h{i}" for i in range(1, len(blocks)))
        self._widths = dict(zip(self.loci, widths, strict=True))
        self.n_outputs = n_outputs

    def width(self, locus: str) -> int:
        return self._widths[locus]

    def _index(self, locus: str) -> int:
        try:
            return self.loci.index(locus)
        except ValueError as exc:
            raise KeyError(f"unknown locus {locus!r}; known: {self.loci}") from exc

    def hidden(self, x: Tensor, locus: str) -> Tensor:
        h = x
        for block in self.blocks[: self._index(locus)]:
            h = block(h)
        return h

    def logits_from_hidden(self, h: Tensor, locus: str) -> Tensor:
        for block in self.blocks[self._index(locus) :]:
            h = block(h)
        return h

    def logits(self, x: Tensor) -> Tensor:
        return self.logits_from_hidden(x, "input")

    def forward(self, x: Tensor) -> Tensor:
        return self.logits(x)


def mlp(
    input_dim: int, hidden_dims: list[int], n_outputs: int, activation: type[nn.Module] = nn.ReLU
) -> LayeredNetwork:
    """Fully connected network with one locus after every hidden activation."""
    dims = [input_dim, *hidden_dims]
    blocks: list[nn.Module] = [
        nn.Sequential(nn.Linear(dims[i], dims[i + 1]), activation())
        for i in range(len(hidden_dims))
    ]
    blocks.append(nn.Linear(dims[-1], n_outputs))
    return LayeredNetwork(blocks, dims, n_outputs)


def reinitialised_twin(network, seed: int):
    """Same architecture, fresh random weights: the untrained-network control.

    Networks that know their own initialisation implement ``reinitialised(seed)``; linear
    layers of a :class:`LayeredNetwork` are redrawn with the PyTorch default bounds.
    """
    import copy

    if hasattr(network, "reinitialised"):
        return network.reinitialised(seed)
    twin = copy.deepcopy(network)
    generator = torch.Generator().manual_seed(seed)
    for module in twin.modules():
        if isinstance(module, nn.Linear):
            bound = 1.0 / module.in_features**0.5
            with torch.no_grad():
                module.weight.uniform_(-bound, bound, generator=generator)
                if module.bias is not None:
                    module.bias.uniform_(-bound, bound, generator=generator)
    return twin
