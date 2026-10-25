"""Invertible alignment maps.

An alignment map ``phi`` sends the hidden state ``h`` at a locus to coordinates
``z = (z_V, z_perp)`` whose first ``d_v`` entries are aligned with the variable ``V``. All
families here are bijections of ``R^d``, so interchange interventions are well defined, and
each reports an upper bound on its distortion ``Lip(phi) Lip(phi^-1)`` for the metric
diagnostics of the plan's Appendix D.

* :class:`Rotation`: ``z = h Q`` with ``Q`` orthogonal (distributed alignment search).
* :class:`AdditiveCoupling`: RevNet-style blocks ``x2 <- x2 + f(x1)`` with alternating
  halves, optionally followed by a rotation. Expressive enough to manufacture alignments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class AlignmentMap(nn.Module, ABC):
    """Bijection of ``R^d`` whose first ``d_v`` output coordinates are the aligned subspace."""

    def __init__(self, d: int, d_v: int):
        super().__init__()
        if not 0 < d_v <= d:
            raise ValueError("need 0 < d_v <= d")
        self.d, self.d_v = d, d_v

    @abstractmethod
    def forward(self, h: Tensor) -> Tensor: ...

    @abstractmethod
    def inverse(self, z: Tensor) -> Tensor: ...

    @abstractmethod
    def lipschitz_bounds(self) -> tuple[float, float]:
        """Upper bounds on ``Lip(phi)`` and ``Lip(phi^-1)``."""

    def distortion_bound(self) -> float:
        a, b = self.lipschitz_bounds()
        return a * b

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def scale_logits(self, logits: Tensor) -> Tensor:
        """The readout's logits as this map's predictor uses them (unchanged; see
        :class:`Tempered`)."""
        return logits

    def interchange(self, h_base: Tensor, h_source: Tensor) -> Tensor:
        """``phi^-1(z_V(source), z_perp(base))``: the intervened hidden state."""
        z_base, z_source = self.forward(h_base), self.forward(h_source)
        z = torch.cat([z_source[:, : self.d_v], z_base[:, self.d_v :]], dim=1)
        return self.inverse(z)

    def interchange_complement(self, h_base: Tensor, h_source: Tensor) -> Tensor:
        """``phi^-1(z_V(base), z_perp(source))``: an intervention on the unmapped coordinates.

        The abstraction predicts that this leaves the output unchanged, because no
        high-level variable was touched; a violation is a faithfulness failure on
        unmapped variables (Meloux et al.).
        """
        z_base, z_source = self.forward(h_base), self.forward(h_source)
        z = torch.cat([z_base[:, : self.d_v], z_source[:, self.d_v :]], dim=1)
        return self.inverse(z)


class Rotation(AlignmentMap):
    """Orthogonal map ``Q = exp(S)`` with ``S`` skew-symmetric, parametrised minimally.

    The parameter vector ``theta`` has ``d (d - 1) / 2`` entries, the strictly lower
    triangle of ``S``; this is the parameter count the Bayes codes see. Rotations inside
    the aligned subspace or inside its complement leave interchange interventions
    unchanged, so the family is singular with respect to the score: the optimal set is a
    Grassmannian of dimension ``d_v (d - d_v)``.
    """

    def __init__(self, d: int, d_v: int, init_scale: float = 0.0):
        super().__init__(d, d_v)
        rows, cols = torch.tril_indices(d, d, offset=-1)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.theta = nn.Parameter(init_scale * torch.randn(len(rows)))

    def skew(self) -> Tensor:
        s = torch.zeros(self.d, self.d, dtype=self.theta.dtype, device=self.theta.device)
        s[self.rows, self.cols] = self.theta
        return s - s.T

    def matrix(self) -> Tensor:
        return torch.linalg.matrix_exp(self.skew())

    def forward(self, h):
        return h @ self.matrix()

    def inverse(self, z):
        return z @ self.matrix().T

    def lipschitz_bounds(self):
        return 1.0, 1.0

    @staticmethod
    def grassmannian_dimension(d: int, d_v: int) -> int:
        """Dimension of the set of ``d_v``-dimensional subspaces of ``R^d``."""
        return d_v * (d - d_v)


class SubspaceMap(AlignmentMap):
    """Alignment by an orthonormal frame of a ``d_v``-dimensional subspace, gauge-free.

    ``W`` (``d x d_v``) is free; ``Q = qr([W | C0])`` with a fixed random completion ``C0``
    gives an orthogonal matrix whose first ``d_v`` columns span ``W``. Predictions depend on
    ``W`` only through its column space, so the ``d_v^2`` directions ``W -> W A`` are flat
    *globally*, not only to first order as for the exponential map of a rotation. The family
    has ``d d_v`` parameters and optimal sets of dimension ``d_v^2`` at a unique optimal
    subspace, hence ``lambda = d_v (d - d_v) / 2``.
    """

    def __init__(self, d: int, d_v: int, init_scale: float = 1.0, seed: int | None = None):
        super().__init__(d, d_v)
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        self.frame = nn.Parameter(init_scale * torch.randn(d, d_v, generator=g))
        completion, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        self.register_buffer("completion", completion[:, : d - d_v])

    @torch.no_grad()
    def retract(self) -> None:
        """Re-orthonormalise the frame in place (same span, unit scale).

        The map depends on the frame only through its span, so a Gaussian prior on the raw
        frame pulls its norm to zero without changing the map, after which the smallest
        perturbation flips the subspace. Learners with a prior call this after every step so
        that the prior is effectively uniform on the Grassmannian and samples stay well
        conditioned.
        """
        q, _ = torch.linalg.qr(self.frame)
        self.frame.copy_(q)

    def matrix(self) -> Tensor:
        q, r = torch.linalg.qr(torch.cat([self.frame, self.completion], dim=1))
        # fix the sign ambiguity of QR so the map is continuous in the frame
        return q * torch.sign(torch.diagonal(r))[None, :]

    def forward(self, h):
        return h @ self.matrix()

    def inverse(self, z):
        return z @ self.matrix().T

    def lipschitz_bounds(self):
        return 1.0, 1.0


class _CouplingBlock(nn.Module):
    """``(x1, x2) -> (x1, x2 + W2 relu(W1 x1 + b))`` on a fixed split; its own inverse subtracts."""

    def __init__(self, d1: int, d2: int, width: int, init_scale: float):
        super().__init__()
        self.lin1 = nn.Linear(d1, width)
        self.lin2 = nn.Linear(width, d2, bias=False)
        with torch.no_grad():
            self.lin1.weight.mul_(init_scale)
            self.lin2.weight.mul_(init_scale)

    def shift(self, x1: Tensor) -> Tensor:
        return self.lin2(torch.relu(self.lin1(x1)))

    def lipschitz_bound(self) -> float:
        norms = torch.linalg.matrix_norm(self.lin1.weight, 2) * torch.linalg.matrix_norm(
            self.lin2.weight, 2
        )
        return float(1.0 + norms.detach())


class AdditiveCoupling(AlignmentMap):
    """``n_blocks`` additive coupling blocks with alternating halves, then an optional rotation.

    Each block and its inverse are ``(1 + ||W1|| ||W2||)``-Lipschitz, so the distortion is at
    most the product over blocks of ``(1 + ||W1|| ||W2||)^2`` (plan, Appendix D.8).
    """

    def __init__(
        self,
        d: int,
        d_v: int,
        n_blocks: int = 2,
        width: int = 32,
        init_scale: float = 0.1,
        rotate: bool = True,
    ):
        super().__init__(d, d_v)
        if d < 2:
            raise ValueError("coupling needs d >= 2")
        self.split = d // 2
        d1, d2 = self.split, d - self.split
        self.blocks = nn.ModuleList(
            _CouplingBlock(d1, d2, width, init_scale)
            if i % 2 == 0
            else _CouplingBlock(d2, d1, width, init_scale)
            for i in range(n_blocks)
        )
        self.rotation = Rotation(d, d_v) if rotate else None

    def forward(self, h):
        x1, x2 = h[:, : self.split], h[:, self.split :]
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                x2 = x2 + block.shift(x1)
            else:
                x1 = x1 + block.shift(x2)
        z = torch.cat([x1, x2], dim=1)
        return self.rotation(z) if self.rotation is not None else z

    def inverse(self, z):
        if self.rotation is not None:
            z = self.rotation.inverse(z)
        x1, x2 = z[:, : self.split], z[:, self.split :]
        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            if i % 2 == 0:
                x2 = x2 - block.shift(x1)
            else:
                x1 = x1 - block.shift(x2)
        return torch.cat([x1, x2], dim=1)

    def lipschitz_bounds(self):
        bound = 1.0
        for block in self.blocks:
            bound *= block.lipschitz_bound()
        return bound, bound


class Tempered(AlignmentMap):
    """A map plus one learned inverse temperature on the frozen readout's logits.

    The algorithm's targets are deterministic behind the noise channel, while a network may
    be less confident than the algorithm; an expressive map can sharpen the readout through
    the hidden state, an orthogonal one cannot. The shared scalar ``exp(log_scale)`` gives
    every family the same calibration freedom, so that the loss term compares causal fits
    rather than calibration. ``log_scale = 0`` is the untempered map; the prior's weight
    decay pulls towards it.
    """

    def __init__(self, inner: AlignmentMap):
        super().__init__(inner.d, inner.d_v)
        self.inner = inner
        self.log_scale = nn.Parameter(torch.zeros(()))

    def scale_logits(self, logits: Tensor) -> Tensor:
        return self.inner.scale_logits(logits) * self.log_scale.exp()

    def forward(self, h):
        return self.inner(h)

    def inverse(self, z):
        return self.inner.inverse(z)

    def lipschitz_bounds(self):
        return self.inner.lipschitz_bounds()

    def retract(self) -> None:
        retract = getattr(self.inner, "retract", None)
        if retract is not None:
            retract()


def sample_distortion(phi: AlignmentMap, h: Tensor, n_pairs: int = 2000, seed: int = 0) -> float:
    """``Dist_H(phi)`` estimated on random pairs of the given states: a lower bound on the
    sample distortion, which is itself at most the certified bound."""
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, len(h), (n_pairs,), generator=g)
    j = torch.randint(0, len(h), (n_pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    with torch.no_grad():
        z = phi(h)
        d_h = torch.linalg.norm(h[i] - h[j], dim=1)
        d_z = torch.linalg.norm(z[i] - z[j], dim=1)
    ratio = d_z / d_h
    return float(ratio.max() * (1.0 / ratio).max())
