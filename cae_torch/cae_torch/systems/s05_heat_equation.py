"""System 05: Brownian particles in a box (``systems/05_heat_equation.py``).

Low-level model ``BrownianParticleSystem``: ``N`` particles on ``[0, L]``; for ``n_steps`` steps
``x <- x + N(0, step_size)`` with ``step_size = sqrt(2 D dt)``, followed by reflection at the
walls (repeated ``x -> -x`` / ``x -> 2L - x`` until inside, as in the original).

Variables: ``T_initial`` (initial positions, ``(batch, N)``) and ``T_final`` (final positions,
``(batch, N)``). Input: ``T_initial``; an intervention on ``T_initial`` takes precedence over
the input; if neither is given the positions are drawn uniformly on ``[0, L]``. If the given
positions have a particle count different from ``N`` they are resampled with replacement to
``N`` (the returned ``T_initial`` is the resampled one), as in the original.

Stochastic: the original uses the global numpy / numba RNG. Here the constructor takes
``generator: torch.Generator | None``; the tests compare the distribution of ``T_final``
(two-sample KS statistic and moments), not samples. :meth:`run_brownian_steps` accepts a
precomputed ``noise`` tensor of shape ``(n_steps, batch, N)`` for deterministic use.

Differentiable: yes, ``T_final`` w.r.t. ``T_initial`` for fixed noise (reparameterisation; the
reflections are piecewise linear with slope +-1, kinks at the walls). The resampling step for a
mismatched particle count is a gather (differentiable w.r.t. the gathered values). Nothing in
the forward pass is thresholded apart from the reflection branches.

High-level abstraction (``test/05_heat_equation.py``): explicit finite-difference heat equation
on ``n_bins = 50`` bins with Neumann boundaries (:func:`heat_equation_solve`), coarse-graining
``T_initial -> histogram`` (:func:`abstract_histogram`), grounding by sampling the histogram as a
PDF; parameters :data:`HL_PARAMS`; ``fail``: diffusion coefficient 0; ``noise``: Gaussian output
noise, std 0.15.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from cae_torch.base import TorchSystem


class BrownianParticleSystem(TorchSystem):
    """Batched torch translation of ``BrownianParticleSystem``."""

    variables = ("T_initial", "T_final")
    input_variables = ("T_initial",)

    def __init__(
        self,
        n_particles: int = 1000,
        n_steps: int = 100,
        diff_coeff: float = 0.1,
        box_len: float = 1.0,
        dt: float = 0.001,
        generator: torch.Generator | None = None,
    ):
        self.N = int(n_particles)
        self.L = float(box_len)
        self.alpha = float(diff_coeff)
        self.dt = float(dt)
        self.step_size = math.sqrt(2 * self.alpha * self.dt)
        self.n_steps = int(n_steps)
        self.generator = generator

    def reflect(self, x: Tensor) -> Tensor:
        """Reflective boundaries, iterated until every particle is inside (as the original)."""
        while bool(((x < 0) | (x > self.L)).any()):
            x = torch.where(x < 0, -x, torch.where(x > self.L, 2 * self.L - x, x))
        return x

    def run_brownian_steps(self, positions: Tensor, noise: Tensor | None = None) -> Tensor:
        """Evolve ``(batch, N)`` positions; ``noise`` is ``(n_steps, batch, N)`` standard normal
        (drawn from ``self.generator`` when ``None``)."""
        x = positions
        for t in range(self.n_steps):
            if noise is None:
                z = torch.randn(x.shape, dtype=self.dtype, generator=self.generator)
            else:
                z = noise[t]
            x = self.reflect(x + self.step_size * z)
        return x

    def forward_with_interventions(
        self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        if "T_initial" in interventions:
            positions = torch.as_tensor(interventions["T_initial"]).to(self.dtype)
        elif "T_initial" in inputs:
            positions = torch.as_tensor(inputs["T_initial"]).to(self.dtype)
        else:
            batch = self.batch_size(inputs, interventions)
            positions = torch.rand((batch, self.N), dtype=self.dtype, generator=self.generator) * self.L
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        batch, current_n = positions.shape
        if current_n != self.N:
            idx = torch.randint(0, current_n, (batch, self.N), generator=self.generator)
            positions = torch.gather(positions, 1, idx)
        final = self.run_brownian_steps(positions)
        return {"T_initial": positions, "T_final": final}


# --------------------------------------------------------------------------------------
# High-level abstraction (plain data / functions)
# --------------------------------------------------------------------------------------

#: Parameters of test/05_heat_equation.py: box, diffusion coefficient, total time, steps, bins, N.
HL_PARAMS = {"L": 1.0, "ALPHA": 0.1, "T_MAX": 0.2, "STEPS": 200, "BINS": 50, "N_PART": 1000}
HL_PARAMS["DT"] = HL_PARAMS["T_MAX"] / HL_PARAMS["STEPS"]
COARSE_GRAINING = {"T_initial": ["T_initial"], "T_final": ["T_final"]}
#: Conditions: high-level diffusion coefficient per condition (``fail`` uses 0); ``noise`` adds
#: Gaussian noise of std 0.15 to the low-level outputs.
HL_DIFFUSION = {"valid": HL_PARAMS["ALPHA"], "fail": 0.0, "noise": HL_PARAMS["ALPHA"]}


def abstract_histogram(positions: Tensor, n_bins: int = 50, box_len: float = 1.0) -> Tensor:
    """``BinningValueMap.abstract``: positions ``(..., N)`` -> normalised histogram ``(..., n_bins)``."""
    x = torch.as_tensor(positions)
    flat = x.reshape(-1, x.shape[-1])
    edges = torch.linspace(0, box_len, n_bins + 1, dtype=x.dtype)
    out = torch.stack([torch.histogram(row.to(torch.float64), bins=edges.to(torch.float64)).hist
                       for row in flat]) / x.shape[-1]
    return out.reshape(*x.shape[:-1], n_bins).to(x.dtype)


def heat_equation_solve(initial_profile: Tensor, n_bins: int = 50, diff_coeff: float = 0.1,
                        box_len: float = 1.0, dt: float = 0.001, n_steps: int = 100) -> Tensor:
    """``HeatEquationSolver.solve``: explicit Euler with Neumann boundaries on ``(..., n_bins)``."""
    u = torch.as_tensor(initial_profile).to(torch.float64)
    dx = box_len / n_bins
    a = diff_coeff * dt / dx**2
    for _ in range(n_steps):
        left = torch.cat([u[..., :1], u[..., :-1]], dim=-1)
        right = torch.cat([u[..., 1:], u[..., -1:]], dim=-1)
        u = u + a * (left + right - 2 * u)
    return u
