"""System 06: 2D phonon lattice (``systems/06_heat_equation_2d.py``).

Low-level model ``PhononLatticeModel``: an ``H x W`` mass-spring lattice (unit rest spacing,
free boundaries, spring constant ``k``, mass 1) integrated with velocity Verlet for
``int(sim_time / dt)`` steps. One atom at ``(int(source_y*H), int(source_x*W))`` (clipped to
``[3, H-4] x [3, W-4]``) receives an initial kick of kinetic energy ``source_E`` in a uniformly
random direction (no kick when ``source_x < 0``). After every step each atom independently, with
probability ``p_scatter``, has its velocity rotated to a uniformly random direction (speed kept;
skipped when the speed is below 1e-9). The reported ``final_temp_map`` is the kinetic-energy map
``0.5 m |v|^2`` averaged over ``n_averages`` independent runs.

Variables: ``source_x``, ``source_y``, ``source_E`` (scalars, ``(batch, 1)``; defaults
``-1, -1, 0`` when absent) and ``final_temp_map`` (``(batch, H, W)``). Interventions take
precedence over inputs (there is no further mechanism to clamp: the three inputs are exogenous).

Stochastic (kick angle and scattering): the constructor takes ``generator``; the equivalence
test compares the deterministic dynamics exactly with ``p_scatter = 0`` by passing the kick
angles drawn by the original's RNG to :meth:`simulate`, and compares the scattering case in
distribution. Arithmetic follows the original's operation order (neighbour sum up, down, left,
right; ``vel += 0.5 * (f / m) * dt``), so the deterministic path agrees to round-off.

Differentiable: ``final_temp_map`` w.r.t. ``source_E`` (through the kick speed
``sqrt(2 E / m)``, the Verlet integration and the speed-preserving rotations). Not
differentiable w.r.t. ``source_x`` / ``source_y`` (integer truncation and clipping select the
kicked atom); these receive no gradient.

High-level abstraction (``test/06_heat_equation_2d.py``): :data:`HL_PARAMS`,
:func:`solve_heat_pde_neumann` (explicit 2D heat equation, Neumann boundaries, sub-stepping
for stability), :func:`hl_predict` (delta at the source cell, Gaussian smoothing with
``sigma = max(0.1, 0.05 * grid)``, normalised to unit mass). The diffusion coefficient
``alpha`` is calibrated at run time against the low-level model (``calibrate_alpha_opt``);
``fail`` uses ``10 * alpha``; ``noise`` adds Gaussian output noise of std 50. The value map
Gaussian-smooths the low-level map with the same ``sigma`` and normalises it.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from cae_torch.base import TorchSystem


class PhononLatticeModel(TorchSystem):
    """Batched torch translation of ``PhononLatticeModel``."""

    variables = ("source_x", "source_y", "source_E", "final_temp_map")
    input_variables = ("source_x", "source_y", "source_E")

    def __init__(
        self,
        grid_h: int = 32,
        grid_w: int = 32,
        k_spring: float = 100.0,
        p_scatter: float = 0.1,
        dt: float = 0.02,
        sim_time: float = 20.0,
        n_averages: int = 50,
        generator: torch.Generator | None = None,
    ):
        self.H, self.W = int(grid_h), int(grid_w)
        self.k = float(k_spring)
        self.p_scatter = float(p_scatter)
        self.dt = float(dt)
        self.sim_time = float(sim_time)
        self.mass = 1.0
        self.n_averages = int(n_averages)
        self.generator = generator
        rr, cc = torch.meshgrid(
            torch.arange(self.H, dtype=self.dtype), torch.arange(self.W, dtype=self.dtype), indexing="ij"
        )
        self.pos0 = torch.stack([cc, rr], dim=-1).reshape(self.H * self.W, 2)  # [c, r]

    # -- physics -------------------------------------------------------------------------
    def compute_forces(self, pos: Tensor) -> Tensor:
        """Harmonic nearest-neighbour forces; ``pos`` is ``(B, H*W, 2)``."""
        B = pos.shape[0]
        p = pos.reshape(B, self.H, self.W, 2)
        f = torch.zeros_like(p)
        zero = torch.zeros((), dtype=pos.dtype)
        # (slice of self, slice of neighbour, rest offset [c, r] - rest_neigh), in the original's order
        specs = (
            ((slice(1, None), slice(None)), (slice(None, -1), slice(None)), (0.0, 1.0)),    # up    (r-1)
            ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), (0.0, -1.0)),   # down  (r+1)
            ((slice(None), slice(1, None)), (slice(None), slice(None, -1)), (1.0, 0.0)),    # left  (c-1)
            ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), (-1.0, 0.0)),   # right (c+1)
        )
        for me, nb, off in specs:
            rest_diff = torch.tensor(off, dtype=pos.dtype)
            disp = (p[:, me[0], me[1]] - p[:, nb[0], nb[1]]) - rest_diff
            term = torch.zeros_like(f)
            term[:, me[0], me[1]] = self.k * disp
            f = f - term  # rows without this neighbour subtract an exact zero
        return f.reshape(B, self.H * self.W, 2)

    def run_scattering_steps(self, pos: Tensor, vel: Tensor, n_steps: int) -> tuple[Tensor, Tensor]:
        for _ in range(n_steps):
            forces = self.compute_forces(pos)
            vel = vel + 0.5 * (forces / self.mass) * self.dt
            pos = pos + vel * self.dt
            forces = self.compute_forces(pos)
            vel = vel + 0.5 * (forces / self.mass) * self.dt
            if self.p_scatter > 0:
                B, N = vel.shape[:2]
                hit = torch.rand((B, N), dtype=self.dtype, generator=self.generator) < self.p_scatter
                angle = torch.rand((B, N), dtype=self.dtype, generator=self.generator) * (2 * math.pi)
                # clamp keeps the gradient finite at exactly-zero speed (those atoms are not
                # rotated: speed > 1e-9 fails, so the forward value is unchanged)
                speed = torch.sqrt(torch.clamp(vel[..., 0] * vel[..., 0] + vel[..., 1] * vel[..., 1], min=1e-300))
                rotated = torch.stack([speed * torch.cos(angle), speed * torch.sin(angle)], dim=-1)
                vel = torch.where((hit & (speed > 1e-9)).unsqueeze(-1), rotated, vel)
        return pos, vel

    def measure_temperature_map(self, vel: Tensor) -> Tensor:
        ke = 0.5 * self.mass * (vel[..., 0] ** 2 + vel[..., 1] ** 2)
        return ke.reshape(vel.shape[0], self.H, self.W)

    def simulate(self, source_x: Tensor, source_y: Tensor, source_E: Tensor,
                 angles: Tensor | None = None) -> Tensor:
        """One run per batch element (``_run_single_instance``); inputs ``(B,)``. ``angles``
        are the kick directions (``(B,)``), drawn uniformly on ``[0, 2 pi)`` when ``None``."""
        B = source_x.shape[0]
        N = self.H * self.W
        pos = self.pos0.unsqueeze(0).expand(B, N, 2).clone()
        if angles is None:
            angles = torch.rand((B,), dtype=self.dtype, generator=self.generator) * (2 * math.pi)
        kick = source_x >= 0
        cx = torch.clamp(torch.trunc(source_x * self.W).long(), 3, self.W - 4)
        cy = torch.clamp(torch.trunc(source_y * self.H).long(), 3, self.H - 4)
        idx = torch.where(kick, cy * self.W + cx, torch.zeros_like(cx))
        v_mag = torch.sqrt(2 * source_E / self.mass)
        kick_vec = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * v_mag.unsqueeze(-1)
        onehot = torch.nn.functional.one_hot(idx, N).to(self.dtype) * kick.to(self.dtype).unsqueeze(-1)
        vel = torch.zeros(B, N, 2, dtype=self.dtype) + onehot.unsqueeze(-1) * kick_vec.unsqueeze(1)
        n_steps = int(self.sim_time / self.dt)
        pos, vel = self.run_scattering_steps(pos, vel, n_steps)
        return self.measure_temperature_map(vel)

    # -- interface -----------------------------------------------------------------------
    def forward_with_interventions(
        self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        B = self.batch_size(inputs, interventions)

        def get(name: str, default: float) -> Tensor:
            if name in interventions:
                v = interventions[name]
            elif name in inputs:
                v = inputs[name]
            else:
                v = torch.full((B,), default, dtype=self.dtype)
            return torch.as_tensor(v).to(self.dtype).reshape(B)

        sx, sy, se = get("source_x", -1.0), get("source_y", -1.0), get("source_E", 0.0)
        n = self.n_averages
        maps = self.simulate(sx.repeat_interleave(n), sy.repeat_interleave(n), se.repeat_interleave(n))
        maps = maps.reshape(B, n, self.H, self.W)
        accum = torch.zeros(B, self.H, self.W, dtype=self.dtype)
        for j in range(n):  # sequential accumulation, as the original
            accum = accum + maps[:, j]
        return {
            "source_x": sx.reshape(B, 1),
            "source_y": sy.reshape(B, 1),
            "source_E": se.reshape(B, 1),
            "final_temp_map": accum / n,
        }


# --------------------------------------------------------------------------------------
# High-level abstraction (plain numpy; the original's test-suite parameters)
# --------------------------------------------------------------------------------------

HL_PARAMS = {"GRID_SIZE": 16, "SIM_TIME": 5.0, "P_SCATTER": 0.2, "N_AVG": 10, "HL_DT": 0.2}
HL_PARAMS["MAP_SIGMA"] = max(0.1, HL_PARAMS["GRID_SIZE"] * 0.05)
#: Domains of the exogenous variables (RectSubspace bounds) and the output.
HL_DOMAINS = {"source_x": (0.3, 0.7), "source_y": (0.3, 0.7), "source_E": (500.0, 1000.0)}
COARSE_GRAINING = {k: [k] for k in ("source_x", "source_y", "source_E", "final_temp_map")}
#: Multiplier of the calibrated alpha per condition; ``noise`` adds Gaussian output noise, std 50.
HL_ALPHA_FACTOR = {"valid": 1.0, "fail": 10.0, "noise": 1.0}


def solve_heat_pde_neumann(grid: np.ndarray, alpha: float, dt: float, steps: int, dx: float = 1.0) -> np.ndarray:
    """Explicit 2D heat equation with Neumann boundaries (verbatim logic of the original)."""
    curr = np.array(grid, dtype=float)
    crit = alpha * dt / dx**2
    if crit > 0.2:
        sub = int(np.ceil(crit / 0.2))
        dt_sub = dt / sub
    else:
        sub, dt_sub = 1, dt
    for _ in range(steps * sub):
        curr[0, :] = curr[1, :]
        curr[-1, :] = curr[-2, :]
        curr[:, 0] = curr[:, 1]
        curr[:, -1] = curr[:, -2]
        lap = np.zeros_like(curr)
        lap[1:-1, 1:-1] = (curr[2:, 1:-1] + curr[:-2, 1:-1] + curr[1:-1, 2:] + curr[1:-1, :-2]
                           - 4 * curr[1:-1, 1:-1])
        curr += (alpha * dt_sub / dx**2) * lap
    return curr


def hl_predict(source_x: float, source_y: float, alpha: float, grid_size: int = 16,
               sim_time: float = 5.0, map_sigma: float | None = None) -> np.ndarray:
    """High-level ``final_temp_map``: smoothed delta at the source cell, diffused, normalised."""
    from scipy.ndimage import gaussian_filter  # lazy: only the high-level model needs scipy

    sigma = max(0.1, grid_size * 0.05) if map_sigma is None else map_sigma
    grid = np.zeros((grid_size, grid_size))
    grid[int(source_y * grid_size), int(source_x * grid_size)] = 1.0
    grid = gaussian_filter(grid, sigma=sigma)
    final = solve_heat_pde_neumann(grid, alpha, dt=0.2, steps=int(sim_time / 0.2))
    t = np.sum(final)
    return final / t if t > 1e-9 else final


def abstract_temperature_map(micro: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """``SpatialValueMap.abstract`` for ``final_temp_map``: Gaussian smoothing and normalisation."""
    from scipy.ndimage import gaussian_filter

    sigma = max(0.1, grid_shape[0] * 0.05)
    out = gaussian_filter(np.asarray(micro, dtype=float), sigma=sigma)
    total = out.sum()
    return out / total if total > 1e-9 else out
