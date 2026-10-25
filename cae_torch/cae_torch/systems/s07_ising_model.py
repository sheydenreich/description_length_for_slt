"""System 07: vibrating-lattice Ising model (``systems/07_ising_model.py``).

Low-level model ``MolecularDynamicsIsingModel(grid_side, vibrational_temp)``: ``grid_side^2``
atoms on a periodic square lattice (spacing ``r0``), spins initialised to +1. Each sweep
performs, if ``vibrational_temp > 1e-9``, one velocity-Verlet step of the pairwise harmonic
springs between atoms closer than ``1.5 r0`` (microcanonical; velocities seeded once from a
Maxwell-Boltzmann distribution), then ``n_atoms`` Metropolis updates at random sites with the
distance-dependent coupling ``J_eff = J0 exp(-j_decay |r - r0|)`` to the four lattice
neighbours, external field ``h`` and spin temperature ``T``
(``dE = 2 s_i (sum_j J_eff s_j + h)``, accept if ``dE < 0`` or ``u < exp(-dE/T)``; at ``T <= 1e-9``
only ``dE < 0`` flips). After ``equil_sweeps`` sweeps, ``|mean spin|`` is averaged over
``measure_sweeps`` further sweeps. ``self.params`` is the same mutable dict as in the original
(``equil_sweeps = 2000, measure_sweeps = 1000, J0 = 1, r0 = 1, j_decay = 2, mass = 1,
spring_constant = 50, dt = 0.01, k_boltzmann = 1``).

Variables: ``Temperature``, ``ExternalField`` (scalars, ``(batch, 1)``, default 0.0; an
intervention takes precedence over the input) and ``PredictedMagnetization`` (``(batch, 1)``).

Randomness. The original seeds its (numba) RNG per call from the inputs,
``seed = int(round(T, 8) * 1e8) + int(round(h + 10, 8) * 1e8)`` (common random numbers; equal
``T + h`` collide). With ``generator=None`` (default) this module does the same: each batch
element gets its own ``torch.Generator`` seeded with :func:`seed_for`, so repeated calls with the
same inputs return the same value, as in the original. Passing a ``generator`` uses it instead
(independent randomness). Streams differ from numba's, so the tests compare distributions.

Differentiable: no. The output is the mean of discrete spins and every dependence on ``T``,
``h`` and the atom positions passes through Metropolis accept/reject comparisons; outputs carry
no autograd graph.

High-level abstraction (``test/07_ising_model.py``): the rigid-lattice Metropolis Ising model
:func:`rigid_ising_magnetization` with ``J = 1`` (``valid``) or ``J = 2`` (``fail``), sweeps
``(500, 200)``, the same seed formula; ``noise`` adds Gaussian output noise of std 0.05.
Domains: :data:`HL_DOMAINS`; identity coarse-graining :data:`COARSE_GRAINING`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from cae_torch.base import TorchSystem


def seed_for(temperature: float, field: float) -> int:
    """The original's deterministic seed from the input conditions."""
    return int(round(float(temperature), 8) * 1e8) + int(round(float(field) + 10, 8) * 1e8)


def _pbc(dr: Tensor, box: float) -> Tensor:
    dr = torch.where(dr > box / 2, dr - box, dr)
    return torch.where(dr < -box / 2, dr + box, dr)


class MolecularDynamicsIsingModel(TorchSystem):
    """Batched torch translation of ``MolecularDynamicsIsingModel``."""

    variables = ("Temperature", "ExternalField", "PredictedMagnetization")
    input_variables = ("Temperature", "ExternalField")

    def __init__(self, grid_side: int = 8, vibrational_temp: float = 1.0,
                 generator: torch.Generator | None = None):
        self.grid_side = int(grid_side)
        self.vibrational_temp = float(vibrational_temp)
        self.generator = generator
        self.params = {
            "equil_sweeps": 2000, "measure_sweeps": 1000,
            "J0": 1.0, "r0": 1.0, "j_decay": 2.0,
            "mass": 1.0, "spring_constant": 50.0, "dt": 0.01, "k_boltzmann": 1.0,
        }
        L = self.grid_side
        i = torch.arange(L * L)
        row, col = i // L, i % L
        self._neighbors = torch.stack([
            ((row + 1) % L) * L + col,
            ((row - 1 + L) % L) * L + col,
            row * L + ((col + 1) % L),
            row * L + ((col - 1 + L) % L),
        ], dim=1)  # (n_atoms, 4)

    # -- physics -------------------------------------------------------------------------
    def compute_forces(self, positions: Tensor) -> Tensor:
        """Pairwise harmonic springs between atoms closer than ``1.5 r0`` (periodic box)."""
        p = self.params
        box = self.grid_side * p["r0"]
        dr = _pbc(positions.unsqueeze(2) - positions.unsqueeze(1), box)  # (B, n, n, 2): i - j
        dist = torch.sqrt((dr * dr).sum(-1))
        n = positions.shape[1]
        mask = (dist < 1.5 * p["r0"]) & ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
        mag = -p["spring_constant"] * (dist - p["r0"])
        f = (mag / (dist + 1e-9)).unsqueeze(-1) * dr
        return torch.where(mask.unsqueeze(-1), f, torch.zeros_like(f)).sum(2)

    def _md_step(self, positions: Tensor, velocities: Tensor) -> tuple[Tensor, Tensor]:
        p = self.params
        forces = self.compute_forces(positions)
        positions = positions + velocities * p["dt"] + 0.5 * (forces / p["mass"]) * p["dt"] ** 2
        new_forces = self.compute_forces(positions)
        velocities = velocities + 0.5 * ((forces + new_forces) / p["mass"]) * p["dt"]
        return positions, velocities

    def _mc_sweep(self, positions: Tensor, spins: Tensor, temp: Tensor, field: Tensor,
                  sites: Tensor, uniforms: Tensor) -> Tensor:
        """One Metropolis sweep; ``sites`` / ``uniforms`` are ``(B, n_atoms)`` pre-drawn."""
        p = self.params
        B = spins.shape[0]
        box = self.grid_side * p["r0"]
        ar = torch.arange(B)
        hot = temp > 1e-9
        safe_t = torch.where(hot, temp, torch.ones_like(temp))
        for s in range(sites.shape[1]):
            i = sites[:, s]
            nb = self._neighbors[i]  # (B, 4)
            dr = _pbc(positions[ar, i].unsqueeze(1) - positions[ar.unsqueeze(1), nb], box)
            dist = torch.sqrt((dr * dr).sum(-1))
            j_eff = p["J0"] * torch.exp(-p["j_decay"] * torch.abs(dist - p["r0"]))
            e_nb = (-j_eff * spins[ar.unsqueeze(1), nb]).sum(1)
            de = 2 * spins[ar, i] * (-e_nb + field)
            accept = (de < 0) | (hot & (uniforms[:, s] < torch.exp(-de / safe_t)))
            spins = spins.index_put((ar, i), torch.where(accept, -spins[ar, i], spins[ar, i]))
        return spins

    def run(self, temp: Tensor, field: Tensor, generators: list[torch.Generator | None]) -> Tensor:
        """``_run_md_mc_simulation`` for a batch; ``temp``/``field`` are ``(B,)``."""
        p = self.params
        B, L = temp.shape[0], self.grid_side
        n = L * L
        n_sweeps = p["equil_sweeps"] + p["measure_sweeps"]
        spins = torch.ones(B, n, dtype=self.dtype)
        i = torch.arange(n, dtype=self.dtype)
        positions = torch.stack([(i % L) * p["r0"], torch.div(i, L, rounding_mode="floor") * p["r0"]], 1)
        positions = positions.unsqueeze(0).repeat(B, 1, 1)
        vibrate = self.vibrational_temp > 1e-9
        sites = torch.stack([torch.randint(0, n, (n_sweeps, n), generator=g) for g in generators])
        uniforms = torch.stack([torch.rand((n_sweeps, n), dtype=self.dtype, generator=g) for g in generators])
        velocities = torch.zeros_like(positions)
        if vibrate:
            scale = math.sqrt(p["k_boltzmann"] * self.vibrational_temp / p["mass"])
            velocities = torch.stack([torch.randn((n, 2), dtype=self.dtype, generator=g) for g in generators]) * scale
        mags = []
        for sweep in range(n_sweeps):
            if vibrate:
                positions, velocities = self._md_step(positions, velocities)
            spins = self._mc_sweep(positions, spins, temp, field, sites[:, sweep], uniforms[:, sweep])
            if sweep >= p["equil_sweeps"]:
                mags.append(spins.mean(1).abs())
        return torch.stack(mags, 1).mean(1)

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

        t, h = get("Temperature", 0.0), get("ExternalField", 0.0)
        if self.generator is None:
            gens = [torch.Generator().manual_seed(seed_for(t[b].item(), h[b].item())) for b in range(B)]
        else:
            gens = [self.generator] * B
        with torch.no_grad():
            mag = self.run(t.detach(), h.detach(), gens)
        return {
            "Temperature": t.reshape(B, 1),
            "ExternalField": h.reshape(B, 1),
            "PredictedMagnetization": mag.reshape(B, 1),
        }


# --------------------------------------------------------------------------------------
# High-level abstraction (plain numpy)
# --------------------------------------------------------------------------------------

HL_PARAMS = {"GRID_SIDE": 8, "EQUIL_SWEEPS": 500, "MEASURE_SWEEPS": 200}
HL_DOMAINS = {"Temperature": (0.5, 4.0), "ExternalField": (-0.5, 0.5)}
COARSE_GRAINING = {k: [k] for k in ("Temperature", "ExternalField", "PredictedMagnetization")}
#: Rigid-lattice coupling per condition; ``noise`` adds Gaussian output noise of std 0.05.
HL_COUPLING = {"valid": 1.0, "fail": 2.0, "noise": 1.0}


def rigid_ising_magnetization(grid_size: int, temp: float, field: float, J: float,
                              equil_sweeps: int, measure_sweeps: int, seed: int) -> float:
    """``_run_rigid_simulation``: textbook Metropolis Ising on a rigid periodic lattice (numpy RNG)."""
    rs = np.random.RandomState(seed)
    spins = np.ones((grid_size, grid_size), dtype=np.int8)
    n_atoms = grid_size * grid_size

    def sweep():
        for _ in range(n_atoms):
            idx = rs.randint(n_atoms)
            x, y = idx // grid_size, idx % grid_size
            s = spins[x, y]
            nb = (spins[(x + 1) % grid_size, y] + spins[(x - 1) % grid_size, y]
                  + spins[x, (y + 1) % grid_size] + spins[x, (y - 1) % grid_size])
            de = 2 * s * (J * nb + field)
            if temp > 1e-9 and (de < 0 or rs.rand() < np.exp(-de / temp)):
                spins[x, y] = -s
            elif de < 0:
                spins[x, y] = -s

    for _ in range(equil_sweeps):
        sweep()
    mags = np.empty(measure_sweeps)
    for i in range(measure_sweeps):
        sweep()
        mags[i] = abs(spins.mean())
    return float(mags.mean())
