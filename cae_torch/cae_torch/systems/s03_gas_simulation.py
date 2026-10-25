"""System 03: Lennard-Jones molecular dynamics (gas-law abstractions).

Original: ``systems/03_gas_simulation.py`` of MelouxM/CAE (``UniversalGasModel`` and the numba
kernels ``compute_forces``, ``apply_langevin``, ``get_p_tail_corr``, ``minimize_energy``,
``run_equilibration``, ``run_measurement``).  Reduced units (eps = sigma = m = kB = 1),
N = 128 particles, dt = 0.002, cutoff 3.0, velocity Verlet, Langevin thermostat
(gamma = 1), Berendsen-like barostat (NPT) and a proportional bath-temperature controller
(PVT), long-range tail correction of the pressure.

Variables (numpy output order): ``pressure``, ``temperature``, ``volume`` (batch, 1),
``positions``, ``velocities`` (batch, 128, 3), ``box_len`` (batch, 1).  Inputs: the
microstate ``positions``, ``velocities``, ``box_len`` (if ``positions`` is absent a jittered
lattice is drawn and energy-minimised, as in the original) and optionally ``volume``,
``temperature``, ``pressure`` (defaults 1000, 2.0, 0.1).

Semantics reproduced exactly (read from the original's ``forward_with_interventions``):

* The simulation mode is chosen from the *set of intervened names* (shared by the batch):
  {volume, temperature} -> NVT (box rescaled to the target volume, positions scaled with it
  when the box changes by more than 1e-4); {pressure, temperature} -> NPT equilibration
  (box reset to 10 if below 1.0) then NVT measurement; {pressure, volume} -> PVT
  (the bath temperature is tuned to reach the target pressure; the found temperature is
  used in the measurement); anything else -> NVT at the resolved temperature with the
  input box.  Values are taken from the interventions first, then the inputs, then the
  defaults.  A resolved value of exactly 0 falls back to the default, because the
  original writes ``resolve(...) or default``.
* Interventions on ``positions`` / ``velocities`` / ``box_len`` are *ignored* by the
  original (it only reads them from the input state); so here.
* Only ``volume``/``temperature``/``pressure`` set the targets; the outputs are the
  time-averaged measured pressure and kinetic temperature, the final volume and microstate.

Randomness: the Langevin thermostat adds Gaussian noise to every velocity component at
every step (equilibration and measurement), so the whole trajectory is stochastic.  Noise
is drawn from ``noise_source(shape)`` (default: ``torch.randn`` on the optional
``generator``); samples therefore differ from the original and only distributions can be
compared, except that the tests inject the original's exact stream (numba's ``np.random``
is bit-identical to numpy's legacy ``RandomState``) and then compare trajectories to 1e-9.

Deviations: batch size is the leading dimension of any given tensor (the original infers
it from the interventions only); the lattice initialisation is not cached across calls
(the original caches ``self.pos``); the barostat / controller clamps, the cutoff mask and
the minimum-image ``round`` make the dynamics piecewise smooth, so gradients w.r.t.
positions, velocities, volume, temperature and pressure exist almost everywhere and flow
through the whole trajectory (``minimize_energy`` is only used when no positions are given).

High-level abstractions (``test/03_gas_simulation.py``): ideal gas
``P = N T**alpha / V`` with ``alpha = 1`` (valid) or ``0.5`` (invalid), Van der Waals
``P = rho T / (1 - b rho) - a rho**2`` with (a, b) fitted on an isothermal sweep at the
Boyle temperature (``perform_full_calibration``), domains +-30 % around
``V = N / rho``, ``T`` for (rho, T) = (0.05, T_BOYLE) (valid), (0.70, T_BOYLE)
(invalid: high density) and (0.05, 1.0) (invalid: low temperature); identity
coarse-graining on volume, temperature, pressure, positions, velocities, box_len.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

from cae_torch.base import TorchSystem

F64 = torch.float64

N_PARTICLES = 128
DT = 0.002
CUTOFF = 3.0
T_BOYLE = 3.417927982

DEFAULT_VOLUME = 1000.0
DEFAULT_TEMPERATURE = 2.0
DEFAULT_PRESSURE = 0.1

# High-level abstraction data (plain Python) ------------------------------------------
LOW_RHO, HIGH_RHO = 0.05, 0.70
MED_TEMP, LOW_TEMP = T_BOYLE, 1.0
ALPHA_VALID, ALPHA_WRONG = 1.0, 0.5
ALL_MICRO = ("volume", "temperature", "pressure", "positions", "velocities", "box_len")
CONDITIONS = {   # name -> ((rho, T), alpha or None, use_vdw)
    "valid_ideal": ((LOW_RHO, MED_TEMP), ALPHA_VALID, False),
    "valid_vdw": ((LOW_RHO, MED_TEMP), None, True),
    "invalid_alpha": ((LOW_RHO, MED_TEMP), ALPHA_WRONG, False),
    "invalid_density": ((HIGH_RHO, MED_TEMP), ALPHA_VALID, False),
    "invalid_temp": ((LOW_RHO, LOW_TEMP), ALPHA_VALID, False),
}
CALIBRATION_RHOS = tuple(0.05 + i * (0.40 - 0.05) / 7 for i in range(8))   # np.linspace(0.05, 0.40, 8)


def domain(rho: float, t_center: float, width: float = 0.30) -> dict[str, tuple[float, float] | None]:
    v = N_PARTICLES / rho
    return {"volume": (v * (1 - width), v * (1 + width)),
            "temperature": (t_center * (1 - width), t_center * (1 + width)), "pressure": None}


def ideal_gas_pressure(volume, temperature, alpha: float = ALPHA_VALID, n: int = N_PARTICLES):
    return n * temperature ** alpha / volume


def vdw_pressure(volume, temperature, a: float, b: float, n: int = N_PARTICLES):
    rho = n / volume
    d = 1.0 - b * rho
    return (rho * temperature) / d - a * rho ** 2 if d > 1e-4 else 0.0


def get_ideal_laws(n: int = N_PARTICLES):
    return (lambda temperature, volume: (n * temperature) / volume,
            lambda pressure, temperature: (n * temperature) / pressure,
            lambda pressure, volume: (pressure * volume) / n)


# Batched physics kernels (box is per item, shape (batch,)) ----------------------------

def compute_forces(pos: Tensor, box: Tensor, cutoff_sq: float = CUTOFF ** 2) -> tuple[Tensor, Tensor]:
    """LJ forces (batch, N, 3) and virial sum_{i<j} f_ij . r_ij (batch,), minimum image."""
    b3 = box.reshape(-1, 1, 1, 1)
    dr = pos.unsqueeze(2) - pos.unsqueeze(1)                 # (B, N, N, 3), dr[i,j] = pos_i - pos_j
    dr = dr - b3 * torch.round(dr / b3)
    r2 = (dr ** 2).sum(-1)                                   # (B, N, N)
    n = pos.shape[1]
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    within = (r2 < cutoff_sq) & ~eye
    safe_r2 = torch.where(within, r2, torch.ones_like(r2))
    inv_r2 = 1.0 / safe_r2
    inv_r6 = inv_r2 ** 3
    f_scalar = torch.where(within, 24.0 * inv_r2 * (2.0 * inv_r6 ** 2 - inv_r6), torch.zeros_like(r2))
    f_vec = f_scalar.unsqueeze(-1) * dr                      # force on i from j
    forces = f_vec.sum(2)
    upper = torch.triu(within, diagonal=1)
    virial = ((f_vec * dr).sum(-1) * upper).sum((1, 2))
    return forces, virial


def apply_langevin(vel: Tensor, dt: float, target_t: Tensor, noise: Tensor, gamma: float = 1.0) -> Tensor:
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(1.0 - c1 ** 2) * torch.sqrt(target_t).reshape(-1, 1, 1)
    return c1 * vel + c2 * noise


def get_p_tail_corr(n: int, vol: Tensor, cutoff: float = CUTOFF) -> Tensor:
    rho = n / vol
    sig, eps = 1.0, 1.0
    term1 = (2.0 / 3.0) * (sig / cutoff) ** 9
    term2 = (sig / cutoff) ** 3
    return (16.0 / 3.0) * math.pi * (rho ** 2) * eps * (sig ** 3) * (term1 - term2)


def minimize_energy(pos: Tensor, box: Tensor, max_steps: int = 500, tol: float = 1.0) -> Tensor:
    """Steepest descent with per-item early stop (an item stops updating once max |f| < tol)."""
    cutoff_sq = CUTOFF ** 2
    dt_desc = 0.001
    active = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    for _ in range(max_steps):
        forces, _ = compute_forces(pos, box, cutoff_sq)
        max_f = torch.sqrt((forces ** 2).sum(-1)).amax(1)
        new_pos = torch.remainder(pos + forces * dt_desc, box.reshape(-1, 1, 1))
        pos = torch.where(active.reshape(-1, 1, 1), new_pos, pos)
        active = active & ~(max_f < tol)
        if not bool(active.any()):
            break
    return pos


def _kinetic_pressure(pos, vel, box, virial, n):
    curr_v = box ** 3
    curr_rho = n / curr_v
    curr_k_temp = (vel ** 2).sum((1, 2)) / (3 * n)
    return (curr_rho * curr_k_temp) + (virial / (3.0 * curr_v)) + get_p_tail_corr(n, curr_v, CUTOFF)


def run_equilibration(pos: Tensor, vel: Tensor, box: Tensor, dt: float, steps: int, mode: int,
                      target_1: Tensor, target_2: Tensor, noise_source: Callable[[tuple[int, ...]], Tensor]
                      ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mode 0 NVT (target_1 = T), 1 NPT (target_1 = T, target_2 = P), 2 PVT (target_1 = P)."""
    n = pos.shape[1]
    cutoff_sq = CUTOFF ** 2
    forces, _ = compute_forces(pos, box, cutoff_sq)
    tau_p = 200 * dt
    gamma = 1.0
    compressibility = 0.05
    smoothed_p = torch.zeros_like(box)
    bath_temp = target_1.clone() if mode in (0, 1) else torch.full_like(box, T_BOYLE)

    for step in range(steps):
        pos = pos + vel * dt + 0.5 * forces * dt ** 2
        if mode == 1 and step % 10 == 0:
            _, curr_vir = compute_forces(pos, box, cutoff_sq)
            curr_p = _kinetic_pressure(pos, vel, box, curr_vir, n)
            scale_p = (1.0 - (dt / tau_p) * (target_2 - curr_p) / compressibility) ** (1.0 / 3.0)
            scale_p = torch.clamp(scale_p, 0.995, 1.005)
            box = box * scale_p
            pos = pos * scale_p.reshape(-1, 1, 1)
        pos = torch.remainder(pos, box.reshape(-1, 1, 1))

        forces_new, virial = compute_forces(pos, box, cutoff_sq)
        vel = vel + 0.5 * (forces + forces_new) * dt
        forces = forces_new

        if mode == 2:
            curr_p = _kinetic_pressure(pos, vel, box, virial, n)
            smoothed_p = curr_p if step == 0 else 0.9 * smoothed_p + 0.1 * curr_p
            p_err = target_1 - smoothed_p
            bath_temp = torch.clamp(bath_temp * (1.0 + 0.005 * p_err), 0.1, 10.0)

        vel = apply_langevin(vel, dt, bath_temp, noise_source(tuple(vel.shape)), gamma)

    return pos, vel, box, bath_temp


def run_measurement(pos: Tensor, vel: Tensor, box: Tensor, dt: float, steps: int, temperature: Tensor,
                    noise_source: Callable[[tuple[int, ...]], Tensor]):
    n = pos.shape[1]
    cutoff_sq = CUTOFF ** 2
    vol = box ** 3
    gamma = 1.0
    p_accum = torch.zeros_like(box)
    t_accum = torch.zeros_like(box)
    forces, _ = compute_forces(pos, box, cutoff_sq)
    for _ in range(steps):
        pos = torch.remainder(pos + vel * dt + 0.5 * forces * dt ** 2, box.reshape(-1, 1, 1))
        forces_new, virial_new = compute_forces(pos, box, cutoff_sq)
        vel = vel + 0.5 * (forces + forces_new) * dt
        forces = forces_new
        vel = apply_langevin(vel, dt, temperature, noise_source(tuple(vel.shape)), gamma)

        curr_t = (vel ** 2).sum((1, 2)) / (3 * n)
        measured_p = (n / vol * curr_t) + (virial_new / (3.0 * vol))
        p_corr = get_p_tail_corr(n, vol, CUTOFF)
        p_accum = p_accum + measured_p + p_corr
        t_accum = t_accum + curr_t
    return p_accum / steps, t_accum / steps, box ** 3, pos, vel


def lattice_positions(box: Tensor, n: int = N_PARTICLES) -> Tensor:
    """Simple cubic lattice of the original ``_init_lattice`` for each box in ``box`` (batch,)."""
    k = int(math.ceil(n ** (1 / 3.0)))
    grid = torch.arange(k, dtype=F64)
    x, y, z = torch.meshgrid(grid, grid, grid, indexing="xy")      # numpy meshgrid default
    unit = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], 1)[:n]
    return unit.unsqueeze(0) * (box / k).reshape(-1, 1, 1)


# Low-level model ----------------------------------------------------------------------

class UniversalGasModel(TorchSystem):
    """Torch translation of ``UniversalGasModel(equil_steps=5000, measure_steps=5000)``.

    Extra keyword arguments (not in the original): ``generator`` for the thermostat noise
    and ``noise_source(shape) -> Tensor`` to supply the standard-normal noise explicitly
    (one call per MD step with shape (batch, N, 3), equilibration steps first).
    """

    variables = ("pressure", "temperature", "volume", "positions", "velocities", "box_len")
    input_variables = ("volume", "temperature", "pressure", "positions", "velocities", "box_len")

    def __init__(self, equil_steps: int = 5000, measure_steps: int = 5000,
                 generator: torch.Generator | None = None,
                 noise_source: Callable[[tuple[int, ...]], Tensor] | None = None):
        self.N = N_PARTICLES
        self.equil_steps = int(equil_steps)
        self.measure_steps = int(measure_steps)
        self.generator = generator
        if noise_source is None:
            def noise_source(shape):
                return torch.randn(shape, generator=self.generator, dtype=F64)
        self.noise_source = noise_source

    def _init_lattice(self, box: Tensor) -> tuple[Tensor, Tensor]:
        batch = box.shape[0]
        k = int(math.ceil(self.N ** (1 / 3.0)))
        s = (box / k).reshape(-1, 1, 1)
        pos = lattice_positions(box, self.N)
        pos = pos + (torch.rand((batch, self.N, 3), generator=self.generator, dtype=F64) - 0.5) * (s * 0.1)
        vel = torch.randn((batch, self.N, 3), generator=self.generator, dtype=F64)
        pos = minimize_energy(pos, box)
        return pos, vel

    @staticmethod
    def _resolve(key: str, inputs, interventions, default: float, batch: int) -> Tensor:
        val = interventions.get(key, inputs.get(key))
        if val is None:
            return torch.full((batch,), default, dtype=F64)
        v = torch.as_tensor(val, dtype=F64).reshape(-1)
        if v.shape[0] == 1 and batch > 1:
            v = v.expand(batch)
        elif v.shape[0] != batch:            # original: val[i] if i < len(val) else val[0]
            v = torch.cat([v, v[:1].expand(batch - v.shape[0])]) if v.shape[0] < batch else v[:batch]
        return torch.where(v == 0, torch.full_like(v, default), v)   # ``resolve(...) or default``

    def forward_with_interventions(self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = self.batch_size(inputs, interventions)
        vol_in = self._resolve("volume", inputs, interventions, DEFAULT_VOLUME, batch)
        temp_in = self._resolve("temperature", inputs, interventions, DEFAULT_TEMPERATURE, batch)
        press_in = self._resolve("pressure", inputs, interventions, DEFAULT_PRESSURE, batch)

        box_data = inputs.get("box_len")
        if box_data is not None:
            curr_box = torch.as_tensor(box_data, dtype=F64).reshape(-1)
            if curr_box.shape[0] == 1 and batch > 1:
                curr_box = curr_box.expand(batch)
            curr_box = curr_box.clone()
        else:
            curr_box = torch.full((batch,), 10.0, dtype=F64)

        if inputs.get("positions") is not None:
            curr_pos = torch.as_tensor(inputs["positions"], dtype=F64)
            curr_vel = torch.as_tensor(inputs["velocities"], dtype=F64)
            if curr_pos.shape[0] == 1 and batch > 1:
                curr_pos = curr_pos.expand(batch, -1, -1)
                curr_vel = curr_vel.expand(batch, -1, -1)
        else:
            curr_pos, curr_vel = self._init_lattice(curr_box)
            curr_pos = minimize_energy(curr_pos, curr_box)

        active = set(interventions.keys())
        mode = 0
        target_1 = temp_in
        target_2 = torch.zeros_like(temp_in)

        def rescale(pos, box, target_box):
            change = (target_box - box).abs() > 1e-4
            factor = torch.where(change, target_box / box, torch.ones_like(box))
            return pos * factor.reshape(-1, 1, 1), torch.where(change, target_box, box)

        if "volume" in active and "temperature" in active:
            mode = 0
            curr_pos, curr_box = rescale(curr_pos, curr_box, vol_in ** (1.0 / 3.0))
            target_1 = temp_in
        elif "pressure" in active and "temperature" in active:
            mode = 1
            target_1 = temp_in
            target_2 = press_in
            curr_box = torch.where(curr_box < 1.0, torch.full_like(curr_box, 10.0), curr_box)
        elif "pressure" in active and "volume" in active:
            mode = 2
            curr_pos, curr_box = rescale(curr_pos, curr_box, vol_in ** (1.0 / 3.0))
            target_1 = press_in

        new_pos, new_vel, fin_box, found_temp = run_equilibration(
            curr_pos, curr_vel, curr_box, DT, self.equil_steps, mode, target_1, target_2, self.noise_source)
        p_avg, t_avg, v_avg, fin_pos, fin_vel = run_measurement(
            new_pos, new_vel, fin_box, DT, self.measure_steps,
            found_temp if mode == 2 else temp_in, self.noise_source)

        return {
            "pressure": p_avg.reshape(-1, 1),
            "temperature": t_avg.reshape(-1, 1),
            "volume": v_avg.reshape(-1, 1),
            "positions": fin_pos,
            "velocities": fin_vel,
            "box_len": fin_box.reshape(-1, 1),
        }
