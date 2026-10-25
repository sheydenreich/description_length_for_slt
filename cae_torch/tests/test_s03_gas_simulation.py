"""Equivalence of ``s03_gas_simulation`` with ``systems/03_gas_simulation.py``.

The Langevin thermostat makes every trajectory stochastic.  Numba's internal ``np.random``
stream is bit-identical to numpy's legacy ``RandomState`` (verified below), so the torch
model is fed the exact noise stream the original consumed (one normal per velocity
component per step, batch items one after the other) and the full trajectories are
compared.  Tolerance 1e-9 (atol and rtol): the only differences are summation orders in the
pairwise force / virial sums and in ``sum(vel**2)``, which are at the 1e-13 level for the
short runs used here.
"""

import numpy as np
import pytest
import torch

from cae_torch.reference import numpy_forward
from cae_torch.systems import s03_gas_simulation as gs
from cae_torch.systems.s03_gas_simulation import UniversalGasModel

N = gs.N_PARTICLES
EQUIL, MEASURE = 12, 5        # 12 equilibration steps include two barostat updates (steps 0 and 10)

_REF = {}


@pytest.fixture
def ref(cae_module):
    """The numpy module, loaded (and its numba kernels compiled) once per session."""
    if "m" not in _REF:
        _REF["m"] = cae_module("03_gas_simulation.py")
    return _REF["m"]


@pytest.fixture(autouse=True, scope="module")
def _few_torch_threads():
    """The (batch, 128, 128, 3) pair tensors are small; on a shared login node a large
    thread pool makes them ~20x slower than two threads."""
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)

MODES = {
    "nvt": ("volume", "temperature"),
    "npt": ("pressure", "temperature"),
    "pvt": ("pressure", "volume"),
    "t_only": ("temperature",),
}
RANGES = {"volume": (700.0, 1300.0), "temperature": (1.5, 4.0), "pressure": (0.05, 0.5)}


def _numba_seed_fn():
    import numba

    @numba.njit
    def seed(s):
        np.random.seed(s)

    return seed


def _microstate(rng, batch, box=10.0):
    k = int(np.ceil(N ** (1 / 3.0)))
    s = box / k
    grid = np.arange(k) * s
    x, y, z = np.meshgrid(grid, grid, grid)
    pos = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T[:N]
    pos = np.stack([pos + rng.normal(0.0, s * 0.05, size=pos.shape) for _ in range(batch)]) % box
    vel = rng.normal(0.0, 1.5, size=(batch, N, 3))
    return pos, vel, np.full(batch, box)


def _replay_noise(seed, batch, steps):
    """Noise source replaying numba's stream: item b consumed its (steps, N, 3) normals in turn."""
    rs = np.random.RandomState(seed)
    per_item = [rs.normal(0.0, 1.0, size=(steps, N, 3)) for _ in range(batch)]
    stream = torch.as_tensor(np.stack(per_item, axis=1))        # (steps, batch, N, 3)
    counter = {"t": 0}

    def source(shape):
        t = counter["t"]
        counter["t"] += 1
        assert tuple(shape) == tuple(stream.shape[1:]), (shape, stream.shape)
        return stream[t]

    return source, counter


def test_numba_normal_stream_matches_randomstate(ref):
    import numba

    @numba.njit
    def normals(n):
        out = np.empty(n)
        for i in range(n):
            out[i] = np.random.normal(0.0, 1.0)
        return out

    _numba_seed_fn()(11)
    assert np.array_equal(normals(1000), np.random.RandomState(11).normal(0.0, 1.0, size=1000))


def test_force_kernels_match(ref):
    """Deterministic kernels: forces, virial, tail correction, energy minimisation."""
    rng = np.random.default_rng(5)
    pos, _, box = _microstate(rng, 3, box=9.0)
    pos[1] = rng.uniform(0, 9.0, size=(N, 3))        # random (overlapping) configuration
    f_t, v_t = gs.compute_forces(torch.as_tensor(pos), torch.as_tensor(box))
    for b in range(3):
        f_n, v_n = ref.compute_forces(pos[b].copy(), N, box[b], gs.CUTOFF ** 2)
        np.testing.assert_allclose(f_t[b].numpy(), f_n, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(v_t[b].item(), v_n, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(gs.get_p_tail_corr(N, torch.as_tensor(box) ** 3)[b].item(),
                                   ref.get_p_tail_corr(N, box[b] ** 3, gs.CUTOFF), rtol=1e-12)
    # minimisation (per-item early stop): run on the random configuration and a lattice
    mins_n = np.stack([ref.minimize_energy(pos[b].copy(), N, box[b], 30, 1.0) for b in range(3)])
    mins_t = gs.minimize_energy(torch.as_tensor(pos), torch.as_tensor(box), max_steps=30, tol=1.0)
    np.testing.assert_allclose(mins_t.numpy()[[0, 2]], mins_n[[0, 2]], atol=1e-9, rtol=1e-9)
    # The random configuration has overlapping particles with LJ forces ~1e8: the 1-ULP
    # differences in the pairwise summation order (numba sequential vs torch reduction) are
    # amplified by the steep repulsion over 30 steps, hence the looser tolerance here only.
    np.testing.assert_allclose(mins_t.numpy()[1], mins_n[1], atol=1e-6, rtol=1e-6)
    # lattice construction equals the original's _init_lattice geometry (before jitter)
    k = int(np.ceil(N ** (1 / 3.0)))
    grid = np.arange(k) * (9.0 / k)
    x, y, z = np.meshgrid(grid, grid, grid)
    np.testing.assert_allclose(gs.lattice_positions(torch.tensor([9.0]))[0].numpy(),
                               np.vstack([x.ravel(), y.ravel(), z.ravel()]).T[:N], atol=1e-12)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("mode", sorted(MODES))
def test_matches_numpy(ref, batch, mode):
    seed = 100 + 10 * batch + list(MODES).index(mode)
    rng = np.random.default_rng(seed)
    pos, vel, box = _microstate(rng, batch)
    inputs_np = {"positions": pos, "velocities": vel, "box_len": box}
    interventions_np = {k: rng.uniform(*RANGES[k], size=batch) for k in MODES[mode]}
    if mode == "t_only":
        interventions_np["positions"] = rng.uniform(0, 10, size=(batch, N, 3))   # ignored by the original

    numpy_model = ref.UniversalGasModel(equil_steps=EQUIL, measure_steps=MEASURE)
    _numba_seed_fn()(seed)
    expected = numpy_forward(numpy_model, inputs_np, interventions_np)

    source, counter = _replay_noise(seed, batch, EQUIL + MEASURE)
    torch_model = UniversalGasModel(equil_steps=EQUIL, measure_steps=MEASURE, noise_source=source)
    inputs = {"positions": torch.as_tensor(pos), "velocities": torch.as_tensor(vel),
              "box_len": torch.as_tensor(box).reshape(-1, 1)}
    interventions = {k: torch.as_tensor(v).reshape(batch, -1) if v.ndim == 1 else torch.as_tensor(v)
                     for k, v in interventions_np.items()}
    out = torch_model.forward_with_interventions(inputs, interventions)

    assert counter["t"] == EQUIL + MEASURE
    assert set(out) == set(torch_model.variables)
    for name in torch_model.variables:
        np.testing.assert_allclose(out[name].numpy(), np.asarray(expected[name], dtype=float),
                                   atol=1e-9, rtol=1e-9, err_msg=f"{mode}/{name}")


def test_zero_falls_back_to_default(ref):
    """The original resolves ``value or default``: an intervention of exactly 0 means default."""
    rng = np.random.default_rng(1)
    pos, vel, box = _microstate(rng, 1)
    inter = {"volume": np.array([0.0]), "temperature": np.array([0.0])}
    _numba_seed_fn()(1)
    expected = numpy_forward(ref.UniversalGasModel(2, 2), {"positions": pos, "velocities": vel, "box_len": box}, inter)
    source, _ = _replay_noise(1, 1, 4)
    out = UniversalGasModel(2, 2, noise_source=source).forward_with_interventions(
        {"positions": torch.as_tensor(pos), "velocities": torch.as_tensor(vel), "box_len": torch.as_tensor(box)},
        {k: torch.as_tensor(v).reshape(1, 1) for k, v in inter.items()})
    for name in out:
        np.testing.assert_allclose(out[name].numpy(), np.asarray(expected[name], dtype=float), atol=1e-9, rtol=1e-9)
    assert abs(out["volume"].item() - 1000.0) < 1e-9


def test_gradients_flow():
    """With the noise fixed, pressure is differentiable in volume, temperature (gradcheck),
    and the gradient w.r.t. the microstate is finite and nonzero."""
    pos = gs.lattice_positions(torch.tensor([10.0]))
    rng = np.random.default_rng(2)
    pos = pos + torch.as_tensor(rng.normal(0, 0.05, size=pos.shape))
    vel = torch.as_tensor(rng.normal(0, 1.5, size=(1, N, 3)))
    fixed = torch.as_tensor(rng.normal(size=(4, 1, N, 3)))
    counter = {"t": 0}

    def source(shape):
        counter["t"] = (counter["t"] + 1) % 4
        return fixed[counter["t"]]

    model = UniversalGasModel(equil_steps=2, measure_steps=2, noise_source=source)

    def f(volume, temperature):
        return model.forward_with_interventions(
            {"positions": pos, "velocities": vel, "box_len": torch.tensor([[10.0]])},
            {"volume": volume, "temperature": temperature})["pressure"]

    volume = torch.tensor([[1100.0]], dtype=torch.float64, requires_grad=True)
    temperature = torch.tensor([[2.5]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (volume, temperature), eps=1e-6, atol=1e-6, rtol=1e-4)

    p = pos.clone().requires_grad_(True)
    v = vel.clone().requires_grad_(True)
    out = model.forward_with_interventions(
        {"positions": p, "velocities": v, "box_len": torch.tensor([[10.0]])},
        {"pressure": torch.tensor([[0.3]]), "volume": torch.tensor([[900.0]])})     # PVT branch
    (out["pressure"].sum() + out["temperature"].sum()).backward()
    for g in (p.grad, v.grad):
        assert torch.isfinite(g).all() and g.abs().sum() > 0
