"""Equivalence of ``cae_torch.systems.s06_heat_equation_2d`` with ``systems/06_heat_equation_2d.py``.

Deterministic path (``p_scatter = 0``): the only random draw of the original is the kick angle,
taken from numpy's global RNG in plain Python; we re-seed numpy, replay those draws and pass them
to ``PhononLatticeModel.simulate(angles=...)``, so the lattice dynamics are compared exactly
(``atol = rtol = 1e-9``; observed difference 0.0). Scattering (``p_scatter > 0``) is compared in
distribution: relative L1 distance of the averaged maps below 0.3 (two independent runs of the
original are ~0.12 apart at 60 averages) and total kinetic energy within 10 %.
Small grid (8 x 8 is the smallest the source clipping allows) and short simulated time.
"""
import numpy as np
import pytest
import torch

from cae_torch.reference import numpy_forward
from cae_torch.systems import s06_heat_equation_2d as s06


@pytest.fixture(scope="module")
def ref_module():
    """Load the numpy reference once per module (its numba kernels take seconds to compile)."""
    from cae_torch.reference import load_numpy_system

    try:
        return load_numpy_system("06_heat_equation_2d.py")
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"numpy reference unavailable: {exc}")

PARAMS = dict(grid_h=8, grid_w=8, k_spring=100.0, p_scatter=0.0, dt=0.02, sim_time=1.0, n_averages=3)


def _replayed_angles(seed, n):
    np.random.seed(seed)
    return torch.tensor([np.random.uniform(0, 2 * np.pi) for _ in range(n)], dtype=torch.float64)


def _torch_deterministic(model, sx, sy, se, angles):
    """forward_with_interventions with the given kick angles (one per run, batch-major)."""
    B, n = sx.shape[0], model.n_averages
    maps = model.simulate(sx.repeat_interleave(n), sy.repeat_interleave(n), se.repeat_interleave(n), angles=angles)
    maps = maps.reshape(B, n, model.H, model.W)
    acc = torch.zeros(B, model.H, model.W, dtype=torch.float64)
    for j in range(n):
        acc = acc + maps[:, j]
    return acc / n


@pytest.mark.parametrize("batch", [1, 8])
def test_deterministic_dynamics_match_numpy(ref_module, batch):
    ref = ref_module
    numpy_model = ref.PhononLatticeModel(**PARAMS)
    torch_model = s06.PhononLatticeModel(**PARAMS)
    rng = np.random.default_rng(batch)
    inputs = {"source_x": rng.uniform(0.0, 1.0, (batch, 1)), "source_y": rng.uniform(0.0, 1.0, (batch, 1))}
    interventions = {"source_E": rng.uniform(500.0, 1000.0, (batch, 1)),
                     "source_y": rng.uniform(0.0, 1.0, (batch, 1))}  # intervention beats the input
    if batch > 1:
        inputs["source_x"][-1, 0] = -1.0  # no kick: zero map
    np.random.seed(11)
    expected = numpy_forward(numpy_model, inputs, interventions)
    out = torch_model.forward_with_interventions(
        {k: torch.as_tensor(v) for k, v in inputs.items()},
        {k: torch.as_tensor(v) for k, v in interventions.items()},
    )
    assert set(out) == set(torch_model.variables)
    for k in ("source_x", "source_y", "source_E"):
        assert out[k].shape == (batch, 1)
        np.testing.assert_allclose(out[k].numpy(), np.asarray(expected[k]).reshape(batch, 1), atol=1e-9, rtol=1e-9)
    assert out["final_temp_map"].shape == (batch, 8, 8)
    exact = _torch_deterministic(
        torch_model, out["source_x"][:, 0], out["source_y"][:, 0], out["source_E"][:, 0],
        _replayed_angles(11, batch * PARAMS["n_averages"]),
    )
    np.testing.assert_allclose(exact.numpy(), expected["final_temp_map"], atol=1e-9, rtol=1e-9)
    if batch > 1:
        assert float(exact[-1].abs().max()) == 0.0
    # the random-angle forward has the same energy content (speed is set by E alone, rotation-free)
    np.testing.assert_allclose(out["final_temp_map"].sum((1, 2)).numpy(), exact.sum((1, 2)).numpy(), rtol=0.15)


def test_scattering_distribution(ref_module):
    ref = ref_module
    params = dict(PARAMS, p_scatter=0.3, n_averages=60)
    numpy_model = ref.PhononLatticeModel(**params)
    torch_model = s06.PhononLatticeModel(**params, generator=torch.Generator().manual_seed(0))
    iv = {"source_x": np.array([[0.55]]), "source_y": np.array([[0.4]]), "source_E": np.array([[800.0]])}
    np.random.seed(1)
    expected = numpy_forward(numpy_model, {}, iv)["final_temp_map"][0]
    out = torch_model.forward_with_interventions({}, {k: torch.as_tensor(v) for k, v in iv.items()})
    got = out["final_temp_map"][0].numpy()
    assert abs(got.sum() - expected.sum()) / expected.sum() < 0.10
    assert np.abs(got - expected).sum() / expected.sum() < 0.3


def test_gradients():
    """final_temp_map is differentiable w.r.t. source_E (kick speed sqrt(2E/m) through Verlet and
    the speed-preserving scattering rotations); source_x / source_y select the kicked atom by
    integer truncation and get no gradient."""
    model = s06.PhononLatticeModel(**dict(PARAMS, sim_time=0.1))
    angles = torch.tensor([0.3, 2.0], dtype=torch.float64)
    sx, sy = torch.tensor([0.5, 0.7], dtype=torch.float64), torch.tensor([0.45, 0.6], dtype=torch.float64)
    se = torch.tensor([700.0, 900.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda e: model.simulate(sx, sy, e, angles=angles), (se,), eps=1e-6, atol=1e-6, rtol=1e-6)

    noisy = s06.PhononLatticeModel(**dict(PARAMS, sim_time=0.1, p_scatter=0.5), generator=torch.Generator().manual_seed(0))
    sx = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    se = torch.tensor([[700.0]], dtype=torch.float64, requires_grad=True)
    out = noisy.forward_with_interventions({"source_x": sx, "source_y": torch.tensor([[0.5]])}, {"source_E": se})
    g_e, g_x = torch.autograd.grad(out["final_temp_map"].sum(), (se, sx), allow_unused=True)
    assert torch.isfinite(g_e).all() and g_e.abs().sum() > 0
    assert g_x is None
