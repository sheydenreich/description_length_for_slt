"""Equivalence of ``cae_torch.systems.s07_ising_model`` with ``systems/07_ising_model.py``.

The model is a Monte-Carlo simulation seeded from its inputs; samples cannot match across RNGs
(numba vs torch), so distributions are compared: the mean of ``PredictedMagnetization`` over 12
nearby conditions (``ExternalField`` shifted by multiples of 1e-4, which changes the seed but not
the physics) must agree within 0.08 at T = 1.5 (ordered, |m| ~ 0.9) and T = 3.5 (disordered),
both for the rigid case (``vibrational_temp = 0``) and the vibrating case (0.2). The per-run
standard deviation of |m| on a 6 x 6 lattice with 60 measurement sweeps is ~0.1, so the standard
error of each mean is ~0.03. The deterministic parts (inputs echoed, seed formula, common random
numbers) are checked exactly.
"""
import numpy as np
import pytest
import torch

from cae_torch.reference import numpy_forward
from cae_torch.systems import s07_ising_model as s07


@pytest.fixture(scope="module")
def ref_module():
    """Load the numpy reference once per module (its numba kernels take seconds to compile)."""
    from cae_torch.reference import load_numpy_system

    try:
        return load_numpy_system("07_ising_model.py")
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"numpy reference unavailable: {exc}")

GRID, SWEEPS = 6, (60, 60)


def _build(ref, vib, generator=None):
    numpy_model = ref.MolecularDynamicsIsingModel(grid_side=GRID, vibrational_temp=vib)
    torch_model = s07.MolecularDynamicsIsingModel(grid_side=GRID, vibrational_temp=vib, generator=generator)
    for m in (numpy_model, torch_model):
        m.params["equil_sweeps"], m.params["measure_sweeps"] = SWEEPS
    return numpy_model, torch_model


def _numpy_batched(model, inputs, interventions):
    batch = next(iter((*inputs.values(), *interventions.values()))).shape[0]
    rows = [numpy_forward(model, {k: v[i : i + 1] for k, v in inputs.items()},
                          {k: v[i : i + 1] for k, v in interventions.items()}) for i in range(batch)]
    return {k: np.stack([np.asarray(r[k]).reshape(-1) for r in rows]) for k in rows[0]}


@pytest.mark.parametrize("vib", [0.0, 0.2])
@pytest.mark.parametrize("temp", [1.5, 3.5])
def test_magnetization_distribution(ref_module, vib, temp):
    ref = ref_module
    numpy_model, torch_model = _build(ref, vib)
    n = 12
    inputs = {"Temperature": np.full((n, 1), temp)}
    interventions = {"ExternalField": (np.arange(n) * 1e-4).reshape(n, 1)}
    expected = _numpy_batched(numpy_model, inputs, interventions)
    out = torch_model.forward_with_interventions(
        {k: torch.as_tensor(v) for k, v in inputs.items()},
        {k: torch.as_tensor(v) for k, v in interventions.items()},
    )
    assert set(out) == set(torch_model.variables)
    for k in ("Temperature", "ExternalField"):
        np.testing.assert_allclose(out[k].numpy(), expected[k], atol=1e-9, rtol=1e-9)
    m_t, m_n = out["PredictedMagnetization"].numpy().ravel(), expected["PredictedMagnetization"].ravel()
    assert m_t.shape == (n,) and np.all((0 <= m_t) & (m_t <= 1))
    assert abs(m_t.mean() - m_n.mean()) < 0.08, (m_t.mean(), m_n.mean())


@pytest.mark.parametrize("batch", [1, 8])
def test_shapes_inputs_and_common_random_numbers(ref_module, batch):
    ref = ref_module
    numpy_model, torch_model = _build(ref, 0.0)
    rng = np.random.default_rng(batch)
    inputs = {"Temperature": rng.uniform(0.5, 4.0, (batch, 1)), "ExternalField": rng.uniform(-0.5, 0.5, (batch, 1))}
    interventions = {"ExternalField": rng.uniform(-0.5, 0.5, (batch, 1))}  # beats the input
    expected = _numpy_batched(numpy_model, inputs, interventions)
    tin = {k: torch.as_tensor(v) for k, v in inputs.items()}
    tiv = {k: torch.as_tensor(v) for k, v in interventions.items()}
    out = torch_model.forward_with_interventions(tin, tiv)
    for k in torch_model.variables:
        assert out[k].shape == (batch, 1)
    np.testing.assert_allclose(out["Temperature"].numpy(), expected["Temperature"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(out["ExternalField"].numpy(), expected["ExternalField"], atol=1e-9, rtol=1e-9)
    # same inputs -> same seed -> identical output, as in the original (common random numbers)
    again = torch_model.forward_with_interventions(tin, tiv)
    torch.testing.assert_close(out["PredictedMagnetization"], again["PredictedMagnetization"], atol=0, rtol=0)
    for b in range(batch):
        t, h = float(inputs["Temperature"][b, 0]), float(interventions["ExternalField"][b, 0])
        assert s07.seed_for(t, h) == int(round(t, 8) * 1e8) + int(round(h + 10, 8) * 1e8)
    # defaults when nothing is given: T = h = 0 -> all spins stay +1
    out0 = torch_model.forward_with_interventions({}, {"Temperature": torch.zeros(batch, 1)})
    assert torch.all(out0["PredictedMagnetization"] == 1.0)
    e0 = numpy_forward(numpy_model, {}, {"Temperature": np.zeros((1, 1))})
    assert float(np.ravel(e0["PredictedMagnetization"])[0]) == 1.0


def test_no_differentiable_path():
    """PredictedMagnetization is a mean of discrete spins; T, h and the atom positions only enter
    through Metropolis accept/reject comparisons, so there is no gradient path."""
    model = s07.MolecularDynamicsIsingModel(grid_side=4, vibrational_temp=0.2)
    model.params["equil_sweeps"], model.params["measure_sweeps"] = 3, 3
    t = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    h = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
    out = model.forward_with_interventions({"Temperature": t}, {"ExternalField": h})
    assert not out["PredictedMagnetization"].requires_grad
    assert out["Temperature"].requires_grad  # the echoed inputs are the inputs themselves
