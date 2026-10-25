"""Equivalence of ``cae_torch.systems.s05_heat_equation`` with ``systems/05_heat_equation.py``.

The original is stochastic (global numpy / numba RNG), so ``T_final`` is compared in
distribution: two-sample KS statistic below ``2.5 * sqrt(2 / n)`` (p ~ 4e-6 under equality;
0.016 for the 8 x 4000 particles of batch 8, 0.045 for batch 1), plus mean / std within 5e-3
(standard errors ~ 6e-4 for batch 8). The
deterministic parts (``T_initial`` pass-through, reflection operator) are compared exactly.
"""
import numpy as np
import pytest
import torch
from scipy import stats

from cae_torch.reference import numpy_forward
from cae_torch.systems import s05_heat_equation as s05


@pytest.fixture(scope="module")
def ref_module():
    """Load the numpy reference once per module (its numba kernels take seconds to compile)."""
    from cae_torch.reference import load_numpy_system

    try:
        return load_numpy_system("05_heat_equation.py")
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"numpy reference unavailable: {exc}")

PARAMS = dict(n_particles=4000, n_steps=50, diff_coeff=0.1, box_len=1.0, dt=0.001)


@pytest.mark.parametrize("batch", [1, 8])
def test_distribution_matches_numpy(ref_module, batch):
    ref = ref_module
    numpy_model = ref.BrownianParticleSystem(**PARAMS)
    torch_model = s05.BrownianParticleSystem(**PARAMS, generator=torch.Generator().manual_seed(0))
    rng = np.random.default_rng(batch)
    # initial positions concentrated near a wall so that reflections matter
    init = rng.uniform(0.0, 0.3, size=(batch, PARAMS["n_particles"]))
    np.random.seed(123)
    expected = numpy_forward(numpy_model, {}, {"T_initial": init})
    out = torch_model.forward_with_interventions({}, {"T_initial": torch.as_tensor(init)})
    assert set(out) == {"T_initial", "T_final"}
    np.testing.assert_allclose(out["T_initial"].numpy(), expected["T_initial"], atol=1e-9, rtol=1e-9)
    a, b = out["T_final"].numpy().ravel(), np.asarray(expected["T_final"]).ravel()
    assert a.min() >= 0 and a.max() <= 1
    assert stats.ks_2samp(a, b).statistic < 2.5 * np.sqrt(2 / a.size)
    assert abs(a.mean() - b.mean()) < 5e-3 and abs(a.std() - b.std()) < 5e-3


def test_input_state_and_resampling(ref_module):
    ref = ref_module
    small = dict(PARAMS, n_particles=500, n_steps=5)
    numpy_model = ref.BrownianParticleSystem(**small)
    torch_model = s05.BrownianParticleSystem(**small, generator=torch.Generator().manual_seed(1))
    rng = np.random.default_rng(5)
    # T_initial from the input state (no intervention): passed through unchanged
    init = rng.uniform(0, 1, size=(8, 500))
    expected = numpy_forward(numpy_model, {"T_initial": init}, {})
    out = torch_model.forward_with_interventions({"T_initial": torch.as_tensor(init)}, {})
    np.testing.assert_allclose(out["T_initial"].numpy(), expected["T_initial"], atol=1e-9, rtol=1e-9)
    # particle-count mismatch: resampled with replacement to N (values drawn from the input)
    init2 = rng.uniform(0, 1, size=(2, 300))
    expected = numpy_forward(numpy_model, {}, {"T_initial": init2})
    out = torch_model.forward_with_interventions({}, {"T_initial": torch.as_tensor(init2)})
    assert out["T_initial"].shape == expected["T_initial"].shape == (2, 500)
    for i in range(2):
        assert set(out["T_initial"][i].tolist()) <= set(init2[i].tolist())
    # no positions at all: uniform initialisation of the right shape
    out = torch_model.forward_with_interventions({}, {})
    assert out["T_initial"].shape == (1, 500) and out["T_final"].shape == (1, 500)


def test_reflection_matches_original_loop():
    """Reflection operator vs a literal transcription of the original's while loop."""
    model = s05.BrownianParticleSystem(n_particles=6, n_steps=3, diff_coeff=5.0, box_len=1.0, dt=0.01)
    rng = np.random.default_rng(2)
    x0 = rng.uniform(0, 1, size=(2, 6))
    noise = rng.standard_normal(size=(3, 2, 6))
    ref = x0.copy()
    for t in range(3):
        ref = ref + model.step_size * noise[t]
        for idx in np.ndindex(ref.shape):
            while ref[idx] < 0 or ref[idx] > 1.0:
                if ref[idx] < 0:
                    ref[idx] = -ref[idx]
                elif ref[idx] > 1.0:
                    ref[idx] = 2 * 1.0 - ref[idx]
    out = model.run_brownian_steps(torch.as_tensor(x0), noise=torch.as_tensor(noise))
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-12, rtol=0)


def test_gradients():
    """T_final is differentiable w.r.t. T_initial for fixed noise (slopes +-1 through reflections)."""
    model = s05.BrownianParticleSystem(n_particles=4, n_steps=3, diff_coeff=2.0, box_len=1.0, dt=0.01)
    noise = torch.randn(3, 1, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    x0 = torch.tensor([[0.05, 0.5, 0.97, 0.3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: model.run_brownian_steps(x, noise=noise), (x0,), eps=1e-7)
    out = model.run_brownian_steps(x0, noise=noise)
    out.sum().backward()
    assert torch.all(x0.grad.abs() == 1.0)
    # the full forward with a generator also carries a graph
    gen_model = s05.BrownianParticleSystem(n_particles=4, n_steps=2, generator=torch.Generator().manual_seed(0))
    x0 = torch.rand(2, 4, dtype=torch.float64, requires_grad=True)
    gen_model.forward_with_interventions({}, {"T_initial": x0})["T_final"].sum().backward()
    assert torch.isfinite(x0.grad).all() and (x0.grad != 0).any()
