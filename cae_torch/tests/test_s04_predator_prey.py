"""Equivalence of ``s04_predator_prey`` with ``systems/04_predator_prey.py``.

Deterministic variants are compared exactly.  The stochastic / spatial variants use numba's
internal ``np.random`` stream, which is bit-identical to numpy's legacy ``RandomState``
(checked in ``test_numba_stream_matches_randomstate``); the torch model is therefore run
with a replaying RNG (one ``RandomState`` per batch item) and compared exactly too.  A small
distribution-level check covers the default ``torch.Generator`` path.
"""

import numpy as np
import pytest
import torch

from cae_torch.reference import numpy_forward
from cae_torch.systems.s04_predator_prey import (
    AgentBasedModel, AgentRNG, BASE_CONFIG, IDEAL_CONFIG, calibrate_high_level_model,
    solve_lotka_volterra,
)

SMALL = dict(BASE_CONFIG, num_steps=6, grid_size=8)
VARIANTS = {
    "ideal": dict(spatial=False, stochastic_reproduction=False, agent_aging=False),
    "aging": dict(spatial=False, stochastic_reproduction=False, agent_aging=True),
    "stochastic": dict(spatial=False, stochastic_reproduction=True, agent_aging=False),
    "stoch_aging": dict(spatial=False, stochastic_reproduction=True, agent_aging=True),
    "spatial": dict(spatial=True, stochastic_reproduction=False, agent_aging=False),
    "spatial_aging": dict(spatial=True, stochastic_reproduction=True, agent_aging=True),
}


class ReplayRNG(AgentRNG):
    """Per-batch-item ``RandomState`` streams, seeded like the numba stream of each original run."""

    def __init__(self, seeds):
        self.states = [np.random.RandomState(int(s)) for s in seeds]

    def binomial(self, n, p, mask):
        out = torch.zeros(n.shape[0], dtype=torch.float64)
        for b in range(n.shape[0]):
            if bool(mask[b]):
                out[b] = float(self.states[b].binomial(int(n[b]), p))
        return out

    def randint(self, b, low, high):
        return int(self.states[b].randint(low, high))

    def rand(self, b):
        return float(self.states[b].rand())

    def permutation(self, b, n):
        return self.states[b].permutation(n).tolist()


_REF = {}


@pytest.fixture
def ref(cae_module):
    """The numpy module, loaded (and its numba kernels compiled) once per session."""
    if "m" not in _REF:
        _REF["m"] = cae_module("04_predator_prey.py")
    return _REF["m"]


def _numba_seed_fn():
    import numba

    @numba.njit
    def seed(s):
        np.random.seed(s)

    return seed


def _draw_inputs(rng, batch, spatial):
    hi_prey, hi_pred = (20, 8) if spatial else (250, 100)   # spatial: must fit an 8x8 grid
    prey = rng.uniform(5, hi_prey, size=(batch, 1))
    pred = rng.uniform(2, hi_pred, size=(batch, 1))
    return prey, pred


def test_numba_stream_matches_randomstate(ref):
    import numba

    @numba.njit
    def draws(n):
        out = np.empty(n)
        for i in range(n):
            k = i % 4
            if k == 0:
                out[i] = np.random.binomial(40, 0.1)
            elif k == 1:
                out[i] = np.random.randint(-1, 2)
            elif k == 2:
                out[i] = np.random.rand()
            else:
                out[i] = np.random.permutation(9)[4]
        return out

    @numba.njit
    def seed(s):
        np.random.seed(s)

    seed(7)
    a = draws(200)
    rs = np.random.RandomState(7)
    b = np.array([rs.binomial(40, 0.1) if i % 4 == 0 else rs.randint(-1, 2) if i % 4 == 1
                  else rs.rand() if i % 4 == 2 else rs.permutation(9)[4] for i in range(200)])
    assert np.array_equal(a, b)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_matches_numpy(ref, batch, variant):
    cfg = dict(SMALL, **VARIANTS[variant])
    numpy_model = ref.AgentBasedModel(cfg)
    rng = np.random.default_rng(hash(variant) % 1000 + batch)
    prey, pred = _draw_inputs(rng, batch, cfg["spatial"])
    # interventions: intervene on predator_t only (prey_t comes from the input state),
    # and on final_populations, which the original ignores
    inputs = {"prey_t": prey, "predator_t": pred}
    interventions = {"predator_t": rng.uniform(2, 8 if cfg["spatial"] else 100, size=(batch, 1)),
                     "final_populations": rng.uniform(0, 10, size=(batch, 2))}

    stochastic = cfg["spatial"] or cfg["stochastic_reproduction"]
    seeds = 1000 * batch + np.arange(batch)
    if stochastic:
        seed = _numba_seed_fn()
        expected = {k: [] for k in ("prey_t", "predator_t", "final_populations")}
        for b in range(batch):      # one original run per item, each with its own seed
            seed(int(seeds[b]))
            out_b = numpy_forward(numpy_model, {k: v[b:b + 1] for k, v in inputs.items()},
                                  {k: v[b:b + 1] for k, v in interventions.items()})
            for k in expected:
                expected[k].append(np.asarray(out_b[k], dtype=float))
        expected = {k: np.concatenate(v, axis=0) for k, v in expected.items()}
        torch_model = AgentBasedModel(cfg, rng=ReplayRNG(seeds))
    else:
        expected = numpy_forward(numpy_model, inputs, interventions)
        torch_model = AgentBasedModel(cfg)

    out = torch_model.forward_with_interventions(
        {k: torch.as_tensor(v) for k, v in inputs.items()},
        {k: torch.as_tensor(v) for k, v in interventions.items()},
    )
    assert set(out) == set(torch_model.variables)
    for name in torch_model.variables:
        np.testing.assert_allclose(out[name].numpy(), np.asarray(expected[name], dtype=float),
                                   atol=1e-9, rtol=1e-9, err_msg=name)


def test_broadcast_and_defaults(ref):
    cfg = dict(SMALL, **VARIANTS["aging"])
    numpy_model, torch_model = ref.AgentBasedModel(cfg), AgentBasedModel(cfg)
    inputs = {"prey_t": np.array([[120.0], [80.0], [60.0]])}      # predator_t defaults to 40
    expected = numpy_forward(numpy_model, inputs, {})
    out = torch_model.forward_with_interventions({k: torch.as_tensor(v) for k, v in inputs.items()}, {})
    for name in torch_model.variables:
        # the defaulted predator_t comes back as (3,) from numpy and (3, 1) here (contract shape)
        np.testing.assert_allclose(out[name].numpy().reshape(3, -1), np.asarray(expected[name]).reshape(3, -1),
                                   atol=1e-9, rtol=1e-9)


def test_default_generator_distribution(ref):
    """Default torch RNG path: mean final populations of the stochastic ABM agree with the
    original within 4 standard errors (n=300 samples each)."""
    cfg = dict(SMALL, **VARIANTS["stoch_aging"], num_steps=10)
    n = 300
    seed = _numba_seed_fn()
    seed(3)
    inputs = {"prey_t": np.full((n, 1), 150.0), "predator_t": np.full((n, 1), 50.0)}
    expected = numpy_forward(ref.AgentBasedModel(cfg), inputs, {})["final_populations"]
    gen = torch.Generator().manual_seed(3)
    out = AgentBasedModel(cfg, generator=gen).forward({k: torch.as_tensor(v) for k, v in inputs.items()})
    got = out["final_populations"].numpy()
    se = np.sqrt(expected.var(axis=0) / n + got.var(axis=0) / n)
    assert np.all(np.abs(got.mean(axis=0) - expected.mean(axis=0)) < 4 * se + 1e-9)


def test_lotka_volterra_and_calibration(ref):
    params = dict(alpha=0.1, beta=0.001, delta=0.0005, gamma=0.05, dt=50)
    assert np.allclose(solve_lotka_volterra(100.0, 40.0, params),
                       ref.solve_lotka_volterra(100.0, 40.0, params), atol=1e-9, rtol=1e-9)
    best = calibrate_high_level_model(AgentBasedModel(IDEAL_CONFIG))
    assert set(best) == {"alpha", "beta", "delta", "gamma", "dt"} and best["dt"] == 50


def test_gradients_deterministic_variant():
    """Non-spatial deterministic ABM is differentiable (a.e.) in prey_t and predator_t."""
    cfg = dict(SMALL, **VARIANTS["aging"], num_steps=4)
    model = AgentBasedModel(cfg)

    def f(prey, pred):
        return model.forward_with_interventions({"prey_t": prey, "predator_t": pred}, {})["final_populations"]

    prey = torch.tensor([[120.0], [70.0]], dtype=torch.float64, requires_grad=True)
    pred = torch.tensor([[30.0], [55.0]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (prey, pred), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_gradients_stochastic_variants():
    """Spatial variant: the grid dynamics are discrete, no differentiable path exists (the
    output does not require grad).  Non-spatial stochastic variant: the binomial draws are
    detached, only the identity carry-over ``prey + births - ...`` propagates, so the
    gradient is the partial (straight-through-like) factor 1 per surviving step."""
    cfg = dict(SMALL, **VARIANTS["spatial"], num_steps=2)
    model = AgentBasedModel(cfg, generator=torch.Generator().manual_seed(0))
    prey = torch.tensor([[12.0]], dtype=torch.float64, requires_grad=True)
    pred = torch.tensor([[5.0]], dtype=torch.float64, requires_grad=True)
    out = model.forward_with_interventions({"prey_t": prey, "predator_t": pred}, {})
    assert not out["final_populations"].requires_grad

    cfg = dict(SMALL, **VARIANTS["stochastic"], num_steps=3)
    model = AgentBasedModel(cfg, generator=torch.Generator().manual_seed(0))
    prey = torch.tensor([[200.0]], dtype=torch.float64, requires_grad=True)
    pred = torch.tensor([[50.0]], dtype=torch.float64, requires_grad=True)
    out = model.forward_with_interventions({"prey_t": prey, "predator_t": pred}, {})
    out["final_populations"][0, 0].backward()
    assert prey.grad.item() == 1.0 and pred.grad.item() == 0.0
