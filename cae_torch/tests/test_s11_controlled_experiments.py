"""Equivalence of ``s11_controlled_experiments`` with ``systems/11_controlled_experiments.py``.

All six systems are deterministic (system 1 wraps the deterministic ideal ABM of system 04),
so every variable is compared with atol = rtol = 1e-9 (the comparisons are in fact exact).
"""

import numpy as np
import pytest
import torch

from cae_torch.reference import numpy_forward
from cae_torch.systems import s11_controlled_experiments as ctrl
from cae_torch.systems.s04_predator_prey import IDEAL_CONFIG, AgentBasedModel
from cae_torch.systems.s11_controlled_experiments import (
    ChainLowLevelModel, ConfounderPPModel, ForkLowLevelModel, IntermediateChainLowLevelModel,
    ThresholdLowLevelModel, XORBackupLowLevelModel, round_digits,
)

SMALL_IDEAL = dict(IDEAL_CONFIG, num_steps=8)

# name -> (numpy builder, torch builder, input draw, intervention draw)
CASES = {
    "xor_leak": (lambda ref: ref.XORBackupLowLevelModel(xor_active=True), lambda: XORBackupLowLevelModel(True),
                 {"x": (-0.2, 1.2)}, {"b1": (-0.2, 1.2), "b2": (-0.2, 1.2), "a": (0, 1), "y": (0, 1)}),
    "xor_valid": (lambda ref: ref.XORBackupLowLevelModel(xor_active=False), lambda: XORBackupLowLevelModel(False),
                  {"x": (-0.2, 1.2)}, {"b1": (-0.2, 1.2), "b2": (-0.2, 1.2)}),
    "intermediate": (lambda ref: ref.IntermediateChainLowLevelModel(), IntermediateChainLowLevelModel,
                     {"x": (0.0, 10.0)}, {"m": (0.0, 30.0), "z": (3.0, 23.0)}),
    "fork": (lambda ref: ref.ForkLowLevelModel(2.5, 0.7, 3.1, 1.9), lambda: ForkLowLevelModel(2.5, 0.7, 3.1, 1.9),
             {"x": (0.0, 10.0)}, {"y": (1.0, 21.0), "z": (2.0, 32.0)}),
    "threshold": (lambda ref: ref.ThresholdLowLevelModel(), ThresholdLowLevelModel,
                  {"x": (-0.5, 1.5)}, {"m": (-0.5, 3.5), "z": (0, 1)}),
    "chain": (lambda ref: ref.ChainLowLevelModel(1.7, 0.3, 1.25, 0.75), lambda: ChainLowLevelModel(1.7, 0.3, 1.25, 0.75),
              {"x": (0.0, 10.0)}, {"y": (1.0, 21.0), "z": (2.0, 32.0)}),
}


_REF = {}


@pytest.fixture
def ref(cae_module):
    if "m" not in _REF:
        _REF["m"] = cae_module("11_controlled_experiments.py")
    return _REF["m"]


@pytest.fixture
def pp(cae_module):
    """System 04 module (numba kernels compiled once per session)."""
    if "pp" not in _REF:
        _REF["pp"] = cae_module("04_predator_prey.py")
    return _REF["pp"]


def _draw(rng, spec, batch):
    return {k: rng.uniform(lo, hi, size=(batch, 1)) for k, (lo, hi) in spec.items()}


def _compare(torch_model, out, expected, batch):
    assert set(out) == set(torch_model.variables)
    for name in torch_model.variables:
        exp = np.asarray(expected[name], dtype=float)
        got = out[name].numpy()
        assert got.shape[0] == batch
        np.testing.assert_allclose(got.reshape(batch, -1), np.broadcast_to(exp.reshape(exp.shape[0], -1), got.reshape(batch, -1).shape),
                                   atol=1e-9, rtol=1e-9, err_msg=name)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("which", ["partial", "all", "none"])
def test_matches_numpy(ref, batch, case, which):
    """Interventions on none, on a random subset (always including an internal variable
    when there is one), or on every intervenable / ignored variable of the case."""
    mk_np, mk_t, in_spec, iv_spec = CASES[case]
    rng = np.random.default_rng(abs(hash((case, which))) % 10_000 + batch)
    inputs = _draw(rng, in_spec, batch)
    if which == "none":
        interventions = {}
    elif which == "all":
        interventions = _draw(rng, iv_spec, batch)
        interventions.update(_draw(rng, in_spec, batch))        # also override x
    else:
        keys = list(iv_spec)[: max(1, len(iv_spec) // 2)]
        interventions = _draw(rng, {k: iv_spec[k] for k in keys}, batch)
    # the threshold system is meant for integer levels: round half of the draws
    if case == "threshold" and batch > 1:
        for d in (inputs, interventions):
            for k in d:
                d[k][::2] = np.round(d[k][::2])
    expected = numpy_forward(mk_np(ref), inputs, interventions)
    out = mk_t().forward_with_interventions({k: torch.as_tensor(v) for k, v in inputs.items()},
                                            {k: torch.as_tensor(v) for k, v in interventions.items()})
    _compare(mk_t(), out, expected, batch)


@pytest.mark.parametrize("batch", [1, 8])
def test_defaults_match_numpy(ref, batch):
    """No x in the inputs: the per-system defaults (5.0 or 0.0) are used; batch from the interventions."""
    for case in ("xor_leak", "intermediate", "fork", "threshold", "chain"):
        mk_np, mk_t, _in, iv_spec = CASES[case]
        rng = np.random.default_rng(batch)
        key = list(iv_spec)[0]
        interventions = _draw(rng, {key: iv_spec[key]}, batch)
        expected = numpy_forward(mk_np(ref), {}, interventions)
        out = mk_t().forward_with_interventions({}, {k: torch.as_tensor(v) for k, v in interventions.items()})
        _compare(mk_t(), out, expected, batch)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("effect", [0.0, 0.5])
def test_confounder_matches_numpy(ref, pp, batch, effect):
    numpy_model = ref.ConfounderPPModel(pp.AgentBasedModel(SMALL_IDEAL), resource_effect=effect)
    torch_model = ConfounderPPModel(AgentBasedModel(SMALL_IDEAL), resource_effect=effect)
    rng = np.random.default_rng(batch)
    inputs = {"prey_t": rng.uniform(50, 250, size=(batch, 1)), "predator_t": rng.uniform(20, 100, size=(batch, 1))}
    for interventions in ({},
                          {"resource_level": rng.uniform(-1, 1, size=(batch, 1))},
                          {"predator_t": rng.uniform(20, 100, size=(batch, 1)),
                           "resource_level": rng.uniform(-1, 1, size=(batch, 1))},
                          {"resource_level": np.array([[-3.0]])}):      # -> negative prey, clamped to 0
        expected = numpy_forward(numpy_model, inputs, interventions)
        out = torch_model.forward_with_interventions({k: torch.as_tensor(v) for k, v in inputs.items()},
                                                     {k: torch.as_tensor(v) for k, v in interventions.items()})
        _compare(torch_model, out, expected, batch)


def test_rounding_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.uniform(-50, 50, size=10_000)
    np.testing.assert_array_equal(round_digits(torch.as_tensor(x)).numpy(), np.round(x, 10))


def test_high_level_models_against_numpy(ref):
    """The plain-Python high-level rules reproduce the original CausalGraph mechanisms."""
    for x in (0.0, 0.37, 5.0, 9.99):
        for ours, theirs in ((ctrl.EXP3["valid"], ref.build_valid_intermediate_high_level_model()),
                             (ctrl.EXP3["wrong"], ref.build_wrong_intermediate_high_level_model())):
            m = ours["m"](x)
            assert m == theirs._nodes["m"]["equation"](x) and ours["z"](m) == theirs._nodes["z"]["equation"](m)
        for ours, theirs in ((ctrl.FORK_HLM, ref.build_fork_high_level_model()),
                             (ctrl.CHAIN_HLM, ref.build_chain_high_level_model()),
                             (ctrl.FORK_HLM, ref.build_wrong_fork_high_level_model())):
            y = ours["y"][1](x)
            assert y == theirs._nodes["y"]["equation"](x)
            arg = x if ours["z"][0] == "x" else y
            assert ours["z"][1](arg) == theirs._nodes["z"]["equation"](arg)
    for M in range(4):
        assert ctrl.EXP5["valid"]["Z"](M) == ref.build_valid_threshold_high_level_model()._nodes["Z"]["equation"](M)
        assert ctrl.EXP5["wrong_threshold"]["Z"](M) == ref.build_wrong_threshold_high_level_model()._nodes["Z"]["equation"](M)


def test_gradients():
    """Continuous systems: exact analytic gradients through the straight-through rounding;
    gated systems: zero gradient through the exact gate, nonzero with ``soft=True``."""
    x = torch.tensor([[1.3], [4.2]], dtype=torch.float64, requires_grad=True)
    z = IntermediateChainLowLevelModel().forward({"x": x})["z"]
    z.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 2.0))          # z = 2x + 3

    x = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    y_iv = torch.tensor([[7.0]], dtype=torch.float64, requires_grad=True)
    out = ChainLowLevelModel().forward_with_interventions({"x": x}, {"y": y_iv})
    out["z"].sum().backward()
    assert x.grad is None and y_iv.grad.item() == 1.5                # z = 1.5 y + 0.5, y clamped
    x = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    y_iv = torch.tensor([[7.0]], dtype=torch.float64, requires_grad=True)
    out = ForkLowLevelModel().forward_with_interventions({"x": x}, {"y": y_iv})
    out["z"].sum().backward()
    assert x.grad.item() == 3.0 and y_iv.grad is None                 # z = 3x + 2, independent of y

    m = torch.tensor([[1.2]], dtype=torch.float64, requires_grad=True)
    z = ThresholdLowLevelModel().forward_with_interventions({}, {"m": m})["z"]
    assert not z.requires_grad                                        # exact threshold: no path
    z = ThresholdLowLevelModel(soft=True).forward_with_interventions({}, {"m": m})["z"]
    z.sum().backward()
    assert m.grad.item() > 0

    b1 = torch.tensor([[0.8]], dtype=torch.float64, requires_grad=True)
    out = XORBackupLowLevelModel(True).forward_with_interventions({"x": torch.tensor([[0.0]])}, {"b1": b1})
    assert not out["y"].requires_grad
    out = XORBackupLowLevelModel(True, soft=True).forward_with_interventions({"x": torch.tensor([[0.0]])}, {"b1": b1})
    out["y"].sum().backward()
    assert b1.grad.item() > 0

    prey = torch.tensor([[120.0]], dtype=torch.float64, requires_grad=True)
    r = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
    model = ConfounderPPModel(AgentBasedModel(SMALL_IDEAL), resource_effect=0.5)
    out = model.forward_with_interventions({"prey_t": prey, "predator_t": torch.tensor([[40.0]])}, {"resource_level": r})
    out["final_populations"][0, 0].backward()
    assert torch.isfinite(prey.grad).all() and prey.grad.item() != 0 and r.grad.item() != 0
