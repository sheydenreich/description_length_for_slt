"""Equivalence of ``cae_torch.systems.s01_logic_circuit`` with ``systems/01_logic_circuit.py``."""
import numpy as np
import pytest
import torch

from cae_torch.reference import numpy_forward
from cae_torch.systems import s01_logic_circuit as s01

INTERNAL = ["bit0_xor1", "C1", "S0", "bit1_and2"]


def _numpy_batched(model, inputs, interventions):
    """Run the numpy model one sample at a time (its own batching reads interventions only)."""
    batch = next(iter((*inputs.values(), *interventions.values()))).shape[0]
    rows = [
        numpy_forward(
            model,
            {k: v[i : i + 1] for k, v in inputs.items()},
            {k: v[i : i + 1] for k, v in interventions.items()},
        )
        for i in range(batch)
    ]
    return {k: np.concatenate([r[k] for r in rows], axis=0) for k in rows[0]}


def _draw(rng, batch, n_interv):
    """Random voltages around the logic levels (threshold semantics exercised), with
    interventions on internal wires."""
    inputs = {w: rng.uniform(-0.3, 1.3, size=(batch, 1)) for w in ("A0", "A1", "B0", "B1", "C0")}
    interventions = {w: rng.uniform(-0.3, 1.3, size=(batch, 1)) for w in INTERNAL[:n_interv]}
    return inputs, interventions


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("n_interv", [0, 2, 4])
def test_matches_numpy(cae_module, batch, n_interv):
    ref = cae_module("01_logic_circuit.py")
    gates, wires = ref.build_2bit_adder()
    numpy_model = ref.NetlistSimulator(gates, wires)
    torch_model = s01.NetlistSimulator(*s01.build_2bit_adder())
    assert torch_model.variables == tuple(numpy_model.all_wires)

    rng = np.random.default_rng(seed=batch * 10 + n_interv)
    inputs, interventions = _draw(rng, batch, n_interv)
    # also feed a gate-driven wire as (non-forced) initial state, as the original allows
    inputs["bit0_and1"] = rng.uniform(-0.3, 1.3, size=(batch, 1))
    expected = _numpy_batched(numpy_model, inputs, interventions)
    out = torch_model.forward_with_interventions(
        {k: torch.as_tensor(v) for k, v in inputs.items()},
        {k: torch.as_tensor(v) for k, v in interventions.items()},
    )
    assert set(out) == set(torch_model.variables)
    for name in torch_model.variables:
        assert out[name].shape == (batch, 1)
        np.testing.assert_allclose(out[name].numpy(), expected[name], atol=1e-9, rtol=1e-9)


def test_truth_table_and_high_level_data():
    torch_model = s01.NetlistSimulator(*s01.build_2bit_adder())
    for a in range(4):
        for b in range(4):
            for cin in range(2):
                iv = {
                    "A0": torch.tensor([[float(a & 1)]]), "A1": torch.tensor([[float(a >> 1)]]),
                    "B0": torch.tensor([[float(b & 1)]]), "B1": torch.tensor([[float(b >> 1)]]),
                    "C0": torch.tensor([[float(cin)]]),
                }
                out = torch_model.forward_with_interventions({}, iv)
                hl = {
                    name: s01.abstract_bits([out[w][0, 0] for w in wires])
                    for name, wires in s01.COARSE_GRAINING.items()
                }
                expect = s01.hl_valid(a, b, cin)
                assert hl["Result_Sum"] == expect["Result_Sum"] == (a + b + cin) % 4
                assert hl["Result_Carry"] == expect["Result_Carry"] == int(a + b + cin >= 4)
                assert hl["Internal_Carries"] == expect["Internal_Carries"]
                inv = s01.hl_inverted_internal(a, b, cin)
                assert inv["Internal_Carries"] == 1 - expect["Internal_Carries"]
                assert inv["Result_Sum"] == expect["Result_Sum"]


def test_soft_relaxation_matches_exact_on_booleans():
    exact = s01.NetlistSimulator(*s01.build_2bit_adder())
    soft = s01.NetlistSimulator(*s01.build_2bit_adder(), soft=True)
    rng = np.random.default_rng(1)
    inputs = {w: torch.as_tensor(rng.integers(0, 2, size=(8, 1)).astype(float))
              for w in ("A0", "A1", "B0", "B1", "C0")}
    iv = {"C1": torch.as_tensor(rng.integers(0, 2, size=(8, 1)).astype(float))}
    a, b = exact.forward_with_interventions(inputs, iv), soft.forward_with_interventions(inputs, iv)
    for name in exact.variables:
        torch.testing.assert_close(a[name], b[name], atol=1e-12, rtol=0)


def test_gradients():
    """Exact mode: no differentiable path (0.5 thresholds and boolean gates); the outputs do
    not even carry a graph. Soft mode: gradcheck passes."""
    exact = s01.NetlistSimulator(*s01.build_2bit_adder())
    x = {w: torch.full((2, 1), 0.7, requires_grad=True) for w in ("A0", "A1", "B0", "B1", "C0")}
    out = exact.forward_with_interventions(x, {"C1": torch.ones(2, 1, requires_grad=True)})
    assert all(not v.requires_grad for v in out.values())

    soft = s01.NetlistSimulator(*s01.build_2bit_adder(), soft=True)
    names = ("A0", "A1", "B0", "B1", "C0")

    def f(*vals):
        out = soft.forward_with_interventions(dict(zip(names, vals)), {"C1": torch.tensor([[0.3]])})
        return torch.cat([out["S0"], out["S1"], out["C2"], out["bit1_and1"]], dim=1)

    args = tuple(torch.rand(1, 1, dtype=torch.float64, requires_grad=True) for _ in names)
    assert torch.autograd.gradcheck(f, args, atol=1e-6, rtol=1e-6)
