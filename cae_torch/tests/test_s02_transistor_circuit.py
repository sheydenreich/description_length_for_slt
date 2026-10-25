"""Equivalence of ``cae_torch.systems.s02_transistor_circuit`` with ``systems/02_transistor_circuit.py``.

The reference needs PySpice and the ngspice shared library (``NGSPICE_LIBRARY_PATH`` or
``LD_LIBRARY_PATH``); the tests skip when they are missing.

Tolerance: ``atol = 1e-2 V``. ngspice stops its Newton iteration at ``reltol = 1e-3`` /
``vntol = 1e-6``, so the reference itself is only converged to a few mV on weakly driven nodes
(the ``n_*`` nodes between stacked NMOS in cutoff; two symmetric cases of the reference differ by
1.5 mV there). Our solve is converged to 1e-13 V; the observed differences are ~2e-3 V on those
nodes and < 1e-7 V on every strongly driven node.
"""
import numpy as np
import pytest
import torch

from cae_torch.reference import numpy_forward
from cae_torch.systems import s02_transistor_circuit as s02

ATOL = 1e-2


@pytest.fixture
def ref(cae_module):
    try:
        mod = cae_module("02_transistor_circuit.py")
    except SystemExit:  # the original calls exit(0) when PySpice is missing
        pytest.skip("PySpice not installed")
    try:
        numpy_forward(mod.GenericSpiceModel(mod.nand_topology, ["a", "b"]), {}, {"a": np.array([[0.0]])})
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"ngspice unavailable: {exc}")
    return mod


def _numpy_batched(model, inputs, interventions):
    """The numpy model handles one sample per call. ngspice occasionally fails to converge
    ("run failed" after gmin stepping) on perfectly regular analog inputs where our Newton solve
    converges; such samples are returned as NaN rows and excluded from the comparison."""
    batch = next(iter((*inputs.values(), *interventions.values()))).shape[0]
    rows = []
    for i in range(batch):
        try:
            rows.append(numpy_forward(
                model,
                {k: v[i : i + 1] for k, v in inputs.items()},
                {k: v[i : i + 1] for k, v in interventions.items()},
            ))
        except Exception:  # NgSpiceCommandError: reference did not converge
            rows.append(None)
    names = next(r for r in rows if r is not None).keys()
    assert sum(r is not None for r in rows) >= max(1, batch // 2)
    return {k: np.concatenate([np.asarray(r[k]).reshape(1, -1) if r is not None else np.full((1, 1), np.nan)
                               for r in rows], axis=0) for k in names}


def _compare(torch_model, numpy_model, inputs, interventions):
    expected = _numpy_batched(numpy_model, inputs, interventions)
    out = torch_model.forward_with_interventions(
        {k: torch.as_tensor(v) for k, v in inputs.items()},
        {k: torch.as_tensor(v) for k, v in interventions.items()},
    )
    assert set(out) == set(torch_model.variables)
    batch = next(iter((*inputs.values(), *interventions.values()))).shape[0]
    for name in torch_model.variables:
        assert out[name].shape == (batch, 1)
        ok = ~np.isnan(expected[name][:, 0])
        np.testing.assert_allclose(out[name].numpy()[ok], expected[name][ok], atol=ATOL, rtol=0, err_msg=name)


@pytest.mark.parametrize("batch", [1, 8])
def test_half_adder_matches_ngspice(ref, batch):
    numpy_model = ref.GenericSpiceModel(ref.half_adder_topology, ["a", "b"])
    torch_model = s02.GenericSpiceModel(s02.half_adder_topology, ["a", "b"])
    assert torch_model.variables == tuple(numpy_model.all_nodes)
    rng = np.random.default_rng(batch)
    # analog input voltages (not only logic levels) and interventions on internal nodes
    inputs = {"a": rng.uniform(-0.5, 5.5, (batch, 1)), "b": rng.uniform(-0.5, 5.5, (batch, 1))}
    interventions = {"nand1_out": rng.uniform(0.0, 5.0, (batch, 1)), "n_2": rng.uniform(0.0, 5.0, (batch, 1))}
    _compare(torch_model, numpy_model, inputs, interventions)
    # no interventions, inputs only
    _compare(torch_model, numpy_model, inputs, {})
    # internal node supplied as *input*: the original drives it too (clamp), as do we
    _compare(torch_model, numpy_model, {**inputs, "sum": rng.uniform(0.0, 5.0, (batch, 1))}, {})


@pytest.mark.parametrize("batch", [1, 8])
def test_nand_matches_ngspice(ref, batch):
    numpy_model = ref.GenericSpiceModel(ref.nand_topology, ["a", "b"])
    torch_model = s02.GenericSpiceModel(s02.nand_topology, ["a", "b"])
    assert torch_model.variables == tuple(numpy_model.all_nodes)
    rng = np.random.default_rng(100 + batch)
    inputs = {"a": rng.uniform(-0.5, 5.5, (batch, 1)), "b": rng.uniform(-0.5, 5.5, (batch, 1))}
    _compare(torch_model, numpy_model, inputs, {"n_1": rng.uniform(0.0, 5.0, (batch, 1))})
    _compare(torch_model, numpy_model, inputs, {})


def test_ground_and_vdd_interventions_follow_original(ref):
    numpy_model = ref.GenericSpiceModel(ref.half_adder_topology, ["a", "b"])
    torch_model = s02.GenericSpiceModel(s02.half_adder_topology, ["a", "b"])
    iv = {"a": np.array([[2.5]]), "b": np.array([[2.5]]), "0": np.array([[0.7]]), "vdd": np.array([[3.0]])}
    _compare(torch_model, numpy_model, {}, iv)
    out = torch_model.forward_with_interventions({}, {k: torch.as_tensor(v) for k, v in iv.items()})
    assert out["0"].item() == 0.7 and out["vdd"].item() == 5.0


def test_logic_levels_and_high_level_data():
    torch_model = s02.GenericSpiceModel(s02.half_adder_topology, ["a", "b"])
    for a in (0, 1):
        for b in (0, 1):
            out = torch_model.forward_with_interventions(
                {}, {"a": torch.tensor([[5.0 * a]]), "b": torch.tensor([[5.0 * b]])}
            )
            logic = {k: int(out[k].item() > 2.5) for k in ("sum", "carry")}
            assert logic == s02.hl_valid(a, b)
            lo, hi = s02.BIT_RANGES[logic["sum"]]
            assert lo <= out["sum"].item() <= hi


def test_gradients():
    """Every node voltage is differentiable w.r.t. the driven voltages through the implicit
    solve (last Newton step taken with autograd). gradcheck on the NAND gate."""
    torch_model = s02.GenericSpiceModel(s02.nand_topology, ["a", "b"])

    def f(a, b, n1):
        out = torch_model.forward_with_interventions({"a": a, "b": b}, {"n_1": n1})
        return torch.cat([out["output"], out["n_1"]], dim=1)

    a = torch.tensor([[2.3], [3.9]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[3.1], [1.2]], dtype=torch.float64, requires_grad=True)
    n1 = torch.tensor([[0.4], [1.0]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (a, b, n1), eps=1e-6, atol=1e-5, rtol=1e-4)

    def g(a, b):
        return torch_model.forward_with_interventions({"a": a, "b": b}, {})["output"]

    a = torch.tensor([[2.2]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[2.9]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(g, (a, b), eps=1e-6, atol=1e-5, rtol=1e-4)
    out = g(a, b)
    out.sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
