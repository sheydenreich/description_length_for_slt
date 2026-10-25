"""Equivalence of ``s09_grn`` with ``systems/09_grn/grn.py`` (deterministic, exact)."""

import numpy as np
import pytest
import torch

from cae_torch.reference import cae_dir, numpy_forward
from cae_torch.systems import s09_grn as grn
from cae_torch.systems.s09_grn import SegmentPolarityLowLevelModel, SegmentPolarityNetwork


def _ginml_path():
    path = cae_dir() / "systems" / "09_grn" / "regulatoryGraph.ginml"
    if not path.exists():
        pytest.skip(f"{path} not found")
    return str(path)


def _numpy_model(ref):
    return ref.SegmentPolarityModel.from_ginml(_ginml_path())


def test_embedded_network_equals_parsed_file(cae_module):
    ref = cae_module("09_grn/grn.py")
    parsed = SegmentPolarityNetwork.from_ginml(_ginml_path())
    embedded = SegmentPolarityNetwork()
    assert parsed.node_order == embedded.node_order == grn.NODE_ORDER
    assert parsed.nodes == embedded.nodes and parsed.edges == embedded.edges
    # and both agree with the original parser
    node_order, nodes, edges = ref.parse_ginml(_ginml_path())
    assert node_order == grn.NODE_ORDER
    for nid, node in nodes.items():
        name, base, maxv, params = grn.NODES[nid]
        assert (node.name, node.basevalue, node.maxvalue) == (name, base, maxv)
        assert [(set(a), v) for a, v in params] == [(set(a), v) for a, v in node.parameters]
    for eid, edge in edges.items():
        assert grn.EDGES[eid] == (edge.source, edge.target, edge.minvalue, edge.maxvalue, edge.sign)


@pytest.mark.parametrize("batch", [1, 8])
def test_matches_numpy(cae_module, batch):
    ref = cae_module("09_grn/grn.py")
    numpy_model = ref.SegmentPolarityLowLevelModel(_numpy_model(ref))
    torch_model = SegmentPolarityLowLevelModel()
    rng = np.random.default_rng(batch)
    # non-integer levels exercise round-half-to-even and the clamp to [0, 2]
    inputs = {"wg_c2": rng.uniform(-0.6, 2.6, size=(batch, 1))}
    interventions = {"wg_c2": np.concatenate([rng.uniform(-0.6, 2.6, size=(batch - 1, 1)), [[1.5]]]),
                     "fz_c1": rng.uniform(0, 1, size=(batch, 1))}       # ignored by the original
    expected = numpy_forward(numpy_model, inputs, interventions)
    out = torch_model.forward_with_interventions(
        {k: torch.as_tensor(v) for k, v in inputs.items()},
        {k: torch.as_tensor(v) for k, v in interventions.items()},
    )
    assert set(out) == set(torch_model.variables)
    for name in torch_model.variables:
        np.testing.assert_allclose(out[name].numpy(), expected[name], atol=1e-9, rtol=1e-9, err_msg=name)
    # input-only path and defaults
    expected = numpy_forward(numpy_model, inputs, {})
    out = torch_model.forward({k: torch.as_tensor(v) for k, v in inputs.items()})
    for name in torch_model.variables:
        np.testing.assert_allclose(out[name].numpy(), expected[name], atol=1e-9, rtol=1e-9, err_msg=name)


def test_all_wg_levels_and_abstraction(cae_module):
    """The three input levels; Wg2 = 2 switches Fz on in the true neighbours (cells 1, 3) only."""
    ref = cae_module("09_grn/grn.py")
    numpy_model = ref.SegmentPolarityLowLevelModel(_numpy_model(ref))
    levels = np.array([[0.0], [1.0], [2.0]])
    expected = numpy_forward(numpy_model, {"wg_c2": levels}, {})
    out = SegmentPolarityLowLevelModel().forward({"wg_c2": torch.as_tensor(levels)})
    for name in out:
        np.testing.assert_array_equal(out[name].numpy(), expected[name])
    assert out["fz_c1"].tolist() == [[0.0], [0.0], [1.0]] and out["fz_c3"].tolist() == [[0.0], [0.0], [1.0]]
    assert out["fz_c4"].abs().sum() == 0 and out["fz_c5"].abs().sum() == 0
    for b in range(3):
        wg = grn.abstract_wg_src(out["wg_c2"][b])
        assert grn.abstract_fz_tgt(torch.cat([out[v][b] for v in grn.VALID_COARSE_GRAINING["fz_tgt"]])) == grn.valid_rule(wg)
        assert grn.abstract_fz_tgt(torch.cat([out[v][b] for v in grn.WRONG_COARSE_GRAINING["fz_tgt"]])) == 0


def test_general_synchronous_step_matches_numpy(cae_module):
    """The batched full-network step equals the original on random 72-node states."""
    ref = cae_module("09_grn/grn.py")
    model = _numpy_model(ref)
    net = SegmentPolarityNetwork()
    rng = np.random.default_rng(9)
    maxv = np.array([grn.NODES[n][2] for n in grn.NODE_ORDER])
    states = np.stack([rng.integers(0, maxv + 1) for _ in range(16)]).astype(np.int32)
    expected = np.stack([model.synchronous_step(s) for s in states])
    got = net.synchronous_step(torch.as_tensor(states, dtype=torch.float64)).numpy()
    np.testing.assert_array_equal(got, expected)


def test_no_differentiable_path():
    """All operations are integer thresholds / set equalities: the outputs are piecewise
    constant in wg_c2 and the output tensors carry no grad_fn."""
    wg = torch.tensor([[1.7]], dtype=torch.float64, requires_grad=True)
    out = SegmentPolarityLowLevelModel().forward({"wg_c2": wg})
    for name in ("fz_c1", "fz_c3", "fz_c4", "fz_c5"):
        assert out[name].grad_fn is None or torch.autograd.grad(out[name].sum(), wg, allow_unused=True)[0] in (None, 0)
