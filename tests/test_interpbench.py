"""Network-dependent: downloads small InterpBench models (a few hundred kB each)."""

import numpy as np
import pytest
import torch

pytest.importorskip("huggingface_hub")
from bitsev.testbeds import interpbench as IB  # noqa: E402


@pytest.fixture(scope="module")
def metadata():
    try:
        return IB.load_metadata()
    except Exception as exc:  # noqa: BLE001 - no network, skip
        pytest.skip(f"InterpBench metadata unavailable: {exc}")


@pytest.mark.parametrize("case", sorted(IB.VERIFIED))
def test_registered_case_reproduces_its_algorithm(case, metadata):
    ib = IB.load_case(case, metadata)
    acc = IB.task_accuracy(ib, n=400)
    assert acc > 0.97, (case, acc)


def test_loci_compose_and_counterfactuals_are_consistent(metadata):
    ib = IB.load_case("13", metadata)  # loads fine even though its program is unverified
    rng = np.random.default_rng(1)
    x = ib.algorithm.tokens.encode(ib.algorithm.tokens.sample(50, rng))
    xt = ib.model.prepare_input(x)
    for locus in ib.model.loci:
        torch.testing.assert_close(
            ib.model.logits_from_hidden(ib.model.hidden(xt, locus), locus), ib.model.logits(xt)
        )
    y = ib.algorithm.counterfactual(x[:25], x[25:], "next")
    assert y.shape == (25,) and set(np.unique(y)) <= {0, 1, 2}
    assert np.array_equal(ib.algorithm.counterfactual(x, x, "next"), ib.algorithm.output(x))
