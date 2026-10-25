"""Jacobian-lens alignment: fixed-frame interchange, final-locus J-vectors, consistency."""

import numpy as np
import torch

from bitsev.alignment import jlens as J


def test_fixed_subspace_interchange_is_the_projection_swap():
    torch.manual_seed(0)
    d, k = 8, 3
    frame, _ = torch.linalg.qr(torch.randn(d, k))
    phi = J.FixedSubspaceMap(frame)
    assert phi.n_params == 0 and phi.d_v == k
    h_b, h_s = torch.randn(5, d), torch.randn(5, d)
    proj = frame @ frame.T
    expected = h_b - h_b @ proj + h_s @ proj
    torch.testing.assert_close(phi.interchange(h_b, h_s), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(phi.inverse(phi(h_b)), h_b, atol=1e-5, rtol=1e-5)


def test_jlens_map_spans_the_unit_vectors_and_truncates_rank():
    torch.manual_seed(1)
    raw = torch.randn(4, 10)
    raw[3] = 2 * raw[0]  # a dependent J-vector
    vec = J.JVectors(raw, "layer1", 1)
    phi = J.jlens_map(vec)
    assert phi.d_v == 3
    basis = phi.basis[:, :3]
    for v in vec.unit:
        residual = v - basis @ (basis.T @ v)
        assert float(residual.norm()) < 1e-5
    assert vec.condition_number >= 1.0


def test_condition_number_of_two_vectors_follows_the_cot_law():
    theta = 0.6
    raw = torch.tensor([[1.0, 0.0], [np.cos(theta), np.sin(theta)]])
    vec = J.JVectors(raw, "layer1", 1)
    assert abs(vec.condition_number - 1 / np.tan(theta / 2)) < 1e-4
