import numpy as np

from bitsev.codes.gain import log_term_vif, variance_inflation
from bitsev.exact.identifiability import vif_over_decades


def test_vif_of_orthogonal_regressors_is_one(rng):
    x = rng.normal(size=(200, 2))
    x -= x.mean(axis=0)  # the VIF is computed after removing a constant
    x[:, 1] -= x[:, 0] * (x[:, 0] @ x[:, 1]) / (x[:, 0] @ x[:, 0])
    np.testing.assert_allclose(variance_inflation(x), 1.0, atol=1e-10)


def test_log_terms_are_nearly_collinear_over_a_decade():
    rows = vif_over_decades(decades=(1, 2, 3))
    assert all(r.vif > 100 for r in rows), [r.vif for r in rows]
    assert all(r.stderr_inflation > 10 for r in rows)


def test_log_term_vif_matches_two_regressor_formula():
    n = np.logspace(2, 3, 21)
    a, b = np.log(n) / n, 1 / n
    rho = np.corrcoef(a, b)[0, 1]
    assert np.isclose(log_term_vif(n), 1 / (1 - rho**2))
