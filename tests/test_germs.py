"""Exact singular germs: reductions against brute force, lambda fits, thermometer table."""

import numpy as np
import pytest

from bitsev.exact import germs as G
from bitsev.exact.batching import batching_rows
from bitsev.exact.thermometer import read_germ

POTENTIALS = {
    "x^2+y^2": lambda x, y: x**2 + y**2,
    "x^2+y^4": lambda x, y: x**2 + y**4,
    "x^2y^2": lambda x, y: x**2 * y**2,
    "(ab-0.5)^2": lambda x, y: (x * y - 0.5) ** 2,
}


@pytest.mark.parametrize("name", list(G.STANDARD_GERMS))
def test_reduced_integrals_match_gauss_legendre(name):
    germ = G.STANDARD_GERMS[name]
    brute = G.GaussLegendre2D(POTENTIALS[name], germ.lam, germ.mult, nodes=600)
    for beta in (1.0, 10.0, 100.0):
        assert np.isclose(
            float(germ.log_partition(beta)), float(brute.log_partition(beta)), atol=1e-8
        )
        m1, v1 = germ.gibbs_moments(beta)
        m2, v2 = brute.gibbs_moments(beta)
        assert np.isclose(m1, m2, rtol=1e-7) and np.isclose(v1, v2, rtol=1e-6)


def test_log_partition_keeps_input_shape():
    germ = G.STANDARD_GERMS["x^2y^2"]
    assert germ.log_partition(3.0).shape == ()
    assert germ.log_partition(np.array([1.0, 2.0, 3.0])).shape == (3,)


@pytest.mark.parametrize("name", ["x^2+y^2", "x^2+y^4", "(ab-0.5)^2"])
def test_lambda_fit_recovers_known_value_for_multiplicity_one(name):
    germ = G.STANDARD_GERMS[name]
    fit = G.fit_lambda(germ, 2.0 ** np.arange(8, 15))
    assert abs(fit.lam - germ.lam) < 0.01, fit


def test_multiplicity_two_germ_shows_loglog_bias():
    """x^2 y^2: without the log log n term the slope is biased low; fixing m = 2 overshoots
    slightly at these n. Both are reported, neither is claimed as an estimate of m."""
    germ = G.STANDARD_GERMS["x^2y^2"]
    sizes = 2.0 ** np.arange(8, 15)
    plain = G.fit_lambda(germ, sizes)
    fixed = G.fit_lambda(germ, sizes, fix_mult=2)
    assert plain.lam < germ.lam - 0.05
    assert abs(fixed.lam - germ.lam) < 0.05
    assert plain.lam < fixed.lam


def test_bayes_excess_decays_like_lambda_over_n():
    for germ in G.STANDARD_GERMS.values():
        m = np.array([1e3, 1e4])
        ratio = germ.bayes_excess(m) * m / germ.lam
        tol = 0.25 if germ.mult > 1 else 0.02  # the 1/(m log m) term is visible for m = 2
        np.testing.assert_allclose(ratio, 1.0, rtol=tol)


PLAN_THERMOMETER = {1e-2: (0.93, 0.37), 1e-3: (0.97, 0.40), 1e-4: (0.98, 0.42)}


def test_thermometer_reproduces_plan_appendix_c():
    for T, (t_ratio, lam_hat) in PLAN_THERMOMETER.items():
        r = read_germ(G.STANDARD_GERMS["x^2y^2"], T)
        assert abs(r.t_ratio - t_ratio) < 0.006 and abs(r.lambda_hat - lam_hat) < 0.006, r
    for name in ("x^2+y^4", "x^2+y^2"):
        for T in PLAN_THERMOMETER:
            r = read_germ(G.STANDARD_GERMS[name], T)
            assert np.isclose(r.t_ratio, 1.0, atol=1e-9) and np.isclose(
                r.lambda_hat, r.lam, atol=1e-9
            )


def test_block_codes_of_bayes_germ_follow_batching_factor():
    """Criterion 3 of G-A0 for a multiplicity-one germ: uncorrected slopes scale with
    (r-1)/ln r and corrected slopes return to lambda."""
    germ = G.STANDARD_GERMS["x^2+y^4"]
    rows = batching_rows(
        germ.name, "bayes", germ.bayes_excess, germ.lam, (1.0, 2.0, 4.0), 2**14, (2**6, 2**14)
    )
    for row in rows:
        assert abs(row.error) < 0.05 * row.predicted / row.kappa, row
        assert abs(row.corrected - germ.lam) < 0.03, row
