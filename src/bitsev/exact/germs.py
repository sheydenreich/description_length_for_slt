"""Exactly integrable singular germs in two parameters.

For a population excess loss ``K(w) >= 0`` with ``min K = 0`` and a prior ``phi`` on a box,
the idealised Bayes free energy

    F(n) = -log  integral  exp(-n K(w)) phi(w) dw

has Watanabe's expansion ``lambda log n - (m - 1) log log n + c + o(1)``. Its increments
``F(n + 1) - F(n)`` are the expected excess of the Bayes predictive at sample size ``n`` in
this idealisation, which is what block schedules sum. Tempering ``exp(-K / T)`` gives the
Gibbs law behind the loss-fluctuation thermometer.

Every germ here reduces to one-dimensional integrals with closed-form inner factors, so
values are exact to quadrature tolerance at any ``n``. A generic Gauss-Legendre
two-dimensional integrator is provided for cross-checks at moderate ``n``.

Germs (uniform prior on ``[-1, 1]^2`` unless stated):

================  =======  ===  ========================================
K(x, y)           lambda   m    reduction
================  =======  ===  ========================================
x^2 + y^2         1        1    product of two monomial factors
x^2 + y^4         3/4      1    product of two monomial factors
x^2 y^2           1/2      2    (ab - c)^2 with c = 0
(ab - c)^2, c!=0  1/2      1    inner Gaussian integral in b, outer quad
================  =======  ===  ========================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import integrate, special

SQRT_PI = np.sqrt(np.pi)


class Germ(ABC):
    """A two-parameter potential with exact (log) partition functions and Gibbs moments."""

    name: str
    lam: float
    mult: int

    @abstractmethod
    def log_partition(self, beta: np.ndarray) -> np.ndarray:
        """``log integral exp(-beta K) phi`` for each inverse temperature (or sample size).

        Returns an array of the same shape as ``beta`` (0-d for a scalar).
        """

    @abstractmethod
    def gibbs_moments(self, beta: float) -> tuple[float, float]:
        """Mean and variance of ``K`` under the Gibbs law ``exp(-beta K) phi / Z``."""

    # Derived quantities -----------------------------------------------------------------

    def free_energy(self, n: np.ndarray) -> np.ndarray:
        """``F(n) = -log Z(n)``; the optimal loss is zero so this is the whole redundancy."""
        return -self.log_partition(np.asarray(n, dtype=float))

    def bayes_excess(self, m: np.ndarray) -> np.ndarray:
        """``F(m + 1) - F(m)``: expected excess of the Bayes predictive after ``m`` items."""
        m = np.asarray(m, dtype=float)
        # F(m) and F(m + 1) overlap on consecutive m: evaluate each distinct size once.
        sizes, inverse = np.unique(
            np.concatenate([m.ravel(), m.ravel() + 1.0]), return_inverse=True
        )
        f = self.free_energy(sizes)
        k = m.size
        return (f[inverse[k:]] - f[inverse[:k]]).reshape(m.shape)

    def asymptotic_free_energy(self, n: np.ndarray, constant: float = 0.0) -> np.ndarray:
        n = np.asarray(n, dtype=float)
        return self.lam * np.log(n) - (self.mult - 1) * np.log(np.log(n)) + constant


# ----------------------------------------------------------------------------------------
# One-dimensional factors


def _monomial_log_integral(beta: np.ndarray, power: int) -> np.ndarray:
    """``log integral_{-1}^{1} exp(-beta |x|^p) dx``, equal to
    ``log[(2/p) beta^{-1/p} gamma_lower(1/p, beta)]``."""
    a = 1.0 / power
    # gammainc is the regularised lower incomplete gamma P(a, beta) = gamma(a, beta) / Gamma(a)
    return (
        np.log(2.0 / power)
        - a * np.log(beta)
        + special.gammaln(a)
        + np.log(special.gammainc(a, beta))
    )


def _monomial_moments(beta: float, power: int) -> tuple[float, float]:
    """Mean and variance of ``|x|^p`` under ``exp(-beta |x|^p)`` on ``[-1, 1]``.

    With ``u = beta x^p`` the moments are ratios of incomplete gamma functions:
    ``E[x^{pk}] = beta^{-k} gamma(a + k, beta) / gamma(a, beta)`` with ``a = 1/p``.
    """
    a = 1.0 / power

    def lower_gamma(s: float) -> float:
        return float(special.gamma(s) * special.gammainc(s, beta))

    g0, g1, g2 = lower_gamma(a), lower_gamma(a + 1), lower_gamma(a + 2)
    mean = g1 / (beta * g0)
    second = g2 / (beta**2 * g0)
    return mean, second - mean**2


def _gaussian_moment_integral(k: int, beta: float, lo: float, hi: float) -> float:
    """``integral_lo^hi t^k exp(-beta t^2) dt`` for ``k`` in {0, 2, 4}."""
    s = np.sqrt(beta)

    def antiderivative(t: float) -> float:
        erf_term = special.erf(s * t)
        e = np.exp(-beta * t * t)
        if k == 0:
            return SQRT_PI / (2 * s) * erf_term
        if k == 2:
            return SQRT_PI / (4 * beta * s) * erf_term - t * e / (2 * beta)
        if k == 4:
            return 3 * SQRT_PI / (8 * beta**2 * s) * erf_term - e * (
                t**3 / (2 * beta) + 3 * t / (4 * beta**2)
            )
        raise ValueError("k must be 0, 2 or 4")

    return antiderivative(hi) - antiderivative(lo)


# ----------------------------------------------------------------------------------------
# Germs


class SeparableMonomial(Germ):
    """``K = |x|^p + |y|^q`` on ``[-1, 1]^2`` with uniform prior; ``lambda = 1/p + 1/q``."""

    def __init__(self, p: int, q: int):
        self.p, self.q = int(p), int(q)
        self.lam, self.mult = 1.0 / p + 1.0 / q, 1
        self.name = f"x^{p}+y^{q}"

    def log_partition(self, beta):
        beta = np.asarray(beta, dtype=float)
        # the uniform prior density on the box is 1/4
        return (
            _monomial_log_integral(beta, self.p)
            + _monomial_log_integral(beta, self.q)
            - np.log(4.0)
        )

    def gibbs_moments(self, beta):
        mx, vx = _monomial_moments(beta, self.p)
        my, vy = _monomial_moments(beta, self.q)
        return mx + my, vx + vy


class ProductSquare(Germ):
    """``K = (ab - c)^2`` on ``[-1, 1]^2`` with uniform prior.

    For fixed ``a`` the inner integral over ``b`` is Gaussian in ``t = ab - c`` on
    ``[-|a| - c, |a| - c]`` with Jacobian ``1/|a|``; the outer integral is one-dimensional.
    ``c = 0`` is the germ ``x^2 y^2`` with ``lambda = 1/2, m = 2``; ``c != 0`` has
    ``lambda = 1/2, m = 1`` (the zero set is a smooth curve).
    """

    def __init__(self, c: float = 0.0, quad_limit: int = 400):
        self.c = float(c)
        self.lam, self.mult = 0.5, 2 if c == 0 else 1
        self.name = "x^2y^2" if c == 0 else f"(ab-{c:g})^2"
        self.quad_limit = quad_limit

    def _outer(self, beta: float, k: int) -> float:
        """``integral_{-1}^{1} da (1/|a|) integral t^k exp(-beta t^2) dt`` (symmetric in a)."""

        def integrand(a: float) -> float:
            if a == 0.0:
                a = 1e-300
            return _gaussian_moment_integral(k, beta, -a - self.c, a - self.c) / a

        points = [self.c] if 0 < self.c < 1 else None
        value, _ = integrate.quad(
            integrand, 0.0, 1.0, points=points, limit=self.quad_limit, epsabs=0, epsrel=1e-11
        )
        return 2.0 * value

    def log_partition(self, beta):
        beta = np.asarray(beta, dtype=float)
        out = np.array([np.log(self._outer(b, 0)) for b in beta.ravel()]) - np.log(4.0)
        return out.reshape(beta.shape)

    def gibbs_moments(self, beta):
        z0, z2, z4 = (self._outer(beta, k) for k in (0, 2, 4))
        mean = z2 / z0
        return mean, z4 / z0 - mean**2


class GaussLegendre2D(Germ):
    """Generic tensor-product Gauss-Legendre integrator on ``[-1, 1]^2`` for cross-checks.

    Accurate only while ``exp(-beta K)`` is resolved by the grid (moderate ``beta``).
    """

    def __init__(self, potential, lam: float, mult: int, nodes: int = 400, name: str = "generic"):
        self.potential, self.lam, self.mult, self.name = potential, lam, mult, name
        x, w = np.polynomial.legendre.leggauss(nodes)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        self.k_values = potential(xx, yy).ravel()
        self.weights = np.outer(w, w).ravel() / 4.0  # uniform prior density 1/4

    def log_partition(self, beta):
        beta = np.asarray(beta, dtype=float)
        log_terms = -beta.ravel()[:, None] * self.k_values[None, :] + np.log(self.weights)[None, :]
        return special.logsumexp(log_terms, axis=1).reshape(beta.shape)

    def gibbs_moments(self, beta):
        log_w = -beta * self.k_values + np.log(self.weights)
        p = np.exp(log_w - special.logsumexp(log_w))
        mean = float(p @ self.k_values)
        return mean, float(p @ self.k_values**2 - mean**2)


# ----------------------------------------------------------------------------------------
# Learning-coefficient fits


@dataclass
class LambdaFit:
    lam: float
    stderr: float
    mult_fixed: int | None
    n_range: tuple[int, int]


def fit_lambda(germ: Germ, sizes: np.ndarray, fix_mult: int | None = None) -> LambdaFit:
    """Fit ``F(n) ~ c + lambda log n`` after optionally removing a known ``(m-1) log log n``."""
    from bitsev.codes.gain import fit_log_slope

    sizes = np.asarray(sizes)
    f = germ.free_energy(sizes)
    if fix_mult is not None:
        f = f + (fix_mult - 1) * np.log(np.log(sizes))
    fit = fit_log_slope(sizes, f)
    return LambdaFit(fit.slope, fit.slope_stderr, fix_mult, fit.n_range)


STANDARD_GERMS: dict[str, Germ] = {
    "x^2+y^2": SeparableMonomial(2, 2),
    "x^2+y^4": SeparableMonomial(2, 4),
    "x^2y^2": ProductSquare(0.0),
    "(ab-0.5)^2": ProductSquare(0.5),
}
