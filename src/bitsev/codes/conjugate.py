"""Conjugate families with Bayes and plug-in learners.

Every family is described by its sufficient statistics. Because the statistics add up,
the predictor after any prefix is a function of a cumulative sum, so a whole prequential
schedule is evaluated in one vectorised pass (:class:`BayesLearner` and
:class:`PlugInLearner` implement :class:`~bitsev.codes.learner.SequentialLearner`).

For the Bayes learner the prequential code equals the mixture code exactly (chain rule),
which the tests check against closed-form marginal likelihoods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from bitsev.codes.schedule import Schedule

_LOG_2PI = np.log(2.0 * np.pi)


def _gaussian_log_pdf(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (_LOG_2PI + np.log(var) + (y - mean) ** 2 / var)


def _safe_log(p: np.ndarray) -> np.ndarray:
    """log p with log 0 = -inf and no warning (plug-in predictors can assign zero mass)."""
    with np.errstate(divide="ignore"):
        return np.log(p)


class ConjugateFamily(ABC):
    """A model whose posterior and maximum-likelihood fit depend on additive statistics.

    Subclasses define the statistics and two predictive log densities, each vectorised
    over items: ``stats[i]`` is the cumulative statistic of the prefix used to predict
    item ``i`` and ``count[i]`` the size of that prefix.
    """

    name: str
    plugin_min_fit_size: int = 1

    @abstractmethod
    def sufficient_stats(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-item statistics, shape ``(n, k)``."""

    @abstractmethod
    def bayes_log_prob(
        self, stats: np.ndarray, count: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Posterior-predictive log probability of ``y[i]`` given prefix statistics ``stats[i]``."""

    @abstractmethod
    def plugin_log_prob(
        self, stats: np.ndarray, count: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Log probability of ``y[i]`` under the maximum-likelihood fit to prefix ``i``."""


class GaussianMean(ConjugateFamily):
    """``y ~ N(mu, sigma^2)`` with known ``sigma`` and prior ``mu ~ N(prior_mean, prior_var)``."""

    def __init__(self, sigma: float = 1.0, prior_mean: float = 0.0, prior_var: float = 1.0):
        self.sigma, self.prior_mean, self.prior_var = (
            float(sigma),
            float(prior_mean),
            float(prior_var),
        )
        self.name = f"gaussian_mean(sigma={sigma:g},prior=N({prior_mean:g},{prior_var:g}))"

    def sufficient_stats(self, x, y):
        return np.asarray(y, dtype=float).reshape(-1, 1)

    def posterior(self, total: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance of ``mu`` given a sum ``total`` of ``count`` items."""
        precision = 1.0 / self.prior_var + count / self.sigma**2
        mean = (self.prior_mean / self.prior_var + total / self.sigma**2) / precision
        return mean, 1.0 / precision

    def bayes_log_prob(self, stats, count, x, y):
        mean, var = self.posterior(stats[:, 0], count)
        return _gaussian_log_pdf(y, mean, var + self.sigma**2)

    def plugin_log_prob(self, stats, count, x, y):
        return _gaussian_log_pdf(y, stats[:, 0] / count, self.sigma**2)

    def log_marginal_likelihood(self, y: np.ndarray) -> float:
        """Closed-form ``log p(y)``: a Gaussian with covariance ``sigma^2 I + prior_var 11^T``."""
        y = np.asarray(y, dtype=float) - self.prior_mean
        n = len(y)
        s2, t2 = self.sigma**2, self.prior_var
        # Sherman-Morrison and matrix determinant lemma.
        quad = (y @ y) / s2 - t2 * y.sum() ** 2 / (s2 * (s2 + n * t2))
        logdet = n * np.log(s2) + np.log(1.0 + n * t2 / s2)
        return -0.5 * (n * _LOG_2PI + logdet + quad)


class BetaBernoulli(ConjugateFamily):
    """``y ~ Bernoulli(p)`` with ``p ~ Beta(a, b)``."""

    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a, self.b = float(a), float(b)
        self.name = f"beta_bernoulli(a={a:g},b={b:g})"

    def sufficient_stats(self, x, y):
        return np.asarray(y, dtype=float).reshape(-1, 1)

    @staticmethod
    def _log_prob_given_p(p: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(y == 1, _safe_log(p), _safe_log(1.0 - p))

    def bayes_log_prob(self, stats, count, x, y):
        p = (self.a + stats[:, 0]) / (self.a + self.b + count)
        return self._log_prob_given_p(p, y)

    def plugin_log_prob(self, stats, count, x, y):
        return self._log_prob_given_p(stats[:, 0] / count, y)

    def log_marginal_likelihood(self, y: np.ndarray) -> float:
        from scipy.special import betaln

        y = np.asarray(y)
        s, n = y.sum(), len(y)
        return float(betaln(self.a + s, self.b + n - s) - betaln(self.a, self.b))


class DirichletCategorical(ConjugateFamily):
    """``y in {0, ..., K-1}`` with ``p ~ Dirichlet(alpha)``.

    ``alpha`` is a scalar or an array of length ``n_classes``.
    """

    def __init__(self, n_classes: int, alpha: float | np.ndarray = 1.0):
        self.n_classes = int(n_classes)
        self.alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (self.n_classes,)).copy()
        self.name = f"dirichlet_categorical(K={n_classes},alpha={np.unique(self.alpha).tolist()})"

    def sufficient_stats(self, x, y):
        return np.eye(self.n_classes)[np.asarray(y, dtype=int)]

    def bayes_log_prob(self, stats, count, x, y):
        counts_y = stats[np.arange(len(y)), y]
        return np.log(self.alpha[y] + counts_y) - np.log(self.alpha.sum() + count)

    def plugin_log_prob(self, stats, count, x, y):
        counts_y = stats[np.arange(len(y)), y]
        return _safe_log(counts_y / count)

    def log_marginal_likelihood(self, y: np.ndarray) -> float:
        from scipy.special import gammaln

        counts = np.bincount(np.asarray(y, dtype=int), minlength=self.n_classes)
        a = self.alpha
        return float(
            gammaln(a.sum())
            - gammaln(a.sum() + counts.sum())
            + (gammaln(a + counts) - gammaln(a)).sum()
        )


class BayesLinearRegression(ConjugateFamily):
    """``y ~ N(w . x, sigma^2)`` with known ``sigma`` and prior ``w ~ N(0, prior_var I)``.

    Statistics are the entries of ``x x^T`` and ``x y``; the posterior for each prefix is a
    batched linear solve, so the cost is ``O(n d^3)`` with small ``d``.
    """

    def __init__(self, n_features: int, sigma: float = 1.0, prior_var: float = 1.0):
        self.d, self.sigma, self.prior_var = int(n_features), float(sigma), float(prior_var)
        self.plugin_min_fit_size = self.d
        self.name = f"bayes_linreg(d={n_features},sigma={sigma:g},prior_var={prior_var:g})"

    def sufficient_stats(self, x, y):
        x = np.asarray(x, dtype=float).reshape(len(y), self.d)
        xx = np.einsum("ni,nj->nij", x, x).reshape(len(y), self.d * self.d)
        return np.concatenate([xx, x * np.asarray(y, dtype=float)[:, None]], axis=1)

    def _unpack(self, stats):
        d = self.d
        return stats[:, : d * d].reshape(-1, d, d), stats[:, d * d :]

    def bayes_log_prob(self, stats, count, x, y):
        xtx, xty = self._unpack(stats)
        x = np.asarray(x, dtype=float).reshape(len(y), self.d)
        precision = np.eye(self.d) / self.prior_var + xtx / self.sigma**2
        mean = np.linalg.solve(precision, xty[..., None] / self.sigma**2)[..., 0]
        cov_x = np.linalg.solve(precision, x[..., None])[..., 0]
        var = self.sigma**2 + np.einsum("ni,ni->n", x, cov_x)
        return _gaussian_log_pdf(y, np.einsum("ni,ni->n", x, mean), var)

    def plugin_log_prob(self, stats, count, x, y):
        xtx, xty = self._unpack(stats)
        x = np.asarray(x, dtype=float).reshape(len(y), self.d)
        w = np.einsum("nij,nj->ni", np.linalg.pinv(xtx), xty)
        return _gaussian_log_pdf(y, np.einsum("ni,ni->n", x, w), self.sigma**2)


# ----------------------------------------------------------------------------------------
# Learners


@dataclass
class _FrozenPredictor:
    """A fitted conjugate predictor: fixed statistics broadcast over the items to predict."""

    family: ConjugateFamily
    stats: np.ndarray
    count: int
    bayes: bool

    def log_prob(self, x, y):
        y = np.asarray(y)
        stats = np.broadcast_to(self.stats, (len(y), self.stats.shape[-1]))
        count = np.full(len(y), self.count, dtype=float)
        if self.bayes:
            return self.family.bayes_log_prob(stats, count, x, y)
        return self.family.plugin_log_prob(stats, count, x, y)


class _ConjugateLearner:
    """Shared machinery of the Bayes and plug-in learners of a conjugate family."""

    bayes: bool

    def __init__(self, family: ConjugateFamily):
        self.family = family
        self.min_fit_size = 0 if self.bayes else family.plugin_min_fit_size
        self.name = f"{'bayes' if self.bayes else 'plugin'}:{family.name}"

    def fit(self, x, y):
        y = np.asarray(y)
        if len(y) < self.min_fit_size:
            raise ValueError(f"{self.name} needs at least {self.min_fit_size} items")
        # Summing a (0, k) array gives zeros(k): an empty prefix is the prior.
        stats = self.family.sufficient_stats(x, y).sum(axis=0)
        return _FrozenPredictor(self.family, stats, len(y), self.bayes)

    def prequential_log_probs(self, x, y, schedule: Schedule) -> np.ndarray:
        y = np.asarray(y)
        if schedule.n != len(y):
            raise ValueError("schedule length does not match the data")
        if schedule.start < self.min_fit_size:
            raise ValueError(f"{self.name} needs a schedule starting at >= {self.min_fit_size}")
        per_item = self.family.sufficient_stats(x, y)
        # cumulative[m] = statistics of the first m items, m = 0 .. n
        cumulative = np.vstack([np.zeros((1, per_item.shape[1])), np.cumsum(per_item, axis=0)])
        fit_sizes = schedule.fit_sizes()
        coded = slice(schedule.start, schedule.n)
        x_coded = None if x is None else np.asarray(x)[coded]
        if self.bayes:
            return self.family.bayes_log_prob(cumulative[fit_sizes], fit_sizes, x_coded, y[coded])
        return self.family.plugin_log_prob(cumulative[fit_sizes], fit_sizes, x_coded, y[coded])


class BayesLearner(_ConjugateLearner):
    """Posterior-predictive learner: its prequential code is the mixture code."""

    bayes = True


class PlugInLearner(_ConjugateLearner):
    """Maximum-likelihood learner: a valid code, but not a Bayes code."""

    bayes = False
