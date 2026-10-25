"""Reduced-rank regression as an alignment-map family with a known learning coefficient.

The family is ``y = U V x + noise`` with ``U`` of shape ``(M, H)`` and ``V`` of shape
``(H, N)``, unit Gaussian noise and inputs, Gaussian prior on every entry. Aoyagi and
Watanabe (2005) give its learning coefficient ``lambda`` and multiplicity ``m`` in closed
form (:func:`bitsev.exact.rrr.aoyagi_watanabe`), so three codes of the same data can be
checked against theory:

* the Bayes code (posterior predictive, sampled by Hamiltonian Monte Carlo on the
  sufficient statistics) has regret ``lambda ln n - (m - 1) ln ln n + O(1)``;
* the least-squares plug-in code has the regular coefficient ``M N / 2`` whatever the true
  rank (:func:`bitsev.exact.rrr.ols_kappa`);
* the local learning coefficient estimated by Langevin sampling of the tempered posterior
  around the least-squares point should read ``lambda``.

Everything is a function of ``S_xx = X^T X``, ``S_xy = Y^T X`` and ``sum ||y||^2``, so the
cost per posterior sample does not grow with ``n``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOG_2PI = float(np.log(2 * np.pi))


@dataclass(frozen=True)
class RrrTask:
    """``y = A* x + e`` with ``A*`` of rank ``r`` (unit singular values) and unit noise."""

    M: int
    N: int
    r: int
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.r <= min(self.M, self.N):
            raise ValueError("rank must lie between 0 and min(M, N)")

    @property
    def truth(self) -> np.ndarray:
        rng = np.random.default_rng(1000 + self.seed)
        left = np.linalg.qr(rng.standard_normal((self.M, self.M)))[0][:, : self.r]
        right = np.linalg.qr(rng.standard_normal((self.N, self.N)))[0][:, : self.r]
        return left @ right.T

    def sample(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, self.N))
        y = x @ self.truth.T + rng.standard_normal((n, self.M))
        return x, y

    def true_log_prob(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return gaussian_log_prob(y, x @ self.truth.T)


def gaussian_log_prob(y: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Log density of ``y`` under ``N(mean, I)``, per row."""
    return -0.5 * np.sum((y - mean) ** 2, axis=-1) - 0.5 * y.shape[-1] * LOG_2PI


@dataclass
class Stats:
    """Sufficient statistics of a prefix: ``S_xx``, ``S_xy = Y^T X``, ``sum ||y||^2``, ``n``."""

    sxx: np.ndarray
    sxy: np.ndarray
    syy: float
    n: int

    @classmethod
    def of(cls, x: np.ndarray, y: np.ndarray) -> Stats:
        return cls(x.T @ x, y.T @ x, float(np.sum(y * y)), len(y))

    def sum_sq_residual(self, a: np.ndarray) -> np.ndarray:
        """``sum_i ||y_i - a x_i||^2`` for a batch of matrices ``a`` of shape ``(..., M, N)``."""
        quad = np.einsum("...mn,nk,...mk->...", a, self.sxx, a)
        cross = np.einsum("...mn,mn->...", a, self.sxy)
        return quad - 2 * cross + self.syy

    def ols(self) -> np.ndarray:
        return np.linalg.solve(self.sxx, self.sxy.T).T


# Learners ------------------------------------------------------------------------------


class LinearPredictor:
    def __init__(self, a: np.ndarray):
        self.a = a

    def log_prob(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return gaussian_log_prob(y, x @ self.a.T)


class OlsLearner:
    """Least squares on the full ``M x N`` matrix: the plug-in code of the family when
    ``H >= min(M, N)``."""

    name = "ols"

    def __init__(self, M: int, N: int):
        self.M, self.N = M, N
        self.min_fit_size = N + 2

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearPredictor:
        return LinearPredictor(Stats.of(x, y).ols())


class MixturePredictor:
    """Monte-Carlo posterior predictive: the mean of ``N(A_s x, I)`` over samples ``A_s``."""

    def __init__(self, samples: np.ndarray, chunk: int = 512):
        self.samples = samples  # (S, M, N)
        self.chunk = chunk

    def log_prob(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty(len(y))
        log_s = np.log(len(self.samples))
        for i in range(0, len(y), self.chunk):
            xs, ys = x[i : i + self.chunk], y[i : i + self.chunk]
            means = np.einsum("smn,bn->sbm", self.samples, xs)
            lp = gaussian_log_prob(ys[None], means)  # (S, b)
            top = lp.max(axis=0)
            out[i : i + self.chunk] = top + np.log(np.exp(lp - top).sum(axis=0)) - log_s
        return out


@dataclass
class HmcSettings:
    chains: int = 64
    warmup: int = 300
    samples: int = 200
    thin: int = 2
    leapfrog: int = 20
    step: float = 0.05
    target_accept: float = 0.7


class RrrBayesLearner:
    """Bayes code of ``y = U V x + e`` with prior ``N(0, prior_var)`` on every entry.

    The posterior given a prefix is sampled by Hamiltonian Monte Carlo on ``(U, V)`` using
    the sufficient statistics; the predictor is the sample average of the Gaussian
    likelihoods. Before any data the predictor is the prior predictive (samples from the
    prior).
    """

    min_fit_size = 0

    def __init__(
        self,
        M: int,
        N: int,
        H: int,
        prior_var: float = 1.0,
        settings: HmcSettings | None = None,
        seed: int = 0,
    ):
        self.M, self.N, self.H = M, N, H
        self.prior_var = prior_var
        self.settings = settings or HmcSettings()
        self.seed = seed
        self.name = f"bayes[H={H},hmc]"
        self.last_accept = float("nan")

    def _split(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mh = self.M * self.H
        u = w[..., :mh].reshape(*w.shape[:-1], self.M, self.H)
        v = w[..., mh:].reshape(*w.shape[:-1], self.H, self.N)
        return u, v

    def energy_grad(self, w: np.ndarray, stats: Stats | None) -> tuple[np.ndarray, np.ndarray]:
        """Negative log posterior (up to a constant) and its gradient, batched over chains."""
        u, v = self._split(w)
        a = u @ v
        energy = 0.5 * np.sum(w * w, axis=-1) / self.prior_var
        grad_w = w / self.prior_var
        if stats is not None and stats.n > 0:
            energy = energy + 0.5 * stats.sum_sq_residual(a)
            grad_a = a @ stats.sxx - stats.sxy  # (..., M, N)
            grad_u = grad_a @ np.swapaxes(v, -1, -2)
            grad_v = np.swapaxes(u, -1, -2) @ grad_a
            grad_w = grad_w + np.concatenate(
                [grad_u.reshape(*w.shape[:-1], -1), grad_v.reshape(*w.shape[:-1], -1)], axis=-1
            )
        return energy, grad_w

    def sample_posterior(self, stats: Stats | None, rng: np.random.Generator) -> np.ndarray:
        """Posterior samples of ``A = U V``, shape ``(S, M, N)``; sets ``last_accept``."""
        s = self.settings
        dim = self.M * self.H + self.H * self.N
        w = rng.standard_normal((s.chains, dim)) * np.sqrt(self.prior_var)
        if stats is not None and stats.n > self.N + 1:
            # start the chains near the least-squares matrix via a balanced factorisation
            a0 = stats.ols()
            left, sing, right = np.linalg.svd(a0, full_matrices=False)
            k = min(self.H, len(sing))
            u0 = np.zeros((self.M, self.H))
            v0 = np.zeros((self.H, self.N))
            u0[:, :k] = left[:, :k] * np.sqrt(sing[:k])
            v0[:k] = np.sqrt(sing[:k])[:, None] * right[:k]
            w = np.concatenate([u0.ravel(), v0.ravel()])[None] + 0.1 * w
        step = s.step
        if stats is not None and stats.n > 0:
            # keep the first leapfrog steps stable: the curvature in A is about the largest
            # eigenvalue of S_xx, and the factorisation can double it
            step = min(step, 0.5 / np.sqrt(float(np.linalg.eigvalsh(stats.sxx).max())))
        energy, grad = self.energy_grad(w, stats)
        kept = []
        accepts = []
        for it in range(s.warmup + s.samples * s.thin):
            p = rng.standard_normal(w.shape)
            h0 = energy + 0.5 * np.sum(p * p, axis=-1)
            w_new, p_new = w.copy(), p - 0.5 * step * grad
            with np.errstate(over="ignore", invalid="ignore"):
                for leap in range(s.leapfrog):
                    w_new = w_new + step * p_new
                    e_new, g_new = self.energy_grad(w_new, stats)
                    p_new = p_new - (step if leap < s.leapfrog - 1 else 0.5 * step) * g_new
                h1 = e_new + 0.5 * np.sum(p_new * p_new, axis=-1)
            log_alpha = np.minimum(0.0, h0 - h1)
            log_alpha = np.where(np.isfinite(log_alpha), log_alpha, -np.inf)
            accept = np.log(rng.uniform(size=len(w))) < log_alpha
            w = np.where(accept[:, None], w_new, w)
            energy = np.where(accept, e_new, energy)
            grad = np.where(accept[:, None], g_new, grad)
            rate = float(accept.mean())
            if it < s.warmup:
                step *= 1.05 if rate > s.target_accept else 0.9
            else:
                accepts.append(rate)
                if (it - s.warmup) % s.thin == 0:
                    kept.append(w.copy())
        self.last_accept = float(np.mean(accepts))
        u, v = self._split(np.concatenate(kept))
        return u @ v

    def fit(self, x: np.ndarray, y: np.ndarray) -> MixturePredictor:
        stats = Stats.of(x, y) if len(y) > 0 else None
        rng = np.random.default_rng(self.seed * 7919 + len(y))
        return MixturePredictor(self.sample_posterior(stats, rng))


def map_point(
    learner: RrrBayesLearner, stats: Stats, rng: np.random.Generator, restarts: int = 8
) -> np.ndarray:
    """MAP of ``(U, V)`` under the learner's prior by gradient descent from several starts."""
    dim = learner.M * learner.H + learner.H * learner.N
    w = rng.standard_normal((restarts, dim)) * 0.5
    lr = 0.5 / max(float(np.linalg.eigvalsh(stats.sxx).max()), 1.0)
    for _ in range(2000):
        energy, grad = learner.energy_grad(w, stats)
        w = w - lr * grad
    energy, _ = learner.energy_grad(w, stats)
    return w[int(np.argmin(energy))]


def local_learning_coefficient(
    learner: RrrBayesLearner,
    stats: Stats,
    w_star: np.ndarray,
    rng: np.random.Generator,
    chains: int = 32,
    steps: int = 4000,
    step: float = 1e-4,
    gamma: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float]:
    """Langevin estimate ``n beta (E[L_n] - L_n(w*))`` of the local learning coefficient at
    ``w*`` (Lau et al. 2023), with localisation ``gamma ||w - w*||^2 / 2`` and
    ``beta = beta0 / ln n``. Returns the estimate and its standard error over chains. At
    step 1e-4 and ``gamma`` 1 the estimate matches Aoyagi–Watanabe within 5% on shapes up to
    (3, 3, 3) at n = 4096; a step of 1e-3 overshoots by a factor of 2-3."""
    n = stats.n
    beta = beta0 / np.log(n)
    like = RrrBayesLearner(learner.M, learner.N, learner.H, prior_var=np.inf)

    def loss(w):  # mean negative log likelihood up to a constant
        u, v = like._split(w)
        return 0.5 * stats.sum_sq_residual(u @ v) / n

    base = loss(w_star[None])[0]
    w = np.repeat(w_star[None], chains, axis=0)
    traces = np.zeros(chains)
    count = 0
    for it in range(steps):
        _, g = like.energy_grad(w, stats)  # gradient of n * loss
        drift = beta * g + gamma * (w - w_star)
        w = w - step * drift + np.sqrt(2 * step) * rng.standard_normal(w.shape)
        if it >= steps // 2:
            traces += n * beta * (loss(w) - base)
            count += 1
    lam = traces / count
    return float(lam.mean()), float(lam.std(ddof=1) / np.sqrt(chains))
