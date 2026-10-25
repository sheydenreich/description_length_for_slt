"""Alignment learners: fit a map on a prefix of targets, predict the next ones.

The gradient learner is the canonical Stage 1 learner: a fresh map (cold start) is trained
by Adam for a fixed budget on the prefix, minimising the cross-entropy of the intervened
network's output against the (noisy) targets. Warm start is a different learner and is
named as such. Both are valid prequential codes; neither is a Bayes code.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from bitsev.alignment.intervene import HiddenCache, intervened_log_probs
from bitsev.alignment.maps import AlignmentMap, Tempered
from bitsev.alignment.targets import NoiseChannel


@dataclass(frozen=True)
class TrainingBudget:
    steps: int = 300
    lr: float = 1e-2
    batch_size: int = 256
    weight_decay: float = 0.0


DEFAULT_BUDGET = TrainingBudget()


@dataclass
class MapPredictor:
    cache: HiddenCache
    phi: AlignmentMap
    channel: NoiseChannel | None = None

    def log_prob(self, pairs: np.ndarray, y: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return intervened_log_probs(self.cache, self.phi, pairs, y, self.channel).cpu().numpy()

    def predicted_probs(self, pairs: np.ndarray) -> np.ndarray:
        from bitsev.alignment.intervene import intervened_logits

        with torch.no_grad():
            return (
                torch.softmax(intervened_logits(self.cache, self.phi, pairs), dim=1).cpu().numpy()
            )


class GradientAlignmentLearner:
    """Train an alignment map on the prefix with a fixed optimisation budget."""

    min_fit_size = 1

    def __init__(
        self,
        cache: HiddenCache,
        map_factory: Callable[[], AlignmentMap],
        budget: TrainingBudget = DEFAULT_BUDGET,
        warm_start: bool = False,
        seed: int = 0,
        family_name: str | None = None,
        channel: NoiseChannel | None = None,
    ):
        self.cache, self.map_factory, self.budget = cache, map_factory, budget
        self.warm_start, self.seed, self.channel = warm_start, seed, channel
        self.family_name = family_name or type(map_factory()).__name__
        start = "warm" if warm_start else "cold"
        self.name = (
            f"grad[{self.family_name},{start},steps={budget.steps},lr={budget.lr:g}"
            f"{_channel_tag(channel)}]"
        )
        self._last: AlignmentMap | None = None
        self._fits = 0

    def _new_map(self) -> AlignmentMap:
        torch.manual_seed(self.seed * 1_000_003 + self._fits)
        return self.map_factory().to(self.cache.states.device)

    def fit(self, pairs: np.ndarray, y: np.ndarray) -> MapPredictor:
        phi = self._last if (self.warm_start and self._last is not None) else self._new_map()
        self._fits += 1
        g = torch.Generator().manual_seed(self.seed * 7_919 + self._fits)
        opt = torch.optim.Adam(
            phi.parameters(), lr=self.budget.lr, weight_decay=self.budget.weight_decay
        )
        n = len(y)
        for _ in range(self.budget.steps):
            idx = torch.randint(0, n, (min(self.budget.batch_size, n),), generator=g).numpy()
            loss = -intervened_log_probs(self.cache, phi, pairs[idx], y[idx], self.channel).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self._last = phi
        return MapPredictor(self.cache, phi, self.channel)


@dataclass
class MixturePredictor:
    """Equal-weight mixture of map predictors: the predictive averages probabilities."""

    members: list[MapPredictor]

    def log_prob(self, pairs: np.ndarray, y: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            lps = torch.stack(
                [intervened_log_probs(m.cache, m.phi, pairs, y, m.channel) for m in self.members]
            )
            return (torch.logsumexp(lps, dim=0) - np.log(len(self.members))).cpu().numpy()

    def predicted_probs(self, pairs: np.ndarray) -> np.ndarray:
        return np.mean([m.predicted_probs(pairs) for m in self.members], axis=0)

    @property
    def phi(self) -> AlignmentMap:
        """The first member, used for size and distortion reporting."""
        return self.members[0].phi


class MixtureAlignmentLearner:
    """``n_members`` cold-start gradient learners whose predictions are averaged.

    A plug-in learner commits to one map per block and is overconfident on small prefixes;
    averaging the predictive distributions of independently initialised maps hedges in the
    way a posterior predictive would, while staying inside the same map family and the
    network's fixed readout. It is a different, named learner, not a correction.
    """

    min_fit_size = 1

    def __init__(self, n_members: int, **kw):
        self.members = [
            GradientAlignmentLearner(**(kw | {"seed": kw.get("seed", 0) * 10_007 + k}))
            for k in range(n_members)
        ]
        base = self.members[0].name
        self.name = f"mixture{n_members}:{base}"

    def fit(self, pairs: np.ndarray, y: np.ndarray) -> MixturePredictor:
        return MixturePredictor([m.fit(pairs, y) for m in self.members])


def _channel_tag(channel: NoiseChannel | None) -> str:
    return f",{channel.label}" if channel is not None else ""


def freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module.eval()


# ----------------------------------------------------------------------------------------
# Laplace-approximate Bayes learner


@dataclass(frozen=True)
class LaplaceBudget:
    """Settings of the MAP fit and the Laplace posterior.

    ``prior_precision`` is the precision of the Gaussian prior on the map parameters; it is
    also what makes the posterior proper along the family's flat (gauge) directions.
    """

    steps: int = 300
    lr: float = 1e-2
    prior_precision: float = 1.0
    n_samples: int = 32  # 0 means the MAP point predictor (no posterior averaging)
    full_hessian_max_params: int = 2000
    fisher_examples: int = 256  # per-example gradients for the diagonal Fisher (subsampled)
    per_example: str = "vmap"  # "vmap" (small networks) or "loop" (hooked HF models)
    sampler: str = "laplace"  # "laplace" (Gaussian at the MAP) or "sgld" (Langevin from the MAP)
    sgld_steps: int = 200  # preconditioned SGLD steps after the MAP; samples from the second half
    sgld_lr: float = 1e-4
    sgld_precondition: bool = False  # RMSprop preconditioning amplifies noise in flat directions
    sgld_scale: str = (
        "map"  # step units: each tensor's scale at the MAP ("map") or at init ("init")
    )
    batch_size: int = 512
    min_epochs: float = 0.0  # Adam: at least this many passes over the prefix (steps grow with n)
    optimizer: str = "lbfgs"  # "lbfgs" (full batch, tight MAP) or "adam" (minibatch)
    chunk: int | None = None  # gradient-accumulation chunk of the full-batch L-BFGS objective


DEFAULT_LAPLACE_BUDGET = LaplaceBudget()


class _Interchange(nn.Module):
    """Wraps a map so ``torch.func.functional_call`` can swap its parameters."""

    def __init__(self, phi: AlignmentMap):
        super().__init__()
        self.phi = phi

    def forward(self, h_base, h_source):
        return self.phi.interchange(h_base, h_source)


def _retract(phi: AlignmentMap) -> None:
    """Project the map back onto its manifold if it defines a retraction (gauge fixing)."""
    retract = getattr(phi, "retract", None)
    if retract is not None:
        retract()


def _param_names(phi: AlignmentMap) -> list[str]:
    return [name for name, p in phi.named_parameters() if p.requires_grad]


def _unflatten(phi: AlignmentMap, vector: torch.Tensor) -> dict[str, torch.Tensor]:
    out, offset = {}, 0
    for name, p in phi.named_parameters():
        if p.requires_grad:
            out["phi." + name] = vector[offset : offset + p.numel()].view_as(p)
            offset += p.numel()
    return out


@dataclass
class LaplacePredictor:
    """Monte Carlo posterior predictive: average over parameter samples of the map."""

    cache: HiddenCache
    phi: AlignmentMap  # holds the MAP parameters
    samples: torch.Tensor  # (S, p) parameter vectors
    channel: NoiseChannel | None = None

    def _log_probs_at(self, vector: torch.Tensor, pairs: np.ndarray, y: np.ndarray) -> torch.Tensor:
        from torch.func import functional_call

        wrapper = _Interchange(self.phi)
        h_base, h_source = self.cache.pair_states(pairs)
        h_new = functional_call(wrapper, _unflatten(self.phi, vector), (h_base, h_source))
        logits = self.phi.scale_logits(self.cache.downstream(h_new, pairs))
        labels = torch.as_tensor(y, device=logits.device)
        log_p = torch.log_softmax(logits, dim=1).gather(1, labels[:, None])[:, 0]
        return self.channel.log_prob(log_p) if self.channel is not None else log_p

    def log_prob(self, pairs: np.ndarray, y: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            lps = torch.stack([self._log_probs_at(v, pairs, y) for v in self.samples])
            return (torch.logsumexp(lps, 0) - np.log(len(self.samples))).cpu().numpy()

    def predicted_probs(self, pairs: np.ndarray) -> np.ndarray:
        from torch.func import functional_call

        wrapper = _Interchange(self.phi)
        h_base, h_source = self.cache.pair_states(pairs)
        with torch.no_grad():
            probs = []
            for v in self.samples:
                h_new = functional_call(wrapper, _unflatten(self.phi, v), (h_base, h_source))
                logits = self.phi.scale_logits(self.cache.downstream(h_new, pairs))
                probs.append(torch.softmax(logits, dim=1))
            return torch.stack(probs).mean(0).cpu().numpy()


class LaplaceAlignmentLearner:
    """MAP fit with a Gaussian prior, Gaussian posterior from the Hessian, MC predictive.

    This is the Bayes code of the map family in the Laplace approximation: the predictive
    distribution averages the network's intervened outputs over maps drawn from
    ``N(theta_MAP, (H + alpha I)^-1)``, where ``H`` is the Hessian of the negative log
    likelihood on the prefix (exact for up to ``full_hessian_max_params`` parameters, the
    empirical-Fisher diagonal above). With a warm start the MAP fit resumes from the previous
    block's solution, which changes the optimiser path but not the definition of the code.
    """

    min_fit_size = 1

    def __init__(
        self,
        cache: HiddenCache,
        map_factory: Callable[[], AlignmentMap],
        budget: LaplaceBudget = DEFAULT_LAPLACE_BUDGET,
        warm_start: bool = True,
        seed: int = 0,
        family_name: str | None = None,
        channel: NoiseChannel | None = None,
    ):
        self.cache, self.map_factory, self.budget = cache, map_factory, budget
        self.warm_start, self.seed, self.channel = warm_start, seed, channel
        self.family_name = family_name or type(map_factory()).__name__
        self.name = (
            f"laplace[{self.family_name},alpha={budget.prior_precision:g},S={budget.n_samples},"
            f"{budget.optimizer},steps={budget.steps},{'warm' if warm_start else 'cold'}"
            f"{',sgld' if budget.sampler == 'sgld' and budget.n_samples else ''}"
            f"{f',epochs>={budget.min_epochs:g}' if budget.min_epochs else ''}"
            f"{_channel_tag(channel)}]"
        )
        self._last: AlignmentMap | None = None
        self._fits = 0

    def _adam_map(self, phi, params, pairs, y, n, steps, lr) -> None:
        """MAP fit with Adam on the mean loss; the prior enters as decoupled weight decay.

        Fed through Adam's normalisation, a negligible prior gradient dominates the near-zero
        data gradient of a saturated readout and shrinks the map to zero within tens of steps.
        """
        b = self.budget
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=b.prior_precision / n)
        g = torch.Generator().manual_seed(self.seed * 7_919 + self._fits + 101)
        for _ in range(steps):
            idx = torch.randint(0, n, (min(b.batch_size, n),), generator=g).numpy()
            nll = -intervened_log_probs(self.cache, phi, pairs[idx], y[idx], self.channel).mean()
            opt.zero_grad()
            nll.backward()
            opt.step()
            _retract(phi)

    def _adam_steps(self, n: int) -> int:
        """``budget.steps``, raised to ``min_epochs`` passes over the ``n`` targets: a fixed step
        count leaves the fits on large prefixes under-optimised."""
        b = self.budget
        return max(b.steps, math.ceil(b.min_epochs * n / min(b.batch_size, n)))

    def _nll(self, phi: AlignmentMap, pairs: np.ndarray, y: np.ndarray) -> torch.Tensor:
        return -intervened_log_probs(self.cache, phi, pairs, y, self.channel).sum()

    def _observed(self, log_p: torch.Tensor) -> torch.Tensor:
        return self.channel.log_prob(log_p) if self.channel is not None else log_p

    def fit(self, pairs: np.ndarray, y: np.ndarray) -> LaplacePredictor:
        from torch.func import functional_call, hessian
        from torch.nn.utils import parameters_to_vector

        b = self.budget
        if self.warm_start and self._last is not None:
            phi = self._last
        else:
            torch.manual_seed(self.seed * 1_000_003 + self._fits)
            phi = self.map_factory().to(self.cache.states.device)
        self._fits += 1
        if b.n_samples and isinstance(phi, Tempered):
            # the predictor reads the logit scale from the live map, not from each sample
            raise NotImplementedError("Tempered maps support the MAP code only (n_samples = 0)")
        params = [p for p in phi.parameters() if p.requires_grad]
        n = len(y)

        def chunks():
            step = b.chunk or n
            for i in range(0, n, step):
                yield pairs[i : i + step], y[i : i + step]

        def prior_term() -> torch.Tensor:
            return 0.5 * b.prior_precision * sum((p**2).sum() for p in params)

        def objective() -> torch.Tensor:
            nll = sum(
                -intervened_log_probs(self.cache, phi, pc, yc, self.channel).sum()
                for pc, yc in chunks()
            )
            return nll + prior_term()

        if b.optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                params,
                lr=1.0,
                max_iter=b.steps,
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
                history_size=20,
                line_search_fn="strong_wolfe",
            )

            def closure():
                # accumulated chunk by chunk so that large networks never hold the activations
                # of the whole target set at once
                opt.zero_grad()
                total = prior_term()
                total.backward()
                total = total.detach()
                for pc, yc in chunks():
                    part = -intervened_log_probs(self.cache, phi, pc, yc, self.channel).sum()
                    part.backward()
                    total = total + part.detach()
                return total

            before = [q.detach().clone() for q in params]
            failed = False
            try:
                opt.step(closure)
            except RuntimeError as err:  # L-BFGS line search overflowed
                if "overflow" not in str(err):
                    raise
                failed = True
            with torch.no_grad():
                if not failed and any(not torch.isfinite(q).all() for q in params):
                    failed = True  # silent NaN/inf from the line search (seen on EPYC)
                if not failed:
                    failed = not torch.isfinite(objective()).all()
                if failed:
                    for q, q0 in zip(params, before, strict=True):
                        q.copy_(q0)
            if failed:  # fall back to Adam from the block's starting point
                self._adam_map(phi, params, pairs, y, n, steps=self._adam_steps(n) * 4, lr=b.lr)
        else:
            self._adam_map(phi, params, pairs, y, n, steps=self._adam_steps(n), lr=b.lr)
        self._last = phi
        _retract(phi)
        theta = parameters_to_vector(params).detach()
        p = theta.numel()
        wrapper = _Interchange(phi)
        h_base, h_source = self.cache.pair_states(pairs)
        labels = torch.as_tensor(y, device=theta.device)

        def total_nll(vector: torch.Tensor) -> torch.Tensor:
            h_new = functional_call(wrapper, _unflatten(phi, vector), (h_base, h_source))
            logits = phi.scale_logits(self.cache.downstream(h_new, pairs))
            log_p = torch.log_softmax(logits, dim=1).gather(1, labels[:, None])[:, 0]
            return -self._observed(log_p).sum()

        alpha = b.prior_precision
        gen = torch.Generator(device="cpu").manual_seed(self.seed * 31 + self._fits)
        if b.n_samples == 0:  # MAP point predictor, for diagnostics
            samples = theta[None]
        elif b.sampler == "sgld":
            samples = self._sgld_samples(phi, params, theta, pairs, y, gen)
        elif p <= b.full_hessian_max_params:
            h = hessian(total_nll)(theta).detach()
            precision = 0.5 * (h + h.T) + alpha * torch.eye(p, dtype=h.dtype)
            evals, evecs = torch.linalg.eigh(precision)
            evals = evals.clamp_min(alpha)  # negative curvature cannot widen the posterior
            scale = evecs * evals.rsqrt()  # columns: eigvec / sqrt(eigval)
            z = torch.randn(b.n_samples, p, generator=gen).to(theta)
            samples = theta[None] + z @ scale.T
        else:
            diag = self._fisher_diagonal(phi, params, theta, pairs, y, gen) + alpha
            z = torch.randn(b.n_samples, p, generator=gen).to(theta)
            samples = theta[None] + z * diag.rsqrt()[None]
        return LaplacePredictor(self.cache, phi, samples.detach(), self.channel)

    def _sgld_samples(self, phi, params, theta, pairs, y, gen) -> torch.Tensor:
        """SGLD started at the MAP (optionally RMSprop-preconditioned, Li et al. 2016).

        The chain targets exp(-(n * mean NLL + prior)); the readout's saturation keeps it
        in the basin where the prefix constrains the map, and it wanders only along the
        directions the prefix does not constrain, which is where a posterior predictive hedges.
        ``n_samples`` parameter vectors are taken evenly from the second half of the chain and
        the MAP is restored in ``phi`` afterwards.
        """
        b = self.budget
        n = len(y)
        burn = b.sgld_steps // 2
        every = max(1, (b.sgld_steps - burn) // b.n_samples)
        v = [torch.zeros_like(q) for q in params]
        # the step is taken in units of each tensor's initialisation scale, so that one step
        # size means the same relative perturbation for a 260-wide and a 768-wide frame
        scales = (
            self._map_scales(params) if b.sgld_scale == "map" else self._init_scales(phi, params)
        )
        noise_gen = torch.Generator(device=theta.device).manual_seed(int(gen.initial_seed()) + 1)
        samples = []
        for step in range(b.sgld_steps):
            idx = torch.randint(0, n, (min(b.batch_size, n),), generator=gen).numpy()
            nll = -intervened_log_probs(self.cache, phi, pairs[idx], y[idx], self.channel).mean()
            energy = nll + 0.5 * b.prior_precision * sum((q**2).sum() for q in params) / n
            grads = [  # gradient of n * mean energy, scaled after the backward pass
                None if g is None else g * n
                for g in torch.autograd.grad(energy, params, allow_unused=True)
            ]
            with torch.no_grad():
                for q, g, vv, sc in zip(params, grads, v, scales, strict=True):
                    if g is None:
                        continue
                    if b.sgld_precondition:
                        vv.mul_(0.99).addcmul_(g, g, value=0.01)
                        pre = 1.0 / (vv.sqrt() + 1e-8)
                    else:
                        pre = torch.ones((), device=q.device, dtype=q.dtype)
                    noise = torch.randn(
                        q.shape, generator=noise_gen, device=q.device, dtype=q.dtype
                    )
                    eta = b.sgld_lr * pre * sc**2
                    q.add_(-eta * g + (2 * eta).sqrt() * noise)
                _retract(phi)
            if step >= burn and (step - burn) % every == 0 and len(samples) < b.n_samples:
                samples.append(torch.cat([q.detach().reshape(-1) for q in params]).clone())
        with torch.no_grad():  # restore the MAP
            offset = 0
            for q in params:
                q.copy_(theta[offset : offset + q.numel()].view_as(q))
                offset += q.numel()
        return torch.stack(samples)

    @staticmethod
    def _map_scales(params) -> list[torch.Tensor]:
        """Standard deviation of each parameter tensor at the MAP (floored).

        Measured at initialisation, a coupling map's weights are so small that the relative
        step gives no hedge at all once the fit has grown them a hundredfold.
        """
        out = []
        for q in params:
            sd = float(q.detach().std()) if q.numel() > 1 else float(q.detach().abs().mean())
            out.append(torch.tensor(max(sd, 1e-3), device=q.device, dtype=q.dtype))
        return out

    def _init_scales(self, phi, params) -> list[torch.Tensor]:
        """Standard deviation of each parameter tensor at a fresh initialisation (floored)."""
        torch.manual_seed(self.seed * 1_000_003 + 999_983)
        fresh = self.map_factory()
        _retract(fresh)  # measure the scale on the manifold the map actually lives on
        fresh_params = [q for q in fresh.parameters() if q.requires_grad]
        out = []
        for q, f in zip(params, fresh_params, strict=True):
            sd = float(f.detach().std()) if f.numel() > 1 else float(f.detach().abs().mean())
            out.append(torch.tensor(max(sd, 1e-3), device=q.device, dtype=q.dtype))
        return out

    def _fisher_diagonal(self, phi, params, theta, pairs, y, gen) -> torch.Tensor:
        """Sum of squared per-example gradients at the MAP, on at most ``fisher_examples``
        examples drawn without replacement and rescaled to the prefix size."""

        b = self.budget
        n = len(y)
        m = min(b.fisher_examples, n)
        idx = torch.randperm(n, generator=gen)[:m].numpy() if m < n else np.arange(n)
        sub_pairs, sub_y = pairs[idx], y[idx]
        if b.per_example == "vmap":
            from torch.func import functional_call, grad, vmap

            wrapper = _Interchange(phi)
            h_base, h_source = self.cache.pair_states(sub_pairs)
            labels = torch.as_tensor(sub_y, device=theta.device)

            def single_nll(vector, hb, hs, onehot):
                h_new = functional_call(wrapper, _unflatten(phi, vector), (hb[None], hs[None]))
                logits = self.cache.network.logits_from_hidden(h_new, self.cache.locus)
                return -self._observed((torch.log_softmax(logits, dim=1)[0] * onehot).sum())

            onehots = torch.nn.functional.one_hot(labels, self.cache.network.n_outputs).to(theta)
            grads = vmap(grad(single_nll), in_dims=(None, 0, 0, 0))(
                theta, h_base, h_source, onehots
            )
            total = (grads**2).sum(0)
        else:
            total = torch.zeros_like(theta)
            for i in range(m):
                nll = -intervened_log_probs(
                    self.cache, phi, sub_pairs[i : i + 1], sub_y[i : i + 1], self.channel
                ).sum()
                g = torch.autograd.grad(nll, params, allow_unused=True)
                g = [
                    torch.zeros_like(q) if gi is None else gi
                    for q, gi in zip(params, g, strict=True)
                ]
                total += torch.cat([gi.reshape(-1) for gi in g]).detach() ** 2
        return total * (n / m)
