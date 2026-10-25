"""The Stage 1 score: prequential code length of fixed targets and gains over controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from bitsev.alignment.controls import LABELS, Configuration, label_only_learner
from bitsev.alignment.hmc import HmcAlignmentLearner, HmcBudget
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import (
    DEFAULT_BUDGET,
    DEFAULT_LAPLACE_BUDGET,
    GradientAlignmentLearner,
    LaplaceAlignmentLearner,
    LaplaceBudget,
    MixtureAlignmentLearner,
    TrainingBudget,
)
from bitsev.alignment.targets import TargetSet
from bitsev.codes.gain import GainCurve, gain
from bitsev.codes.prequential import CodeLength, prequential_code
from bitsev.codes.schedule import Schedule


@dataclass
class Faithfulness:
    """Log loss and calibration of the final predictor on held-out targets."""

    log_loss: float  # mean nats per pair against the noisy labels
    iia: float  # agreement of the argmax with the clean counterfactual label
    expected_calibration_error: float
    n: int


def faithfulness(predictor, targets: TargetSet, n_bins: int = 10) -> Faithfulness:
    probs = targets.channel.probs(predictor.predicted_probs(targets.pairs).astype(np.float64))
    log_loss = float(-np.mean(predictor.log_prob(targets.pairs, targets.y).astype(np.float64)))
    pred = probs.argmax(1)
    confidence = probs.max(1)
    correct = pred == targets.y
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return Faithfulness(log_loss, float(np.mean(pred == targets.y_clean)), float(ece), len(targets))


@dataclass
class ScoreResult:
    configuration: str
    learner: str
    sees: list[str]
    locus: str
    map_family: str
    targets_hash: str
    schedule: str
    code: CodeLength
    faithfulness: dict[str, Faithfulness] = field(default_factory=dict)
    map_params: int | None = None
    distortion_bound: float | None = None
    extra: dict = field(default_factory=dict)
    final_map: object = field(default=None, repr=False, compare=False)

    def record(self) -> dict:
        """JSON-serialisable record with the raw per-item losses."""
        skip = ("code", "faithfulness", "final_map")
        d = {k: v for k, v in asdict(self).items() if k not in skip}
        d["faithfulness"] = {k: asdict(v) for k, v in self.faithfulness.items()}
        d["boundaries"] = self.code.schedule.boundaries.tolist()
        d["growth"] = self.code.schedule.growth
        d["item_losses"] = self.code.item_losses.tolist()
        d["total_nats"] = self.code.total
        return d


def score_configuration(
    config: Configuration,
    targets: TargetSet,
    x_pool: np.ndarray,
    schedule: Schedule,
    budget: TrainingBudget = DEFAULT_BUDGET,
    warm_start: bool = False,
    seed: int = 0,
    eval_targets: dict[str, TargetSet] | None = None,
    device: str | torch.device = "cpu",
    n_members: int = 1,
    learner_kind: str = "mixture",
    laplace_budget: LaplaceBudget | None = None,
    hmc_budget: HmcBudget | None = None,
    prior_var: float = 1.0,
) -> ScoreResult:
    """Prequential code of ``targets`` under ``config`` with a gradient learner.

    ``eval_targets`` maps split names (for example ``"iid"`` and ``"shifted"``) to held-out
    target sets on which the final map's faithfulness is measured.
    """
    cache = HiddenCache.build(config.network, config.locus, x_pool, device)
    kw = dict(
        cache=cache,
        map_factory=config.make_map,
        budget=budget,
        warm_start=warm_start,
        seed=seed,
        family_name=config.map_family,
        channel=targets.channel,
    )
    if learner_kind == "laplace":
        kw.pop("budget")
        learner = LaplaceAlignmentLearner(budget=laplace_budget or DEFAULT_LAPLACE_BUDGET, **kw)
    elif learner_kind == "hmc":
        kw.pop("budget")
        learner = HmcAlignmentLearner(budget=hmc_budget, prior_var=prior_var, **kw)
    elif n_members > 1:
        learner = MixtureAlignmentLearner(n_members, **kw)
    else:
        learner = GradientAlignmentLearner(**kw)
    code = prequential_code(learner, targets.pairs, targets.y, schedule)
    code.meta = {"targets_hash": targets.content_hash(), "configuration": config.name}
    final = learner.fit(targets.pairs, targets.y)
    faith = {name: faithfulness(final, t) for name, t in (eval_targets or {}).items()}
    final_item_nats = (-final.log_prob(targets.pairs, targets.y)).astype(float).tolist()
    return ScoreResult(
        config.name,
        learner.name,
        sorted(config.sees),
        config.locus,
        config.map_family,
        targets.content_hash(),
        schedule.label(),
        code,
        faith,
        map_params=final.phi.n_params,
        distortion_bound=final.phi.distortion_bound(),
        extra={"final_item_nats": final_item_nats}
        | (
            {"hmc_accept": learner.last_accept, "hmc_step": learner.last_step}
            if hasattr(learner, "last_accept")
            else {}
        ),
        final_map=final.phi,
    )


def label_only_code(targets: TargetSet, schedule: Schedule) -> ScoreResult:
    learner = label_only_learner(targets.n_outputs)
    code = prequential_code(learner, None, targets.y, schedule)
    code.meta = {"targets_hash": targets.content_hash(), "configuration": "label-only"}
    return ScoreResult(
        "label-only",
        learner.name,
        [LABELS],
        "-",
        "-",
        targets.content_hash(),
        schedule.label(),
        code,
    )


def gains(hypothesis: ScoreResult, controls: list[ScoreResult]) -> dict[str, GainCurve]:
    """Gain of the hypothesis over each control on the same targets and schedule."""
    out = {}
    for c in controls:
        if c.targets_hash != hypothesis.targets_hash:
            raise ValueError(f"{c.configuration} was scored on different targets")
        out[c.configuration] = gain(hypothesis.code, c.code)
    return out
