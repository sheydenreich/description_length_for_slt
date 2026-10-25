"""Configurations and their controls, with information access made explicit.

A configuration is scored only against controls that differ from it in one respect, and
every learner declares what it sees. The label-only code sees neither the network nor
the algorithm, so it normalises for the difficulty of the targets alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from bitsev.alignment.maps import AlignmentMap
from bitsev.alignment.network import Network
from bitsev.codes.conjugate import BayesLearner, DirichletCategorical

NETWORK, LOCUS, LABELS, ALGORITHM = "network", "locus", "labels", "algorithm"
STANDARD_ACCESS = frozenset({NETWORK, LOCUS, LABELS})


@dataclass
class Configuration:
    """Network, locus and map family; the learner is attached at scoring time."""

    name: str
    network: Network
    locus: str
    map_family: str
    map_factory: Callable[[int], AlignmentMap]  # width -> map
    sees: frozenset[str] = STANDARD_ACCESS
    differs_from_hypothesis_in: str | None = None
    notes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if ALGORITHM in self.sees:
            raise ValueError("no Stage 1 learner may see the algorithm")

    def make_map(self) -> AlignmentMap:
        return self.map_factory(self.network.width(self.locus))


def untrained_control(hypothesis: Configuration, untrained: Network) -> Configuration:
    return Configuration(
        f"{hypothesis.name}/untrained",
        untrained,
        hypothesis.locus,
        hypothesis.map_family,
        hypothesis.map_factory,
        hypothesis.sees,
        differs_from_hypothesis_in="network",
    )


def wrong_locus_control(hypothesis: Configuration, locus: str) -> Configuration:
    return Configuration(
        f"{hypothesis.name}/locus={locus}",
        hypothesis.network,
        locus,
        hypothesis.map_family,
        hypothesis.map_factory,
        hypothesis.sees,
        differs_from_hypothesis_in="locus",
    )


def other_map_control(
    hypothesis: Configuration, family: str, factory: Callable[[int], AlignmentMap]
) -> Configuration:
    return Configuration(
        f"{hypothesis.name}/map={family}",
        hypothesis.network,
        hypothesis.locus,
        family,
        factory,
        hypothesis.sees,
        differs_from_hypothesis_in="map_family",
    )


def label_only_learner(n_outputs: int) -> BayesLearner:
    """Dirichlet-categorical Bayes code of the labels: sees labels only."""
    learner = BayesLearner(DirichletCategorical(n_outputs))
    learner.name = f"label-only:{learner.name}"
    return learner


# ----------------------------------------------------------------------------------------
# Choosing the wrong locus: where V is not linearly decodable


def probe_accuracies(
    network: Network,
    x_pool: np.ndarray,
    values: np.ndarray,
    loci=None,
    seed: int = 0,
    l2: float = 1e-2,
) -> dict[str, float]:
    """Held-out accuracy of a logistic probe for ``values`` (the variable V) at each locus.

    Half of the pool trains the probe, the other half scores it. Used to define the
    wrong-locus control as the locus where V is least linearly decodable: a locus
    "before V is computed" does not exist for copied variables, so the plan's notion is
    operationalised by decodability (preregistration note).
    """
    from bitsev.alignment.intervene import HiddenCache
    from bitsev.baselines.probing_mdl import OnlineProbeLearner

    rng = np.random.default_rng(seed)
    n = len(values)
    perm = rng.permutation(n)
    train, test = perm[: n // 2], perm[n // 2 :]
    codes = np.unique(values, return_inverse=True)[1]
    out = {}
    for locus in loci or network.loci:
        states = HiddenCache.build(network, locus, x_pool).states.cpu().numpy()
        probe = OnlineProbeLearner(int(codes.max()) + 1, l2=l2).fit(states[train], codes[train])
        logits = np.concatenate([states[test], np.ones((len(test), 1))], axis=1) @ probe.weights
        out[locus] = float(np.mean(logits.argmax(1) == codes[test]))
    return out


def least_decodable_locus(
    network: Network, x_pool: np.ndarray, values: np.ndarray, exclude=(), **kw
) -> tuple[str, dict[str, float]]:
    """The locus with the lowest held-out probe accuracy for V, and all accuracies."""
    acc = probe_accuracies(network, x_pool, values, **kw)
    candidates = {k: v for k, v in acc.items() if k not in exclude}
    return min(candidates, key=candidates.get), acc


def least_intervenable_locus(
    network: Network,
    make_map: Callable[[int], AlignmentMap],
    targets,
    eval_targets,
    x_pool: np.ndarray,
    n_fit: int | None = None,
    budget=None,
    device: str = "cpu",
    seeds: int = 2,
    exclude=(),
    tolerance: float = 0.02,
) -> tuple[str, dict[str, float]]:
    """The locus where the map family can least realise the counterfactual (causal rule).

    For every locus, ``seeds`` cold-start maps of the hypothesis's family are fitted on the
    first ``n_fit`` targets (default: all) with the plug-in budget, and the attainable
    accuracy is the best held-out interchange intervention accuracy among them (a single fit
    can get stuck: InterpBench case 52, resid_2, 0.18 for one seed against 1.0 in the full
    run). The wrong-locus control is the locus with the lowest attainable accuracy; loci
    within ``tolerance`` of the minimum count as tied and the earliest of them wins (GPT-2:
    0.072 / 0.069 / 0.093 at layers 0 / 6 / 9 are all chance level). Unlike linear
    decodability this cannot be fooled by a variable that is a copy of an input token and
    therefore decodable everywhere (InterpBench case 52); a full-state swap would not
    discriminate either, since on an MLP it reproduces the source's output at every locus.
    """
    from bitsev.alignment.intervene import HiddenCache
    from bitsev.alignment.learners import DEFAULT_BUDGET, GradientAlignmentLearner
    from bitsev.alignment.score import faithfulness

    budget = budget or DEFAULT_BUDGET
    n = len(targets) if n_fit is None else n_fit
    iia: dict[str, float] = {}
    for locus in network.loci:
        cache = HiddenCache.build(network, locus, x_pool, device)
        best = 0.0
        for seed in range(seeds):
            learner = GradientAlignmentLearner(
                cache, lambda w=cache.width: make_map(w), budget, seed=seed, channel=targets.channel
            )
            predictor = learner.fit(targets.pairs[:n], targets.y[:n])
            best = max(best, faithfulness(predictor, eval_targets).iia)
        iia[locus] = round(best, 3)
    candidates = {k: v for k, v in iia.items() if k not in exclude}
    return _earliest_within(candidates, list(network.loci), tolerance), iia


def _earliest_within(scores: dict[str, float], order: list[str], tolerance: float) -> str:
    """The earliest key (in ``order``) whose score is within ``tolerance`` of the minimum."""
    floor = min(scores.values())
    return next(k for k in order if k in scores and scores[k] <= floor + tolerance)
