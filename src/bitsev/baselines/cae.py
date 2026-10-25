"""Causal abstraction error with interventions on unmapped variables.

Meloux et al. (2026) found that only causal metrics which also test faithfulness on
*unmapped* variables separate valid from invalid abstractions reliably. For an alignment
that maps the variable ``V`` to a subspace, the protocol here evaluates three kinds of
intervention on (base, source) pairs and compares the network's output with what the
abstraction predicts:

* **mapped**: interchange on the aligned subspace; prediction ``A(b; V <- V(s))``;
* **complement**: interchange on the complement subspace; prediction ``A(s; V <- V(b))``.
  An alignment of one variable leaves every other variable of the algorithm in the
  complement, so swapping the complement sets all of them to their source values. The
  intervened state ``phi^-1(z_V(b), z_perp(s))`` is the mapped interchange with base and
  source exchanged, so this error is one minus the IIA on the reversed pairs and adds no
  information beyond IIA. (An earlier version predicted ``A(b)`` here, which is wrong
  whenever another variable feeds the output: on hierarchical equality it scored every
  faithful alignment at about 0.5.) A test on genuinely unmapped coordinates needs every
  variable of the algorithm aligned, so that the residual complement carries none;
* **random direction** (reference): interchange on a random subspace of the same width
  under a random rotation, scored against ``A(b)``. A random subspace may carry part of
  any variable, so this is the output's sensitivity to an arbitrary swap of that width,
  not an error of the abstraction.

The error of each kind is the disagreement rate of the most probable output with the
prediction and the mean log loss of the prediction; the causal abstraction error is the
average over mapped and complement interventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from bitsev.alignment.algorithms import Algorithm
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.maps import AlignmentMap, Rotation
from bitsev.alignment.targets import TargetSet


@dataclass
class InterventionError:
    kind: str
    disagreement: float  # 1 - accuracy against the abstraction's prediction
    log_loss: float  # mean nats the network assigns to the predicted label
    n: int


@dataclass
class CausalAbstractionError:
    mapped: InterventionError
    complement: InterventionError
    random_direction: InterventionError

    @property
    def error(self) -> float:
        """Average disagreement over mapped and complement interventions."""
        return 0.5 * (self.mapped.disagreement + self.complement.disagreement)

    def record(self) -> dict:
        return {
            "mapped": asdict(self.mapped),
            "complement": asdict(self.complement),
            "random_direction": asdict(self.random_direction),
            "error": self.error,
        }


def _score(
    cache: HiddenCache, h_new: torch.Tensor, predicted: np.ndarray, kind: str
) -> InterventionError:
    with torch.no_grad():
        logits = cache.network.logits_from_hidden(h_new, cache.locus)
        log_probs = torch.log_softmax(logits, dim=1)
    pred_t = torch.as_tensor(predicted, device=logits.device)
    lp = log_probs.gather(1, pred_t[:, None])[:, 0]
    disagreement = float((logits.argmax(1) != pred_t).float().mean())
    return InterventionError(kind, disagreement, float(-lp.mean()), len(predicted))


def causal_abstraction_error(
    cache: HiddenCache,
    phi: AlignmentMap,
    algorithm: Algorithm,
    x_pool: np.ndarray,
    targets: TargetSet,
    seed: int = 0,
) -> CausalAbstractionError:
    """Evaluate the three intervention kinds on the pairs of ``targets`` (clean labels)."""
    h_base, h_source = cache.pair_states(targets.pairs)
    base_output = algorithm.output(x_pool[targets.base])
    complement_output = algorithm.counterfactual(
        x_pool[targets.source], x_pool[targets.base], targets.variable
    )
    with torch.no_grad():
        mapped = _score(cache, phi.interchange(h_base, h_source), targets.y_clean, "mapped")
        complement = _score(
            cache, phi.interchange_complement(h_base, h_source), complement_output, "complement"
        )
        torch.manual_seed(seed)
        random_phi = Rotation(phi.d, phi.d_v, init_scale=1.0).to(h_base.device)
        random = _score(
            cache, random_phi.interchange(h_base, h_source), base_output, "random_direction"
        )
    return CausalAbstractionError(mapped, complement, random)
