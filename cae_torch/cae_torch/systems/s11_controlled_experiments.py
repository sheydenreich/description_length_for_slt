"""System 11: controlled experiments for metric failure-mode analysis.

Original: ``systems/11_controlled_experiments.py`` of MelouxM/CAE.  Six small structural
systems, each paired with a valid and an invalid high-level model (``test/11_controlled_experiments.py``):

1. :class:`ConfounderPPModel` -- wraps a predator-prey ABM (system 04) and multiplies the
   prey output by ``1 + resource_effect * resource_level``; ``resource_level`` is a hidden
   (unmapped) variable read from the interventions only (the original ignores it in the
   input state), default 0.  Conditions: ``valid`` (effect 0) vs ``confounder`` (effect 0.5),
   phi-noise std 1.0; LV high-level model calibrated as in system 04.
2. :class:`XORBackupLowLevelModel` -- ``x -> a = x, b1 = x, b2 = x, y = bool(a) OR (bool(b1) XOR bool(b2))``
   (booleanised at 0.5; ``a`` is reported un-booleanised, equal to ``x``).  ``xor_active=False``
   disconnects the backup path (valid), ``True`` is the ``xor_leak`` condition.  Only
   ``x``, ``b1``, ``b2`` can be intervened (an intervention on ``a`` or ``y`` is ignored, as in
   the original).  Coarse-graining ``X <- x, M <- a, Y <- y``, bit value map
   (0: (-0.5, 0.5), 1: (0.5, 1.5)), high-level model ``M = X, Y = M`` with priors 1/2.
3. :class:`IntermediateChainLowLevelModel` -- ``x -> m = 2x -> z = round(m + 3, 10)``;
   ``m`` can be intervened, ``z`` cannot (always recomputed).  Valid high-level model: same
   equations; wrong: ``m = 3x, z = round(2/3 m + 3, 10)`` (same z, wrong mechanism).
   Domains x [0, 10], m [0, 30], z [3, 23].
4. :class:`ForkLowLevelModel` -- ``y = a x + b``, ``z = round(c x + d, 10)`` (defaults 2, 1, 3, 2);
   ``y`` and ``z`` can be intervened; ``z`` ignores ``y``.  Valid fork high-level model
   ``y = 2x + 1, z = round(3x + 2)``; wrong chain high-level model ``z = round(1.5 y + 0.5)``.
   Domains x [0, 10], y [1, 21], z [2, 32].
5. :class:`ThresholdLowLevelModel` -- ``x in {0, 1} -> m = 2x -> z = float(m >= 1.5)``; ``m``
   can be intervened, ``z`` cannot.  Coarse-graining ``X <- x, M <- m, Z <- z``; value map
   bit for X, Z and ``M`` labels 0..3 in unit rects around 0, 1, 2, 3.  Valid high-level
   model ``M = 2X, Z = int(M >= 2)``; wrong ``Z = int(M >= 1)`` (differs only at the
   input-unreachable ``M = 1``).  Priors X, Z 1/2 each, M 1/4 each.
6. :class:`ChainLowLevelModel` -- ``y = a x + b``, ``z = round(c y + d, 10)`` (defaults 2, 1, 1.5, 0.5);
   ``y`` and ``z`` can be intervened.  Valid chain high-level model as in 4; wrong fork
   high-level model ``z = round(3x + 2)`` from x.

All variables are scalars, shape (batch, 1); the input is ``x`` (default 5.0 for 3, 4, 6 and
0.0 for 2, 5) or the ABM inputs for 1.  The 10-decimal rounding of the original (which
removes IEEE-754 ULP differences between algebraically equal high-level equations) is
reproduced bitwise as ``round(x * 1e10) / 1e10`` and implemented *straight-through*: the
forward value is rounded, the gradient is that of the unrounded expression (the original's
rounding is numerical hygiene, not a mechanism).  Deterministic; differentiable except the
``> 0.5`` booleanisation of 2 and the ``>= 1.5`` threshold of 5, which are exact by default
and replaced by ``sigmoid(soft_slope * (v - threshold))`` when ``soft=True``.  System 1
inherits the stochasticity / differentiability of the wrapped ABM.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from cae_torch.base import TorchSystem

F64 = torch.float64
ROUND_DIGITS = 10
_ROUND_FACTOR = 10.0 ** ROUND_DIGITS


def round_digits(x: Tensor) -> Tensor:
    """``np.round(x, 10)`` in the forward pass, identity gradient (straight-through)."""
    rounded = torch.round(x * _ROUND_FACTOR) / _ROUND_FACTOR
    return x + (rounded - x).detach()


def _as_2d(value, default: float | None = None) -> Tensor:
    if value is None:
        return torch.tensor([[default]], dtype=F64)
    t = torch.as_tensor(value, dtype=F64)
    return t.reshape(1, 1) if t.ndim == 0 else (t.reshape(-1, 1) if t.ndim == 1 else t)


def _finish(out: dict[str, Tensor]) -> dict[str, Tensor]:
    """Broadcast size-1 variables to the batch (the original returns e.g. a (1, 1) default
    ``x`` next to (batch, 1) outputs; the contract wants every variable as (batch, ...))."""
    batch = max(t.shape[0] for t in out.values())
    return {k: t.expand(batch, *t.shape[1:]) if t.shape[0] == 1 and batch > 1 else t for k, t in out.items()}


def _gate(v: Tensor, threshold: float, soft: bool, slope: float) -> Tensor:
    if soft:
        return torch.sigmoid(slope * (v - threshold))
    return (v > threshold).to(F64) if threshold == 0.5 else (v >= threshold).to(F64)


# High-level abstraction data (plain Python) ------------------------------------------

BIT_SPECS = {0: (-0.5, 0.5), 1: (0.5, 1.5)}
HALF = {0: 0.5, 1: 0.5}

EXP1 = dict(conditions={"valid": 0.0, "confounder": 0.5}, phi_noise_std=1.0,
            coarse_graining={"prey_t": ["prey_t"], "predator_t": ["predator_t"],
                             "final_populations": ["final_populations"]},
            domains={"prey_t": (50.0, 250.0), "predator_t": (20.0, 100.0)},
            intervention_domain=["prey_t", "predator_t"])

EXP2 = dict(conditions={"valid": False, "xor_leak": True}, phi_noise_std=1.0,
            coarse_graining={"X": ["x"], "M": ["a"], "Y": ["y"]},
            value_map={"X": BIT_SPECS, "M": BIT_SPECS, "Y": BIT_SPECS},
            high_level_model={"X": (None, HALF), "M": (lambda X: X, HALF), "Y": (lambda M: M, HALF)})

EXP3 = dict(domains={"x": (0.0, 10.0), "m": (0.0, 30.0), "z": (3.0, 23.0)},
            coarse_graining={"x": ["x"], "m": ["m"], "z": ["z"]},
            intervention_domain=["x", "m"],
            valid={"m": lambda x: 2.0 * x, "z": lambda m: round(m + 3.0, ROUND_DIGITS)},
            wrong={"m": lambda x: 3.0 * x, "z": lambda m: round((2.0 / 3.0) * m + 3.0, ROUND_DIGITS)})

FORK_DOMAINS = {"x": (0.0, 10.0), "y": (1.0, 21.0), "z": (2.0, 32.0)}
FORK_HLM = {"y": ("x", lambda x: 2.0 * x + 1.0), "z": ("x", lambda x: round(3.0 * x + 2.0, ROUND_DIGITS))}
CHAIN_HLM = {"y": ("x", lambda x: 2.0 * x + 1.0), "z": ("y", lambda y: round(1.5 * y + 0.5, ROUND_DIGITS))}
EXP4 = dict(domains=FORK_DOMAINS, coarse_graining={"x": ["x"], "y": ["y"], "z": ["z"]},
            intervention_domain=["x", "y"], valid_fork=FORK_HLM, spurious_chain=CHAIN_HLM)
EXP6 = dict(domains=FORK_DOMAINS, coarse_graining={"x": ["x"], "y": ["y"], "z": ["z"]},
            intervention_domain=["x", "y"], valid_chain=CHAIN_HLM, wrong_fork=FORK_HLM)

EXP5 = dict(coarse_graining={"X": ["x"], "M": ["m"], "Z": ["z"]},
            value_map={"X": BIT_SPECS, "Z": BIT_SPECS,
                       "M": {0: (-0.5, 0.5), 1: (0.5, 1.5), 2: (1.5, 2.5), 3: (2.5, 3.5)}},
            distributions={"X": HALF, "M": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}, "Z": HALF},
            intervention_domain=["M"],
            valid={"M": lambda X: 2 * X, "Z": lambda M: int(M >= 2)},
            wrong_threshold={"M": lambda X: 2 * X, "Z": lambda M: int(M >= 1)})


# Experiment 1 ---------------------------------------------------------------------------

class ConfounderPPModel(TorchSystem):
    """Hidden resource confounder on top of a predator-prey :class:`TorchSystem`.

    Deviation: the output ``resource_level`` is always (batch, 1) (broadcast of the given
    value, default 0); the original returns the un-broadcast rows it was given.
    """

    variables = ("prey_t", "predator_t", "resource_level", "final_populations")
    input_variables = ("prey_t", "predator_t")

    def __init__(self, base_model: TorchSystem, resource_effect: float = 0.0):
        self.base = base_model
        self.resource_effect = float(resource_effect)

    def forward_with_interventions(self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> dict[str, Tensor]:
        clean = {k: v for k, v in interventions.items() if k != "resource_level"}
        result = dict(self.base.forward_with_interventions(inputs, clean))
        pops = result["final_populations"]
        batch = pops.shape[0]
        if "resource_level" in interventions:
            r = torch.as_tensor(interventions["resource_level"], dtype=F64).reshape(-1)
            if r.shape[0] < batch:                       # r_flat[min(i, len - 1)]
                r = torch.cat([r, r[-1:].expand(batch - r.shape[0])])
            r = r[:batch]
        else:
            r = torch.zeros(batch, dtype=F64)
        prey = pops[:, 0] * (1.0 + self.resource_effect * r)
        pops = torch.stack([prey, pops[:, 1]], dim=1)
        result["final_populations"] = torch.clamp(pops, min=0.0)
        result["resource_level"] = r.reshape(-1, 1)
        return _finish(result)


# Experiment 2 ---------------------------------------------------------------------------

class XORBackupLowLevelModel(TorchSystem):
    variables = ("x", "a", "b1", "b2", "y")
    input_variables = ("x",)

    def __init__(self, xor_active: bool = True, soft: bool = False, soft_slope: float = 10.0):
        self.xor_active = bool(xor_active)
        self.soft = bool(soft)
        self.soft_slope = float(soft_slope)

    def forward_with_interventions(self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> dict[str, Tensor]:
        x = _as_2d(interventions.get("x", inputs.get("x")), 0.0)
        a = x
        b1 = _as_2d(interventions["b1"]) if "b1" in interventions else x
        b2 = _as_2d(interventions["b2"]) if "b2" in interventions else x
        a_b = _gate(a, 0.5, self.soft, self.soft_slope)
        b1_b = _gate(b1, 0.5, self.soft, self.soft_slope)
        b2_b = _gate(b2, 0.5, self.soft, self.soft_slope)
        if self.xor_active:
            y = torch.maximum(a_b, torch.abs(b1_b - b2_b))
        else:
            y = a_b
        return _finish({"x": x, "a": a, "b1": b1, "b2": b2, "y": y})


# Experiment 3 ---------------------------------------------------------------------------

class IntermediateChainLowLevelModel(TorchSystem):
    variables = ("x", "m", "z")
    input_variables = ("x",)

    def forward_with_interventions(self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> dict[str, Tensor]:
        x = _as_2d(interventions.get("x", inputs.get("x")), 5.0)
        m = _as_2d(interventions["m"]) if "m" in interventions else 2.0 * x
        z = round_digits(m + 3.0)
        return _finish({"x": x, "m": m, "z": z})


# Experiment 4 ---------------------------------------------------------------------------

class ForkLowLevelModel(TorchSystem):
    variables = ("x", "y", "z")
    input_variables = ("x",)

    def __init__(self, a: float = 2.0, b: float = 1.0, c: float = 3.0, d: float = 2.0):
        self.a, self.b, self.c, self.d = float(a), float(b), float(c), float(d)

    def forward_with_interventions(self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> dict[str, Tensor]:
        x = _as_2d(interventions.get("x", inputs.get("x")), 5.0)
        y = _as_2d(interventions["y"]) if "y" in interventions else self.a * x + self.b
        z = _as_2d(interventions["z"]) if "z" in interventions else round_digits(self.c * x + self.d)
        return _finish({"x": x, "y": y, "z": z})


# Experiment 5 ---------------------------------------------------------------------------

class ThresholdLowLevelModel(TorchSystem):
    variables = ("x", "m", "z")
    input_variables = ("x",)

    def __init__(self, soft: bool = False, soft_slope: float = 10.0):
        self.soft = bool(soft)
        self.soft_slope = float(soft_slope)

    def forward_with_interventions(self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> dict[str, Tensor]:
        x = _as_2d(interventions.get("x", inputs.get("x")), 0.0)
        m = _as_2d(interventions["m"]) if "m" in interventions else 2.0 * x
        z = _gate(m, 1.5, self.soft, self.soft_slope)
        return _finish({"x": x, "m": m, "z": z})


# Experiment 6 ---------------------------------------------------------------------------

class ChainLowLevelModel(TorchSystem):
    variables = ("x", "y", "z")
    input_variables = ("x",)

    def __init__(self, a: float = 2.0, b: float = 1.0, c: float = 1.5, d: float = 0.5):
        self.a, self.b, self.c, self.d = float(a), float(b), float(c), float(d)

    def forward_with_interventions(self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> dict[str, Tensor]:
        x = _as_2d(interventions.get("x", inputs.get("x")), 5.0)
        y = _as_2d(interventions["y"]) if "y" in interventions else self.a * x + self.b
        z = _as_2d(interventions["z"]) if "z" in interventions else round_digits(self.c * y + self.d)
        return _finish({"x": x, "y": y, "z": z})
