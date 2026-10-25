"""System 01: 2-bit adder as a gate-level netlist (``systems/01_logic_circuit.py``).

Low-level model ``NetlistSimulator``: a Gauss-Seidel fixed-point evaluation of a netlist of
AND / OR / XOR gates on boolean wires, reported as floats 0.0 / 1.0.

Variables (sorted wire names, the numpy model's order)::

    A0 A1 B0 B1 C0 C1 C2 S0 S1
    bit0_and1 bit0_and2 bit0_xor1 bit1_and1 bit1_and2 bit1_xor1

Inputs: ``A0 A1 B0 B1 C0`` (the undriven wires). Any wire given in ``inputs`` is used as the
*initial* state of the fixed-point iteration (so a gate-driven wire given as input is simply
overwritten); any wire given in ``interventions`` is *forced* (its gate is skipped) for the
whole computation. Every value is thresholded at 0.5 on entry, exactly as the original.

Non-differentiable steps (``soft=False``): the 0.5 threshold on entry and the boolean gates.
There is no differentiable path at all in the exact mode (outputs do not even carry
``grad_fn``). With ``soft=True`` (not in the original) no thresholding is applied and the
gates become the product relaxations ``AND = prod(x)``, ``OR = 1 - prod(1 - x)``,
``XOR = a + b - 2ab`` (folded), which reproduce the exact gates on {0, 1} inputs and are
differentiable everywhere.

Deviation from the original: the numpy model infers the batch size from the interventions
only (a batched input state without interventions silently yields a batch of 1 built from
row 0); here the batch size is taken from inputs or interventions (``TorchSystem.batch_size``).
For every call where the numpy model does see the right batch size the outputs are identical.

High-level abstractions (``test/01_logic_circuit.py`` conditions ``valid``, ``fail``,
``inv_internal``, ``noise``) are given below as plain data / functions: :data:`COARSE_GRAINING`,
:data:`INTERNAL_WIRES`, :data:`N_BITS`, :data:`VALUE_RANGES`, :func:`abstract_bits` and
:data:`HIGH_LEVEL_MODELS`. The ``noise`` condition uses the valid high-level model with the
low-level outputs perturbed by Gaussian noise of std 0.4 (``NoisyLowLevelModel``).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from cae_torch.base import TorchSystem

# --------------------------------------------------------------------------------------
# Circuit topology (verbatim from the original)
# --------------------------------------------------------------------------------------


def build_2bit_adder() -> tuple[list[dict], list[str]]:
    """Return ``(gates, all_wires)`` of the ripple-carry 2-bit adder of the original."""
    gates: list[dict] = []
    all_wires = ["C0", "C1", "C2", "A0", "A1", "B0", "B1", "S0", "S1"]
    for i in range(2):
        cin, cout = f"C{i}", f"C{i + 1}"
        a, b, s = f"A{i}", f"B{i}", f"S{i}"
        xor1, and1, and2 = f"bit{i}_xor1", f"bit{i}_and1", f"bit{i}_and2"
        all_wires.extend([xor1, and1, and2])
        gates.append({"type": "XOR", "in": [a, b], "out": xor1})
        gates.append({"type": "XOR", "in": [xor1, cin], "out": s})
        gates.append({"type": "AND", "in": [xor1, cin], "out": and1})
        gates.append({"type": "AND", "in": [a, b], "out": and2})
        gates.append({"type": "OR", "in": [and1, and2], "out": cout})
    return gates, sorted(set(all_wires))


# --------------------------------------------------------------------------------------
# Low-level model
# --------------------------------------------------------------------------------------


def _and(xs: list[Tensor]) -> Tensor:
    out = xs[0]
    for x in xs[1:]:
        out = out * x
    return out


def _or(xs: list[Tensor]) -> Tensor:
    out = 1.0 - xs[0]
    for x in xs[1:]:
        out = out * (1.0 - x)
    return 1.0 - out


def _xor_hard(xs: list[Tensor]) -> Tensor:
    s = xs[0]
    for x in xs[1:]:
        s = s + x
    return torch.remainder(s, 2.0)


def _xor_soft(xs: list[Tensor]) -> Tensor:
    out = xs[0]
    for x in xs[1:]:
        out = out + x - 2.0 * out * x
    return out


class NetlistSimulator(TorchSystem):
    """Batched torch translation of ``NetlistSimulator(gates, all_wires)``."""

    def __init__(self, gates: list[dict], all_wires: list[str], soft: bool = False):
        self.gates = list(gates)
        self.all_wires = sorted(set(all_wires))
        self.soft = bool(soft)
        self.variables = tuple(self.all_wires)
        driven = {g["out"] for g in self.gates}
        self.input_variables = tuple(w for w in self.all_wires if w not in driven)
        self._gate_fns: dict[str, Callable[[list[Tensor]], Tensor]] = {
            "AND": _and,
            "OR": _or,
            "XOR": _xor_soft if self.soft else _xor_hard,
        }

    def _coerce(self, v: Tensor, batch: int) -> Tensor:
        v = torch.as_tensor(v).reshape(batch, 1)
        if self.soft:
            return v.to(self.dtype)
        # exact: threshold at 0.5, as `_scalar_to_bool` in the original
        return (v.to(self.dtype) > 0.5).to(self.dtype)

    def forward_with_interventions(
        self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        batch = self.batch_size(inputs, interventions)
        zeros = torch.zeros(batch, 1, dtype=self.dtype)
        state: dict[str, Tensor] = {w: zeros for w in self.all_wires}
        for k, v in inputs.items():
            if k in state:
                state[k] = self._coerce(v, batch)
        forced: set[str] = set()
        for k, v in interventions.items():
            if k in state:
                state[k] = self._coerce(v, batch)
                forced.add(k)
        # The original runs at most len(gates) + 2 Gauss-Seidel passes and stops early once
        # a pass changes nothing; running the full count gives identical values.
        for _ in range(len(self.gates) + 2):
            for g in self.gates:
                if g["out"] in forced:
                    continue
                fn = self._gate_fns.get(g["type"])
                if fn is None:  # unknown gate type evaluates to False in the original
                    state[g["out"]] = zeros
                    continue
                state[g["out"]] = fn([state[w] for w in g["in"]])
        return dict(state)


# --------------------------------------------------------------------------------------
# High-level abstractions (plain data; no causal_abstraction dependency)
# --------------------------------------------------------------------------------------

#: Coarse-graining map: high-level variable -> low-level wires (bit 0 first).
COARSE_GRAINING: dict[str, list[str]] = {
    "Operand_A": ["A0", "A1"],
    "Operand_B": ["B0", "B1"],
    "Carry_In": ["C0"],
    "Internal_Carries": ["C1"],
    "Result_Sum": ["S0", "S1"],
    "Result_Carry": ["C2"],
}
#: Wires declared `internal_variables` of the coarse-graining (redundant mechanism wires).
INTERNAL_WIRES: list[str] = [
    "bit0_xor1", "bit0_and1", "bit0_and2", "bit1_xor1", "bit1_and1", "bit1_and2",
]
#: Number of bits of each high-level variable (``BinaryValueMap``).
N_BITS: dict[str, int] = {
    "Operand_A": 2, "Operand_B": 2, "Carry_In": 1,
    "Internal_Carries": 1, "Result_Sum": 2, "Result_Carry": 1,
}
#: Value-map intervals of one wire for bit value 0 / 1 (a high-level value is the product
#: rectangle of its bits).
VALUE_RANGES: dict[int, tuple[float, float]] = {0: (-0.1, 0.1), 1: (0.9, 1.1)}
#: Uniform priors of the high-level exogenous variables.
HL_DISTRIBUTIONS: dict[str, dict[int, float]] = {
    "Operand_A": {i: 0.25 for i in range(4)},
    "Operand_B": {i: 0.25 for i in range(4)},
    "Carry_In": {0: 0.5, 1: 0.5},
}


def abstract_bits(values) -> int:
    """``BinaryValueMap.abstract`` for one sample: wires (bit 0 first) -> integer, threshold 0.5."""
    out = 0
    for b, v in enumerate(values):
        if float(v) > 0.5:
            out |= 1 << b
    return out


def hl_valid(Operand_A: int, Operand_B: int, Carry_In: int) -> dict[str, int]:
    """Correct 2-bit adder (``build_valid_high_level_model``)."""
    a0, b0 = Operand_A & 1, Operand_B & 1
    c1 = (a0 & b0) | (Carry_In & (a0 ^ b0))
    s = 0
    for i, c in enumerate([Carry_In, c1]):
        s |= (((Operand_A >> i) & 1) ^ ((Operand_B >> i) & 1) ^ c) << i
    a1, b1 = (Operand_A >> 1) & 1, (Operand_B >> 1) & 1
    c2 = (a1 & b1) | (c1 & (a1 ^ b1))
    return {"Internal_Carries": c1, "Result_Sum": s, "Result_Carry": c2}


def hl_failing(Operand_A: int, Operand_B: int, Carry_In: int) -> dict[str, int]:
    """Wrong model: OR instead of XOR in the carry and sum (``build_failing_high_level_model``)."""
    a0, b0 = Operand_A & 1, Operand_B & 1
    c1 = (a0 & b0) | (Carry_In & (a0 | b0))
    s = 0
    for i, c in enumerate([Carry_In, c1]):
        s |= (((Operand_A >> i) & 1) | ((Operand_B >> i) & 1) | c) << i
    a1, b1 = (Operand_A >> 1) & 1, (Operand_B >> 1) & 1
    c2 = (a1 & b1) | (c1 & (a1 ^ b1))
    return {"Internal_Carries": c1, "Result_Sum": s, "Result_Carry": c2}


def hl_inverted_internal(Operand_A: int, Operand_B: int, Carry_In: int) -> dict[str, int]:
    """Internal carry inverted, consumers compensate (``build_inverted_internal_high_level_model``)."""
    valid = hl_valid(Operand_A, Operand_B, Carry_In)
    return {
        "Internal_Carries": 1 - valid["Internal_Carries"],
        "Result_Sum": valid["Result_Sum"],
        "Result_Carry": valid["Result_Carry"],
    }


#: High-level structural equations per condition of ``test/01_logic_circuit.py``.
HIGH_LEVEL_MODELS: dict[str, Callable[[int, int, int], dict[str, int]]] = {
    "valid": hl_valid,
    "fail": hl_failing,
    "inv_internal": hl_inverted_internal,
    "noise": hl_valid,  # low-level outputs get Gaussian noise, std 0.4
}
