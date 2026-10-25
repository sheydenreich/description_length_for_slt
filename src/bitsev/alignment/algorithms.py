"""High-level causal models: an algorithm with named intermediate variables.

The Stage 1 targets are the algorithm's counterfactual outputs ``A(b; V <- V(s))``: run the
algorithm on base input ``b`` with variable ``V`` replaced by its value on source ``s``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from bitsev.testbeds import hierarchical_equality as he


@runtime_checkable
class Algorithm(Protocol):
    name: str
    variables: tuple[str, ...]
    n_outputs: int

    def variable_values(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Value of each intermediate variable on every input."""
        ...

    def output(self, x: np.ndarray) -> np.ndarray: ...

    def counterfactual(self, base: np.ndarray, source: np.ndarray, variable: str) -> np.ndarray:
        """``A(base; variable <- variable(source))`` for aligned rows of ``base`` and ``source``."""
        ...


class HierarchicalEquality:
    """``V1 = [a == b]``, ``V2 = [c == d]``, output ``[V1 == V2]``."""

    name = "hierarchical_equality"
    variables = ("V1", "V2")
    n_outputs = 2

    def __init__(self, dim: int):
        self.dim = dim

    def variable_values(self, x):
        v1, v2 = he.variables_from_input(x, self.dim)
        return {"V1": v1, "V2": v2}

    def output(self, x):
        v = self.variable_values(x)
        return (v["V1"] == v["V2"]).astype(np.int64)

    def counterfactual(self, base, source, variable):
        vb, vs = self.variable_values(base), self.variable_values(source)
        if variable not in self.variables:
            raise KeyError(variable)
        vb = dict(vb)
        vb[variable] = vs[variable]
        return (vb["V1"] == vb["V2"]).astype(np.int64)
