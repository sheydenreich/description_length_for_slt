"""The interface every translated system implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class TorchSystem(ABC):
    """A batched, torch-native structural model with interventions on micro-variables.

    Subclasses set ``variables`` (all micro-variable names, in the numpy model's order) and
    ``input_variables`` (what the numpy model expects in the input state), and implement
    :meth:`forward_with_interventions`. See ``README.md`` for the full contract.
    """

    variables: tuple[str, ...]
    input_variables: tuple[str, ...]
    dtype: torch.dtype = torch.float64

    @abstractmethod
    def forward_with_interventions(
        self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Run the system on a batch of inputs with some micro-variables clamped.

        Returns every variable as a ``(batch, ...)`` tensor.
        """

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.forward_with_interventions(inputs, {})

    @staticmethod
    def batch_size(inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> int:
        for v in (*inputs.values(), *interventions.values()):
            return int(v.shape[0])
        return 1
