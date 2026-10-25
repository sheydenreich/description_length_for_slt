"""Training loop for toy networks with dense checkpoints (needed for H-A2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass
class Checkpoint:
    step: int
    state_dict: dict
    train_loss: float
    train_accuracy: float


@dataclass
class TrainingRun:
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def steps(self) -> list[int]:
        return [c.step for c in self.checkpoints]

    def load(self, network: nn.Module, step: int) -> nn.Module:
        """Load the checkpoint at ``step`` into ``network`` (in place) and return it."""
        for c in self.checkpoints:
            if c.step == step:
                network.load_state_dict(c.state_dict)
                return network
        raise KeyError(f"no checkpoint at step {step}; have {self.steps()}")


def _clone_state(network: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in network.state_dict().items()}


def train_classifier(
    network: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    lr: float = 1e-2,
    batch_size: int = 128,
    seed: int = 0,
    checkpoint_steps: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> TrainingRun:
    """Adam on cross-entropy with checkpoints at the given steps.

    Default checkpoints: step 0, powers of two, and the final step.
    """
    g = torch.Generator().manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    network.to(device)
    opt = torch.optim.Adam(network.parameters(), lr=lr)
    if checkpoint_steps is None:
        checkpoint_steps = sorted(
            {0, steps, *(int(2**k) for k in range(int(np.log2(max(steps, 1))) + 1))}
        )
    wanted = set(checkpoint_steps)
    run = TrainingRun()

    def record(step: int) -> None:
        with torch.no_grad():
            logits = network(xt)
            loss = nn.functional.cross_entropy(logits, yt).item()
            acc = (logits.argmax(1) == yt).float().mean().item()
        run.checkpoints.append(Checkpoint(step, _clone_state(network), loss, acc))

    if 0 in wanted:
        record(0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(xt), (min(batch_size, len(xt)),), generator=g).to(device)
        loss = nn.functional.cross_entropy(network(xt[idx]), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in wanted:
            record(step)
    return run
