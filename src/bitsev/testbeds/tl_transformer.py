"""A minimal transformer in the TransformerLens 1.x parameter layout.

InterpBench ships its semi-synthetic models as TransformerLens 1.19 state dicts (no layer
norm, GELU MLPs, learned positional embeddings, optional bidirectional attention).
TransformerLens 4 removed ``HookedTransformer``, so this module re-implements exactly the
forward pass those weights need, with the same parameter names so ``load_state_dict`` works
unchanged. The forward was validated on InterpBench cases 52 and 60 (100% task accuracy).

The network exposes the residual stream after the embedding and after every block as loci,
flattened over positions, so the Stage 1 machinery (which needs ``hidden`` and
``logits_from_hidden``) applies without hooks. The output is the logit vector at a fixed
position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TLConfig:
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    d_vocab: int
    d_vocab_out: int
    n_ctx: int
    act_fn: str = "gelu"
    attention_dir: str = "causal"
    initializer_range: float = 0.02

    @classmethod
    def from_tl_dict(cls, cfg: dict, d_vocab_out: int) -> TLConfig:
        if cfg.get("normalization_type") not in (None, "None"):
            raise ValueError("only models without layer norm are supported")
        return cls(
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_model=cfg["d_model"],
            d_head=cfg["d_head"],
            d_mlp=cfg["d_mlp"],
            d_vocab=cfg["d_vocab"],
            d_vocab_out=d_vocab_out,
            n_ctx=cfg["n_ctx"],
            act_fn=cfg.get("act_fn", "gelu"),
            attention_dir=cfg.get("attention_dir", "causal"),
            initializer_range=float(cfg.get("initializer_range", 0.02)),
        )


class _Embed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_E = nn.Parameter(torch.zeros(d_vocab, d_model))


class _PosEmbed(nn.Module):
    def __init__(self, n_ctx: int, d_model: int):
        super().__init__()
        self.W_pos = nn.Parameter(torch.zeros(n_ctx, d_model))


class _Attention(nn.Module):
    def __init__(self, cfg: TLConfig):
        super().__init__()
        h, d, e = cfg.n_heads, cfg.d_model, cfg.d_head
        self.W_Q, self.W_K, self.W_V = (nn.Parameter(torch.zeros(h, d, e)) for _ in range(3))
        self.W_O = nn.Parameter(torch.zeros(h, e, d))
        self.b_Q, self.b_K, self.b_V = (nn.Parameter(torch.zeros(h, e)) for _ in range(3))
        self.b_O = nn.Parameter(torch.zeros(d))
        self.register_buffer("mask", torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))
        self.register_buffer("IGNORE", torch.tensor(-torch.inf))
        self.causal = cfg.attention_dir == "causal"
        self.scale = math.sqrt(cfg.d_head)

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[1]
        q = torch.einsum("btd,hde->bthe", x, self.W_Q) + self.b_Q
        k = torch.einsum("btd,hde->bthe", x, self.W_K) + self.b_K
        v = torch.einsum("btd,hde->bthe", x, self.W_V) + self.b_V
        scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / self.scale
        if self.causal:
            causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~causal, float("-inf"))
        z = torch.einsum("bhqk,bkhe->bqhe", scores.softmax(-1), v)
        return torch.einsum("bqhe,hed->bqd", z, self.W_O) + self.b_O


class _MLP(nn.Module):
    def __init__(self, cfg: TLConfig):
        super().__init__()
        self.W_in = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.zeros(cfg.d_mlp, cfg.d_model))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))
        self.act = {
            "gelu": nn.functional.gelu,
            "gelu_new": lambda h: nn.functional.gelu(h, approximate="tanh"),
            "relu": torch.relu,
        }[cfg.act_fn]

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x @ self.W_in + self.b_in) @ self.W_out + self.b_out


class _Block(nn.Module):
    def __init__(self, cfg: TLConfig):
        super().__init__()
        self.attn = _Attention(cfg)
        self.mlp = _MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class _Unembed(nn.Module):
    def __init__(self, d_model: int, d_vocab_out: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.zeros(d_model, d_vocab_out))
        self.b_U = nn.Parameter(torch.zeros(d_vocab_out))


class TLTransformer(nn.Module):
    """Residual-stream loci ``"resid_0"`` (after embedding) ... ``"resid_L"`` (after block L).

    ``hidden`` returns the residual stream flattened over positions, ``(n, n_ctx * d_model)``;
    ``logits_from_hidden`` runs the remaining blocks and returns the logits at
    ``output_position`` (default: the last position).
    """

    def __init__(self, cfg: TLConfig, output_position: int = -1):
        super().__init__()
        self.cfg = cfg
        self.embed = _Embed(cfg.d_vocab, cfg.d_model)
        self.pos_embed = _PosEmbed(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.unembed = _Unembed(cfg.d_model, cfg.d_vocab_out)
        self.output_position = output_position
        self.loci = tuple(f"resid_{i}" for i in range(cfg.n_layers + 1))
        self.n_outputs = cfg.d_vocab_out

    # Network protocol -----------------------------------------------------------------

    def prepare_input(self, tokens) -> Tensor:
        return torch.as_tensor(tokens, dtype=torch.long)

    def width(self, locus: str) -> int:
        return self.cfg.n_ctx * self.cfg.d_model

    def _index(self, locus: str) -> int:
        try:
            return self.loci.index(locus)
        except ValueError as exc:
            raise KeyError(f"unknown locus {locus!r}; known: {self.loci}") from exc

    def residual(self, tokens: Tensor, upto: int) -> Tensor:
        x = self.embed.W_E[tokens] + self.pos_embed.W_pos[: tokens.shape[1]][None]
        for block in self.blocks[:upto]:
            x = block(x)
        return x

    def hidden(self, tokens: Tensor, locus: str) -> Tensor:
        return self.residual(tokens, self._index(locus)).flatten(1)

    def logits_from_hidden(self, h: Tensor, locus: str) -> Tensor:
        x = h.view(h.shape[0], self.cfg.n_ctx, self.cfg.d_model)
        for block in self.blocks[self._index(locus) :]:
            x = block(x)
        return (x @ self.unembed.W_U + self.unembed.b_U)[:, self.output_position]

    def reinitialised(self, seed: int) -> TLTransformer:
        """Same architecture with fresh ``N(0, initializer_range^2)`` weights and zero biases."""
        import copy

        twin = copy.deepcopy(self)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, param in twin.named_parameters():
                if name.split(".")[-1].startswith("b_"):
                    param.zero_()
                else:
                    param.copy_(torch.randn(param.shape, generator=g) * self.cfg.initializer_range)
        return twin.eval()

    def all_logits(self, tokens: Tensor) -> Tensor:
        """Logits at every position, ``(n, n_ctx, d_vocab_out)``."""
        x = self.residual(tokens, self.cfg.n_layers)
        return x @ self.unembed.W_U + self.unembed.b_U

    def logits(self, tokens: Tensor) -> Tensor:
        return self.all_logits(tokens)[:, self.output_position]

    def forward(self, tokens: Tensor) -> Tensor:
        return self.logits(tokens)


def from_state_dict(cfg_dict: dict, state_dict: dict, output_position: int = -1) -> TLTransformer:
    cfg = TLConfig.from_tl_dict(cfg_dict, d_vocab_out=state_dict["unembed.W_U"].shape[1])
    model = TLTransformer(cfg, output_position)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected or any(not k.endswith(("mask", "IGNORE")) for k in missing):
        raise ValueError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval()
