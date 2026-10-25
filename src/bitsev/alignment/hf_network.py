"""Hugging Face causal language models as networks with position-specific residual loci.

Locus ``"layer{l}"`` is the residual stream entering decoder layer ``l`` (``l = 0`` is the
embedding output; ``l = L`` is the final, post-norm state) at a fixed position (default:
the last token). ``logits_from_hidden``
re-runs the model on the base prompt with a forward pre-hook on layer ``l`` that overwrites
that position's residual with the given state, and returns the logits restricted to the
output token ids (for IOI, the name tokens). Interventions therefore cost one forward pass
per batch of pairs; the hidden states of a pool are cached once.

Tested with GPT-NeoX (Pythia); other architectures need only the ``layers`` attribute path.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class HFNetwork(nn.Module):
    needs_base_input = True

    def __init__(
        self,
        model,
        output_token_ids: np.ndarray,
        layers: list[int] | None = None,
        position: int = -1,
        device="cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # the frozen model may run in bfloat16 (7B-class models on a 40 GB card); the maps,
        # the cached hidden states and all log-probabilities stay in float32
        self.dtype = dtype
        self.model = model.to(device).to(dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self.decoder_layers = self._find_layers(model)
        n_layers = len(self.decoder_layers)
        self.layer_indices = list(range(n_layers + 1)) if layers is None else list(layers)
        self.loci = tuple(f"layer{i}" for i in self.layer_indices)
        self.position = position
        self.register_buffer(
            "output_token_ids", torch.as_tensor(output_token_ids, dtype=torch.long)
        )
        self.n_outputs = len(output_token_ids)
        self.d_model = model.config.hidden_size

    @staticmethod
    def _find_layers(model) -> nn.ModuleList:
        for attr in ("gpt_neox.layers", "model.layers", "transformer.h"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise ValueError("cannot find the decoder layers of this model")

    def width(self, locus: str) -> int:
        return self.d_model

    def _layer(self, locus: str) -> int:
        return int(locus.removeprefix("layer"))

    def prepare_input(self, tokens) -> Tensor:
        return torch.as_tensor(np.asarray(tokens), dtype=torch.long)

    @torch.no_grad()
    def hidden(self, tokens: Tensor, locus: str, batch_size: int = 256) -> Tensor:
        out = []
        for i in range(0, len(tokens), batch_size):
            res = self.model(tokens[i : i + batch_size].to(self.device), output_hidden_states=True)
            out.append(res.hidden_states[self._layer(locus)][:, self.position, :].float())
        return torch.cat(out)

    forward_chunk: int = 1024  # prompts per intervened forward pass

    def logits_from_hidden(
        self, h: Tensor, locus: str, base_tokens: Tensor, batch_size: int | None = None
    ) -> Tensor:
        """Run the base prompts with the residual at ``locus``/``position`` replaced by ``h``.

        Evaluated in chunks of ``batch_size`` prompts: a prequential block can hold tens of
        thousands of items, and one forward pass over all of them overflows a 40 GB GPU.
        """
        layer = self._layer(locus)
        if layer >= len(self.decoder_layers):
            # the final residual: apply the final norm and the unembedding directly
            final = self._final_logits(h)
            return final
        batch_size = batch_size or self.forward_chunk
        if len(h) > batch_size:
            return torch.cat(
                [
                    self.logits_from_hidden(
                        h[i : i + batch_size], locus, base_tokens[i : i + batch_size]
                    )
                    for i in range(0, len(h), batch_size)
                ]
            )
        target = self.decoder_layers[layer]

        def pre_hook(module, args, kwargs):
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            hs = hs.clone()
            hs[:, self.position, :] = h.to(hs.dtype)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = hs
                return args, kwargs
            return (hs, *args[1:]), kwargs

        handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            logits = self.model(base_tokens.to(self.device)).logits[:, self.position, :]
        finally:
            handle.remove()
        return logits[:, self.output_token_ids].float()

    def _final_logits(self, h: Tensor) -> Tensor:
        """Unembedding of the final hidden state.

        Hugging Face's last ``hidden_states`` entry is already normalised by the final layer
        norm (GPT-NeoX, GPT-2), so the final locus is that post-norm state and only the
        unembedding is applied.
        """
        h = h.to(next(self.model.parameters()).dtype)
        return self.model.get_output_embeddings()(h)[:, self.output_token_ids].float()

    @torch.no_grad()
    def logits(self, tokens: Tensor) -> Tensor:
        logits = self.model(tokens.to(self.device)).logits[:, self.position, self.output_token_ids]
        return logits.float()

    def reinitialised(self, seed: int) -> HFNetwork:
        """Same architecture with freshly initialised weights (the untrained control)."""
        from transformers import AutoModelForCausalLM

        torch.manual_seed(seed)
        fresh = AutoModelForCausalLM.from_config(self.model.config)
        return HFNetwork(
            fresh,
            self.output_token_ids.cpu().numpy(),
            self.layer_indices,
            self.position,
            self.device,
            dtype=self.dtype,
        )
