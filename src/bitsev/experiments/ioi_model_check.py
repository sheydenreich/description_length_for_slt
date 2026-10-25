"""Which larger models solve IOI on our prompts?

Accuracy among names, name pool, prompt length, memory.
"""

import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bitsev.testbeds import ioi

for name in sys.argv[1:]:
    t0 = time.time()
    try:
        # "repo@revision" selects a training checkpoint (e.g. EleutherAI/pythia-1.4b@step4000)
        repo, _, revision = name.partition("@")
        tok = AutoTokenizer.from_pretrained(repo, revision=revision or None)
        model = (
            AutoModelForCausalLM.from_pretrained(
                repo,
                revision=revision or None,
                torch_dtype=torch.bfloat16 if os.environ.get("BITSEV_BF16") else torch.float32,
            )
            .to("cuda")
            .eval()
        )
        names = ioi.single_token_names(tok)
        rng = np.random.default_rng(0)
        prompts = ioi.build_prompts(tok, tuple(names[:16]), 400, rng)
        ids = torch.as_tensor(prompts.tokens, device="cuda")
        with torch.no_grad():
            logits = model(ids).logits[:, -1, :]
        name_ids = torch.as_tensor(
            [tok.encode(" " + nm, add_special_tokens=False)[0] for nm in names[:16]], device="cuda"
        )
        pred = logits[:, name_ids].argmax(1).cpu().numpy()
        acc = float((pred == prompts.io).mean())
        top1 = float((logits.argmax(1).cpu().numpy() == name_ids[prompts.io].cpu().numpy()).mean())
        cfg = model.config
        print(
            f"{name:<28} IO-over-S among 16 names {acc:.3f}; top-1 over vocab {top1:.3f}; "
            f"single-token names {len(names)}; "
            f"layers {cfg.num_hidden_layers}, width {cfg.hidden_size}; "
            f"prompt tokens {prompts.tokens.shape[1]}; "
            f"GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB; {time.time() - t0:.0f}s",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print(f"{name:<28} FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
