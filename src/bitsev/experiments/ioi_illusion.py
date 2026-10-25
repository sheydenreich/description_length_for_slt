"""Subspace-illusion check (Makelov et al. 2024) for the fitted IOI subspaces.

An interchange along a subspace ``U`` can succeed without ``U`` being the variable's
representation: ``U`` may combine a *disconnected* direction that varies with the variable
but is ignored downstream with a *dormant* direction that the readout responds to but that
barely varies on the data. Patching the combination writes the variable's variation into
the dormant pathway. The code length cannot exclude this, because the readout really uses
that pathway.

Makelov et al. split ``U`` by the row space of the next linear map. At a residual-stream
locus every later layer reads the state, so the split here is by the readout's own
sensitivity: the per-prompt Jacobians ``J_i = d log softmax(names) / d h`` (``K x d``) at the
locus, stacked over prompts, define by their SVD the *output-connected* subspace ``C_tau``
holding a share ``tau`` of the Jacobian energy. For a fitted ``U`` the check reports

* the share of ``U`` in ``C_tau`` against the share ``k_tau / d`` of a random subspace;
* IIA when patching ``U``, its connected part ``orth(P_C U)`` and its disconnected part
  ``orth((I - P_C) U)`` separately (an illusion needs both parts: neither alone works);
* per direction of ``U``, ordered by readout sensitivity, the sensitivity, the variance
  across prompts (dormancy) and the share of that variance explained by the IO name, each
  relative to random directions. The illusion's signature is that the sensitive directions
  are dormant and the variable-carrying ones insensitive.

Run on a GPU:
  sbatch slurm/gpu.sbatch python -m bitsev.experiments.ioi_illusion --out results/illusion
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from bitsev.alignment import maps as M
from bitsev.alignment.hf_network import HFNetwork
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import GradientAlignmentLearner, TrainingBudget
from bitsev.alignment.targets import TargetSet
from bitsev.experiments import run_stage1_ioi as R

THRESHOLDS = (0.9, 0.99, 0.999)


def readout_jacobians(
    network: HFNetwork, tokens: Tensor, states: Tensor, locus: str, batch_size: int = 64
) -> Tensor:
    """Per-prompt Jacobians of the log-softmax over the output names with respect to the
    residual at ``locus`` and the read position, stacked to ``(n_prompts * K, d)``.
    ``states`` are the prompts' residuals at the locus (used at the final locus, where the
    readout is the unembedding of the state itself)."""
    if network._layer(locus) >= len(network.decoder_layers):
        h = states.detach().clone().requires_grad_(True)
        log_p = torch.log_softmax(network._final_logits(h), dim=1)
        jac = torch.stack(
            [
                torch.autograd.grad(log_p[:, j].sum(), h, retain_graph=True)[0]
                for j in range(log_p.shape[1])
            ],
            dim=1,
        )
        return jac.reshape(-1, network.d_model).float()
    target = network.decoder_layers[network._layer(locus)]
    pos, out_ids = network.position, network.output_token_ids
    captured: list[Tensor] = []

    def pre_hook(module, args, kwargs):
        hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        hs = hs.detach().clone().requires_grad_(True)
        captured.append(hs)
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs
            return args, kwargs
        return (hs, *args[1:]), kwargs

    rows = []
    handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        for i in range(0, len(tokens), batch_size):
            captured.clear()
            batch = tokens[i : i + batch_size].to(network.device)
            logits = network.model(batch).logits[:, pos, :][:, out_ids].float()
            log_p = torch.log_softmax(logits, dim=1)
            hs = captured[0]
            jac = torch.empty(len(batch), len(out_ids), network.d_model, device=network.device)
            for j in range(len(out_ids)):
                (g,) = torch.autograd.grad(log_p[:, j].sum(), hs, retain_graph=j < len(out_ids) - 1)
                jac[:, j] = g[:, pos, :].float()
            rows.append(jac.reshape(-1, network.d_model))
    finally:
        handle.remove()
    return torch.cat(rows)


def patch_iia(cache: HiddenCache, basis: Tensor, targets: TargetSet) -> float:
    """IIA of the interchange along the orthonormal ``basis`` (``d x k``)."""
    h_b, h_s = cache.pair_states(targets.pairs)
    with torch.no_grad():
        h_new = h_b + (h_s - h_b) @ basis @ basis.T
        logits = cache.downstream(h_new, targets.pairs)
    return float((logits.argmax(1).cpu().numpy() == targets.y_clean).mean())


def orth(a: Tensor, tol: float = 1e-6) -> Tensor:
    """Orthonormal basis of the column span of ``a``."""
    u, s, _ = torch.linalg.svd(a, full_matrices=False)
    return u[:, s > tol * s[0]]


def direction_profile(states: Tensor, labels: np.ndarray, jac: Tensor, dirs: Tensor) -> dict:
    """Sensitivity, variance and variance explained by the label along unit ``dirs``."""
    proj = states @ dirs  # (n, m)
    var = proj.var(0)
    lab = torch.as_tensor(labels, device=states.device)
    means = torch.zeros(int(lab.max()) + 1, dirs.shape[1], device=states.device)
    counts = torch.bincount(lab, minlength=len(means)).float()
    means.index_add_(0, lab, proj)
    means = means / counts.clamp_min(1)[:, None]
    between = ((means[lab] - proj.mean(0)) ** 2).mean(0)
    sens = (jac @ dirs).pow(2).sum(0) / len(jac)
    return {
        "sensitivity": sens.cpu().numpy(),
        "variance": var.cpu().numpy(),
        "label_share": (between / var.clamp_min(1e-12)).cpu().numpy(),
    }


def analyse(net, alg, pool, targets, evals, locus: str, args, seed: int) -> dict:
    device = args.device
    t0 = time.time()
    cache = HiddenCache.build(net, locus, pool, device)
    d = cache.width
    learner = GradientAlignmentLearner(
        cache,
        lambda: M.SubspaceMap(d, args.d_v),
        TrainingBudget(args.map_steps, args.map_lr, args.map_batch),
        seed=seed,
        family_name="subspace",
        channel=targets.channel,
    )
    phi = learner.fit(targets.pairs, targets.y).phi
    with torch.no_grad():
        u = phi.matrix()[:, : args.d_v].detach()
    print(f"{locus} seed {seed}: subspace fitted ({time.time() - t0:.0f}s)", flush=True)

    # output-connected subspace of the locus from the readout Jacobians of training prompts
    bases = np.unique(targets.base)
    prompt_idx = bases[: args.jacobian_prompts] if args.jacobian_prompts > 0 else bases
    jac = readout_jacobians(
        net, net.prepare_input(pool[prompt_idx]), cache.states[prompt_idx], locus
    )
    _, s, vh = torch.linalg.svd(jac, full_matrices=False)
    energy = torch.cumsum(s**2, 0) / (s**2).sum()

    states = cache.states[bases]
    labels = alg.variable_values(pool[bases])[targets.variable]
    g = torch.Generator(device="cpu").manual_seed(seed)
    random_dirs = torch.nn.functional.normalize(torch.randn(d, 256, generator=g), dim=0).to(device)
    ref = direction_profile(states, labels, jac, random_dirs)
    ref_sens, ref_var = float(np.median(ref["sensitivity"])), float(np.median(ref["variance"]))

    out = {"locus": locus, "seed": seed, "d": d, "d_v": args.d_v}
    out["n_jacobian_prompts"] = len(prompt_idx)
    out["iia_full"] = {name: patch_iia(cache, u, t) for name, t in evals.items()}
    out["split"] = {}
    for tau in THRESHOLDS:
        k = int(torch.searchsorted(energy, torch.tensor(tau, device=energy.device)).item()) + 1
        c = vh[:k].T
        u_c = c @ (c.T @ u)
        share = float((u_c**2).sum() / args.d_v)
        b_c, b_n = orth(u_c), orth(u - u_c)
        out["split"][str(tau)] = {
            "k": k,
            "share_in_connected": share,
            "random_share": k / d,
            "iia_connected_part": {n: patch_iia(cache, b_c, t) for n, t in evals.items()},
            "iia_disconnected_part": {n: patch_iia(cache, b_n, t) for n, t in evals.items()},
        }

    # directions of U ordered by the readout's sensitivity to them
    _, _, rh = torch.linalg.svd(jac @ u, full_matrices=False)
    dirs = u @ rh.T
    prof = direction_profile(states, labels, jac, dirs)
    out["directions"] = {
        "sensitivity_rel": (prof["sensitivity"] / ref_sens).tolist(),
        "variance_rel": (prof["variance"] / ref_var).tolist(),
        "label_share": prof["label_share"].tolist(),
        "random_label_share_median": float(np.median(ref["label_share"])),
    }
    out["iia_top_sensitive"] = {
        str(m): {n: patch_iia(cache, dirs[:, :m], t) for n, t in evals.items()}
        for m in (1, 2, 4, 8, args.d_v)
        if m <= args.d_v
    }
    print(
        f"{locus} seed {seed}: IIA full {out['iia_full']['iid']:.3f}; "
        + "; ".join(
            f"tau {t}: k {v['k']}, share {v['share_in_connected']:.3f} "
            f"(random {v['random_share']:.3f}), IIA conn {v['iia_connected_part']['iid']:.3f}, "
            f"disc {v['iia_disconnected_part']['iid']:.3f}"
            for t, v in out["split"].items()
        )
        + f" ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--layers", type=int, nargs="+", default=[10, 11, 12])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--d-v", type=int, default=16)
    p.add_argument("--map-steps", type=int, default=400)
    p.add_argument("--map-lr", type=float, default=1e-2)
    p.add_argument("--map-batch", type=int, default=256)
    p.add_argument("--jacobian-prompts", type=int, default=256, help="0: every training base")
    p.add_argument("--out", type=Path, default=Path("results/ioi_illusion_gpt2"))
    args = p.parse_args(argv)
    ioi_args = R.parse(["--model", args.model, "--layers", *map(str, args.layers)])
    args.device = ioi_args.device
    net, alg, pool, targets, evals, _ = R.build(ioi_args)
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for locus in net.loci:
        for seed in range(args.seeds):
            report.append(analyse(net, alg, pool, targets, evals, locus, args, seed))
            (args.out / "illusion.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
