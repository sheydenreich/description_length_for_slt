"""Hessian and Fisher spectra of the subspace map's loss at its MAP on IOI (GPT-2).

At the fitted subspace the mean loss's Hessian (top eigenpairs by Lanczos on Hessian-vector
products) and the empirical Fisher (covariance of per-example gradients) are compared
direction by direction, as on the oracle (``reports/oracle/hessian_vs_fisher.txt``): the
plug-in code's prequential coefficient is ``(1/2) tr(F H^-1)`` over the resolved directions,
the Laplace prediction for a Bayes code is ``(1/2) sum n h / (n h + alpha)``. Run on a GPU:
  sbatch slurm/gpu.sbatch python -m bitsev.experiments.ioi_curvature --out results/curv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from bitsev.alignment import maps as M
from bitsev.alignment.intervene import HiddenCache
from bitsev.alignment.learners import (
    LaplaceAlignmentLearner,
    LaplaceBudget,
    _Interchange,
    _unflatten,
)
from bitsev.experiments import run_stage1_ioi as R


def analyse_locus(
    net, pool, targets, locus: str, args, device, fisher_targets=None, hessian_targets=None
) -> dict:
    """``fisher_targets``: targets on which the per-example gradients are taken (default: the
    fitted targets; fresh targets give the generalisation Fisher). ``hessian_targets``: targets
    of the mean loss whose Hessian is taken at the MAP (default: the fitted targets; fresh
    targets estimate the population Hessian, free of the fit's interpolation)."""
    t0 = time.time()
    cache = HiddenCache.build(net, locus, pool, device)
    learner = LaplaceAlignmentLearner(
        cache,
        lambda d=cache.width: M.SubspaceMap(d, args.d_v),
        LaplaceBudget(steps=args.map_steps, prior_precision=args.prior_precision, n_samples=0),
        warm_start=False,
        seed=0,
        channel=targets.channel,
    )
    pred = learner.fit(targets.pairs, targets.y)
    phi = pred.phi
    nll = float(-pred.log_prob(targets.pairs, targets.y).mean())
    print(f"{locus}: MAP fitted, train NLL {nll:.4f} ({time.time() - t0:.0f}s)", flush=True)
    params = [q for q in phi.parameters() if q.requires_grad]
    theta = torch.cat([q.detach().reshape(-1) for q in params])
    wrapper = _Interchange(phi)
    ht = targets if hessian_targets is None else hessian_targets
    h_b, h_s = cache.pair_states(ht.pairs)
    labels = torch.as_tensor(ht.y, device=device)

    def mean_loss(vec, idx=None):
        if idx is None:
            b, s, y, pairs = h_b, h_s, labels, ht.pairs
        else:
            b, s, y, pairs = h_b[idx], h_s[idx], labels[idx], ht.pairs[idx]
        h_new = functional_call(wrapper, _unflatten(phi, vec), (b, s))
        logits = cache.downstream(h_new, pairs)
        log_p = torch.log_softmax(logits, 1).gather(1, y[:, None])[:, 0]
        return -targets.channel.log_prob(log_p).mean()

    # per-example gradients (vmap does not work through the hooked HF model)
    t0 = time.time()
    ft = targets if fisher_targets is None else fisher_targets
    fb, fs = cache.pair_states(ft.pairs)
    flabels = torch.as_tensor(ft.y, device=device)

    def example_loss(vec, i):
        h_new = functional_call(wrapper, _unflatten(phi, vec), (fb[i : i + 1], fs[i : i + 1]))
        logits = cache.downstream(h_new, ft.pairs[i : i + 1])
        log_p = torch.log_softmax(logits, 1).gather(1, flabels[i : i + 1, None])[:, 0]
        return -ft.channel.log_prob(log_p).mean()

    grads = torch.empty(len(ft.y), theta.numel(), device=device)
    for i in range(len(ft.y)):
        v = theta.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(example_loss(v, i), v)
        grads[i] = g
    mean_grad = grads.mean(0)
    print(
        f"{locus}: per-example gradients ({time.time() - t0:.0f}s); "
        f"|mean grad| {mean_grad.norm():.2e}",
        flush=True,
    )

    # Hessian-vector products of the mean loss, Lanczos for the top eigenpairs
    t0 = time.time()
    v0 = theta.clone().requires_grad_(True)
    counter = {"n": 0}
    # the fused attention kernels have no double backward; the math backend does
    with sdpa_kernel(SDPBackend.MATH):
        (g0,) = torch.autograd.grad(mean_loss(v0), v0, create_graph=True)

        def hvp(x):
            x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
            (hv,) = torch.autograd.grad(g0, v0, grad_outputs=x_t, retain_graph=True)
            counter["n"] += 1
            return hv.detach().cpu().numpy().astype(np.float64)

        if args.full:
            # the dense Hessian from one product per coordinate, then an exact eigendecomposition
            dim = theta.numel()
            hess = np.empty((dim, dim), dtype=np.float32)
            for j in range(dim):
                e = np.zeros(dim, dtype=np.float32)
                e[j] = 1.0
                hess[j] = hvp(e)
                if j % 2000 == 0:
                    print(
                        f"{locus}: {j}/{dim} Hessian columns ({time.time() - t0:.0f}s)", flush=True
                    )
            hess = 0.5 * (hess + hess.T)
            h, u = np.linalg.eigh(hess.astype(np.float64))
        else:
            op = LinearOperator((theta.numel(), theta.numel()), matvec=hvp, dtype=np.float64)
            h, u = eigsh(op, k=min(args.top, theta.numel() - 2), which="LA", tol=1e-4, maxiter=4000)
    order = np.argsort(-h)
    h, u = h[order], u[:, order]
    print(
        f"{locus}: {len(h)} Hessian eigenpairs from {counter['n']} products "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    u_t = torch.as_tensor(u.astype(np.float32), device=device)
    f = ((grads @ u_t) ** 2).mean(0).cpu().numpy().astype(np.float64)  # u_i^T F u_i
    ratio = np.where(h > 1e-8, f / np.maximum(h, 1e-12), np.nan)
    alpha = args.prior_precision
    pred = {}
    for nn in (1024, 8192, 32768):
        resolved = int(np.sum(nn * np.maximum(h, 0) > alpha))
        pred[str(nn)] = {
            "resolved_directions": resolved,
            "laplace_bayes_slope": float(
                0.5 * np.sum(nn * np.maximum(h, 0) / (nn * np.maximum(h, 0) + alpha))
            ),
            "plugin_slope_tr_FHinv": float(0.5 * np.nansum(ratio[:resolved])),
        }
    print(f"{locus}: top Hessian eigenvalues {np.array2string(h[:12], precision=3)}")
    print(f"{locus}: F/H in those directions  {np.array2string(ratio[:12], precision=3)}")
    counts = [int(np.sum(h > tol * h[0])) for tol in (1e-2, 1e-3, 1e-4, 1e-5)]
    print(
        f"{locus}: eigenvalues above 1e-2 / 1e-3 / 1e-4 / 1e-5 of the top: {counts}; "
        f"negative: {int(np.sum(h < 0))}"
    )
    for nn, pr in pred.items():
        print(
            f"{locus}: n = {nn}: resolved {pr['resolved_directions']}, Laplace Bayes slope "
            f"{pr['laplace_bayes_slope']:.1f}, plug-in (1/2) tr(F H^-1) "
            f"{pr['plugin_slope_tr_FHinv']:.1f}"
        )
    return {
        "train_nll": nll,
        "hessian_targets": len(ht.y),
        "n_params": int(theta.numel()),
        "gauge_dim": args.d_v**2,
        "hessian_eigenvalues": h.tolist(),
        "fisher_in_hessian_directions": f.tolist(),
        "ratio_F_over_H": ratio.tolist(),
        "mean_grad_norm": float(mean_grad.norm()),
        "predictions": pred,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--layers", type=int, nargs="+", default=[11])
    p.add_argument("--n-targets", type=int, default=1024)
    p.add_argument("--d-v", type=int, default=16)
    p.add_argument("--prior-precision", type=float, default=0.01)
    p.add_argument("--top", type=int, default=200, help="Hessian eigenpairs by Lanczos")
    p.add_argument("--full", action="store_true", help="dense Hessian, all eigenpairs")
    p.add_argument(
        "--fisher-split",
        choices=["train", "iid"],
        default="train",
        help="targets for the Fisher: the fitted ones or the held-out i.i.d. split",
    )
    p.add_argument(
        "--hessian-split",
        choices=["train", "iid"],
        default="train",
        help="targets of the loss whose Hessian is taken at the MAP",
    )
    p.add_argument("--n-eval", type=int, default=1000, help="size of the held-out i.i.d. split")
    p.add_argument("--map-steps", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("results/ioi_curvature"))
    args = p.parse_args(argv)
    ioi_args = R.parse(
        ["--model", args.model, "--layers", *map(str, args.layers)]
        + ["--n-targets", str(args.n_targets), "--n-eval", str(args.n_eval)]
    )
    net, alg, pool, targets, evals, _ = R.build(ioi_args)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for locus in net.loci:
        report[locus] = analyse_locus(
            net,
            pool,
            targets,
            locus,
            args,
            ioi_args.device,
            fisher_targets=evals["iid"] if args.fisher_split == "iid" else None,
            hessian_targets=evals["iid"] if args.hessian_split == "iid" else None,
        )
        (args.out / "curvature.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
