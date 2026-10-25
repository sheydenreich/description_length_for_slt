"""(c) GPT-2 layer 11, learned d_V = 16 subspace at its MAP (1024 targets): the information equality fails.
Left: Hessian eigenvalues h_i and the held-out empirical Fisher F_ii in the same directions.
Right: the two coefficients accumulated over directions (sorted by curvature): a Bayes code pays 1/2 for
every resolved curved direction (Laplace count -> 6016); the plug-in pays (1/2) sum F_ii / h_i (-> 70.6).

Source: reports/stage1_ioi/curvature_gpt2_layer11_full_heldout_fisher.json (keys hessian_eigenvalues,
fisher_in_hessian_directions; eigenvalues sorted in decreasing order; 256 gauge zeros excluded).
"""
import numpy as np
import matplotlib.pyplot as plt

from common import style, save, load_json, gpt2_map_slopes, PLUGIN, BAYES, MUTED, INK, INK2


def panels(axes):
    d = load_json("stage1_ioi/curvature_gpt2_layer11_full_heldout_fisher.json")["layer11"]
    h = np.array(d["hessian_eigenvalues"])
    F = np.array(d["fisher_in_hessian_directions"])
    pos = h > 0
    h, F = h[pos], F[pos]
    order = np.argsort(-h)
    h, F = h[order], F[order]
    rank = np.arange(1, len(h) + 1)

    ax = axes[0]
    ax.scatter(rank, np.clip(F, 1e-9, None), s=4, color=PLUGIN, alpha=0.35, lw=0, label="Fisher F_ii (held-out)")
    ax.plot(rank, h, color=INK, lw=2.5, label="Hessian eigenvalue h_i")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 3)
    ax.set_xlabel("direction (sorted by curvature)")
    ax.set_ylabel("curvature / gradient variance")
    leg = ax.legend(loc="lower left", markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ratio = F.sum() / h.sum()
    ax.text(0.97, 0.95, f"tr F / tr H = {ratio:.1%}\nmedian F_ii/h_i = {np.median(F / h):.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=15, color=INK)

    ax = axes[1]
    bayes = 0.5 * rank
    plug = 0.5 * np.cumsum(F / h)
    ax.plot(rank, bayes, color=BAYES, lw=3, label="Bayes (Laplace): ½ per resolved direction")
    ax.plot(rank, plug, color=PLUGIN, lw=3, label="plug-in: ½ Σ F_ii / h_i")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 2e4)
    ax.set_xlabel("number of directions included")
    ax.set_ylabel("accumulated coefficient per ln n")
    ax.annotate(f"λ = {bayes[-1]:,.0f}", xy=(rank[-1], bayes[-1]), xytext=(rank[-1] * 0.08, bayes[-1] * 1.6),
                fontsize=16, color=BAYES, arrowprops=dict(arrowstyle="-", color=BAYES))
    kv, ke = gpt2_map_slopes()["valid"]
    ax.annotate(f"κ predicted {plug[-1]:.1f}\n(measured {kv:.0f} ± {ke:.0f})", xy=(rank[-1], plug[-1]),
                xytext=(rank[-1] * 0.04, plug[-1] * 0.03), fontsize=16, color=PLUGIN,
                arrowprops=dict(arrowstyle="-", color=PLUGIN))
    ax.legend(loc="lower right")


def main():
    style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"wspace": 0.28})
    panels(axes)
    save(fig, "gpt2_fisher_vs_hessian_slide")


if __name__ == "__main__":
    main()
