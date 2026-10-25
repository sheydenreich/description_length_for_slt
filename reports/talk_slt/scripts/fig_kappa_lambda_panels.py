"""The kappa-vs-lambda slide of the 7-minute deck as one 2 x 2 figure: (a) kappa against d_V
(fig_kappa_vs_dv), (b) predicted against measured kappa (fig_predicted_vs_measured), (c, d) Hessian
against Fisher on GPT-2 layer 11 and the accumulated coefficients (fig_fisher_hessian). Each panel is drawn
by the function of its own script, so the numbers and sources are those named there.

The figure is drawn at about the size of its slot on the slide (12.3 x 5.3 in once cropped), so font sizes
are close to slide points.
"""
import matplotlib.pyplot as plt

from common import style, save
import fig_fisher_hessian
import fig_kappa_vs_dv
import fig_predicted_vs_measured

TITLES = [
    "κ stays flat in d_V; the nominal λ grows",
    "½ tr(F H⁻¹) gets the order of κ; λ is 20–400× off",
    "GPT-2: the Fisher is 2 % of the Hessian",
    "so the plug-in pays ~1 % of the Laplace count",
]
TEXT_SCALE = 0.62  # annotations of the full-size panels, scaled to the slot
LEGEND_AT = ["upper left", "lower right", "lower left", "upper left"]  # clear of each panel's annotations


def main():
    style()
    plt.rcParams.update({"axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "lines.markersize": 6, "lines.linewidth": 1.6})
    fig, axes = plt.subplots(2, 2, figsize=(14.3, 5.6), gridspec_kw={"hspace": 0.62, "wspace": 0.2})
    fig_kappa_vs_dv.panel(axes[0][0], compact=True)
    fig_predicted_vs_measured.panel(axes[0][1], compact=True)
    fig_fisher_hessian.panels(axes[1])
    for ax, tag, title, loc in zip(axes.flat, "abcd", TITLES, LEGEND_AT):
        ax.set_title(f"({tag})  {title}", loc="left", fontsize=12, pad=6)
        for t in ax.texts:
            t.set_fontsize(t.get_fontsize() * TEXT_SCALE)
            if tag == "c":  # the ratio text would sit on the Hessian curve: move it to the empty lower left
                t.set_position((0.03, 0.56))
                t.set_horizontalalignment("left")
        for line in ax.lines:
            line.set_markersize(line.get_markersize() * 0.7)
        leg = ax.legend(loc=loc, fontsize=9.5, handlelength=1.6, markerscale=2.5 if tag == "c" else 1.0)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    save(fig, "kappa_lambda_panels")


if __name__ == "__main__":
    main()
