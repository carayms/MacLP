"""Figure 1: coverage gap of weighted conformal prediction with estimated weights.

Reads the two result files of ``coverage_gap_data.py`` (m = n = 100 and
m = n = 300) and writes ``figures/figure1.pdf``.

Example:
    python synthetic/coverage_gap_plot.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import FIGURES_ROOT, RESULTS_ROOT
from exputils.plotting import (fig1_plot_classes, fig1_plot_weight_boxes,
                               plot_coverage_histograms, plot_result_box, set_paper_style)

CLASS_COLORS = ("#47abcc", "#ffba00")
SMALL_COLOR, LARGE_COLOR, ORACLE_COLOR = "#cc4b5f", "#cc2525", "#263238"


def draw_arrow(fig, axA, axB, posA, posB, rad=-0.05):
    style = mpatch.ArrowStyle("Fancy", head_length=0.4, head_width=0.4, tail_width=0.1)
    fig.add_artist(ConnectionPatch(xyA=posA, xyB=posB, coordsA="axes fraction", coordsB="axes fraction",
                                   axesA=axA, axesB=axB, arrowstyle=style, color="0.3",
                                   connectionstyle=f"arc3,rad={rad}", shrinkA=8, shrinkB=8, capstyle="round"))


def load(results_dir, mn):
    with open(Path(results_dir) / f"coverage_gap_m.{mn}_n.{mn}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT / "synthetic"))
    parser.add_argument("--fig-dir", default=str(FIGURES_ROOT))
    parser.add_argument("--small", type=int, default=100)
    parser.add_argument("--large", type=int, default=300)
    args = parser.parse_args()

    small, large = load(args.results_dir, args.small), load(args.results_dir, args.large)
    set_paper_style()
    fig = plt.figure(figsize=(7.5, 3))
    gs = gridspec.GridSpec(8, 5, figure=fig, hspace=0.2, wspace=0.5)

    # outer columns: source / target samples for the weights
    ax_src_s, ax_tgt_s = fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[2:4, 0])
    ax_src_l, ax_tgt_l = fig.add_subplot(gs[0:2, 4]), fig.add_subplot(gs[2:4, 4])
    fig1_plot_classes(ax_src_s, small["example"]["src"], f"Source ($P$)\n$m$={args.small}", CLASS_COLORS,
                      pos_title=[-0.42, 0.2])
    fig1_plot_classes(ax_tgt_s, small["example"]["tgt"], f"Target ($Q$)\n$n$={args.small}", CLASS_COLORS,
                      pos_title=[-0.42, 0.2])
    fig1_plot_classes(ax_src_l, large["example"]["src"], f"Source ($P$)\n$m$={args.large}", CLASS_COLORS,
                      pos_title=[1.45, 0.2])
    fig1_plot_classes(ax_tgt_l, large["example"]["tgt"], f"Target ($Q$)\n$n$={args.large}", CLASS_COLORS,
                      pos_title=[1.45, 0.2])

    # estimated weights; black dots are the true weights
    ax_w_s, ax_w_l = fig.add_subplot(gs[1:3, 1]), fig.add_subplot(gs[1:3, 3])
    for ax, res in [(ax_w_s, small), (ax_w_l, large)]:
        fig1_plot_weight_boxes(ax, res["bbse_wts"], "Est. Weights", CLASS_COLORS)
        ax.plot([1, 2], res["true_wts"], ".", color="black", ms=4, zorder=3, alpha=0.7)

    # center: calibration and test sets
    ax_cal, ax_test = fig.add_subplot(gs[0:2, 2]), fig.add_subplot(gs[3:5, 2])
    fig1_plot_classes(ax_cal, small["example"]["calib"], "Calibration Set ($P$)", CLASS_COLORS)
    fig1_plot_classes(ax_test, small["example"]["testing"], "Testing Set ($Q$)", CLASS_COLORS, pos_title=[0.5, 1.0])

    # bottom center: prediction-set sizes
    ax_box = fig.add_subplot(gs[6:, 2])
    for spine in ["left", "right", "top"]:
        ax_box.spines[spine].set_visible(False)
    plot_result_box(ax_box, small["set_size"]["estimated"], pos=1, color=SMALL_COLOR)
    plot_result_box(ax_box, large["set_size"]["estimated"], pos=2, color=LARGE_COLOR)
    plot_result_box(ax_box, small["set_size"]["oracle"], pos=3, color=ORACLE_COLOR)
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels([f"{args.small}\n$m, n$", f"{args.large}\n$m, n$"], fontsize=7)
    ax_box.text(0.92, 1.05, "Oracle", transform=ax_box.transAxes, ha="center", va="bottom", fontsize=9)
    ax_box.text(0.15, 1.05, "Finite-sample", transform=ax_box.transAxes, ha="center", va="bottom", fontsize=9)
    ax_box.text(0.5, -0.8, "Prediction Set Size", transform=ax_box.transAxes, ha="center", va="bottom", fontsize=9)
    ax_box.axvline(x=2.5, color="black", linestyle="--", lw=1)
    ax_box.tick_params(axis="y", direction="inout", length=6)

    # bottom left / right: coverage histograms
    ax_cov_l = fig.add_subplot(gs[5:, 0:2])
    cov_small = [small["coverage"]["oracle"], small["coverage"]["estimated"]]
    plot_coverage_histograms(ax_cov_l, cov_small, ["--", "-"], ["oracle", f"$m,n$={args.small}"],
                             [ORACLE_COLOR, SMALL_COLOR], lgn_pos="upper left", hist_range=(0.8, 1.0), n_bins=35)
    ax_cov_l.axvspan(cov_small[1].mean(), cov_small[0].mean(), color="#bfbdbd", zorder=0)
    ax_cov_l.text(0.82, 1.05, "Under coverage", transform=ax_cov_l.transAxes, ha="center", va="bottom", fontsize=11)

    ax_cov_r = fig.add_subplot(gs[5:, 3:])
    plot_coverage_histograms(ax_cov_r, [large["coverage"]["oracle"], large["coverage"]["estimated"]], ["--", "-"],
                             ["oracle", f"$m,n$={args.large}"], [ORACLE_COLOR, LARGE_COLOR],
                             lgn_pos="upper left", hist_range=(0.8, 1.0), n_bins=35)

    draw_arrow(fig, ax_w_s, ax_cal, (1, 0.5), (0, 0.5), rad=-0.15)
    draw_arrow(fig, ax_w_l, ax_cal, (0, 0.5), (1, 0.5), rad=+0.15)
    draw_arrow(fig, ax_cal, ax_test, (0.5, 0.1), (0.5, 1.1), rad=0.0)
    draw_arrow(fig, ax_test, ax_box, (0.5, 0.15), (0.5, 1.15), rad=0.0)
    draw_arrow(fig, ax_w_s, ax_cov_l, (0.0, -0.95), (0.485, 0.9), rad=+0.5)

    plt.subplots_adjust(top=0.93, bottom=0.15, left=0.1, right=0.9)
    pos = ax_cov_r.get_position()
    ax_cov_r.set_position([pos.x0 + 0.06, pos.y0, pos.width, pos.height])

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "figure1.pdf")
    print(f"Figure saved to {fig_dir / 'figure1.pdf'}")


if __name__ == "__main__":
    main()
