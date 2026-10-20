"""Figure 4 (nuImages) and summary tables.

Reads the pickles written by ``real_world_shift_experiment.py`` and draws
``figures/nuImagesRes.pdf`` with four panels:

* upper left: volcano plot of the GE/LP interval-length ratios.  For each
  sample size, r_k = log2 of the GE/LP length ratio of class k averaged over
  repetitions; the x-axis is mean_k r_k and the y-axis is -log10 p of a
  one-sided one-sample t-test of {r_k} (H0: mean <= 0);
* lower left: Monte Carlo volume ratio vol(Omega)/vol(Omega_LP);
* upper right: empirical coverage; lower right: average set size.

The same numbers are printed as Markdown tables (mean (std) over repetitions).

Example:
    python nuimages/analyze_results.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import FIGURES_ROOT, RESULTS_ROOT
from exputils.io import load_results
from exputils.plotting import generate_stats_markdown_table, set_paper_style

METHOD_LABELS = {
    "oracle": r"$\boldsymbol{\omega}^*$",
    "estG.estWts": r"$\widehat{\boldsymbol{\omega}}$",
    "estG.Omega": r"$\boldsymbol{\Omega}$",
    "estG.Omega_LP": r"$\boldsymbol{\Omega}_{\mathrm{LP}}$",
    "estG.Omega_GE": r"$\boldsymbol{\Omega}_{\mathrm{GE}}$",
}
METHOD_COLORS = ["#de285f", "#457B9D", "#2A9D8F", "#8338EC", "#F4A261"]
PAPER_SAMPLE_SIZES = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]
# x-ranges of the two halves of the broken volcano axis in the paper figure
PAPER_VOLCANO_XLIMS = ((0, 2.2), (4.6, 6.2))


def find_result(results_dir, nominal_coverage, itvl_type, sz, n_calib, n_testing):
    pattern = (f"experiType.real-world_CI.{int(round(nominal_coverage * 100))}.{itvl_type}"
               f"_sampleSize.{sz}_lbsftAlp.real-world-shift_nCalib.{n_calib}_nTesting.{n_testing}_*.pkl")
    matches = sorted(Path(results_dir).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file matching {pattern} in {results_dir}, found {len(matches)}")
    return matches[0]


def collect(results_dir, sample_sizes, nominal_coverage, itvl_type, n_calib, n_testing):
    """Arrays (n_sizes, n_repts, K), (n_sizes, n_repts) and (n_sizes, n_methods, 2, n_repts)."""
    lengths, volumes, conformal = [], [], []
    for sz in sample_sizes:
        path = find_result(results_dir, nominal_coverage, itvl_type, sz, n_calib, n_testing)
        _, LPvsGE_length, vol_ratio, _, _, _, size_repts, coverage_repts = load_results(path)
        lengths.append(LPvsGE_length[1] / LPvsGE_length[0])        # GE / LP
        volumes.append(vol_ratio)
        conformal.append([[coverage_repts[m], size_repts[m].mean(axis=1)] for m in METHOD_LABELS])
    return np.array(lengths), np.array(volumes), np.array(conformal)


def draw_volcano(fig, pos, x, y, sample_sizes, cmap, norm):
    """Volcano panel; a broken x-axis as in the paper when the points fit its ranges."""
    scatter_kw = dict(c=sample_sizes, cmap=cmap, norm=norm, s=42, alpha=0.9,
                      edgecolors="black", linewidth=0.6, zorder=3)
    (l0, l1), (r0, r1) = PAPER_VOLCANO_XLIMS
    fits_paper_axis = np.all(((x >= l0) & (x <= l1)) | ((x >= r0) & (x <= r1)))
    shift = 0.06

    if not fits_paper_axis:
        ax = fig.add_axes([pos.x0, pos.y0 + shift, pos.width, pos.height])
        ax.scatter(x, y, **scatter_kw)
        ax.axhline(-np.log10(0.05), color="black", linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel(r"$-\log_{10}$(p-val)", fontsize=11)
        ax.set_xlabel(r"Mean $\log_2$Ratio (GE/LP)", fontsize=11)
        return

    gap = 0.04
    w_left = pos.width * (1 - gap) / 3
    w_right = pos.width * (1 - gap) * 2 / 3
    ax_left = fig.add_axes([pos.x0, pos.y0 + shift, w_left, pos.height])
    ax_right = fig.add_axes([pos.x0 + w_left + pos.width * gap, pos.y0 + shift, w_right, pos.height],
                            sharey=ax_left)
    for ax in (ax_left, ax_right):
        ax.scatter(x, y, **scatter_kw)
        ax.axhline(-np.log10(0.05), color="black", linestyle="--")
        ax.spines["top"].set_visible(False)
    ax_left.set_xlim(l0, l1)
    ax_right.set_xlim(r0, r1)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines[["left", "right"]].set_visible(False)
    ax_right.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

    d = 0.025  # diagonal break marks
    ax_left.plot((1 - d, 1 + d), (-d, d), transform=ax_left.transAxes, color="k", clip_on=False)
    ax_right.plot((-d, d), (-d, d), transform=ax_right.transAxes, color="k", clip_on=False)

    ax_left.set_ylabel(r"$-\log_{10}$(p-val)", fontsize=11)
    ax_right.set_xlabel(r"Mean $\log_2$Ratio (GE/LP)", fontsize=11)
    ax_right.xaxis.set_label_coords(0.25, -0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT / "nuimages"))
    parser.add_argument("--fig-dir", default=str(FIGURES_ROOT))
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=PAPER_SAMPLE_SIZES)
    parser.add_argument("--n-calib", type=int, default=5000)
    parser.add_argument("--n-testing", type=int, default=2500)
    parser.add_argument("--nominal-coverage", type=float, default=0.90)
    parser.add_argument("--itvl-type", default="Clopper-Pearson")
    args = parser.parse_args()

    sample_sizes = np.array(sorted(args.sample_sizes))
    n_sizes = sample_sizes.size
    methods = list(METHOD_LABELS)
    lengths, volumes, conformal = collect(args.results_dir, sample_sizes, args.nominal_coverage,
                                          args.itvl_type, args.n_calib, args.n_testing)

    # length ratios: per class, average over repetitions, then log2
    log2_ratio = np.log2(lengths.mean(axis=1))                     # (n_sizes, K)
    log2_mean = log2_ratio.mean(axis=1)
    log2_std = log2_ratio.std(axis=1, ddof=1)
    neg_log_p = np.array([-np.log10(ttest_1samp(r, 0, alternative="greater").pvalue + 1e-10)
                          for r in log2_ratio])
    vol_mean, vol_std = volumes.mean(axis=1), volumes.std(axis=1, ddof=1)
    cvg_mean, cvg_std = conformal[:, :, 0].mean(axis=2).T, conformal[:, :, 0].std(axis=2, ddof=1).T
    size_mean, size_std = conformal[:, :, 1].mean(axis=2).T, conformal[:, :, 1].std(axis=2, ddof=1).T

    header = ["sample size"] + [f"{sz / 1000:g}k" for sz in sample_sizes]
    print(generate_stats_markdown_table(log2_mean[None], log2_std[None], header,
                                        ["mean log2 ratio (GE/LP), std over classes"]))
    print(generate_stats_markdown_table(neg_log_p[None], None, header, ["-log10 p (t-test)"]))
    print(generate_stats_markdown_table(100 * vol_mean[None], 100 * vol_std[None], header,
                                        ["vol(Omega) / vol(Omega_LP) (%)"]))
    print("\nEmpirical coverage")
    print(generate_stats_markdown_table(cvg_mean, cvg_std, header, methods))
    print("\nAverage set size")
    print(generate_stats_markdown_table(size_mean, size_std, header, methods))

    # --- figure ---------------------------------------------------------------
    set_paper_style()
    col_sz = plt.get_cmap("Blues")(np.linspace(0.2, 0.8, n_sizes))
    cmap = mcolors.ListedColormap(col_sz)
    n_step = (sample_sizes[1] - sample_sizes[0]) / 2 if n_sizes > 1 else 500
    norm = mcolors.BoundaryNorm(np.append(sample_sizes - n_step, sample_sizes[-1] + n_step), cmap.N)
    size_ticks = [f"{sz / 1000:.0f}" for sz in sample_sizes]

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.45, height_ratios=[1, 5, 5], width_ratios=[1, 1.2])
    ax_volcano = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[2, 0])
    ax_cvg = fig.add_subplot(gs[1, 1])
    ax_set = fig.add_subplot(gs[2, 1])

    pos = ax_volcano.get_position()
    ax_volcano.remove()
    draw_volcano(fig, pos, log2_mean, neg_log_p, sample_sizes, cmap, norm)

    ax_bar.bar(np.arange(n_sizes), vol_mean, color=col_sz, edgecolor="black", linewidth=0.5,
               yerr=vol_std, alpha=0.8, capsize=3,
               error_kw={"capthick": 0.5, "elinewidth": 0.5, "ecolor": "black"})
    ax_bar.set_xticks(np.arange(n_sizes), size_ticks)
    ax_bar.set_xlabel(r"Sample Size ($\times 10^3$)", fontsize=11, labelpad=2)
    ax_bar.set_ylabel("Volume ratio", fontsize=11, labelpad=0)
    ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    for i_m, m in enumerate(methods):
        ax_cvg.plot(sample_sizes, cvg_mean[i_m], "D-", mfc="none", color=METHOD_COLORS[i_m], ms=5,
                    label=METHOD_LABELS[m])
        ax_set.plot(sample_sizes, size_mean[i_m], "D-", mfc="none", color=METHOD_COLORS[i_m], ms=5)
    ax_cvg.plot(sample_sizes, [args.nominal_coverage] * n_sizes, "--", color="black", alpha=0.9,
                label="Nominal Level")
    ax_cvg.set_ylabel("Empirical Coverage", fontsize=11)
    ax_cvg.set_xticks(sample_sizes, size_ticks)
    ax_cvg.legend(loc="upper center", bbox_to_anchor=(0.39, 1.65), ncol=2, frameon=False,
                  fontsize=10, columnspacing=1.5)
    ax_set.set_ylabel("Avg. Set Size", fontsize=11)
    ax_set.set_xlabel(r"Sample Size ($\times 10^3$)", fontsize=11, labelpad=2)
    ax_set.set_xticks(sample_sizes, size_ticks)

    handles = [plt.scatter([], [], marker="s", s=40, color=col_sz[i], edgecolors="black", linewidth=0.2,
                           label=f"{sz / 1000:g}") for i, sz in enumerate(sample_sizes)]
    fig.legend(handles=handles, loc="lower left", ncol=4, bbox_to_anchor=(-0.01, 0.82), fontsize=9,
               frameon=False, title=r"Sample Size ($\times 10^3$)", title_fontsize=10,
               columnspacing=1, handletextpad=0.2)
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.13, right=0.99)

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "nuImagesRes.pdf")
    plt.close(fig)
    print(f"\nFigure saved to {fig_dir / 'nuImagesRes.pdf'}")


if __name__ == "__main__":
    main()
