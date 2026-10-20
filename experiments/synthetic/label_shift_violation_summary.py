"""Tables S8 / S9 and Figure S10 (Appendix K.6) from the results of
``label_shift_violation.py``.

Cells are "mean (standard error)" over the repetitions, with standard error
std (ddof = 1) / sqrt(number of repetitions).  Prints Markdown and LaTeX
tables and writes ``figures/violAsumptn_violvl<s>.pdf`` (the file names used
in the paper) for every covariance scale found.  The figure follows the
original plotting script: Omega_GE is left out, error bars are one standard
error and the dashed line is the nominal coverage.

Example:
    python synthetic/label_shift_violation_summary.py
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import FIGURES_ROOT, RESULTS_ROOT
from exputils.plotting import generate_stats_latex_table, generate_stats_markdown_table, set_paper_style

# methods reported in the paper: (key in the results, table label, legend label, color)
METHODS = [("oracle", "Oracle", r"$\boldsymbol{\omega}^*$", "#de285f"),
           ("estG.estWts", "BBSE", r"$\widehat{\boldsymbol{\omega}}$", "#457B9D"),
           ("estG.Omega", "Omega", r"$\boldsymbol{\Omega}$", "#2A9D8F"),
           ("estG.Omega_LP", "Omega_LP", r"$\boldsymbol{\Omega}_{\mathrm{LP}}$", "#8338EC")]
TABLE_SIZES = [50, 500, 5000, 50000, 100000]
FILE_PATTERN = re.compile(r"violation_sigma\.(?P<s>[\d.]+)_m\.(?P<m>\d+)\.pkl")


def load_results(results_dir):
    """{sigma_scale: {m: results}}"""
    out = {}
    for path in sorted(Path(results_dir).glob("violation_sigma.*_m.*.pkl")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        with open(path, "rb") as f:
            out.setdefault(float(match["s"]), {})[int(match["m"])] = pickle.load(f)
    return out


def mean_se(values):
    values = np.asarray(values)
    return values.mean(), values.std(ddof=1) / np.sqrt(values.size)


def print_tables(res_by_m, sigma_scale):
    sizes = [m for m in TABLE_SIZES if m in res_by_m] or sorted(res_by_m)
    title = "label shift holds" if sigma_scale == 1.0 else f"target covariance {sigma_scale} I"
    for metric, name in [("coverage", "Coverage"), ("set_size", "Set size")]:
        means = np.zeros((len(METHODS), len(sizes)))
        ses = np.zeros_like(means)
        for i, (key, *_rest) in enumerate(METHODS):
            for j, m in enumerate(sizes):
                means[i, j], ses[i, j] = mean_se(res_by_m[m][metric][key])
        headers = [name] + [str(m) for m in sizes]
        labels = [label for _, label, _, _ in METHODS]
        print(f"\n### {name}, {title}\n")
        print(generate_stats_markdown_table(means, ses, headers, labels, digits=3))
        print()
        print(generate_stats_latex_table(means, ses, headers, labels, digits=3))


def plot_coverage(res_by_m, nominal, fig_path):
    sizes = np.array(sorted(res_by_m))
    with plt.rc_context({"font.size": 10}):
        fig, ax = plt.subplots(figsize=(3, 2))
        for key, _, legend, color in METHODS:
            stats = np.array([mean_se(res_by_m[m]["coverage"][key]) for m in sizes])
            ax.errorbar(sizes, stats[:, 0], yerr=stats[:, 1], fmt="o", capsize=3, linestyle="-",
                        mfc="none", mec="none", color=color, label=legend)
        ax.legend(frameon=False)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Empirical Coverage")
        ax.set_xscale("log")
        ax.plot(sizes, np.full(len(sizes), nominal), "k--")
        fig.subplots_adjust(left=0.2, right=0.98, top=0.99, bottom=0.25)
        fig.savefig(fig_path)
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT / "violation"))
    parser.add_argument("--fig-dir", default=str(FIGURES_ROOT))
    parser.add_argument("--nominal-coverage", type=float, default=0.90)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        sys.exit(f"No result files in {args.results_dir}; run label_shift_violation.py first.")
    set_paper_style()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for sigma_scale, res_by_m in sorted(results.items()):
        print_tables(res_by_m, sigma_scale)
        plot_coverage(res_by_m, args.nominal_coverage, fig_dir / f"violAsumptn_violvl{sigma_scale}.pdf")


if __name__ == "__main__":
    main()
