"""Figures for the benchmark experiment (Figures 2-3 and S4-S9).

Reads the pickles written by ``dirichlet_shift_experiment.py`` and draws

* ``lpres_<dataset>_ci0.9.pdf``: volcano plot of the GE/LP interval-length
  ratios and the Monte Carlo volume ratio vol(Omega)/vol(Omega_LP);
* ``cpres_<dataset>_ci0.9.pdf``: empirical coverage and average set size of
  weighted conformal prediction for every (a, n) setting.

Length ratio statistic: for each class k, r_k = log2 of the GE/LP length
ratio averaged over repetitions; the volcano plot shows mean_k r_k against
-log10 p of a one-sided one-sample t-test of {r_k} (H0: mean <= 0).

Example:
    python benchmarks/analyze_results.py --dataset n24news
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import FIGURES_ROOT, RESULTS_ROOT
from exputils.io import load_results
from exputils.plotting import (plot_combined_volcano_compass_final,
                               plot_final_compact_gradient, set_paper_style)

METHOD_LABELS = {
    "oracle": r"$\boldsymbol{\omega}^*$",
    "estG.estWts": r"$\widehat{\boldsymbol{\omega}}$",
    "estG.Omega": r"$\boldsymbol{\Omega}$",
    "estG.Omega_LP": r"$\boldsymbol{\Omega}_{\mathrm{LP}}$",
    "estG.Omega_GE": r"$\boldsymbol{\Omega}_{\mathrm{GE}}$",
}

# x-axis break of the volcano plot used for the paper figures
VOLCANO_BREAKS = {"mnist": (0.118, 1.68), "cifar10": (0.217, 3.9)}


def find_result(results_dir, nominal_coverage, itvl_type, sz, alp, n_calib, n_testing):
    pattern = (f"experiType.dirichlet-drift_CI.{int(round(nominal_coverage * 100))}.{itvl_type}"
               f"_sampleSize.{sz}_lbsftAlp.{alp}_nCalib.{n_calib}_nTesting.{n_testing}_*.pkl")
    matches = sorted(Path(results_dir).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file matching {pattern} in {results_dir}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results-dir", default=None, help="default: results/benchmarks/<dataset>")
    parser.add_argument("--fig-dir", default=str(FIGURES_ROOT))
    parser.add_argument("--n-calib", type=int, default=5000)
    parser.add_argument("--n-testing", type=int, default=2500)
    parser.add_argument("--sample-sizes", type=int, nargs="+",
                        default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
    parser.add_argument("--alphas", type=float, nargs="+", default=list(np.linspace(-3, 3, 7)))
    parser.add_argument("--nominal-coverage", type=float, default=0.90)
    parser.add_argument("--itvl-type", default="Clopper-Pearson")
    parser.add_argument("--show", action="store_true", help="also open the figures in a window")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_ROOT / "benchmarks" / args.dataset
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    alphas = np.array(args.alphas)
    sample_sizes = np.array(sorted(args.sample_sizes))
    methods = list(METHOD_LABELS)
    metrics = ["coverage", "set_size"]

    conformal_storage = length_storage = vol_storage = None
    for i_a, alp in enumerate(alphas):
        for i_n, sz in enumerate(sample_sizes):
            path = find_result(results_dir, args.nominal_coverage, args.itvl_type, sz, alp,
                               args.n_calib, args.n_testing)
            (_, LPvsGE_length, vol_ratio, _, _, file_methods,
             size_repts, coverage_repts) = load_results(path)
            n_repts, num_classes = LPvsGE_length.shape[1:]

            if conformal_storage is None:
                conformal_storage = np.full((len(alphas), len(sample_sizes), len(methods), 2, n_repts), np.nan)
                length_storage = np.full((len(alphas), len(sample_sizes), n_repts, num_classes), np.nan)
                vol_storage = np.full((len(alphas), len(sample_sizes), n_repts), np.nan)

            length_storage[i_a, i_n] = LPvsGE_length[1] / LPvsGE_length[0]   # GE / LP
            vol_storage[i_a, i_n] = vol_ratio
            for i_m, m in enumerate(methods):
                conformal_storage[i_a, i_n, i_m, 0] = coverage_repts[m]
                conformal_storage[i_a, i_n, i_m, 1] = size_repts[m].mean(axis=1)

    # text summary
    cvg_mean = conformal_storage[:, :, :, 0].mean(axis=3)
    size_mean = conformal_storage[:, :, :, 1].mean(axis=3)
    print(f"rows: a = {alphas.tolist()}, columns: n = {sample_sizes.tolist()}")
    for i_m, m in enumerate(methods):
        print(f"\n{m}  coverage\n{cvg_mean[:, :, i_m].round(3)}\n{m}  set size\n{size_mean[:, :, i_m].round(3)}")

    coords_rep = np.arange(conformal_storage.shape[-1])
    da_cp = xr.DataArray(conformal_storage, dims=["alpha", "n", "methods", "metric", "n_repts"],
                         coords={"alpha": alphas, "n": sample_sizes, "methods": methods,
                                 "metric": metrics, "n_repts": coords_rep})
    da_volume = xr.DataArray(vol_storage, dims=["alpha", "n", "n_repts"],
                             coords={"alpha": alphas, "n": sample_sizes, "n_repts": coords_rep})
    da_length = xr.DataArray(length_storage, dims=["alpha", "n", "n_repts", "num_classes"],
                             coords={"alpha": alphas, "n": sample_sizes, "n_repts": coords_rep,
                                     "num_classes": np.arange(length_storage.shape[-1])})

    tag = f"{args.dataset}_ci{args.nominal_coverage}"
    plot_final_compact_gradient(da_cp, METHOD_LABELS, nominal_coverage=args.nominal_coverage,
                                fig_path=fig_dir / f"cpres_{tag}.pdf", show=args.show)
    break_low, break_high = VOLCANO_BREAKS.get(args.dataset, (1.0, 1.0))
    plot_combined_volcano_compass_final(da_length, da_volume, break_low=break_low, break_high=break_high,
                                        fig_path=fig_dir / f"lpres_{tag}.pdf", show=args.show)


if __name__ == "__main__":
    main()
