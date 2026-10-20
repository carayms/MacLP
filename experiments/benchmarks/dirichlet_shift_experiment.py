"""Label-shift experiment on the benchmark datasets (paper, Section 4).

For every shift intensity a and sample size n this script simulates label
shift from the analysis-set predictions of the trained classifiers and
records, for each repetition,

* the interval lengths of Omega_LP and Omega_GE        (Figure 2 / S4-S6, left),
* the Monte Carlo volume ratio vol(Omega)/vol(Omega_LP) (Figure 2 / S4-S6, right),
* coverage and set size of weighted conformal prediction with
  w*, the BBSE estimate, Omega, Omega_LP and Omega_GE   (Figure 3 / S7-S9).

Design (Section 4.1):
    p ~ Dir(1_K * 1e4), q ~ Dir(1_K * 10^a), a in {-3, ..., 3};
    calibration set: 5,000 rows from p, without replacement;
    test set: 2,500 rows from q, with replacement;
    estimation sets: n rows from p and n rows from q, with replacement,
    n in {1000, ..., 8000};
    1,000 repetitions per setting, split evenly over the classifiers.

Input: a folder with one prediction file per classifier (see exputils/io.py).
Output: one pickle per (n, a) setting, read by ``analyze_results.py``.

Example (full run for N24News):
    python benchmarks/dirichlet_shift_experiment.py --pred-dir data/predictions/n24news --dataset n24news

Quick check with synthetic predictions:
    python tools/make_synthetic_predictions.py --out-dir data/predictions/synthetic --n-files 2 --n-rows 8000
    python benchmarks/dirichlet_shift_experiment.py --pred-dir data/predictions/synthetic --dataset synthetic --smoke
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import RESULTS_ROOT
from exputils.conformal import COMPARED_METHODS, run_label_shift_trial
from exputils.io import (PRED_IDX, SOFT_IDX, TRUE_IDX, list_prediction_files,
                         load_predi_data, save_results)
from exputils.sampling import sample_with_rng, seed_sequence

PAPER_SAMPLE_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
PAPER_ALPHAS = list(np.linspace(-3, 3, 7))


def result_file_name(out_dir, nominal_coverage, itvl_type, sz, alp, n_calib, n_testing, n_lbsft):
    """File name shared with analyze_results.py."""
    return Path(out_dir) / (
        f"experiType.dirichlet-drift_CI.{int(round(nominal_coverage * 100))}.{itvl_type}"
        f"_sampleSize.{sz}_lbsftAlp.{alp}_nCalib.{n_calib}_nTesting.{n_testing}"
        f"_nLbsft.{n_lbsft}_complx.pkl"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", required=True, help="folder with the prediction files")
    parser.add_argument("--dataset", required=True, help="name used for the output folder, e.g. n24news")
    parser.add_argument("--out-dir", default=None, help="default: results/benchmarks/<dataset>")
    parser.add_argument("--n-calib", type=int, default=5000)
    parser.add_argument("--n-testing", type=int, default=2500)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=PAPER_SAMPLE_SIZES)
    parser.add_argument("--alphas", type=float, nargs="+", default=PAPER_ALPHAS,
                        help="log10 of the target Dirichlet concentration")
    parser.add_argument("--n-repts", type=int, default=1000, help="repetitions per setting (all classifiers)")
    parser.add_argument("--nominal-coverage", type=float, default=0.90, help="1 - alpha")
    parser.add_argument("--delta", type=float, default=None, help="default: 1 / n_calib")
    parser.add_argument("--itvl-type", default="Clopper-Pearson",
                        choices=["Clopper-Pearson", "Jeffreys-Prior", "Uniform-Prior"])
    parser.add_argument("--n-mc", type=int, default=1000, help="points of Omega for the MC supremum")
    parser.add_argument("--n-mc-pool", type=int, default=10000, help="uniform draws from Omega_LP")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run (2 sizes, 2 shifts, 1 repetition per file) to check the pipeline")
    args = parser.parse_args()

    if args.smoke:
        args.n_calib, args.n_testing = 1000, 500
        args.sample_sizes, args.alphas = [1000, 2000], [-1.0, 1.0]
        args.n_mc, args.n_mc_pool = 200, 2000
        args.n_repts = None  # set to one repetition per file below
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / "benchmarks" / args.dataset

    file_names = list_prediction_files(args.pred_dir)
    predi_collection = [load_predi_data(Path(args.pred_dir) / f)[1] for f in file_names]
    n_files = len(predi_collection)
    num_classes = predi_collection[0].shape[1] - SOFT_IDX
    print(f"{n_files} prediction files, K = {num_classes}")

    n_repts = args.n_repts if args.n_repts is not None else n_files
    if n_repts % n_files != 0:
        raise ValueError(f"n_repts ({n_repts}) must be a multiple of the number of files ({n_files}).")
    sub_repts = n_repts // n_files
    delta = args.delta if args.delta is not None else 1 / args.n_calib

    # reproducibility: proportions are drawn in loop order from one generator
    rng_prop = np.random.default_rng(149)
    n_seed = int(n_repts * 1.05)
    seed_estiw = seed_sequence(1652, n_seed)
    seed_cali = seed_sequence(528, n_seed)
    seed_cps = seed_sequence(963, n_seed)
    seed_testing = seed_sequence(147, n_seed)
    seed_omg = seed_sequence(1991, n_seed)

    src_alpha = 1e4 * np.ones(num_classes)

    for sz in args.sample_sizes:
        src_p = rng_prop.dirichlet(src_alpha)
        for alp in args.alphas:
            tgt_p = rng_prop.dirichlet(10**alp * np.ones(num_classes))
            true_wts = tgt_p / src_p
            print(f"n = {sz}, a = {alp}")
            t0 = time.time()

            estimated_wts = np.zeros((n_repts, num_classes))
            LPvsGE_length = np.zeros((2, n_repts, num_classes))
            vol_ratio = np.zeros(n_repts)
            coverage_repts = {m: np.zeros(n_repts, dtype=np.float32) for m in COMPARED_METHODS}
            size_repts = {m: np.zeros((n_repts, args.n_testing), dtype=np.float32) for m in COMPARED_METHODS}

            for idx_file, table in enumerate(predi_collection):
                for i_sub in range(sub_repts):
                    i_simu = idx_file * sub_repts + i_sub

                    # calibration set (source proportions, without replacement)
                    rng_cali = np.random.default_rng(seed_cali[i_simu])
                    XY_cali, rest = sample_with_rng(table, args.n_calib, src_p, rng_cali, TRUE_IDX,
                                                    min_smp=1, return_both=True)
                    # test set (target proportions)
                    rng_testing = np.random.default_rng(seed_testing[i_simu])
                    XY_testing = sample_with_rng(rest, args.n_testing, tgt_p, rng_testing, TRUE_IDX,
                                                 min_smp=0, return_both=False)
                    # estimation sets for the weights
                    rng_simu = np.random.default_rng(seed_estiw[i_simu])
                    XY_src = sample_with_rng(rest, sz, src_p, rng_simu, TRUE_IDX, min_smp=1, return_both=False)
                    XY_tgt = sample_with_rng(rest, sz, tgt_p, rng_simu, TRUE_IDX, min_smp=0, return_both=False)
                    n_lbsft = rest.shape[0]

                    res = run_label_shift_trial(
                        XY_cali, XY_testing, XY_src, XY_tgt, sz, num_classes, true_wts,
                        delta, args.nominal_coverage, args.itvl_type, args.n_mc, args.n_mc_pool,
                        seed_omg[i_simu], seed_cps[i_simu], cols=(TRUE_IDX, PRED_IDX, SOFT_IDX))

                    estimated_wts[i_simu] = res["bbse_wts"]
                    LPvsGE_length[0][i_simu] = res["lp_length"]
                    LPvsGE_length[1][i_simu] = res["ge_length"]
                    vol_ratio[i_simu] = res["vol_ratio"]
                    for m in COMPARED_METHODS:
                        coverage_repts[m][i_simu] = res["coverage"][m].mean()
                        size_repts[m][i_simu] = res["set_size"][m]

            for m in COMPARED_METHODS:
                print(f"  {m:<14} coverage {coverage_repts[m].mean():.4f}   "
                      f"set size {size_repts[m].mean(axis=1).mean():.4f} ({size_repts[m].mean(axis=1).std():.4f})")

            path = result_file_name(out_dir, args.nominal_coverage, args.itvl_type, sz, alp,
                                    args.n_calib, args.n_testing, n_lbsft)
            save_results(path, LPvsGE_length, vol_ratio, true_wts, estimated_wts,
                         COMPARED_METHODS, size_repts, coverage_repts)
            print(f"  saved {path.name}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
