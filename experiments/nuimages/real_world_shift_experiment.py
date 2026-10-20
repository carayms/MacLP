"""Real-world label shift on nuImages: Singapore (source) -> Boston (target).

Same procedure as ``benchmarks/dirichlet_shift_experiment.py`` (Section 4.3,
Figure 4), except that the shift is the natural one between the two cities:

* calibration set: 5,000 Singapore rows (analysis half), without replacement;
* test set: 2,500 Boston rows, with replacement;
* estimation sets: n Singapore rows and n Boston rows, with replacement;
* p(y), q(y): class proportions of the Singapore / Boston training split
  (Table S2); w* = q(y) / p(y) is used as the "oracle".

Input: the prediction files written by ``train_predict.py``, i.e. pairs
``Singapore_rep<r>_*.tsv`` (analysis half of Singapore) and
``Boston_rep<r>_*.tsv`` (Boston), one pair per trained classifier.

Example:
    python nuimages/real_world_shift_experiment.py --pred-dir data/predictions/nuimages
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

# Class proportions of the nuImages training split (Table S2), in the class
# order of SELECTED_CLASSES in crop_objects.py.
PAPER_SRC_P = np.array([0.0889208, 0.0153803, 0.0040728, 0.1806684, 0.0081875, 0.0103442,
                        0.0421308, 0.007664, 0.0546738, 0.0201545, 0.4009234, 0.0230023,
                        0.0622225, 0.0014553, 0.0801993])
PAPER_TGT_P = np.array([5.989000e-02, 2.035700e-03, 1.900000e-05, 7.590890e-02,
                        2.111800e-03, 1.769300e-03, 5.142400e-02, 1.065390e-02,
                        1.792140e-02, 1.729350e-02, 5.614406e-01, 1.430660e-02,
                        1.160510e-02, 3.772620e-02, 1.358941e-01])

PAPER_SAMPLE_SIZES = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]


def result_file_name(out_dir, nominal_coverage, itvl_type, sz, n_calib, n_testing, n_lbsft, n_mc):
    """File name shared with analyze_results.py."""
    return Path(out_dir) / (
        f"experiType.real-world_CI.{int(round(nominal_coverage * 100))}.{itvl_type}"
        f"_sampleSize.{sz}_lbsftAlp.real-world-shift_nCalib.{n_calib}_nTesting.{n_testing}"
        f"_nLbsft.{n_lbsft}_MC.{n_mc // 1000}k_complx.pkl"
    )


def class_proportions(tables, num_classes):
    labels = np.concatenate([t[:, TRUE_IDX] for t in tables]).astype(int)
    return np.bincount(labels, minlength=num_classes) / labels.size


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", default=None, help="default: results/nuimages")
    parser.add_argument("--proportions", choices=["paper", "data"], default="paper",
                        help="'paper': p(y), q(y) of Table S2; 'data': label proportions of the prediction files")
    parser.add_argument("--n-calib", type=int, default=5000)
    parser.add_argument("--n-testing", type=int, default=2500)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=PAPER_SAMPLE_SIZES)
    parser.add_argument("--n-repts", type=int, default=1000)
    parser.add_argument("--nominal-coverage", type=float, default=0.90)
    parser.add_argument("--delta", type=float, default=None, help="default: 1 / n_calib")
    parser.add_argument("--itvl-type", default="Clopper-Pearson",
                        choices=["Clopper-Pearson", "Jeffreys-Prior", "Uniform-Prior"])
    parser.add_argument("--n-mc", type=int, default=10000, help="points of Omega for the MC supremum")
    parser.add_argument("--n-mc-pool", type=int, default=60000, help="uniform draws from Omega_LP")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    args = parser.parse_args()
    if args.smoke:
        args.n_calib, args.n_testing = 1000, 500
        args.sample_sizes = [1000, 2000]
        args.n_mc, args.n_mc_pool = 1000, 6000
        args.n_repts = None
        args.proportions = "data"
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / "nuimages"

    names = list_prediction_files(args.pred_dir)
    src_files = [f for f in names if f.startswith("Singapore")]
    tgt_files = [f for f in names if f.startswith("Boston")]
    if len(src_files) != len(tgt_files) or not src_files:
        raise ValueError("Expected matching Singapore_* and Boston_* prediction files.")
    src_tables = [load_predi_data(Path(args.pred_dir) / f)[1] for f in src_files]
    tgt_tables = [load_predi_data(Path(args.pred_dir) / f)[1] for f in tgt_files]
    n_files = len(src_tables)
    num_classes = src_tables[0].shape[1] - SOFT_IDX
    for s, t in zip(src_files, tgt_files):
        print(f"source: {s} | target: {t}")

    if args.proportions == "paper":
        if num_classes != PAPER_SRC_P.size:
            raise ValueError("The paper proportions are for 15 classes; use --proportions data.")
        src_p, tgt_p = PAPER_SRC_P, PAPER_TGT_P
    else:
        src_p = class_proportions(src_tables, num_classes)
        tgt_p = class_proportions(tgt_tables, num_classes)
    true_wts = tgt_p / src_p
    print(f"p(y): {src_p.round(4)}\nq(y): {tgt_p.round(4)}")

    n_repts = args.n_repts if args.n_repts is not None else n_files
    if n_repts % n_files != 0:
        raise ValueError(f"n_repts ({n_repts}) must be a multiple of the number of classifiers ({n_files}).")
    sub_repts = n_repts // n_files
    delta = args.delta if args.delta is not None else 1 / args.n_calib

    n_seed = int(n_repts * 1.05)
    seed_estiw = seed_sequence(1652, n_seed)
    seed_cali = seed_sequence(528, n_seed)
    seed_cps = seed_sequence(963, n_seed)
    seed_testing = seed_sequence(147, n_seed)
    seed_omg = seed_sequence(1991, n_seed)

    for sz in args.sample_sizes:
        print(f"n = {sz}")
        t0 = time.time()
        estimated_wts = np.zeros((n_repts, num_classes))
        LPvsGE_length = np.zeros((2, n_repts, num_classes))
        vol_ratio = np.zeros(n_repts)
        coverage_repts = {m: np.zeros(n_repts, dtype=np.float32) for m in COMPARED_METHODS}
        size_repts = {m: np.zeros((n_repts, args.n_testing), dtype=np.float32) for m in COMPARED_METHODS}

        for idx_file, (src_table, tgt_table) in enumerate(zip(src_tables, tgt_tables)):
            for i_sub in range(sub_repts):
                i_simu = idx_file * sub_repts + i_sub

                rng_cali = np.random.default_rng(seed_cali[i_simu])
                XY_cali, src_rest = sample_with_rng(src_table, args.n_calib, src_p, rng_cali, TRUE_IDX,
                                                    min_smp=1, return_both=True)
                rng_testing = np.random.default_rng(seed_testing[i_simu])
                XY_testing = sample_with_rng(tgt_table, args.n_testing, tgt_p, rng_testing, TRUE_IDX,
                                             min_smp=0, return_both=False)
                rng_simu = np.random.default_rng(seed_estiw[i_simu])
                XY_src = sample_with_rng(src_rest, sz, src_p, rng_simu, TRUE_IDX, min_smp=1, return_both=False)
                XY_tgt = sample_with_rng(tgt_table, sz, tgt_p, rng_simu, TRUE_IDX, min_smp=0, return_both=False)
                n_lbsft = src_rest.shape[0] + tgt_table.shape[0]

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

        path = result_file_name(out_dir, args.nominal_coverage, args.itvl_type, sz,
                                args.n_calib, args.n_testing, n_lbsft, args.n_mc)
        save_results(path, LPvsGE_length, vol_ratio, true_wts, estimated_wts,
                     COMPARED_METHODS, size_repts, coverage_repts)
        print(f"  saved {path.name}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
