"""Controlled experiment under label shift violation (Appendix K.6).

Two-dimensional Gaussian mixture with K = 5 classes, means (0, 0), (2, 2),
(-1, 2), (4, 0), (-2, -2), covariance I, p(y) = (0.1, 0.2, 0.2, 0.15, 0.35)
and q(y) = (0.37, 0.23, 0.19, 0.2, 0.01).  With ``--sigma-scale 1.2`` the
target covariance is inflated to 1.2 I, so p(x | y) != q(x | y).

In each of 100 repetitions, for every sample size m = n:

* the Bayes classifier of the source (known means, covariance I and p) gives
  the confusion matrix, the target prediction counts and the scores;
* 1,000 calibration points come from the source and 5,000 test points from
  the target;
* weighted conformal prediction (90%) is run with the oracle weights q / p,
  the BBSE estimate and the regions Omega (Monte Carlo), Omega_LP and Omega_GE
  (delta = 1 / 1000).

Writes ``results/violation/violation_sigma.<s>_m.<m>.pkl`` (one file per
sample size).  ``label_shift_violation_summary.py`` makes Tables S8 / S9 and
Figure S10.

Example:
    python synthetic/label_shift_violation.py --sigma-scale 1.0
    python synthetic/label_shift_violation.py --sigma-scale 1.2
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import RESULTS_ROOT
from exputils.conformal import COMPARED_METHODS, run_label_shift_trial
from exputils.regions import Omega_LPML, Omega_LPML_WarmStart2
from exputils.sampling import bayes_classifier, seed_sequence, toy_data_generating
from exputils.synthetic import prediction_table

P_SRC = np.array([0.1, 0.2, 0.2, 0.15, 0.35])
Q_TGT = np.array([0.37, 0.23, 0.19, 0.2, 0.01])
MUS = np.array([[0.0, 0.0], [2.0, 2.0], [-1.0, 2.0], [4.0, 0.0], [-2.0, -2.0]])
SIGMA = np.diag([1.0, 1.0])
# Tables S8 / S9 use 50, 500, 5000, 50000 and 100000; Figure S10 uses all of them.
SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
LP_SOLVERS = {"scipy": Omega_LPML, "cpp": Omega_LPML_WarmStart2}


def run_sample_size(M, args):
    N = M
    K = len(P_SRC)
    tgt_sigma = SIGMA * args.sigma_scale
    true_wts = Q_TGT / P_SRC
    delta = 1 / args.n_calib
    n_seed = int(args.n_repts * 1.05)
    seed_simu = seed_sequence(1652, n_seed)
    seed_omg = seed_sequence(147, n_seed)
    seed_cps = seed_sequence(963, n_seed)
    seed_cali = seed_sequence(528, n_seed)

    def bayes_table(XY):
        # the classifier always uses the source parameters
        return prediction_table(XY, bayes_classifier(XY[:, :-1], MUS, SIGMA, P_SRC, "soft"))

    out = {"bbse_wts": np.zeros((args.n_repts, K)), "lp_length": np.zeros((args.n_repts, K)),
           "ge_length": np.zeros((args.n_repts, K)), "vol_ratio": np.zeros(args.n_repts),
           "coverage": {m: np.zeros(args.n_repts) for m in COMPARED_METHODS},
           "set_size": {m: np.zeros(args.n_repts) for m in COMPARED_METHODS}}
    step = max(1, args.n_repts // 10)
    for i in range(args.n_repts):
        if i % step == 0:
            print(f"  m = n = {M}: repetition {i}")
        XY_cali = toy_data_generating(MUS, SIGMA, args.n_calib, P_SRC, np.random.default_rng(seed_cali[i]))
        rng_simu = np.random.default_rng(seed_simu[i])
        XY_src = toy_data_generating(MUS, SIGMA, M, P_SRC, rng_simu)
        XY_tgt = toy_data_generating(MUS, tgt_sigma, N, Q_TGT, rng_simu)
        XY_testing = toy_data_generating(MUS, tgt_sigma, args.n_testing, Q_TGT, rng_simu)

        res = run_label_shift_trial(bayes_table(XY_cali), bayes_table(XY_testing), bayes_table(XY_src),
                                    bayes_table(XY_tgt), M, K, true_wts, delta, args.nominal_coverage,
                                    "Clopper-Pearson", 1000, 10000, seed_omg[i], seed_cps[i],
                                    lp_solver=LP_SOLVERS[args.lp_solver])
        for key in ["bbse_wts", "lp_length", "ge_length", "vol_ratio"]:
            out[key][i] = res[key]
        for method in COMPARED_METHODS:
            out["coverage"][method][i] = res["coverage"][method].mean()
            out["set_size"][method][i] = res["set_size"][method].mean()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sigma-scale", type=float, default=1.0, help="1.0: label shift holds; 1.2: violated")
    parser.add_argument("--m", type=int, nargs="+", default=SAMPLE_SIZES, help="sample sizes m = n")
    parser.add_argument("--n-repts", type=int, default=100)
    parser.add_argument("--n-calib", type=int, default=1000)
    parser.add_argument("--n-testing", type=int, default=5000)
    parser.add_argument("--nominal-coverage", type=float, default=0.90)
    parser.add_argument("--lp-solver", choices=sorted(LP_SOLVERS), default="scipy",
                        help="'cpp' uses the compiled solver of the maclp package")
    parser.add_argument("--out-dir", default=str(RESULTS_ROOT / "violation"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"true weights q/p = {np.round(Q_TGT / P_SRC, 3)}, target covariance {args.sigma_scale} I")
    for M in args.m:
        t0 = time.time()
        res = run_sample_size(M, args)
        res.update({"m": M, "sigma_scale": args.sigma_scale, "true_wts": Q_TGT / P_SRC})
        path = out_dir / f"violation_sigma.{args.sigma_scale}_m.{M}.pkl"
        with open(path, "wb") as f:
            pickle.dump(res, f)
        summary = "  ".join(f"{m} {res['coverage'][m].mean():.3f}" for m in COMPARED_METHODS)
        print(f"m = n = {M} ({time.time() - t0:.0f} s): coverage {summary}\n  saved {path}")


if __name__ == "__main__":
    main()
