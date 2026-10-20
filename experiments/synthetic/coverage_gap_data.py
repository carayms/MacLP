"""Simulation for Figure 1: undercoverage of weighted conformal prediction
when the estimated weights replace the true ones.

Two-class Gaussian mixture with means (0, 0) and (1, 1), covariance 1.5 I,
p(y) = (0.3, 0.7) and q(y) = (0.8, 0.2).  In each of 1,000 repetitions:

* a Gaussian naive Bayes classifier is trained on 500 source points and
  gives the nonconformity scores of 1,000 calibration points (source) and
  1,000 test points (target);
* the BBSE weights are estimated from m source and n target points with the
  Bayes classifier (known means, covariance and p);
* weighted conformal prediction (90%) is run with the true and the estimated
  weights.

Writes ``results/synthetic/coverage_gap_m.<m>_n.<n>.pkl``; run it for m = n = 100
and m = n = 300, then ``coverage_gap_plot.py``.

Example:
    python synthetic/coverage_gap_data.py --m 100
    python synthetic/coverage_gap_data.py --m 300
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import RESULTS_ROOT
from exputils.conformal import run_label_shift_trial
from exputils.sampling import bayes_classifier, seed_sequence, toy_data_generating
from exputils.synthetic import prediction_table

P_SRC = np.array([0.3, 0.7])
Q_TGT = np.array([0.8, 0.2])
MUS = np.array([[0.0, 0.0], [1.0, 1.0]])
SIGMA = np.diag([1.0, 1.0]) * 1.5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, required=True, help="source sample size for the weights (n = m)")
    parser.add_argument("--n-repts", type=int, default=1000)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-calib", type=int, default=1000)
    parser.add_argument("--n-testing", type=int, default=1000)
    parser.add_argument("--nominal-coverage", type=float, default=0.90)
    parser.add_argument("--example-rep", type=int, default=100, help="repetition whose samples are drawn in Figure 1")
    parser.add_argument("--out-dir", default=str(RESULTS_ROOT / "synthetic"))
    args = parser.parse_args()

    M = N = args.m
    K = len(P_SRC)
    delta = 1 / args.n_calib
    true_wts = Q_TGT / P_SRC
    n_seed = int(args.n_repts * 1.05)
    seed_simu = seed_sequence(1652, n_seed)
    seed_omg = seed_sequence(147, n_seed)
    seed_cps = seed_sequence(963, n_seed)
    seed_train = seed_sequence(528, n_seed)
    example_rep = min(args.example_rep, args.n_repts - 1)

    bbse_wts = np.zeros((args.n_repts, K))
    coverage = {"oracle": np.zeros(args.n_repts), "estimated": np.zeros(args.n_repts)}
    set_size = {"oracle": np.zeros(args.n_repts), "estimated": np.zeros(args.n_repts)}
    example = None

    for i in range(args.n_repts):
        if i % max(1, args.n_repts // 10) == 0:
            print(f"repetition {i}")
        # classifier g (naive Bayes) and calibration data from the source
        rng_train = np.random.default_rng(seed_train[i])
        XY_train = toy_data_generating(MUS, SIGMA, args.n_train, P_SRC, rng_train)
        XY_cali = toy_data_generating(MUS, SIGMA, args.n_calib, P_SRC, rng_train)
        clf = GaussianNB().fit(XY_train[:, :-1], XY_train[:, -1].astype(int))

        # samples for the weights and test data from the target
        rng_simu = np.random.default_rng(seed_simu[i])
        XY_src = toy_data_generating(MUS, SIGMA, M, P_SRC, rng_simu)
        XY_tgt = toy_data_generating(MUS, SIGMA, N, Q_TGT, rng_simu)
        XY_testing = toy_data_generating(MUS, SIGMA, args.n_testing, Q_TGT, rng_simu)

        # BBSE with the Bayes classifier; conformal scores with the naive Bayes classifier
        res = run_label_shift_trial(
            prediction_table(XY_cali, clf.predict_proba(XY_cali[:, :-1])),
            prediction_table(XY_testing, clf.predict_proba(XY_testing[:, :-1])),
            prediction_table(XY_src, bayes_classifier(XY_src[:, :-1], MUS, SIGMA, P_SRC, "soft")),
            prediction_table(XY_tgt, bayes_classifier(XY_tgt[:, :-1], MUS, SIGMA, P_SRC, "soft")),
            M, K, true_wts, delta, args.nominal_coverage, "Clopper-Pearson", 1000, 10000,
            seed_omg[i], seed_cps[i])

        bbse_wts[i] = res["bbse_wts"]
        for key, method in [("oracle", "oracle"), ("estimated", "estG.estWts")]:
            coverage[key][i] = res["coverage"][method].mean()
            set_size[key][i] = res["set_size"][method].mean()
        if i == example_rep:
            example = {"src": XY_src, "tgt": XY_tgt, "calib": XY_cali, "testing": XY_testing}

    for key in coverage:
        print(f"{key:<10} coverage {coverage[key].mean():.4f}   set size {set_size[key].mean():.4f}")

    out = Path(args.out_dir) / f"coverage_gap_m.{M}_n.{N}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"true_wts": true_wts, "bbse_wts": bbse_wts, "coverage": coverage, "set_size": set_size,
                     "example": example, "example_rep": example_rep}, f)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
