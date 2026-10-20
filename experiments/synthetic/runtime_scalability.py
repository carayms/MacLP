"""Runtime and interval length of Omega_LP and Omega_GE across the number of
classes (Appendix K.4, Tables S5 and S6).

For each K, two-dimensional Gaussian mixture data with covariance 0.5 I and
means at the origin and on concentric circles (``synthetic.circle_means``),
p(y) = 1/K and q(y) ~ Dir(0.1 * 1_K).  In each repetition a Gaussian naive
Bayes classifier is trained on 2000K source points; m = n = max(2000, 1000K)
points give the confusion matrix and the target prediction counts;
Clopper-Pearson intervals use delta = 1 / 5000.  The script times the BBSE
estimate, Omega_LP and Omega_GE and stores both boxes.

The paper uses the compiled solver of the maclp package (``--lp-solver cpp``)
with 5 OpenMP threads; set ``OMP_NUM_THREADS`` to change the number of
threads.  The first repetition warms up numba and the solver and is dropped by
``runtime_summary.py``.

Writes ``results/runtime/runtime_K.<K>.pkl``.

Example:
    python synthetic/runtime_scalability.py --K 2 5 10 20 50 100 200 500
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "5")   # read when OpenMP / BLAS start

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import RESULTS_ROOT
from exputils.intervals import find_C_itvls, find_q_itvls
from exputils.regions import Omega_GEKH, Omega_LPML, Omega_LPML_WarmStart2
from exputils.sampling import q_samp_dist, seed_sequence, toy_data_generating
from exputils.synthetic import circle_means

LP_SOLVERS = {"scipy": Omega_LPML, "cpp": Omega_LPML_WarmStart2}
SIGMA = np.diag([1.0, 1.0]) * 0.5


def run_K(K, args):
    mus = circle_means(K)
    p = np.ones(K) / K
    n_train = 2000 * K
    M = N = max(2000, 1000 * K)
    delta = 1 / args.n_calib
    lp_solver = LP_SOLVERS[args.lp_solver]

    n_seed = int(args.n_repts * 1.05)
    seed_simu = seed_sequence(1652, n_seed)
    seed_train = seed_sequence(528, n_seed)
    seed_tgt_prop = seed_sequence(777, n_seed)

    runtime = {key: np.zeros(args.n_repts) for key in ["bbse", "lp", "ge"]}
    boxes = {key: np.zeros((args.n_repts, K, 2)) for key in ["lp", "ge"]}
    bbse_wts = np.zeros((args.n_repts, K))
    true_wts = np.zeros((args.n_repts, K))
    step = max(1, args.n_repts // 5)
    for i in range(args.n_repts):
        if i % step == 0:
            print(f"  K = {K}: repetition {i}")
        q = np.random.default_rng(seed_tgt_prop[i]).dirichlet(np.ones(K) * 0.1)
        true_wts[i] = q / p

        XY_train = toy_data_generating(mus, SIGMA, n_train, p, np.random.default_rng(seed_train[i]))
        clf = GaussianNB().fit(XY_train[:, :-1], XY_train[:, -1].astype(int))

        rng_simu = np.random.default_rng(seed_simu[i])
        XY_src = toy_data_generating(mus, SIGMA, M, p, rng_simu)
        XY_tgt = toy_data_generating(mus, SIGMA, N, q, rng_simu)
        C_mtx = confusion_matrix(XY_src[:, -1].astype(int), clf.predict(XY_src[:, :-1]), labels=np.arange(K)).T
        q_dist = q_samp_dist(clf.predict(XY_tgt[:, :-1]), K)

        start = time.perf_counter()
        bbse_wts[i] = np.maximum(np.linalg.pinv(C_mtx / M) @ (q_dist / N), 0)
        runtime["bbse"][i] = time.perf_counter() - start

        C_rgn = find_C_itvls(C_mtx, M, K, alpha_matrix=delta, itvl_type="Clopper-Pearson")
        q_rgn = find_q_itvls(q_dist, N, K, alpha_vec=delta, itvl_type="Clopper-Pearson")
        region_args = (C_rgn["C_low"], C_rgn["C_up"], q_rgn["q_low"], q_rgn["q_up"])

        start = time.perf_counter()
        boxes["lp"][i] = lp_solver(*region_args)
        runtime["lp"][i] = time.perf_counter() - start

        start = time.perf_counter()
        boxes["ge"][i] = Omega_GEKH(*region_args)
        runtime["ge"][i] = time.perf_counter() - start

    return {"K": K, "m": M, "lp_solver": args.lp_solver, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "runtime": runtime, "boxes": boxes, "bbse_wts": bbse_wts, "true_wts": true_wts}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--K", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100, 200, 500])
    parser.add_argument("--n-repts", type=int, default=51, help="the first repetition is a warm-up")
    parser.add_argument("--n-calib", type=int, default=5000, help="delta = 1 / n_calib")
    parser.add_argument("--lp-solver", choices=sorted(LP_SOLVERS), default="cpp",
                        help="'cpp' needs the compiled maclp package (pip install . at the repository root)")
    parser.add_argument("--out-dir", default=str(RESULTS_ROOT / "runtime"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"LP solver: {args.lp_solver}, OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")
    for K in args.K:
        res = run_K(K, args)
        path = out_dir / f"runtime_K.{K}.pkl"
        with open(path, "wb") as f:
            pickle.dump(res, f)
        rt = {key: val[1:].mean() if val.size > 1 else val.mean() for key, val in res["runtime"].items()}
        print(f"K = {K}: mean seconds  LP {rt['lp']:.5f}  GE {rt['ge']:.5f}  BBSE {rt['bbse']:.6f}\n  saved {path}")


if __name__ == "__main__":
    main()
