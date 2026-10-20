import os
os.environ["OMP_NUM_THREADS"] = "5"

import numpy as np
from sklearn.metrics import confusion_matrix
import maclp

np.set_printoptions(suppress=True, precision=4)

if __name__ == "__main__":
    # =========================================================================
    # 1. Simulation Parameter Setup and True Weights Computation
    # =========================================================================
    K = 5  # Number of classes
    p = np.array([0.1, 0.2, 0.2, 0.15, 0.35])
    q = np.array([0.37, 0.23, 0.19, 0.2, 0.01])

    TRUE_wts = q / p
    print(f"TRUE_wts       -> {TRUE_wts}")
    print(f"||TRUE_wts||_2 -> {np.linalg.norm(TRUE_wts):.5f}\n")

    mus = np.array([[0.0, 0.0], [2.0, 2.0], [-1.0, 2.0], [4.0, 0.0], [-2.0, -2.0]])
    Sigma = np.diag([1.0, 1.0])

    M = 100  # Source domain sample size
    N = M    # Target domain sample size
    delta = 1.0 / 1000

    # =========================================================================
    # 2. Generate Synthetic Data and Classifier Predictions
    # =========================================================================
    rng = np.random.default_rng(1652)

    XY_src = maclp.toy_data_generating(mus, Sigma, M, p, rng=rng)
    XY_tgt = maclp.toy_data_generating(mus, Sigma, N, q, rng=rng)

    Y_src_true = XY_src[:, -1].astype(int)
    Y_src_pred = maclp.bayes_classifier(XY_src[:, :-1], mus, Sigma, p, out_type='hard')
    Y_tgt_pred = maclp.bayes_classifier(XY_tgt[:, :-1], mus, Sigma, p, out_type='hard')

    # Compute confusion matrix and target domain prediction distribution
    C_mtx = confusion_matrix(Y_src_true, Y_src_pred).T
    q_dist = maclp.utils.q_samp_dist(Y_tgt_pred, K)

    # =========================================================================
    # 3. Standard Data Test: One-step solving via fit_from_data()
    # =========================================================================
    print("=" * 60)
    print("[Test 1: One-step solver from raw data (fit_from_data)]")
    print("=" * 60)

    # 3.1 BBSE Point Estimator
    bbse = maclp.BBSE()
    wts_bbse = bbse.fit(C_mtx, q_dist, M, N)

    # 3.2 LP Bounds Solver
    lp_solver = maclp.LP(bound_single=(0.0, 500.0))
    wts_lp_bounds = lp_solver.fit_from_data(C_mtx, q_dist, M, N, delta=delta)

    # 3.3 GE Heuristic Solver
    ge_solver = maclp.GE(upper_limit=500.0)
    wts_ge_bounds = ge_solver.fit_from_data(C_mtx, q_dist, M, N, delta=delta)

    print("BBSE Weights:\n", wts_bbse)
    print("\nLP Bounds:\n", wts_lp_bounds)
    print("\nGE Bounds:\n", wts_ge_bounds)

    # =========================================================================
    # 4. Infeasible/Edge Case Test: Pass manual intervals to verify Warnings
    # =========================================================================
    print("\n" + "=" * 60)
    print("[Test 2: Infeasible/Empty region intervals (triggers Warning via fit())]")
    print("=" * 60)

    # Construct test intervals representing an empty feasible region
    Clow_empty = np.array([[0.15, 0.70], [0.93, 0.75]])
    Chigh_empty = np.array([[0.25, 0.85], [0.98, 0.95]])

    qlow_empty = np.array([0.60, 0.10])
    qhigh_empty = np.array([0.80, 0.25])

    # 4.1 LP Test (triggers warning automatically when feasible region is empty)
    wts_rgnLP = lp_solver.fit(Clow_empty, Chigh_empty, qlow_empty, qhigh_empty)
    print("\nLP Bounds (Empty Region Result):\n", wts_rgnLP)

    # 4.2 GE Test
    wts_rgnGE = ge_solver.fit(Clow_empty, Chigh_empty, qlow_empty, qhigh_empty)
    print("\nGE Bounds (Empty Region Result):\n", wts_rgnGE)
