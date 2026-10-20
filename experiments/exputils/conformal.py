"""Weighted split conformal prediction under label shift (paper, Section 3.4).

Nonconformity score (randomized APS):
    r(x, y) = sum_{y': p(y'|x) > p(y|x)} p(y'|x) + u * p(y|x),   u ~ U(0, 1).

For a weight vector w, the threshold tau_CP(y0; w) is the weighted
(1 - alpha) quantile of the calibration scores plus a point mass at 1 that
carries weight w_{y0}.  For a region of weights the threshold is the
supremum over the region:

* Omega_LP / Omega_GE (boxes): exact supremum at a vertex,
  ``max_qtl_rxy_on_rectangular`` (Appendix J.3.1).
* Omega (polytope): Monte Carlo maximum over points sampled from Omega
  (Remark 3.8 and Appendix J.3.2); this approximation can only
  underestimate the exact supremum.
"""

import warnings

import numpy as np
from numba import njit, prange
from sklearn.metrics import confusion_matrix

from .intervals import find_C_itvls, find_q_itvls
from .regions import Omega_GEKH, Omega_LPML, validate_constraints
from .sampling import q_samp_dist

# Method keys used in the result files.
#   oracle         true weights w*
#   estG.estWts    BBSE point estimate (estimated classifier g)
#   estG.Omega     MaC polytope Omega (Monte Carlo supremum)
#   estG.Omega_LP  Omega_LP (exact supremum)
#   estG.Omega_GE  Omega_GE (exact supremum)
COMPARED_METHODS = ["oracle", "estG.estWts", "estG.Omega", "estG.Omega_LP", "estG.Omega_GE"]


# ---------------------------------------------------------------------------
# Scores and weighted quantiles
# ---------------------------------------------------------------------------
@njit(parallel=True)
def compute_rho_optimal_numba(probs):
    """rho[i, y] = sum of the probabilities that are larger than probs[i, y]."""
    n, k = probs.shape
    rho = np.zeros((n, k))
    for i in prange(n):
        p = probs[i]
        for y in range(k):
            total = 0.0
            py = p[y]
            for yp in range(k):
                if p[yp] > py:
                    total += p[yp]
            rho[i, y] = total
    return rho


def compute_randomized_nonconformity_optimal(probs, labels, u):
    """Randomized APS score r(x, y) for each row of ``probs``.

    Parameters
    ----------
    probs : array of shape (n, K)
        Predicted class probabilities.
    labels : array of shape (n,) or None
        Labels at which to evaluate the score; None returns all K scores.
    u : array of shape (n,) or None
        Uniform random numbers; drawn with ``np.random`` if None.
    """
    if u is None:
        u = np.random.uniform(0, 1, size=probs.shape[0])

    rho = compute_rho_optimal_numba(probs)
    if labels is not None:
        rows = np.arange(probs.shape[0])
        return rho[rows, labels] + u * probs[rows, labels]
    return rho + u[:, None] * probs


@njit(cache=True)
def quantile_optimal(values, val_idx, weights, cutoff, prob=0.90):
    """Weighted upper quantile of ``values``.

    ``val_idx`` sorts ``values`` in decreasing order.  Weights are
    accumulated from the largest value down until the running sum reaches
    ``cutoff`` (= (1 - prob) * total weight); the last value added is
    returned.  ``prob`` is not used and is kept for readability of the calls.
    """
    cumsum = 0.0
    i = 0
    for i, idx in enumerate(val_idx):
        if cumsum < cutoff:
            cumsum += weights[idx]
        else:
            break
    return values[val_idx[i - 1]]


def cp_score_weights_generate(ipt_wts, labels, idx_cali):
    """Per-observation weights ``ipt_wts[labels]`` (``idx_cali`` = arange(n))."""
    wts_reps = np.tile(ipt_wts, (labels.size, 1))
    return wts_reps[idx_cali, labels]


# Name used by the experiment scripts of version 0.0.0.
cp_score_weights_optimal = cp_score_weights_generate


def max_qtl_rxy_on_rectangular(r_xy_y0, Y_cali_true, K, input_omega_rgn, q_level, idx_calib_plus1, r_xy_y0_srt_idx):
    """Exact sup of tau_CP(y0; w) over a box of weights, for every y0.

    The weighted quantile is monotone in each w_k, so the supremum is
    attained at a vertex: classes whose group-wise quantile is small get
    their lower bound and the others their upper bound.  The split point is
    found by scanning the classes ordered by their group-wise quantile.

    Parameters
    ----------
    r_xy_y0 : array of shape (m + 1,)
        Calibration scores with a trailing 1 for the test point.
    Y_cali_true : array of shape (m,)
        Calibration labels.
    K : int
    input_omega_rgn : array of shape (K, 2)
        The box [w_low, w_up].
    q_level : float
        Quantile level (1 - alpha + delta in the experiments).
    idx_calib_plus1 : arange(m + 1)
    r_xy_y0_srt_idx : argsort of r_xy_y0 in decreasing order.

    Returns
    -------
    array of shape (K,) with tau_CP(y0; box) for y0 = 0..K-1.
    """
    tau_CP = np.zeros(K, dtype=np.float32)
    for y_knw in range(K):
        optimal_weights = np.zeros(K)
        curr_y_cali_true = np.append(Y_cali_true, y_knw)

        # 1. sort by label, then by score
        srt_idx = np.lexsort((r_xy_y0, curr_y_cali_true))
        s_val = r_xy_y0[srt_idx]

        # 2. start position and size of each label group
        _, starts, counts = np.unique(curr_y_cali_true[srt_idx], return_index=True, return_counts=True)

        # 3. group-wise 'inverted_cdf' quantiles (0-based index)
        q_idx = np.ceil(q_level * counts).astype(int) - 1
        grp_qtls = s_val[starts + np.clip(q_idx, 0, counts - 1)]
        order_qtls = np.argsort(grp_qtls)

        for t in range(K - 1, -1, -1):
            small_ones = order_qtls[:t + 1]
            large_ones = order_qtls[t + 1:]
            optimal_weights[small_ones] = input_omega_rgn[small_ones, 0]
            optimal_weights[large_ones] = input_omega_rgn[large_ones, 1]

            if np.all(optimal_weights == 0):
                continue

            loop_wts = cp_score_weights_generate(optimal_weights, curr_y_cali_true, idx_calib_plus1)
            curr_ctf_wts = loop_wts.sum() * (1 - q_level)
            q_grp_k = quantile_optimal(r_xy_y0, r_xy_y0_srt_idx, loop_wts, curr_ctf_wts, q_level)

            if q_grp_k >= grp_qtls[order_qtls[t]]:
                break

        tau_CP[y_knw] = q_grp_k
    return tau_CP


# ---------------------------------------------------------------------------
# One repetition of the label-shift experiment (benchmarks and nuImages)
# ---------------------------------------------------------------------------
def run_label_shift_trial(XY_cali, XY_testing, XY_src, XY_tgt, sz, num_classes, true_wts,
                          delta, nominal_coverage, itvl_type, n_mc, n_mc_pool,
                          seed_omega, seed_cp, cols=(1, 3, 4), lp_solver=Omega_LPML):
    """Estimate the weight regions and evaluate weighted conformal prediction.

    Parameters
    ----------
    XY_cali, XY_testing, XY_src, XY_tgt : prediction tables
        Rows are observations; columns follow the prediction-file layout
        ``[orig_idx, true, true, hard_pred, soft_pred_0..K-1]``.
    sz : int
        Size of the estimation samples (m = n = sz).
    num_classes : int
    true_wts : array of shape (K,)
        Oracle weights q(y) / p(y).
    delta : float
        Error level of the weight regions.
    nominal_coverage : float
        1 - alpha.  The region-based thresholds use 1 - alpha + delta.
    itvl_type : str
        Binomial interval, see ``intervals.binomial_prop_itvls``.
    n_mc : int
        Number of points of Omega used for the Monte Carlo supremum.
    n_mc_pool : int
        Number of uniform draws from Omega_LP; the fraction inside Omega is
        the Monte Carlo volume ratio vol(Omega) / vol(Omega_LP).
    seed_omega, seed_cp : int
        Seeds for the Monte Carlo draws and for the randomized scores.
    cols : (true_idx, pred_idx, soft_idx)
    lp_solver : function
        Solver for Omega_LP with the interface of ``regions.Omega_LPML``.

    Returns
    -------
    dict with the BBSE weights, the LP / GE interval lengths, the volume
    ratio, and the coverage indicator / set size of every test point for
    each method in ``COMPARED_METHODS``.
    """
    true_idx, pred_idx, soft_idx = cols
    n_calib = XY_cali.shape[0]
    n_testing = XY_testing.shape[0]
    idx_calibration = np.arange(n_calib)
    idx_calib_plus1 = np.arange(n_calib + 1)
    idx_testing = np.arange(n_testing)

    Y_cali_true = XY_cali.T[true_idx].astype(int)
    Y_cali_soft = XY_cali[:, soft_idx:]
    Y_testing_true = XY_testing.T[true_idx].astype(int)
    Y_testing_soft = XY_testing[:, soft_idx:]
    Y_src_true = XY_src.T[true_idx].astype(int)
    Y_src_pred = XY_src.T[pred_idx].astype(int)
    Y_tgt_pred = XY_tgt.T[pred_idx].astype(int)

    # --- confusion matrix (C[i, j] = #{pred = i, true = j}) and q ---------
    C_mtx = confusion_matrix(Y_src_true, Y_src_pred, labels=np.arange(num_classes)).T
    q_dist = q_samp_dist(Y_tgt_pred, num_classes)

    # --- BBSE point estimate, projected onto the nonnegative orthant -------
    bbse_wts = np.maximum(np.linalg.pinv(C_mtx / sz) @ (q_dist / sz), 0)

    # --- weight regions -----------------------------------------------------
    C_rgn = find_C_itvls(C_mtx, sz, num_classes, alpha_matrix=delta, itvl_type=itvl_type)
    q_rgn = find_q_itvls(q_dist, sz, num_classes, alpha_vec=delta, itvl_type=itvl_type)
    region_args = (C_rgn["C_low"], C_rgn["C_up"], q_rgn["q_low"], q_rgn["q_up"])

    wts_rgnLP = lp_solver(*region_args)
    wts_rgnGE = Omega_GEKH(*region_args)
    lp_length = wts_rgnLP.T[1] - wts_rgnLP.T[0]
    ge_length = wts_rgnGE.T[1] - wts_rgnGE.T[0]

    # --- Monte Carlo points of Omega and volume ratio ----------------------
    omega_sampling = np.random.default_rng(seed_omega)
    union_samples = omega_sampling.uniform(low=wts_rgnLP.T[0], high=wts_rgnLP.T[1],
                                           size=(n_mc_pool, num_classes))
    inside = validate_constraints(*region_args, union_samples.T)
    candidate_omega = union_samples[inside[:n_mc]]
    vol_ratio = inside.size / n_mc_pool
    if candidate_omega.shape[0] == 0:
        warnings.warn("No Monte Carlo point fell inside Omega; its prediction sets will be empty.")

    # --- calibration weights for the point methods and the Omega samples --
    point_wts = {"oracle": true_wts, "estG.estWts": bbse_wts}
    cali_wts = {m: cp_score_weights_optimal(w, Y_cali_true, idx_calibration) for m, w in point_wts.items()}
    omega_cali_wts = np.zeros((candidate_omega.shape[0], n_calib))
    for i_w, w in enumerate(candidate_omega):
        omega_cali_wts[i_w] = cp_score_weights_optimal(w, Y_cali_true, idx_calibration)

    # --- calibration scores --------------------------------------------------
    rng_cp = np.random.default_rng(seed_cp)
    us = rng_cp.uniform(0, 1, size=n_calib)
    r_xy = compute_randomized_nonconformity_optimal(Y_cali_soft, Y_cali_true, us)
    r_xy_y0 = np.append(r_xy, 1)                       # point mass at 1 for the test point
    r_xy_y0_srt_idx = np.argsort(r_xy_y0)[::-1]

    # --- thresholds tau_CP(y0) for every method ------------------------------
    tau_CP = {m: np.zeros(num_classes, dtype=np.float32) for m in COMPARED_METHODS}
    region_level = nominal_coverage + delta
    tau_CP["estG.Omega_LP"] = max_qtl_rxy_on_rectangular(r_xy_y0, Y_cali_true, num_classes, wts_rgnLP,
                                                         region_level, idx_calib_plus1, r_xy_y0_srt_idx)
    tau_CP["estG.Omega_GE"] = max_qtl_rxy_on_rectangular(r_xy_y0, Y_cali_true, num_classes, wts_rgnGE,
                                                         region_level, idx_calib_plus1, r_xy_y0_srt_idx)

    for y0 in range(num_classes):
        for method, w in point_wts.items():
            w_ext = np.append(cali_wts[method], w[y0])
            cutoff = w_ext.sum() * (1 - nominal_coverage)
            tau_CP[method][y0] = quantile_optimal(r_xy_y0, r_xy_y0_srt_idx, w_ext, cutoff, nominal_coverage)

        # Monte Carlo supremum over Omega
        for i_w, sample_wts in enumerate(omega_cali_wts):
            if np.all(sample_wts == 0):
                sample_wts += 1.0
            w_ext = np.append(sample_wts, candidate_omega[i_w][y0])
            cutoff = w_ext.sum() * (1 - region_level)
            tau = quantile_optimal(r_xy_y0, r_xy_y0_srt_idx, w_ext, cutoff, region_level)
            if tau > tau_CP["estG.Omega"][y0]:
                tau_CP["estG.Omega"][y0] = tau

    # Since Omega is inside Omega_LP, its (MC) threshold cannot exceed the
    # exact Omega_LP threshold; flag the trial if it does for every class.
    if (tau_CP["estG.Omega"] > tau_CP["estG.Omega_LP"]).sum() == num_classes:
        warnings.warn("tau_CP(Omega) > tau_CP(Omega_LP) for all classes; check this trial.")

    # --- prediction sets on the target test sample ----------------------------
    r_x_testing = np.zeros((n_testing, num_classes), dtype=np.float32)
    for ctg in range(num_classes):
        us = rng_cp.uniform(0, 1, size=n_testing)
        r_x_testing[:, ctg] = compute_randomized_nonconformity_optimal(
            Y_testing_soft, np.full(n_testing, ctg), us)

    coverage, set_size = {}, {}
    for method in COMPARED_METHODS:
        pred_sets = (r_x_testing <= tau_CP[method]).astype(int)
        set_size[method] = pred_sets.sum(axis=1)
        coverage[method] = pred_sets[idx_testing, Y_testing_true]

    return {
        "bbse_wts": bbse_wts,
        "lp_length": lp_length,
        "ge_length": ge_length,
        "vol_ratio": vol_ratio,
        "coverage": coverage,
        "set_size": set_size,
        "tau_CP": tau_CP,
    }
