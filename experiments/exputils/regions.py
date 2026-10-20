"""Confidence regions for the importance weights.

Notation (paper, Section 3.2): given entry-wise intervals C* in [C_low, C_up]
and q* in [q_low, q_up], the MaC region is

    Omega = { w >= 0 : C_up @ w >= q_low,  C_low @ w <= q_up }   (Theorem 3.2)

* ``Omega_LPML``   -- Omega_LP, the smallest axis-aligned box containing Omega
                      (2K linear programs, eqs. (2)-(3)), solved with SciPy/HiGHS.
                      ``Omega_LPML_WarmStart`` (highspy) and ``Omega_LPML_WarmStart2``
                      (C++ solver of the maclp package) return the same box faster.
* ``Omega_GEKH``   -- Omega_GE, interval Gaussian elimination of Si et al.
                      (Appendix I).
* ``validate_constraints`` -- membership test for Omega, used for the Monte
                      Carlo volume ratio and for sampling points of Omega.

Most functions keep the argument order (A1, A2, b2, b1) =
(C_low, C_up, q_low, q_up) of the original code.
"""

import numpy as np
from numba import njit
from scipy.optimize import linprog

# Cap for the upper bounds of the weights (Appendix K.5, "Unbounded regions and rare source classes").
CI_UPPER_LIMIT = 500


# ---------------------------------------------------------------------------
# Omega_LP
# ---------------------------------------------------------------------------
def Omega_LPML(A1, A2, b2, b1, bound_single=(0, CI_UPPER_LIMIT)):
    """Omega_LP via 2K independent linear programs (SciPy / HiGHS).

    Parameters
    ----------
    A1 : C_low, shape (K, K)
    A2 : C_up, shape (K, K)
    b2 : q_low, shape (K,)
    b1 : q_up, shape (K,)
    bound_single : (lower, upper) bound applied to every w_k.

    Returns
    -------
    array of shape (K, 2) with rows [w_k_min, w_k_max]; NaN if an LP fails.
    """
    n = A1.shape[1]
    bounds = [bound_single] * n

    # A1 @ w <= b1  and  -A2 @ w <= -b2
    A_combined = np.vstack([A1, -A2])
    b_combined = np.hstack([b1, -b2])
    unit = np.eye(n)

    results = []
    for i in range(n):
        res_min = linprog(unit[i], A_ub=A_combined, b_ub=b_combined, bounds=bounds, method="highs")
        l_i = res_min.fun if res_min.success else np.nan

        res_max = linprog(-unit[i], A_ub=A_combined, b_ub=b_combined, bounds=bounds, method="highs")
        u_i = -res_max.fun if res_max.success else np.nan

        results.append((l_i, u_i))
    return np.array(results)


def Omega_LPML_WarmStart(A1, A2, b2, b1, bound_single=(0, CI_UPPER_LIMIT)):
    """Omega_LP with a single HiGHS model that is re-solved 2K times.

    Only the objective changes between the 2K programs, so HiGHS warm-starts
    each solve from the previous basis.  Pure-Python counterpart of the
    C++/OpenMP solver shipped in ``maclp._fast_aabb_cpp``.
    """
    import highspy   # imported here: its bundled HiGHS can clash with the one linked into maclp

    n = A1.shape[1]
    A_combined = np.vstack([A1, -A2])
    b_ub = np.hstack([b1, -b2])

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)

    lb_val = bound_single[0] if bound_single[0] is not None else -np.inf
    ub_val = bound_single[1] if bound_single[1] is not None else np.inf
    highs.addVars(n, np.full(n, lb_val), np.full(n, ub_val))
    for j in range(A_combined.shape[0]):
        highs.addRow(-np.inf, b_ub[j], n, np.arange(n, dtype=np.int32), A_combined[j])

    col_indices = np.arange(n, dtype=np.int32)
    results = []
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1.0
        highs.changeColsCost(n, col_indices, c)
        highs.run()
        optimal = highs.getModelStatus() == highspy.HighsModelStatus.kOptimal
        l_i = highs.getSolution().col_value[i] if optimal else -np.inf

        c[i] = -1.0
        highs.changeColsCost(n, col_indices, c)
        highs.run()
        optimal = highs.getModelStatus() == highspy.HighsModelStatus.kOptimal
        u_i = highs.getSolution().col_value[i] if optimal else np.inf

        results.append((l_i, u_i))
    return np.array(results)


def Omega_LPML_WarmStart2(A1, A2, b2, b1, bound_single=(0, CI_UPPER_LIMIT)):
    """Omega_LP with the C++ solver of the ``maclp`` package (Appendix K.4).

    The 2K programs are solved by HiGHS (dual simplex, warm start) in C++,
    distributed over CPU cores with OpenMP; set OMP_NUM_THREADS to control
    the number of cores.  Same interface and output as ``Omega_LPML``.
    Requires the ``maclp`` package (``pip install .`` in the repository root).
    """
    try:
        from maclp import _fast_aabb_cpp
    except ImportError as err:
        raise ImportError("Omega_LPML_WarmStart2 needs the compiled maclp package; "
                          "install it from the repository root with `pip install .`") from err

    A_combined = np.ascontiguousarray(np.vstack([A1, -A2]), dtype=np.float64)
    b_ub = np.ascontiguousarray(np.hstack([b1, -b2]), dtype=np.float64)
    lb_val = bound_single[0] if bound_single[0] is not None else -1e30
    ub_val = bound_single[1] if bound_single[1] is not None else 1e30
    return _fast_aabb_cpp.solve_aabb(A_combined, b_ub, lb_val=lb_val, ub_val=ub_val)


def Omega_LPML_VariableElastic(A1, A2, b2, b1, tol=1e-7, verbose=True):
    """Phase-1 LP that measures how far Omega is from the nonnegative orthant.

    Solves  min sum_k s_k  s.t.  A1 @ w <= b1,  A2 @ w >= b2,  w >= -s,  s >= 0
    with w free.  The optimum is 0 iff Omega is nonempty; a positive s_k
    flags w_k as a coordinate that is forced negative at the optimal vertex.

    Returns the slack vector s* of shape (K,).
    """
    m1, n = A1.shape
    m2 = A2.shape[0]
    A_combined = np.vstack([A1, -A2])
    b_combined = np.hstack([b1, -b2])

    # decision variables z = [w_1..w_K, s_1..s_K]
    c_p1 = np.hstack([np.zeros(n), np.ones(n)])
    A_p1 = np.vstack([
        np.hstack([A_combined, np.zeros((m1 + m2, n))]),
        np.hstack([-np.eye(n), -np.eye(n)]),  # -w - s <= 0
    ])
    b_p1 = np.hstack([b_combined, np.zeros(n)])
    bounds_p1 = [(None, None)] * n + [(0, None)] * n

    res_p1 = linprog(c_p1, A_ub=A_p1, b_ub=b_p1, bounds=bounds_p1, method="highs")
    if not res_p1.success:
        raise RuntimeError(
            f"Phase 1 optimization failed: {res_p1.message}. "
            "Check whether q_low exceeds q_up."
        )

    s_star = res_p1.x[n:]
    s_star[s_star < tol] = 0.0
    if verbose:
        print(f"[phase 1] sum of slack s* = {s_star.sum():.4f}")
    return s_star


def Omega_LowBD(A1, A2, b2, b1, verbose=True):
    """Largest depth of the unconstrained region into the negative orthant.

    Solves  max sum_k s_k  s.t.  A1 @ w <= b1,  A2 @ w >= b2,  w + s <= 0
    with w and s free.  s_k > 0 means the region reaches w_k = -s_k.

    Returns (s_star, w_star).
    """
    m1, n = A1.shape
    m2 = A2.shape[0]
    A_combined = np.vstack([A1, -A2])
    b_combined = np.hstack([b1, -b2])

    # decision variables z = [w_1..w_n, s_1..s_n];  max sum(s) <=> min -sum(s)
    c = np.hstack([np.zeros(n), -np.ones(n)])
    A_ub = np.vstack([
        np.hstack([A_combined, np.zeros((m1 + m2, n))]),
        np.hstack([np.eye(n), np.eye(n)]),  # w + s <= 0
    ])
    b_ub = np.hstack([b_combined, np.zeros(n)])
    bounds = [(None, None)] * (2 * n)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    w_star = res.x[:n]
    s_star = res.x[n:]
    if verbose:
        print(f"[negative depth] status: {res.message}")
        for k in range(n):
            if s_star[k] > 0:
                print(f"  omega_{k + 1}: reaches {-s_star[k]:.4f} in the negative direction")
            else:
                print(f"  omega_{k + 1}: contains nonnegative values (s = {s_star[k]:.4f})")
    return s_star, w_star


# ---------------------------------------------------------------------------
# Omega_GE: interval Gaussian elimination (Si et al., Appendix I)
# ---------------------------------------------------------------------------
@njit
def Omega_GEKH_pre(C_low, C_up, q_low, q_up):
    """Forward elimination and back substitution with interval arithmetic.

    Implements the recursions of Appendix I.  Zero or negative intermediate
    lower bounds are floored at a tiny positive value (1e-20) so that the
    divisions stay finite; the final bounds are floored at 0.

    Returns (w_low, w_up), each of shape (K,).
    """
    K = len(q_low)
    if C_low.shape != (K, K):
        raise ValueError("C_low must be K x K.")
    if C_up.shape != (K, K):
        raise ValueError("C_up must be K x K.")
    if len(q_low) != K or len(q_up) != K:
        raise ValueError("q_low and q_up must be length K.")

    # [t, i, j]: intervals after t elimination steps
    C_low_t = np.zeros((K, K, K), dtype=np.float64)
    C_up_t = np.zeros((K, K, K), dtype=np.float64)
    q_low_t = np.zeros((K, K), dtype=np.float64)
    q_up_t = np.zeros((K, K), dtype=np.float64)

    C_low_t[0, :, :] = C_low
    C_up_t[0, :, :] = C_up
    q_low_t[0, :] = q_low
    q_up_t[0, :] = q_up

    tiny = 1e-20

    # forward elimination (0-based: step t eliminates column t below row t)
    for t in range(K - 1):
        for i in range(K):
            for j in range(K):
                if i > t and j <= t:
                    C_low_t[t + 1, i, j] = 0.0
                    C_up_t[t + 1, i, j] = 0.0
                elif i > t and j > t:
                    C_low_t[t + 1, i, j] = max(
                        C_low_t[t, i, j] - C_up_t[t, i, t] * C_up_t[t, t, j] / max(tiny, C_low_t[t, t, t]),
                        tiny,
                    )
                    C_up_t[t + 1, i, j] = max(
                        C_up_t[t, i, j] - C_low_t[t, i, t] * C_low_t[t, t, j] / C_up_t[t, t, t],
                        tiny,
                    )
                else:
                    C_low_t[t + 1, i, j] = C_low_t[t, i, j]
                    C_up_t[t + 1, i, j] = C_up_t[t, i, j]

            if i > t:
                q_low_t[t + 1, i] = max(
                    q_low_t[t, i] - C_up_t[t, i, t] * q_up_t[t, t] / max(tiny, C_low_t[t, t, t]),
                    tiny,
                )
                q_up_t[t + 1, i] = max(
                    q_up_t[t, i] - C_low_t[t, i, t] * q_low_t[t, t] / C_up_t[t, t, t],
                    tiny,
                )
            else:
                q_low_t[t + 1, i] = q_low_t[t, i]
                q_up_t[t + 1, i] = q_up_t[t, i]

    # back substitution
    w_low = np.zeros(K, dtype=np.float64)
    w_up = np.zeros(K, dtype=np.float64)

    w_low[K - 1] = q_low_t[K - 1, K - 1] / C_up_t[K - 1, K - 1, K - 1]
    w_up[K - 1] = q_up_t[K - 1, K - 1] / max(C_low_t[K - 1, K - 1, K - 1], 1e-6)

    for i_idx in range(K - 2, -1, -1):
        s_low = np.sum(C_low_t[K - 1, i_idx, (i_idx + 1):K] * w_low[(i_idx + 1):K])
        s_up = np.sum(C_up_t[K - 1, i_idx, (i_idx + 1):K] * w_up[(i_idx + 1):K])
        w_low[i_idx] = max((q_low_t[K - 1, i_idx] - s_up) / C_up_t[K - 1, i_idx, i_idx], 0)
        w_up[i_idx] = max((q_up_t[K - 1, i_idx] - s_low) / max(tiny, C_low_t[K - 1, i_idx, i_idx]), 0)
    return w_low, w_up


def Omega_GEKH(C_low, C_up, q_low, q_up, upper_limit=CI_UPPER_LIMIT):
    """Omega_GE as an array of shape (K, 2), clipped to [0, upper_limit]."""
    w_low, w_up = Omega_GEKH_pre(C_low, C_up, q_low, q_up)
    return np.clip(np.column_stack([w_low, w_up]), 0, upper_limit)


def myGauss(lm, um):
    """Interval Gaussian elimination, list-based reference version of Si et al.

    ``lm`` and ``um`` are the lower / upper augmented matrices [C | q].
    Returns (lower, upper) lists for w.  Kept for cross-checking Omega_GEKH.
    """
    # forward elimination
    for col in range(len(lm[0])):
        for row in range(col + 1, len(lm)):
            factor_l, factor_u = lm[row][col] / um[col][col], um[row][col] / lm[col][col]
            r_l = [(v * -factor_l) for v in lm[col]]  # subtract with the lower row
            r_u = [(v * -factor_u) for v in um[col]]  # subtract with the upper row
            um[row] = [sum(pair) for pair in zip(um[row], r_l)]
            um[row][col] = 0
            lm[row] = [sum(pair) for pair in zip(lm[row], r_u)]
            lm[row][col] = 0

    # back substitution, starting from the last row
    ans_l, ans_u = [], []
    um = um[::-1]
    lm = lm[::-1]
    for sol in range(len(lm)):
        if sol == 0:
            ans_l.append(lm[sol][-1] / um[sol][-2])
            ans_u.append(um[sol][-1] / lm[sol][-2])
        else:
            inner_l, inner_u = 0, 0
            for x in range(sol):
                inner_l += ans_l[x] * lm[sol][-2 - x]
                inner_u += ans_u[x] * um[sol][-2 - x]
            # the row now reads a * w + b = c, so w = (c - b) / a
            ans_u.append((um[sol][-1] - inner_l) / lm[sol][-sol - 2])
            ans_l.append((lm[sol][-1] - inner_u) / um[sol][-sol - 2])
    ans_l.reverse()
    ans_u.reverse()
    return ans_l, ans_u


def exe_myGauss(C_low, C_up, q_itv):
    """Wrapper of ``myGauss``; ``q_itv`` has shape (K, 2). Returns shape (K, 2)."""
    lm = np.concatenate([C_low, np.asarray(q_itv[:, 0])[:, None]], 1)
    um = np.concatenate([C_up, np.asarray(q_itv[:, 1])[:, None]], 1)
    g1, g2 = myGauss(lm, um)
    return np.column_stack([np.array(g1), np.array(g2)])


# ---------------------------------------------------------------------------
# Membership in Omega
# ---------------------------------------------------------------------------
def validate_constraints(A1, A2, b2, b1, X):
    """Indices of the columns of X (shape (K, n)) that lie in Omega.

    A point w is in Omega iff A1 @ w <= b1, A2 @ w >= b2 and w >= 0,
    with (A1, A2, b2, b1) = (C_low, C_up, q_low, q_up).
    """
    satisfies_A1 = np.all((A1 @ X).T <= b1, axis=1)
    satisfies_A2 = np.all((A2 @ X).T >= b2, axis=1)
    satisfies_nonneg = np.all(X.T >= 0, axis=1)
    return np.where(satisfies_A1 & satisfies_A2 & satisfies_nonneg)[0]
