import warnings
import numpy as np
from scipy.optimize import linprog

from . import _fast_aabb_cpp  # Import low-level C++ solver
from .intervals import find_C_itvls, find_q_itvls
from .utils import Omega_GEKH_pre  # Contains Numba-accelerated elimination logic


class BBSE:
    """Black Box Shift Estimator (Point Estimation)"""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.weights_ = None

    def fit(self, C_mtx: np.ndarray, q_dist: np.ndarray, M: int, N: int):
        """Fit BBSE point estimator.

        Parameters
        ----------
        C_mtx : np.ndarray
            Confusion matrix of shape (K, K).
        q_dist : np.ndarray
            Target domain prediction distribution of shape (K,).
        M : int
            Source domain sample size.
        N : int
            Target domain sample size.
        """
        P_C = C_mtx / M
        P_q = q_dist / N
        wts = np.linalg.pinv(P_C) @ P_q
        self.weights_ = np.maximum(wts, 0.0)
        return self.weights_


class LP:
    """Polytope AABB Solver using C++, OpenMP, and HiGHS Warm-Start."""

    def __init__(
        self,
        bound_single: tuple = (0.0, 500.0),
        tol: float = 1e-7,
        itvl_type: str = "Clopper-Pearson",
    ):
        self.bound_single = bound_single
        self.tol = tol
        self.itvl_type = itvl_type

        # Initializing fitted attributes (following Scikit-Learn naming convention)
        self.bounds_ = None
        self.slack_sum_ = 0.0
        self.slack_vector_ = None
        self.possible_negative_dims_ = np.array([], dtype=int)  # Indices of potential negative dimensions
        self.is_feasible_ = True

    def _check_slack(self, C_low, C_up, q_low, q_up):
        """Phase 1: Compute optimal slack vector s* minimizing L1 norm distance to R_+^n."""
        m1, n = C_low.shape
        m2 = C_up.shape[0]

        A_combined = np.vstack([C_low, -C_up])
        b_combined = np.hstack([q_up, -q_low])

        c_p1 = np.hstack([np.zeros(n), np.ones(n)])
        A_p1 = np.vstack(
            [
                np.hstack([A_combined, np.zeros((m1 + m2, n))]),
                np.hstack([-np.eye(n), -np.eye(n)]),
            ]
        )
        b_p1 = np.hstack([b_combined, np.zeros(n)])
        bounds_p1 = [(None, None)] * n + [(0, None)] * n

        res_p1 = linprog(
            c_p1, A_ub=A_p1, b_ub=b_p1, bounds=bounds_p1, method="highs"
        )

        if res_p1.success:
            s_star = res_p1.x[n:]
            # Clean up floating-point numerical noise below tolerance
            s_star[s_star < self.tol] = 0.0
            return s_star.sum(), s_star

        # If solver fails, return inf values
        return np.inf, np.full(n, np.inf)

    def fit(
        self,
        C_low: np.ndarray,
        C_up: np.ndarray,
        q_low: np.ndarray,
        q_up: np.ndarray,
    ):
        """Solve LP bounds given interval matrices and vectors."""
        # 1. Perform Phase 1 check to compute minimal L1 slack sum and slack vector
        self.slack_sum_, self.slack_vector_ = self._check_slack(
            C_low, C_up, q_low, q_up
        )

        # 2. Necessary & Sufficient condition: slack_sum_ == 0 iff feasible region in R_+^n is non-empty
        self.is_feasible_ = self.slack_sum_ == 0.0

        # 3. Record candidate negative dimensions (where s_i* > 0 under current optimal vertex)
        self.possible_negative_dims_ = np.where(self.slack_vector_ > 0.0)[0]

        # 4. Set solver bounds based on feasibility
        if self.is_feasible_:
            lb, ub = self.bound_single
        else:
            lb, ub = -1e30, 1e30

        A_combined = np.ascontiguousarray(
            np.vstack([C_low, -C_up]), dtype=np.float64
        )
        b_ub = np.ascontiguousarray(
            np.hstack([q_up, -q_low]), dtype=np.float64
        )

        # 5. Call low-level C++ solver
        results = _fast_aabb_cpp.solve_aabb(
            A_combined, b_ub, lb_val=lb, ub_val=ub
        )

        # 6. If infeasible (slack_sum > 0), issue warning and clip output bounds
        if not self.is_feasible_:
            warnings.warn(
                f"\nWarning:\nFeasible region in R_+^{A_combined.shape[1]} is empty. "
                f"\n(slack_sum = {self.slack_sum_:.6f} > 0). "
                f"\nCandidate negative dimension(s) under current vertex: {self.possible_negative_dims_}. "
                "\nOutput bounds have been clipped to arg=bound_single.",
                UserWarning,
            )
            results = np.clip(
                results, self.bound_single[0], self.bound_single[1]
            )

        self.bounds_ = results
        return self.bounds_

    def fit_from_data(
        self, C_mtx, q_dist, M, N, delta, itvl_type: str | None = None
    ):
        """Automatically compute confidence intervals and solve LP bounds."""
        effective_itvl_type = itvl_type if itvl_type is not None else self.itvl_type
        K = len(q_dist)
        C_rgn = find_C_itvls(
            C_mtx, M, K, alpha_matrix=delta, itvl_type=effective_itvl_type
        )
        q_rgn = find_q_itvls(
            q_dist, N, K, alpha_vec=delta, itvl_type=effective_itvl_type
        )
        return self.fit(
            C_rgn["C_low"], C_rgn["C_up"], q_rgn["q_low"], q_rgn["q_up"]
        )


class GE:
    """Gaussian Elimination Estimator"""

    def __init__(
        self,
        upper_limit: float = 500.0,
        itvl_type: str = "Clopper-Pearson",
    ):
        self.upper_limit = upper_limit
        self.itvl_type = itvl_type
        self.bounds_ = None

    def fit(
        self,
        C_low: np.ndarray,
        C_up: np.ndarray,
        q_low: np.ndarray,
        q_up: np.ndarray,
    ):
        w_low, w_up = Omega_GEKH_pre(
            C_low, C_up, q_low, q_up, mal_upper_limit=self.upper_limit
        )

        # Check if lower bound exceeds upper bound
        if np.any(w_low > w_up):
            warnings.warn(
                "\nWarning:\nInvalid interval detected (w_low > w_up). Setting all interval bounds to 0.",
                UserWarning,
            )
            self.bounds_ = np.zeros((len(w_low), 2), dtype=np.float64)
        else:
            rtn = np.column_stack([w_low, w_up])
            self.bounds_ = np.clip(rtn, 0, self.upper_limit)

        return self.bounds_

    def fit_from_data(
        self, C_mtx, q_dist, M, N, delta, itvl_type: str | None = None
    ):
        """Automatically compute confidence intervals and solve GE bounds."""
        effective_itvl_type = itvl_type if itvl_type is not None else self.itvl_type
        K = len(q_dist)
        C_rgn = find_C_itvls(
            C_mtx, M, K, alpha_matrix=delta, itvl_type=effective_itvl_type
        )
        q_rgn = find_q_itvls(
            q_dist, N, K, alpha_vec=delta, itvl_type=effective_itvl_type
        )
        return self.fit(
            C_rgn["C_low"], C_rgn["C_up"], q_rgn["q_low"], q_rgn["q_up"]
        )