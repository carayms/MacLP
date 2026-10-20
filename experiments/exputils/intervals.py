"""Element-wise confidence intervals for the confusion matrix C and the vector q.

Error-budget convention (as in the paper): when ``alpha_matrix`` (or
``alpha_vec``) is a float ``delta``, each of the K**2 entries of C and the K
entries of q gets level 1 - delta / (K**2 + K).  By the union bound the joint
level of (C, q) is therefore 1 - delta.
"""

import numpy as np
import scipy.stats as sps


def binomial_prop_itvls(x, n, alpha, itvl_type="Clopper-Pearson"):
    """Two-sided (1 - alpha) interval for a binomial proportion.

    Parameters
    ----------
    x : int
        Number of successes.
    n : int
        Number of trials.
    alpha : float
        Error level of the interval.
    itvl_type : {"Clopper-Pearson", "Jeffreys-Prior", "Uniform-Prior"}
        "Clopper-Pearson" is the exact interval used in the paper; the two
        Bayesian intervals (Beta(1/2, 1/2) and Beta(1, 1) priors) are kept
        for comparison.
    """
    if itvl_type == "Clopper-Pearson":
        if x == 0:
            lower = 0.0
            upper = 1 - (0.5 * alpha) ** (1 / n)
        elif x == n:
            lower = (0.5 * alpha) ** (1 / n)
            upper = 1.0
        else:
            lower = sps.beta.ppf(alpha / 2.0, x, n - x + 1)
            upper = sps.beta.ppf(1.0 - alpha / 2.0, x + 1, n - x)

    elif itvl_type == "Jeffreys-Prior":
        lower = sps.beta.ppf(alpha / 2.0, x + 0.5, n - x + 0.5)
        upper = sps.beta.ppf(1.0 - alpha / 2.0, x + 0.5, n - x + 0.5)

    elif itvl_type == "Uniform-Prior":
        lower = sps.beta.ppf(alpha / 2.0, x + 1.0, n - x + 1.0)
        upper = sps.beta.ppf(1.0 - alpha / 2.0, x + 1.0, n - x + 1.0)

    else:
        raise ValueError(f"Unknown interval type: {itvl_type}")
    return np.array([lower, upper])


def find_C_itvls(confusion_matrix, sample_size, K, alpha_matrix=None, itvl_type="Clopper-Pearson"):
    """Entry-wise intervals for C* from the K x K confusion counts.

    ``alpha_matrix`` is either a K x K array of per-entry error levels or a
    float delta, which is split as delta / (K**2 + K) per entry.
    Returns ``{"C_low": ..., "C_up": ...}`` (proportions, not counts).
    """
    if isinstance(alpha_matrix, float):
        alpha_matrix = np.full((K, K), alpha_matrix / (K**2 + K))

    C_low = np.empty((K, K))
    C_up = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            C_low[i, j], C_up[i, j] = binomial_prop_itvls(
                confusion_matrix[i, j], sample_size, alpha_matrix[i, j], itvl_type
            )
    return {"C_low": C_low, "C_up": C_up}


def find_q_itvls(q_est, sample_size, K, alpha_vec=None, itvl_type="Clopper-Pearson"):
    """Entry-wise intervals for q* from the K predicted-label counts.

    ``alpha_vec`` is either a length-K array of per-entry error levels or a
    float delta, which is split as delta / (K**2 + K) per entry.
    Returns ``{"q_low": ..., "q_up": ...}`` (proportions, not counts).
    """
    if isinstance(alpha_vec, float):
        alpha_vec = np.full(K, alpha_vec / (K**2 + K))

    q_low = np.zeros(K)
    q_up = np.zeros(K)
    for i in range(K):
        # q_est holds counts of the 0-based predicted labels
        q_low[i], q_up[i] = binomial_prop_itvls(int(q_est[i]), sample_size, alpha_vec[i], itvl_type)
    return {"q_low": q_low, "q_up": q_up}
