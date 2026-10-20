import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue" # "workqueue" / "tbb" / "omp"
import warnings
from collections import Counter
import numpy as np
import scipy.stats as sps
from numba import njit, prange

CI_UPPER_LIMIT = np.inf


import numpy as np

def toy_data_generating(mus, Sigma, sample_size, proportions, rng=None, min_samples_per_class=0):
    """Generate synthetic data from a Gaussian Mixture Model.

    Parameters
    ----------
    mus : np.ndarray
        Array of mean vectors for each component class, shape (K, d).
    Sigma : np.ndarray
        Covariance matrix. Can be shape (d, d) for shared covariance
        or (K, d, d) for class-specific covariances.
    sample_size : int
        Total number of samples to generate.
    proportions : np.ndarray
        Prior class proportions vector, shape (K,).
    min_samples_per_class : int, optional
        Minimum number of samples guaranteed for each class (default is 0).
    rng : np.random.Generator or int, optional
        Random number generator instance or seed.

    Returns
    -------
    np.ndarray
        Combined feature matrix and label array of shape (N, d + 1),
        where the last column contains 0-based integer class labels.
    """
    K = len(mus)
    min_total = K * min_samples_per_class
    if sample_size < min_total:
        raise ValueError(
            f"sample_size ({sample_size}) must be at least "
            f"min_samples_per_class * K ({min_total})."
        )

    if Sigma.ndim == 2:
        cov_type = "shared"
    else:
        cov_type = "per_class"

    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    # 1. keep min samples
    remaining_size = sample_size - min_total
    counts = rng.multinomial(remaining_size, proportions) + min_samples_per_class

    data_list = []

    for k, mu in enumerate(mus):
        n_k = counts[k]
        if n_k == 0:
            continue

        cov_k = Sigma if cov_type == "shared" else Sigma[k]
        x_k = rng.multivariate_normal(mean=mu, cov=cov_k, size=n_k)
        y_k = np.full((n_k, 1), k, dtype=np.float64)

        data_list.append(np.column_stack([x_k, y_k]))
    min_samples_per_class = 0
    return np.vstack(data_list)


def bayes_classifier(x, mus, sigma_mtx, p_cls, out_type="hard"):
    """Evaluate optimal Bayes Classifier prediction under Gaussian assumptions.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix of shape (N, d).
    mus : np.ndarray
        Class means of shape (K, d).
    sigma_mtx : np.ndarray
        Covariance matrix (d, d) or (K, d, d).
    p_cls : np.ndarray
        Prior class probabilities of shape (K,).
    out_type : str, default='hard'
        If 'hard', returns class indices (N,).
        If 'soft', returns posterior probabilities matrix (N, K).

    Returns
    -------
    np.ndarray
        Class predictions or posterior probability matrix.
    """
    cov_type = "shared" if sigma_mtx.ndim == 2 else "per_class"
    post_probs = []

    for k, prior in enumerate(p_cls):
        cov_k = sigma_mtx if cov_type == "shared" else sigma_mtx[k]
        likelihood = sps.multivariate_normal.pdf(x, mean=mus[k], cov=cov_k)
        post_probs.append(likelihood * prior)

    post_probs = np.array(post_probs)

    if out_type == "soft":
        return (post_probs / post_probs.sum(axis=0)).T

    return np.argmax(post_probs, axis=0)

@njit(parallel=True)
def confusion_mtx(true_labs, pred_labs, num_cls):
    """Compute confusion matrix indexed by C[i, j] = count(pred == i, true == j).

    Note: This is the transpose of `sklearn.metrics.confusion_matrix`.
    """
    mtx = np.zeros((num_cls, num_cls), dtype=np.float32)
    for i, j in zip(pred_labs, true_labs):
        mtx[i, j] += 1.0
    return mtx


def q_samp_dist(tgt_labs, num_cls):
    """Compute empirical label counts for target domain predictions."""
    q_est = np.zeros(num_cls, dtype=np.float64)
    for cls_idx, count in Counter(tgt_labs).items():
        q_est[cls_idx] = count
    return q_est


@njit(parallel=True)
def Omega_GEKH_pre(C_low, C_up, q_low, q_up, mal_upper_limit=CI_UPPER_LIMIT):
    """Perform Gaussian Elimination interval bounds precomputation (Numba accelerated).

    Parameters
    ----------
    C_low : np.ndarray
        Lower bound of confusion matrix (K, K).
    C_up : np.ndarray
        Upper bound of confusion matrix (K, K).
    q_low : np.ndarray
        Lower bound of target prediction counts (K,).
    q_up : np.ndarray
        Upper bound of target prediction counts (K,).
    mal_upper_limit : float, optional
        Upper bound truncation limit for elements.

    Returns
    -------
    tuple of (w_low, w_up)
        Estimated lower and upper weight vectors of shape (K,).
    """
    K = len(q_low)
    if C_low.shape != (K, K) or C_up.shape != (K, K):
        raise ValueError("C_low and C_up must be K x K matrices.")
    if len(q_up) != K:
        raise ValueError("q_low and q_up must be vectors of length K.")

    C_low_temp = np.zeros((K, K, K), dtype=np.float64)
    C_up_temp = np.zeros((K, K, K), dtype=np.float64)
    q_low_temp = np.zeros((K, K), dtype=np.float64)
    q_up_temp = np.zeros((K, K), dtype=np.float64)

    C_low_temp[0, :, :] = C_low
    C_up_temp[0, :, :] = C_up
    q_low_temp[0, :] = q_low
    q_up_temp[0, :] = q_up

    rgu_inf = 1e-20

    for t in range(K - 1):
        for i in range(K):
            for j in range(K):
                if (i + 1 > t + 1) and (j + 1 <= t + 1):
                    C_low_temp[t + 1, i, j] = 0.0
                    C_up_temp[t + 1, i, j] = 0.0
                elif (i + 1 > t + 1) and (j + 1 > t + 1):
                    C_low_temp[t + 1, i, j] = max(
                        C_low_temp[t, i, j]
                        - (C_up_temp[t, i, t] * C_up_temp[t, t, j])
                        / max(rgu_inf, C_low_temp[t, t, t]),
                        rgu_inf,
                    )
                    C_up_temp[t + 1, i, j] = max(
                        C_up_temp[t, i, j]
                        - (C_low_temp[t, i, t] * C_low_temp[t, t, j])
                        / C_up_temp[t, t, t],
                        rgu_inf,
                    )
                else:
                    C_low_temp[t + 1, i, j] = C_low_temp[t, i, j]
                    C_up_temp[t + 1, i, j] = C_up_temp[t, i, j]

            if i + 1 > t + 1:
                q_low_temp[t + 1, i] = max(
                    q_low_temp[t, i]
                    - (C_up_temp[t, i, t] * q_up_temp[t, t])
                    / max(rgu_inf, C_low_temp[t, t, t]),
                    rgu_inf,
                )
                q_up_temp[t + 1, i] = max(
                    q_up_temp[t, i]
                    - (C_low_temp[t, i, t] * q_low_temp[t, t])
                    / C_up_temp[t, t, t],
                    rgu_inf,
                )
            else:
                q_low_temp[t + 1, i] = q_low_temp[t, i]
                q_up_temp[t + 1, i] = q_up_temp[t, i]

    w_low = np.zeros(K, dtype=np.float64)
    w_up = np.zeros(K, dtype=np.float64)

    w_low[K - 1] = q_low_temp[K - 1, K - 1] / C_up_temp[K - 1, K - 1, K - 1]
    w_up[K - 1] = q_up_temp[K - 1, K - 1] / max(
        C_low_temp[K - 1, K - 1, K - 1], 1e-6
    )

    for i_r in range(K - 1, 0, -1):
        i_idx = i_r - 1
        if i_idx + 1 < K:
            s_low = np.sum(
                C_low_temp[K - 1, i_idx, (i_idx + 1) : K]
                * w_low[(i_idx + 1) : K]
            )
            s_up = np.sum(
                C_up_temp[K - 1, i_idx, (i_idx + 1) : K]
                * w_up[(i_idx + 1) : K]
            )
        else:
            s_low = 0.0
            s_up = 0.0

        w_low[i_idx] = max(
            (q_low_temp[K - 1, i_idx] - s_up) / C_up_temp[K - 1, i_idx, i_idx],
            0.0,
        )
        w_up[i_idx] = max(
            (q_up_temp[K - 1, i_idx] - s_low)
            / max(rgu_inf, C_low_temp[K - 1, i_idx, i_idx]),
            0.0,
        )

    return w_low, w_up


def validate_constraints(A1, A2, b2, b1, X):
    """Filter point set X to find indices satisfying linear constraints A1 @ X <= b1 and A2 @ X >= b2."""
    satisfies_A1 = np.all((A1 @ X).T <= b1, axis=1)
    satisfies_A2 = np.all((A2 @ X).T >= b2, axis=1)
    satisfies_nonneg = np.all(X.T >= 0, axis=1)

    is_valid = satisfies_A1 & satisfies_A2 & satisfies_nonneg
    return np.where(is_valid)[0]


def sample_with_rng(data, m, class_probs, rng, class_col_idx=1, min_smp = 0, return_both=False):
    """
    data: n x c numpy array
    m: Total samples to draw
    class_probs: List of probabilities (must sum to 1.0)
    rng: numpy random generator (np.random.default_rng)
    return_both: 
        If True: Samples WITHOUT replacement (unique rows) and returns 
                 (sampled_data, remaining_data).
        If False: Samples WITH replacement (allows duplicates) and returns 
                  sampled_data only.
    """
    n_rows = data.shape[0]
    unique_classes = np.unique(data[:, class_col_idx])
    
    # Decide how many rows to take from each class bucket
    counts = rng.multinomial(m, class_probs)
    zero_cnts = counts == 0
    counts[zero_cnts] = 1

    counts[np.argmax(counts)] -= zero_cnts.sum()

    
    selected_indices = []
    
    # We use return_both to decide if we allow duplicates
    use_replacement = not return_both

    for i, cls in enumerate(unique_classes):
        num_to_sample = max(counts[i], min_smp)
        #if num_to_sample == 0:
        #    continue
            
        class_indices = np.where(data[:, class_col_idx] == cls)[0]
        
        # Validation: If sampling without replacement, ensure we have enough rows
        if return_both and num_to_sample > len(class_indices):
            raise ValueError(f"Class {cls} has only {len(class_indices)} rows, "
                             f"but multinomial requested {num_to_sample} unique samples.")

        if len(class_indices) > 0:
            sampled_block = rng.choice(
                class_indices, 
                size=num_to_sample, 
                replace=use_replacement
            )
            selected_indices.extend(sampled_block)

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    sampled_data = data[selected_indices]

    if not return_both:
        return sampled_data

    # Calculate remaining data (only works/makes sense when replace=False)
    all_indices = np.arange(n_rows)
    remaining_mask = np.setdiff1d(all_indices, selected_indices)
    remaining_data = data[remaining_mask]

    return sampled_data, remaining_data



if __name__ == "__main__":
    print("MaC-LP Utility Functions Module.")



    