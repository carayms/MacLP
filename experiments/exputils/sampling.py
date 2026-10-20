"""Data generation and sampling utilities."""

from collections import Counter

import numpy as np
import scipy.stats as sps
from numba import njit


def toy_data_generating(mus, Sigma, sample_size, proportions, rng=None):
    """Draw a labeled sample from a Gaussian mixture.

    The class counts follow Multinomial(sample_size, proportions) and the
    features of class k follow N(mus[k], Sigma) (or N(mus[k], Sigma[k]) when a
    stack of K covariance matrices is given).

    Parameters
    ----------
    mus : array of shape (K, d)
        Class means.
    Sigma : array of shape (d, d) or (K, d, d)
        Shared or class-specific covariance matrix.
    sample_size : int
        Total number of observations.
    proportions : array of shape (K,)
        Class proportions.
    rng : np.random.Generator or int
        Random generator or seed.

    Returns
    -------
    array of shape (sample_size, d + 1)
        Features followed by the 0-based class label in the last column.
    """
    shared_cov = Sigma.ndim == 2

    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)

    counts = rng.multinomial(sample_size, proportions)
    blocks = []
    for k, mu in enumerate(mus):
        cov_k = Sigma if shared_cov else Sigma[k]
        x_k = rng.multivariate_normal(mean=mu, cov=cov_k, size=counts[k])
        blocks.append(np.column_stack([x_k, np.full(counts[k], k)]))
    return np.vstack(blocks)


def bayes_classifier(x, mus, sigma_mtx, p_cls, out_type="hard"):
    """Bayes classifier for a Gaussian mixture with known parameters.

    Parameters
    ----------
    x : array of shape (n, d)
    mus : array of shape (K, d)
    sigma_mtx : array of shape (d, d) or (K, d, d)
    p_cls : array of shape (K,)
        Class priors (the source proportions p(y)).
    out_type : {"hard", "soft"}
        "hard" returns 0-based labels of shape (n,), "soft" returns the
        posterior probabilities of shape (n, K).
    """
    shared_cov = sigma_mtx.ndim == 2
    post_probs = []
    for k, prior in enumerate(p_cls):
        cov_k = sigma_mtx if shared_cov else sigma_mtx[k]
        post_probs.append(sps.multivariate_normal.pdf(x, mean=mus[k], cov=cov_k) * prior)
    post_probs = np.array(post_probs)

    if out_type == "soft":
        return (post_probs / post_probs.sum(axis=0)).T
    return np.argmax(post_probs, axis=0)


@njit
def confusion_mtx(true_labs, pred_labs, num_cls):
    """Confusion counts with C[i, j] = #{pred == i, true == j}.

    This is the transpose of ``sklearn.metrics.confusion_matrix``.
    """
    mtx = np.zeros((num_cls, num_cls), dtype=np.float32)
    for i, j in zip(pred_labs, true_labs):
        mtx[i, j] += 1
    return mtx


def q_samp_dist(tgt_labs, num_cls):
    """Counts of the predicted labels in the target sample, shape (num_cls,)."""
    q_est = np.zeros(num_cls)
    for label, count in Counter(tgt_labs).items():
        q_est[label] = count
    return q_est


def sample_with_rng(data, m, class_probs, rng, class_col_idx=1, min_smp=0, return_both=False):
    """Stratified (multinomial) sampling of rows from a labeled table.

    The per-class counts are drawn from Multinomial(m, class_probs).  Classes
    with a zero count are given one observation, which is taken from the
    largest class so that the total stays m.  ``min_smp`` is an additional
    lower bound on the number of rows per class.

    Parameters
    ----------
    data : array of shape (n, c)
        Table to sample from; ``data[:, class_col_idx]`` holds the labels.
    m : int
        Number of rows to draw.
    class_probs : array of shape (K,)
        Class proportions (must sum to one).
    rng : np.random.Generator
    class_col_idx : int
        Column with the class labels.
    min_smp : int
        Minimum number of rows per class.
    return_both : bool
        If True, sample WITHOUT replacement and return (sample, remaining rows).
        If False, sample WITH replacement and return the sample only.
    """
    n_rows = data.shape[0]
    unique_classes = np.unique(data[:, class_col_idx])
    # counts[i] belongs to the i-th label present in the table, so every class must be present
    if unique_classes.size != len(class_probs):
        raise ValueError(f"The table contains {unique_classes.size} of the {len(class_probs)} classes; "
                         "every class needs at least one row.")

    counts = rng.multinomial(m, class_probs)
    zero_cnts = counts == 0
    counts[zero_cnts] = 1
    counts[np.argmax(counts)] -= zero_cnts.sum()

    selected_indices = []
    use_replacement = not return_both

    for i, cls in enumerate(unique_classes):
        num_to_sample = max(counts[i], min_smp)
        class_indices = np.where(data[:, class_col_idx] == cls)[0]

        if return_both and num_to_sample > len(class_indices):
            raise ValueError(
                f"Class {cls} has only {len(class_indices)} rows, "
                f"but {num_to_sample} rows without replacement were requested."
            )

        if len(class_indices) > 0:
            selected_indices.extend(
                rng.choice(class_indices, size=num_to_sample, replace=use_replacement)
            )

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)
    sampled_data = data[selected_indices]

    if not return_both:
        return sampled_data

    remaining = np.setdiff1d(np.arange(n_rows), selected_indices)
    return sampled_data, data[remaining]


def seed_sequence(master_seed, size, high=9999999):
    """Draw ``size`` distinct integer seeds from a master seed."""
    return np.random.default_rng(master_seed).choice(np.arange(high), size=size, replace=False)
