"""Helpers for the simulations with Gaussian mixture data."""

import numpy as np


def prediction_table(XY, probs):
    """Prediction-file layout ``[idx, true, true, hard_pred, probs...]`` for simulated data.

    ``XY`` has the label in its last column (``sampling.toy_data_generating``)
    and ``probs`` holds the predicted class probabilities; the hard prediction
    is their argmax.  The table can be passed to ``conformal.run_label_shift_trial``.
    """
    y = XY[:, -1]
    return np.column_stack([np.arange(len(y)), y, y, probs.argmax(axis=1), probs])


def circle_means(n_class):
    """Class means for the runtime study (Appendix K.4): the origin, then circles of
    radius 1, 2, 3, ... with at most eight means 45 degrees apart on each."""
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    n_circles = int(np.ceil((n_class - 1) / 8))
    points = [np.zeros((1, 2))] + [r * np.column_stack((np.cos(angles), np.sin(angles)))
                                   for r in range(1, n_circles + 1)]
    return np.vstack(points)[:n_class]
