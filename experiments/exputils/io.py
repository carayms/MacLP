"""Reading prediction files and writing result files.

Prediction files (one per trained classifier) are tab-separated with a
header line and the columns

    orgi.Idx  true.Cls.Vrfy  true.Cls  hard.Pred  soft.Pr(y=0) ... soft.Pr(y=K-1)

i.e. the original index of the observation, the true label (twice), the
predicted label and the K predicted probabilities.  The classifier scripts
in ``classifiers/`` and ``nuimages/`` write this layout.
"""

import os
import pickle
from pathlib import Path

import numpy as np

# Column positions in a prediction table.
TRUE_IDX = 1
PRED_IDX = 3
SOFT_IDX = 4
PREDICTION_HEADER = ["orgi.Idx", "true.Cls.Vrfy", "true.Cls", "hard.Pred"]


def load_predi_data(path):
    """Read one prediction file; returns (header, float32 array)."""
    with open(path, "r") as f:
        header = f.readline().split()
        rows = [np.array(line.strip("\n").split("\t")) for line in f.readlines()]
    return header, np.array(rows).astype(np.float32)


def list_prediction_files(folder, suffixes=(".tsv", ".xls")):
    """Sorted prediction files in ``folder`` (``.xls`` is the old suffix)."""
    names = sorted(f for f in os.listdir(folder) if f.endswith(tuple(suffixes)))
    if not names:
        raise FileNotFoundError(f"No prediction files ({', '.join(suffixes)}) in {folder}")
    return names


def write_prediction_file(path, orig_idx, true_labels, probs):
    """Write a prediction table in the layout described above."""
    probs = np.asarray(probs)
    hard_pred = np.argmax(probs, axis=1)
    header = PREDICTION_HEADER + [f"soft.Pr(y={k})" for k in range(probs.shape[1])]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for i in range(probs.shape[0]):
            fields = [str(int(orig_idx[i])), str(int(true_labels[i])), str(int(true_labels[i])),
                      str(int(hard_pred[i]))] + [repr(float(p)) for p in probs[i]]
            f.write("\t".join(fields) + "\n")


RESULT_DESCRIPTION = "\n".join([
    "LPvsGE_length       array (2, n_repts, K): interval lengths of Omega_LP (index 0) and Omega_GE (index 1)",
    "omega_omegaLP_volR  array (n_repts,): Monte Carlo estimate of vol(Omega) / vol(Omega_LP)",
    "TRUE_wts            array (K,): true importance weights q(y) / p(y)",
    "estimated_wts       array (n_repts, K): BBSE estimates",
    "compr_methods       list: names of the compared methods",
    "size_repts          dict method -> array (n_repts, n_testing): prediction-set sizes",
    "coverage_repts      dict method -> array (n_repts,): fraction of test points covered",
])


def save_results(path, LPvsGE_length, vol_ratio, true_wts, estimated_wts, methods, size_repts, coverage_repts):
    """Pickle the results of one setting as the list read by the analysis scripts."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [RESULT_DESCRIPTION, LPvsGE_length, vol_ratio, true_wts, estimated_wts,
            methods, size_repts, coverage_repts]
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_results(path):
    """Inverse of ``save_results``; returns the 8-element list."""
    with open(path, "rb") as f:
        return pickle.load(f)
