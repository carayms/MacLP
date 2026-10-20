"""Table S4: accuracy of the 10 classifiers trained on each dataset.

Table S4 reports the test accuracy of each classifier at model selection,
which the classifier scripts write into the prediction file names
(``..._acc.<acc>_...``).  This script reads those numbers and also computes
the accuracy on the analysis set from the prediction files themselves.
``--paper`` prints the table from the accuracies recorded for the paper.

Expects ``<pred-root>/{agnews,cifar10,mnist,n24news,nuimages}``; for nuImages
only the Singapore (source) files are used.

Example:
    python diagnostics/accuracy_table.py --pred-root data/predictions
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import prediction_files
from exputils.io import PRED_IDX, TRUE_IDX, load_predi_data

TABLE_DATASETS = {"AGNews": "AGNews", "CIFAR-10": "CIFAR-10", "MNIST": "MNIST", "N24News": "N24News",
                  "nuImages": "nuImages.SG"}
# test accuracies of the 10 classifiers per dataset behind Table S4
PAPER_TEST_ACCURACY = {
    "AGNews": [.9468, .9475, .9479, .9457, .9482, .9471, .9491, .9471, .9507, .9487],
    "CIFAR-10": [.9170, .9229, .9043, .9188, .9149, .9221, .9279, .9267, .9032, .9054],
    "MNIST": [.9747, .9767, .9685, .9779, .9761, .9799, .9724, .9692, .9767, .9770],
    "N24News": [.8643, .8610, .8605, .8581, .8589, .8528, .8600, .8674, .8545, .8612],
    "nuImages": [.9494, .9488, .9533, .9528, .9527, .9445, .9536, .9504, .9508, .9512],
}
ACC_IN_NAME = re.compile(r"_acc\.?(\d+(?:\.\d+)?)_")


def accuracy_from_name(path):
    """Test accuracy in percent from the file name (None if absent)."""
    match = ACC_IN_NAME.search(Path(path).name)
    if match is None:
        return None
    acc = float(match.group(1))
    return acc * 100 if acc <= 1 else acc       # nuImages files store a fraction


def mean_std(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return "-"
    std = values.std(ddof=1) if values.size > 1 else 0.0
    return f"{values.mean():.2f}% ({std:.2f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-root", default="data/predictions")
    parser.add_argument("--paper", action="store_true", help="print Table S4 from the recorded accuracies")
    args = parser.parse_args()

    if args.paper:
        print("| Dataset | " + " | ".join(PAPER_TEST_ACCURACY) + " |")
        print("|" + " --- |" * (len(PAPER_TEST_ACCURACY) + 1))
        print("| Accuracy mean (std) | " + " | ".join(mean_std(np.array(v) * 100)
                                                     for v in PAPER_TEST_ACCURACY.values()) + " |")
        return

    print("| Dataset | models | test accuracy (file names) | analysis-set accuracy |")
    print("| --- | --- | --- | --- |")
    for label, name in TABLE_DATASETS.items():
        files = prediction_files(args.pred_root, name)
        test_acc = [a for a in map(accuracy_from_name, files) if a is not None]
        analysis_acc = []
        for path in files:
            _, tab = load_predi_data(path)
            analysis_acc.append(100 * np.mean(tab[:, TRUE_IDX] == tab[:, PRED_IDX]))
        print(f"| {label} | {len(files)} | {mean_std(test_acc)} | {mean_std(analysis_acc)} |")


if __name__ == "__main__":
    main()
