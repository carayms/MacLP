"""Write synthetic prediction files for testing the benchmark / nuImages pipelines.

The real prediction files come from the trained classifiers in
``classifiers/`` and ``nuimages/``.  This script produces files with the
same layout from a Gaussian mixture and a Gaussian naive Bayes classifier,
so that the downstream experiments can be run end to end without the
datasets or a GPU.

Examples
--------
Ten files of 30,000 rows with 10 classes (benchmark layout):

    python tools/make_synthetic_predictions.py --out-dir data/predictions/synthetic \
        --num-classes 10 --n-rows 30000 --n-files 10

Source/target pairs in the nuImages layout (Singapore_* and Boston_* files):

    python tools/make_synthetic_predictions.py --out-dir data/predictions/synthetic_nuimages \
        --num-classes 15 --n-rows 20000 --n-files 10 --nuimages-layout
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils.io import write_prediction_file
from exputils.sampling import toy_data_generating


def class_means(num_classes, radius=3.0):
    """Class means on a circle (2-d features)."""
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def predictions_for(clf, XY, start_idx=0):
    probs = clf.predict_proba(XY[:, :-1])
    orig_idx = np.arange(start_idx, start_idx + XY.shape[0])
    return orig_idx, XY[:, -1].astype(int), probs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--n-rows", type=int, default=30000, help="rows per prediction file")
    parser.add_argument("--n-files", type=int, default=10, help="number of 'classifiers'")
    parser.add_argument("--noise", type=float, default=1.0, help="std of the class-conditional Gaussians")
    parser.add_argument("--nuimages-layout", action="store_true",
                        help="write Singapore_*/Boston_* pairs with different label proportions")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    K = args.num_classes
    mus = class_means(K)
    Sigma = np.eye(2) * args.noise**2
    uniform = np.ones(K) / K
    out_dir = Path(args.out_dir)

    for rep in range(args.n_files):
        XY_train = toy_data_generating(mus, Sigma, 2000 * K, uniform, rng)
        clf = GaussianNB().fit(XY_train[:, :-1], XY_train[:, -1].astype(int))

        if not args.nuimages_layout:
            XY = toy_data_generating(mus, Sigma, args.n_rows, uniform, rng)
            write_prediction_file(out_dir / f"Synthetic_rep{rep}_pred.holdout.tsv", *predictions_for(clf, XY))
        else:
            # source: roughly uniform labels; target: skewed labels.  The
            # uniform component keeps every class present in both files.
            p_src = 0.8 * rng.dirichlet(np.ones(K) * 10) + 0.2 * uniform
            p_tgt = 0.8 * rng.dirichlet(np.ones(K)) + 0.2 * uniform
            XY_src = toy_data_generating(mus, Sigma, args.n_rows, p_src, rng)
            XY_tgt = toy_data_generating(mus, Sigma, args.n_rows, p_tgt, rng)
            write_prediction_file(out_dir / f"Singapore_rep{rep}_pred.holdout.tsv", *predictions_for(clf, XY_src))
            write_prediction_file(out_dir / f"Boston_rep{rep}_pred.target.tsv", *predictions_for(clf, XY_tgt))

    print(f"Wrote {args.n_files} synthetic prediction file(s) to {out_dir}")


if __name__ == "__main__":
    main()
