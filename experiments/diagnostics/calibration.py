"""Figure S3: reliability diagrams and expected calibration error (ECE) of the
classifiers.

For every prediction file, the confidence is the predicted probability of the
predicted label; confidences are grouped into 10 equal-width bins and
ECE = sum_b n_b |conf_b - acc_b| / n.  Each panel shows one dataset with one
marker color per classifier and the largest ECE among them.  The layout
follows the original plotting script; the only difference is that empty bins
are not drawn (the original plotted them at (0, 0)).

Expects ``<pred-root>/{agnews,cifar10,mnist,n24news,nuimages}``; nuImages is
split into the Singapore (source, SG) and Boston (target, BOS) files.

Example:
    python diagnostics/calibration.py --pred-root data/predictions
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import prediction_files
from exputils import FIGURES_ROOT
from exputils.io import PRED_IDX, SOFT_IDX, TRUE_IDX, load_predi_data
from exputils.plotting import set_paper_style

PANELS = ["AGNews", "N24News", "MNIST", "CIFAR-10", "nuImages.SG", "nuImages.BOS"]


def compute_ece(correct, confidence, n_bins=10):
    """ECE with equal-width bins; returns (ece, bin confidence, bin accuracy, bin counts)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, bins) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)
    conf = np.bincount(bin_idx, weights=confidence, minlength=n_bins) / np.maximum(counts, 1)
    acc = np.bincount(bin_idx, weights=correct, minlength=n_bins) / np.maximum(counts, 1)
    ece = np.sum(counts * np.abs(conf - acc)) / counts.sum()
    return ece, conf, acc, counts


def file_ece(path, n_bins):
    _, tab = load_predi_data(path)
    pred = tab[:, PRED_IDX].astype(int)
    correct = (tab[:, TRUE_IDX] == tab[:, PRED_IDX]).astype(float)
    confidence = tab[np.arange(len(tab)), SOFT_IDX + pred].astype(float)
    return compute_ece(correct, confidence, n_bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-root", default="data/predictions")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--fig-dir", default=str(FIGURES_ROOT))
    args = parser.parse_args()

    set_paper_style()
    plt.rcParams["font.size"] = 10
    ticks = np.linspace(0, 1, 6)
    fig = plt.figure(figsize=(5.6, 7.5))
    grid = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    handles = {}
    for idx, name in enumerate(PANELS):
        i, j = divmod(idx, 2)
        ax = fig.add_subplot(grid[i, j])
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        files = prediction_files(args.pred_root, name)
        eces = []
        for k, path in enumerate(files):
            ece, conf, acc, counts = file_ece(path, args.n_bins)
            eces.append(ece)
            keep = counts > 0
            (line,) = ax.plot(conf[keep], acc[keep], "s", color=f"C{k % 10}", mfc="none", mew=1.5, ms=5)
            handles.setdefault(f"trial {k + 1}", line)
        if i == 2:
            ax.set_xlabel("Confidence")
        if j == 0:
            ax.set_ylabel("Accuracy")
        ax.set_title(name)
        if eces:
            print(f"{name:<13} ECE per classifier: {np.round(eces, 3).tolist()}")
            ax.text(0, 0.87, f"max(ECE)={max(eces):.3f}", weight="bold", fontsize=12)
        else:
            ax.text(0.5, 0.5, "no prediction files", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(visible=True, axis="both", color="gray", linestyle="--", linewidth=0.5)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="center right", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, handletextpad=0.4, fontsize=12)
    fig.subplots_adjust(top=0.97, bottom=0.07, left=0.1, right=0.8)

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "ece6dat.pdf")
    print(f"Figure saved to {fig_dir / 'ece6dat.pdf'}")


if __name__ == "__main__":
    main()
