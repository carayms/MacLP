"""Predict the MNIST analysis set with every trained classifier.

Reads ``<model-dir>/*_best.pth`` from ``mnist_train.py`` and writes one
prediction file per classifier (layout: ``exputils/io.py``) for the 30,000
analysis images.  The images are only normalized (no augmentation).

Example:
    python classifiers/mnist_predict.py --data-root data/torchvision \
        --model-dir data/models/mnist --out-dir data/predictions/mnist
"""

import argparse
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset

from common import pick_device, predict_probs, write_prediction_file
from mnist_train import TEST_TRANSFORM, analysis_indices
from models import MnistCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/torchvision")
    parser.add_argument("--model-dir", default="data/models/mnist")
    parser.add_argument("--out-dir", default="data/predictions/mnist")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    dataset = torchvision.datasets.MNIST(args.data_root, train=True, download=True, transform=TEST_TRANSFORM)
    src_idx = analysis_indices(len(dataset))
    loader = DataLoader(Subset(dataset, src_idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    checkpoints = sorted(Path(args.model_dir).glob("*_best.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No *_best.pth files in {args.model_dir}")
    for path in checkpoints:
        ckpt = torch.load(path, map_location="cpu")
        net = MnistCNN()
        net.load_state_dict(ckpt["net"])
        probs, labels = predict_probs(net.to(device), loader, device)
        out = Path(args.out_dir) / (f"{ckpt['arch']}_seed.{ckpt['seed']}_epoch.{ckpt['epoch']}"
                                    f"_acc.{ckpt['acc']:.2f}_pred.holdout.tsv")
        write_prediction_file(out, src_idx, labels, probs)
        print(f"{path.name}: analysis accuracy {100 * (probs.argmax(1) == labels).mean():.2f}% -> {out.name}")


if __name__ == "__main__":
    main()
