"""Predict the CIFAR-10 analysis sets with every trained classifier.

Reads ``<model-dir>/*_best.pth`` from ``cifar10_train.py``; for each
classifier, the analysis set is the 25,000 training images drawn with its
own seed.  The images are only normalized (no augmentation).

Example:
    python classifiers/cifar10_predict.py --data-root data/torchvision \
        --model-dir data/models/cifar10 --out-dir data/predictions/cifar10
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from cifar10_train import TEST_TRANSFORM, analysis_indices, load_cifar10
from common import pick_device, predict_probs, write_prediction_file
from models import build_cifar10_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/torchvision")
    parser.add_argument("--model-dir", default="data/models/cifar10")
    parser.add_argument("--out-dir", default="data/predictions/cifar10")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fake-data", action="store_true", help="random images instead of CIFAR-10 (testing only)")
    args = parser.parse_args()

    device = pick_device(args.device)
    dataset = load_cifar10(args.data_root, True, TEST_TRANSFORM, args.fake_data)
    checkpoints = sorted(Path(args.model_dir).glob("*_best.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No *_best.pth files in {args.model_dir}")

    for path in checkpoints:
        ckpt = torch.load(path, map_location="cpu")
        net = build_cifar10_model(ckpt["arch"])
        net.load_state_dict(ckpt["net"])
        src_idx = analysis_indices(ckpt["seed"], len(dataset))
        loader = DataLoader(Subset(dataset, src_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        probs, labels = predict_probs(net.to(device), loader, device)
        out = Path(args.out_dir) / (f"{ckpt['arch']}_seed.{ckpt['seed']}_epoch.{ckpt['epoch']}"
                                    f"_acc.{ckpt['acc']:.2f}_pred.holdout.tsv")
        write_prediction_file(out, src_idx, labels, probs)
        print(f"{path.name}: analysis accuracy {100 * (probs.argmax(1) == labels).mean():.2f}% -> {out.name}")


if __name__ == "__main__":
    main()
