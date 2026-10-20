"""Train the 10 MNIST classifiers (Appendix J.1).

* Analysis set: 30,000 of the 60,000 training images, drawn once with
  ``default_rng(1652)``; they are never used for training.
* Training: the other 30,000 images; model selection on the 10,000 test images.
* Model: four-layer CNN (``models/mnist_cnn.py``), SGD (lr 0.1, momentum 0.5),
  batch size 256, 10 epochs; the epoch with the best test accuracy is kept.
* 10 seeds drawn with ``default_rng(1652)``.

Writes ``<out-dir>/CNN_seed.<seed>_best.pth`` per seed and the analysis
indices to ``<out-dir>/analysis_idx.txt``.  Then run ``mnist_predict.py``.

Example:
    python classifiers/mnist_train.py --data-root data/torchvision --out-dir data/models/mnist
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from common import accuracy, pick_device
from models import MnistCNN

ANALYSIS_SEED = 1652
ANALYSIS_SIZE = 30000
NORMALIZE = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
# Training augmentation used for the paper's classifiers.  Predictions use
# TEST_TRANSFORM (see mnist_predict.py).
TRAIN_TRANSFORM = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORMALIZE])
TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), NORMALIZE])


def analysis_indices(total_size=60000):
    return np.random.default_rng(ANALYSIS_SEED).choice(total_size, size=ANALYSIS_SIZE, replace=False)


def classifier_seeds(n):
    return np.random.default_rng(ANALYSIS_SEED).choice(9999999, size=500, replace=False)[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/torchvision")
    parser.add_argument("--out-dir", default="data/models/mnist")
    parser.add_argument("--n-models", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.enabled = False
    print(f"Using device: {device}")

    train_full = torchvision.datasets.MNIST(args.data_root, train=True, download=True, transform=TRAIN_TRANSFORM)
    test_data = torchvision.datasets.MNIST(args.data_root, train=False, download=True, transform=TEST_TRANSFORM)

    src_idx = analysis_indices(len(train_full))
    np.savetxt(out_dir / "analysis_idx.txt", src_idx, fmt="%d")
    train_idx = np.setdiff1d(np.arange(len(train_full)), src_idx)
    train_data = Subset(train_full, train_idx)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=args.num_workers)
    print(f"training images: {len(train_data)}, analysis images: {len(src_idx)}, test images: {len(test_data)}")

    for seed in classifier_seeds(args.n_models):
        seed = int(seed)
        print(f"\nseed {seed}")
        torch.manual_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, generator=g)

        net = MnistCNN().to(device)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
        criterion = nn.CrossEntropyLoss()
        best_acc = -1.0
        for epoch in range(1, args.epochs + 1):
            net.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                loss = criterion(net(data.to(device)), target.to(device))
                loss.backward()
                optimizer.step()
            acc = accuracy(net, test_loader, device)
            print(f"epoch {epoch}: loss {loss.item():.4f}, test accuracy {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save({"net": net.state_dict(), "acc": acc, "epoch": epoch, "seed": seed, "arch": "CNN"},
                           out_dir / f"CNN_seed.{seed}_best.pth")
        print(f"best test accuracy {best_acc:.2f}%")


if __name__ == "__main__":
    main()
