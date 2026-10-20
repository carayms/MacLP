"""Train the CIFAR-10 classifiers (Appendix J.1).

* Architectures: VGG16, ResNet18, PreActResNet18, MobileNetV2, RegNetX_200MF
  (``models/``), two seeds each in the paper.
* Seed of a classifier: ``default_rng(1651).choice(9999999, 200)[i]`` with
  i in (1, 37, 93, 156), selected by ``--seed-indices`` (0..3).
* Analysis set: 25,000 of the 50,000 training images, drawn with
  ``default_rng(seed)``, so it differs between seeds; the other 25,000 are
  used for training and the 10,000 test images for model selection.
* SGD (lr 0.1, momentum 0.9, weight decay 5e-4), cosine schedule, batch size
  256, 200 epochs, random-crop and horizontal-flip augmentation.

Writes ``<out-dir>/<arch>_seed.<seed>_best.pth`` (best test accuracy) and
``<arch>_seed.<seed>_last.pth`` (to resume with ``--resume``).

Example (all 10 classifiers; one at a time is also possible):
    python classifiers/cifar10_train.py --data-root data/torchvision --out-dir data/models/cifar10
    python classifiers/cifar10_train.py --archs ResNet --seed-indices 0
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
from tqdm import tqdm

from common import accuracy, pick_device
from models import CIFAR10_ARCHITECTURES, build_cifar10_model

SEED_POSITIONS = [1, 37, 93, 156]
ANALYSIS_SIZE = 25000
NORMALIZE = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))
TRAIN_TRANSFORM = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), NORMALIZE])
TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), NORMALIZE])


def classifier_seed(seed_index):
    seeds = np.random.default_rng(1651).choice(9999999, size=200, replace=False)
    return int(seeds[SEED_POSITIONS[seed_index]])


def analysis_indices(seed, total_size=50000):
    return np.random.default_rng(seed).choice(total_size, size=ANALYSIS_SIZE, replace=False)


def load_cifar10(data_root, train, transform, fake=False):
    if fake:   # tiny random images, only for testing the scripts
        n = 50000 if train else 10000
        return torchvision.datasets.FakeData(size=n, image_size=(3, 32, 32), num_classes=10,
                                             transform=transform, random_offset=0 if train else n)
    return torchvision.datasets.CIFAR10(data_root, train=train, download=True, transform=transform)


def train_one(arch, seed, args, device):
    out_dir = Path(args.out_dir)
    best_path = out_dir / f"{arch}_seed.{seed}_best.pth"
    last_path = out_dir / f"{arch}_seed.{seed}_last.pth"

    train_full = load_cifar10(args.data_root, True, TRAIN_TRANSFORM, args.fake_data)
    test_data = load_cifar10(args.data_root, False, TEST_TRANSFORM, args.fake_data)
    train_idx = np.setdiff1d(np.arange(len(train_full)), analysis_indices(seed, len(train_full)))
    if args.max_train is not None:
        train_idx, test_data = train_idx[:args.max_train], Subset(test_data, np.arange(args.max_train))

    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=g)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=args.num_workers)

    net = build_cifar10_model(arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, best_acc = 0, -1.0
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        net.load_state_dict(state["net"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_acc = state["epoch"] + 1, state["best_acc"]
        print(f"resuming {arch} seed {seed} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        net.train()
        for inputs, targets in tqdm(train_loader, desc=f"{arch} seed {seed} epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = criterion(net(inputs.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = accuracy(net, test_loader, device)
        print(f"{arch} seed {seed} epoch {epoch}: test accuracy {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save({"net": net.state_dict(), "acc": acc, "epoch": epoch, "seed": seed, "arch": arch}, best_path)
        torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_acc": best_acc}, last_path)
    print(f"{arch} seed {seed}: best test accuracy {best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/torchvision")
    parser.add_argument("--out-dir", default="data/models/cifar10")
    parser.add_argument("--archs", nargs="+", default=list(CIFAR10_ARCHITECTURES), choices=list(CIFAR10_ARCHITECTURES))
    parser.add_argument("--seed-indices", type=int, nargs="+", default=[0, 1], choices=range(len(SEED_POSITIONS)))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="continue from the *_last.pth checkpoints")
    parser.add_argument("--fake-data", action="store_true", help="random images instead of CIFAR-10 (testing only)")
    parser.add_argument("--max-train", type=int, default=None, help="use only this many images (testing only)")
    args = parser.parse_args()

    device = pick_device(args.device)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    for arch in args.archs:
        for i in args.seed_indices:
            train_one(arch, classifier_seed(i), args, device)


if __name__ == "__main__":
    main()
