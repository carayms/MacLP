"""Train the nuImages classifiers and write their predictions (Appendix J.2.2).

For each of 10 replications (seeds drawn from ``default_rng(2024)``):

1. split the Singapore training crops 50/50, stratified by class;
2. fine-tune an ImageNet-pretrained ResNet-50 (IMAGENET1K_V1) on the first
   half for 5 epochs (Adam, lr 1e-4, batch size 128), keeping the epoch with
   the best accuracy on the Singapore validation crops;
3. predict the second half of Singapore (analysis set, used for C) and the
   Boston training crops (target, used for q).

Inputs are the resized crops of ``resize_crops.py``.  Outputs (one pair per
replication, read by ``real_world_shift_experiment.py``):

    Singapore_rep<r>_seed<s>_acc<a>_pred.holdout.tsv
    Boston_rep<r>_seed<s>_pred.target.tsv

Example:
    python nuimages/train_predict.py --train-root data/nuimages/crops_train_224 \
        --val-root data/nuimages/crops_val_224 --out-dir data/predictions/nuimages
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils.io import write_prediction_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crop_objects import SELECTED_CLASSES

CLASS_TO_IDX = {name: i for i, name in enumerate(SELECTED_CLASSES)}

# Crops are already 224 x 224; only ImageNet normalization is applied.
BASE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FolderToClassIndex:
    """Maps an ImageFolder label to the 0..14 coding of SELECTED_CLASSES (picklable for workers)."""

    def __init__(self, folder_classes):
        self.folder_classes = folder_classes

    def __call__(self, folder_idx):
        return CLASS_TO_IDX[self.folder_classes[folder_idx]]


def filtered_dataset(root_dir, city):
    """ImageFolder of one city restricted to the selected classes, labels 0..14.

    Returns the subset, its labels and the ImageFolder indices of its images.
    """
    full_ds = datasets.ImageFolder(str(Path(root_dir) / city), transform=BASE_TRANSFORM)
    full_ds.target_transform = FolderToClassIndex(full_ds.classes)
    indices = [i for i, (_, folder_idx) in enumerate(full_ds.samples) if full_ds.classes[folder_idx] in CLASS_TO_IDX]
    labels = [CLASS_TO_IDX[full_ds.classes[full_ds.samples[i][1]]] for i in indices]
    return Subset(full_ds, indices), np.array(labels), np.array(indices)


def build_model(pretrained):
    model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(SELECTED_CLASSES))
    return model


@torch.no_grad()
def predict(model, dataset, device, batch_size, num_workers):
    """Softmax probabilities and true labels of ``dataset`` (in order)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, labels = [], []
    for imgs, y in loader:
        probs.append(torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-root", required=True, help="resized crops of v1.0-train (both cities)")
    parser.add_argument("--val-root", required=True, help="resized crops of v1.0-val (Singapore is used)")
    parser.add_argument("--out-dir", required=True, help="folder for the prediction files")
    parser.add_argument("--model-dir", default=None, help="folder for the checkpoints (default: <out-dir>/models)")
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random initialization (only for testing without the ImageNet weights)")
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir) if args.model_dir else out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    src_train, src_labels, src_orig_idx = filtered_dataset(args.train_root, "singapore")
    src_val, _, _ = filtered_dataset(args.val_root, "singapore")
    tgt_all, _, tgt_orig_idx = filtered_dataset(args.train_root, "boston")
    print(f"Singapore train: {len(src_train)}, Singapore val: {len(src_val)}, Boston: {len(tgt_all)}")

    seeds = np.random.default_rng(2024).choice(100000, size=args.n_reps, replace=False)
    pin = device.type == "cuda"

    for rep, seed in enumerate(seeds):
        print(f"\n>>> replication {rep} (seed {seed})")
        torch.manual_seed(int(seed))

        # 1. 50/50 stratified split of the Singapore training crops
        train_idx, analysis_idx = train_test_split(np.arange(len(src_train)), test_size=0.5,
                                                   random_state=int(seed), stratify=src_labels)
        train_loader = DataLoader(Subset(src_train, train_idx), batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin)

        # 2. fine-tune, keep the best epoch on the Singapore validation crops
        model = build_model(not args.no_pretrained).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        ckpt = model_dir / f"rep_{rep}_best.pth"
        best_acc = -1.0
        for epoch in range(args.epochs):
            model.train()
            for imgs, labels in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                optimizer.zero_grad()
                loss = criterion(model(imgs.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
            val_probs, val_labels = predict(model, src_val, device, args.batch_size, args.num_workers)
            acc = accuracy_score(val_labels, val_probs.argmax(axis=1))
            print(f"epoch {epoch}: validation accuracy {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), ckpt)

        # 3. predictions of the best model
        model.load_state_dict(torch.load(ckpt, map_location=device))
        probs, labels = predict(model, Subset(src_train, analysis_idx), device, args.batch_size, args.num_workers)
        write_prediction_file(out_dir / f"Singapore_rep{rep}_seed{seed}_acc{best_acc:.2f}_pred.holdout.tsv",
                              src_orig_idx[analysis_idx], labels, probs)
        probs, labels = predict(model, tgt_all, device, args.batch_size, args.num_workers)
        write_prediction_file(out_dir / f"Boston_rep{rep}_seed{seed}_pred.target.tsv", tgt_orig_idx, labels, probs)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nPredictions of {args.n_reps} replications written to {out_dir}")


if __name__ == "__main__":
    main()
