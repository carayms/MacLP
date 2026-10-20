"""Helpers shared by the classifier scripts."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils.io import write_prediction_file  # re-exported for the classifier scripts


def pick_device(name="auto"):
    """'auto' picks CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_probs(model, loader, device):
    """Softmax probabilities and labels for all batches of ``loader`` (in order)."""
    model.eval()
    probs, labels = [], []
    for inputs, targets in loader:
        probs.append(F.softmax(model(inputs.to(device)), dim=1).cpu().numpy())
        labels.append(np.asarray(targets))
    return np.concatenate(probs), np.concatenate(labels)


def accuracy(model, loader, device):
    probs, labels = predict_probs(model, loader, device)
    return float(100.0 * np.mean(probs.argmax(axis=1) == labels))
