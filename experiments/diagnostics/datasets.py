"""Where the diagnostics scripts look for the prediction files of each dataset."""

from pathlib import Path

from exputils.io import list_prediction_files

# name in the paper: (sub-folder of --pred-root, file-name prefix or None)
PREDICTION_SETS = {
    "AGNews": ("agnews", None),
    "CIFAR-10": ("cifar10", None),
    "MNIST": ("mnist", None),
    "N24News": ("n24news", None),
    "nuImages.SG": ("nuimages", "Singapore_"),
    "nuImages.BOS": ("nuimages", "Boston_"),
}


def prediction_files(pred_root, name):
    """Prediction files of one dataset (empty list if the folder is missing)."""
    folder, prefix = PREDICTION_SETS[name]
    folder = Path(pred_root) / folder
    if not folder.is_dir():
        return []
    try:
        names = list_prediction_files(folder)
    except FileNotFoundError:
        return []
    return [folder / f for f in names if prefix is None or f.startswith(prefix)]
