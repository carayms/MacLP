"""Predict the AG News analysis set with every fine-tuned RoBERTa model.

Reads ``<model-dir>/*.pth`` and ``<model-dir>/analysis_idx.txt`` from
``agnews_train.py`` and writes one prediction file per model (layout:
``exputils/io.py``).

Example:
    python classifiers/agnews_predict.py --model-dir data/models/agnews --out-dir data/predictions/agnews
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from agnews_train import MAX_LENGTH, NET_NAME, NUM_CLASSES
from common import pick_device, write_prediction_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="data/models/agnews")
    parser.add_argument("--out-dir", default="data/predictions/agnews")
    parser.add_argument("--dataset", default="ag_news")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    model_dir = Path(args.model_dir)
    src_idx = np.loadtxt(model_dir / "analysis_idx.txt", dtype=int)
    checkpoints = sorted(model_dir.glob(f"{NET_NAME}_seed.*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No {NET_NAME}_seed.*.pth files in {model_dir}")

    raw = load_dataset(args.dataset)
    holdout = raw["train"].select(src_idx.tolist())
    labels = np.array(holdout["label"])

    tokenizer = None
    for path in checkpoints:
        ckpt = torch.load(path, map_location="cpu")
        model_name = ckpt.get("model_name", "roberta-base")
        if tokenizer is None:
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            encoded = holdout.map(lambda ex: tokenizer(ex["text"], padding="max_length", truncation=True,
                                                       max_length=MAX_LENGTH), batched=True)
            encoded.set_format("torch", columns=["input_ids", "attention_mask"])
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        probs = []
        with torch.no_grad():
            for i in range(0, len(encoded), args.batch_size):
                batch = {k: encoded[k][i:i + args.batch_size].to(device) for k in ["input_ids", "attention_mask"]}
                probs.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
        probs = np.vstack(probs)

        out = Path(args.out_dir) / f"{NET_NAME}_seed.{ckpt['seed']}_acc.{ckpt['accuracy']:.2f}_pred.holdout.tsv"
        write_prediction_file(out, src_idx, labels, probs)
        print(f"{path.name}: analysis accuracy {100 * (probs.argmax(1) == labels).mean():.2f}% -> {out.name}")


if __name__ == "__main__":
    main()
