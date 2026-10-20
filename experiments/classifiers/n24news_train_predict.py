"""Train the 10 multimodal N24News classifiers and predict their analysis sets
(Appendix J.1).

* Data: ``<data-dir>/news/nytimes_dataset.json`` and ``<data-dir>/imgs/<image_id>.jpg``
  (N24News, https://github.com/billywzh717/N24News).  Text = headline +
  abstract + caption; label = section (24 classes).
* Model: late fusion of the 768-d BERT (bert-base-uncased) and ViT
  (google/vit-base-patch16-224) pooled outputs, dropout 0.1, linear layer.
* For each of 10 seeds (``default_rng(1651)``): a stratified 50/50 split into
  the analysis set and the training pool; the pool is split 80/20 into
  training and test (model selection) data.  3 epochs, AdamW (lr 3e-5,
  weight decay 0.01), 500 warm-up steps, batch size 16, fp16 on CUDA.
* Images are resized once to 224 x 224 into ``--resized-dir``.

Writes ``<pred-dir>/BERT_ViT_Fusion_rep<r>_seed.<seed>_acc.<acc>_pred.holdout.tsv``
and the analysis indices of each replication.

Example:
    python classifiers/n24news_train_predict.py --data-dir data/n24news \
        --out-dir data/models/n24news --pred-dir data/predictions/n24news
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from tqdm import tqdm
from transformers import (BertModel, BertTokenizerFast, Trainer, TrainingArguments, ViTImageProcessor, ViTModel,
                          set_seed)

from common import write_prediction_file

NET_NAME = "BERT_ViT_Fusion"
TEXT_MODEL = "bert-base-uncased"
IMAGE_MODEL = "google/vit-base-patch16-224"
MAX_LENGTH = 128


class MultimodalClassifier(nn.Module):
    def __init__(self, num_labels, text_model=TEXT_MODEL, image_model=IMAGE_MODEL):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(text_model)
        self.image_encoder = ViTModel.from_pretrained(image_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size,
                                    num_labels)

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_out = self.image_encoder(pixel_values=pixel_values).pooler_output
        logits = self.classifier(self.dropout(torch.cat((text_out, image_out), dim=1)))
        if labels is None:
            return {"logits": logits}
        return {"loss": nn.CrossEntropyLoss()(logits, labels), "logits": logits}


def resize_images(image_ids, image_dir, resized_dir, size=224):
    """Resize every image once (LANCZOS); missing or unreadable images are skipped."""
    resized_dir.mkdir(parents=True, exist_ok=True)
    for img_id in tqdm(image_ids, desc="Resizing images"):
        if not img_id:
            continue
        src, dst = image_dir / f"{img_id}.jpg", resized_dir / f"{img_id}.jpg"
        if dst.exists() or not src.exists():
            continue
        try:
            with Image.open(src) as img:
                img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(dst, "JPEG", quality=90)
        except OSError:
            continue


def build_records(data, resized_dir):
    """Texts, image paths (None if missing) and section labels."""
    texts, image_paths, sections = [], [], []
    for item in data:
        headline = item["headline"]["main"] if isinstance(item["headline"], dict) else item.get("headline", "")
        texts.append(f"{headline} {item.get('abstract', '')} {item.get('caption', '')}".strip())
        img_id = item.get("image_id")
        path = resized_dir / f"{img_id}.jpg" if img_id else None
        image_paths.append(str(path) if path is not None and path.exists() else None)
        sections.append(item.get("section", "Unknown"))
    return texts, image_paths, sections


def make_collate_fn(image_processor):
    def collate_fn(batch):
        images = []
        for item in batch:
            pixel_values = torch.zeros(3, 224, 224)       # missing image
            if item.get("image_path"):
                try:
                    with Image.open(item["image_path"]) as img:
                        pixel_values = image_processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
                except OSError:
                    pass
            images.append(pixel_values)
        return {
            "input_ids": torch.stack([torch.as_tensor(item["input_ids"]) for item in batch]),
            "attention_mask": torch.stack([torch.as_tensor(item["attention_mask"]) for item in batch]),
            "pixel_values": torch.stack(images),
            "labels": torch.tensor([item["labels"] for item in batch]),
        }
    return collate_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="folder with news/nytimes_dataset.json and imgs/")
    parser.add_argument("--resized-dir", default=None, help="default: <data-dir>/resized_224")
    parser.add_argument("--out-dir", default="data/models/n24news", help="models and analysis indices")
    parser.add_argument("--pred-dir", default="data/predictions/n24news")
    parser.add_argument("--n-models", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--text-model", default=TEXT_MODEL)
    parser.add_argument("--image-model", default=IMAGE_MODEL)
    parser.add_argument("--no-save-models", action="store_true", help="only write the predictions")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    resized_dir = Path(args.resized_dir) if args.resized_dir else data_dir / "resized_224"
    out_dir, pred_dir = Path(args.out_dir), Path(args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(data_dir / "news" / "nytimes_dataset.json", "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    resize_images([item.get("image_id") for item in data], data_dir / "imgs", resized_dir)
    texts, image_paths, sections = build_records(data, resized_dir)
    labels = LabelEncoder().fit_transform(sections)
    num_classes = int(labels.max()) + 1
    print(f"{len(texts)} articles, {num_classes} classes, {sum(p is not None for p in image_paths)} images")

    tokenizer = BertTokenizerFast.from_pretrained(args.text_model)
    image_processor = ViTImageProcessor.from_pretrained(args.image_model)
    dataset = Dataset.from_dict({"text": texts, "image_path": image_paths, "labels": labels})
    dataset = dataset.map(lambda ex: tokenizer(ex["text"], padding="max_length", truncation=True,
                                               max_length=MAX_LENGTH), batched=True, remove_columns=["text"])
    collate_fn = make_collate_fn(image_processor)

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        return {"accuracy": accuracy_score(y, np.argmax(logits, axis=-1))}

    seeds = np.random.default_rng(1651).choice(999999, size=args.n_models, replace=False)
    for rep, seed in enumerate(seeds):
        seed = int(seed)
        print(f"\n>>> replication {rep} (seed {seed})")
        set_seed(seed)

        indices = np.arange(len(dataset))
        pool_idx, analysis_idx = train_test_split(indices, test_size=0.5, random_state=seed, stratify=labels)
        np.savetxt(out_dir / f"N24_Analysis_Idx_Rep{rep}_Seed{seed}.txt", analysis_idx, fmt="%d")
        train_idx, test_idx = train_test_split(pool_idx, test_size=0.2, random_state=seed,
                                               stratify=labels[pool_idx])

        model = MultimodalClassifier(num_classes, args.text_model, args.image_model)
        work_dir = out_dir / f"trainer_rep{rep}_seed{seed}"
        training_args = TrainingArguments(
            output_dir=str(work_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=[],
            seed=seed,
            fp16=torch.cuda.is_available(),
            # keep "image_path": it is not an argument of forward(), and the
            # Trainer would otherwise drop it before the collator sees it
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset.select(train_idx),
                          eval_dataset=dataset.select(test_idx), data_collator=collate_fn,
                          compute_metrics=compute_metrics)
        trainer.train()
        acc = trainer.evaluate()["eval_accuracy"] * 100
        print(f"test accuracy {acc:.2f}%")

        # predictions on the analysis set
        model.eval()
        analysis_ds = dataset.select(analysis_idx)
        probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(analysis_ds), args.eval_batch_size), desc="Predicting"):
                batch = collate_fn([analysis_ds[j] for j in range(i, min(i + args.eval_batch_size, len(analysis_ds)))])
                out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                            pixel_values=batch["pixel_values"].to(device))
                probs.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())
        probs = np.vstack(probs)
        write_prediction_file(pred_dir / f"{NET_NAME}_rep{rep}_seed.{seed}_acc.{acc:.2f}_pred.holdout.tsv",
                              analysis_idx, labels[analysis_idx], probs)
        print(f"analysis accuracy {100 * (probs.argmax(1) == labels[analysis_idx]).mean():.2f}%")

        if not args.no_save_models:
            torch.save({"model_state_dict": model.state_dict(), "accuracy": acc, "seed": seed},
                       out_dir / f"{NET_NAME}_rep{rep}_seed.{seed}_acc.{acc:.2f}.pth")
        shutil.rmtree(work_dir, ignore_errors=True)
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
