"""Fine-tune the 10 RoBERTa classifiers for AG News (Appendix J.1).

* Analysis set: 60,000 of the 120,000 training texts (15,000 per class),
  drawn once with ``default_rng(1651)`` and saved to
  ``<out-dir>/analysis_idx.txt``; never used for training.
* Training: the other 60,000 texts; the 7,600 test texts select the best epoch.
* roberta-base, max length 128, AdamW (lr 3e-5, weight decay 0.01), 500
  warm-up steps, batch sizes 32 / 64, fp16 on CUDA.
* 10 seeds drawn with ``default_rng(1651)``.

Writes ``<out-dir>/RoBERTa_seed.<seed>_acc.<acc>.pth``.  Then run
``agnews_predict.py``.

Example:
    python classifiers/agnews_train.py --out-dir data/models/agnews
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (RobertaForSequenceClassification, RobertaTokenizerFast, Trainer, TrainingArguments,
                          set_seed)

ANALYSIS_SEED = 1651
ANALYSIS_SIZE = 60000
NUM_CLASSES = 4
MAX_LENGTH = 128
NET_NAME = "RoBERTa"


def analysis_indices(train_labels, out_dir, size=ANALYSIS_SIZE):
    """Stratified analysis indices (saved on first use, then reloaded)."""
    path = Path(out_dir) / "analysis_idx.txt"
    if path.exists():
        return np.loadtxt(path, dtype=int)
    rng = np.random.default_rng(ANALYSIS_SEED)
    idx = np.concatenate([rng.choice(np.where(train_labels == k)[0], size=size // NUM_CLASSES,
                                     replace=False) for k in range(NUM_CLASSES)])
    rng.shuffle(idx)
    np.savetxt(path, idx, fmt="%d")
    return idx


def tokenize(raw_datasets, tokenizer):
    def fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    return raw_datasets.map(fn, batched=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data/models/agnews")
    parser.add_argument("--dataset", default="ag_news", help="Hugging Face dataset name")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--n-models", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10, help="training epochs (10 in the original script)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--analysis-size", type=int, default=ANALYSIS_SIZE)
    parser.add_argument("--max-train", type=int, default=None, help="use only this many texts (testing only)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("GPU available:", torch.cuda.is_available())

    raw = load_dataset(args.dataset)
    train_labels = np.array(raw["train"]["label"])
    src_idx = analysis_indices(train_labels, out_dir, args.analysis_size)
    train_idx = np.setdiff1d(np.arange(len(train_labels)), src_idx)

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
    tokenized = tokenize(raw, tokenizer).rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_set = tokenized["train"].select(train_idx.tolist())
    test_set = tokenized["test"]
    if args.max_train is not None:
        train_set = train_set.select(range(args.max_train))
        test_set = test_set.select(range(min(args.max_train, len(test_set))))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": accuracy_score(labels, np.argmax(logits, axis=-1))}

    seeds = np.random.default_rng(ANALYSIS_SEED).choice(9999999, size=50, replace=False)[:args.n_models]
    for rep, seed in enumerate(seeds):
        seed = int(seed)
        print(f"\n=== model {rep + 1}/{args.n_models}, seed {seed} ===")
        set_seed(seed)
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=NUM_CLASSES)
        work_dir = out_dir / f"trainer_rep{rep}_seed{seed}"
        training_args = TrainingArguments(
            output_dir=str(work_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=[],
            seed=seed,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=test_set,
                          compute_metrics=compute_metrics)
        trainer.train()
        acc = trainer.evaluate()["eval_accuracy"] * 100
        print(f"test accuracy {acc:.2f}%")

        path = out_dir / f"{NET_NAME}_seed.{seed}_acc.{acc:.2f}.pth"
        torch.save({"model_state_dict": trainer.model.state_dict(), "accuracy": acc, "seed": seed,
                    "model_name": args.model_name}, path)
        shutil.rmtree(work_dir, ignore_errors=True)    # intermediate Trainer checkpoints
        print(f"saved {path}")


if __name__ == "__main__":
    main()
