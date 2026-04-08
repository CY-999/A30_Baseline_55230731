import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import (
    compute_metrics_from_predictions,
    load_records,
    pretty_print_metrics,
    save_json,
    set_seed,
)


def parse_args():
    default_model = "hfl/chinese-roberta-wwm-ext"
    local_model_dir = os.path.join("models", "chinese-roberta-wwm-ext")
    if os.path.isdir(local_model_dir):
        default_model = local_model_dir

    parser = argparse.ArgumentParser(
        description="Train Chinese RoBERTa baseline for A30 text scam detection."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/demo",
        help="Directory containing train/val/test files. Defaults to demo data.",
    )
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default=default_model,
    )
    parser.add_argument("-o", "--output_dir", type=str, default="outputs/baseline")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("-e", "--epochs", type=int, default=3)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument(
        "--lr",
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=2e-5,
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=55230731)
    return parser.parse_args()


def resolve_data_files(args) -> Tuple[str, str, str]:
    train_file = args.train_file or os.path.join(args.data_dir, "train.jsonl")
    val_file = args.val_file or os.path.join(args.data_dir, "val.jsonl")
    test_file = args.test_file or os.path.join(args.data_dir, "test.jsonl")
    return train_file, val_file, test_file


def build_dataset(file_path: str) -> Dataset:
    records = load_records(file_path)
    texts = [r["text"] for r in records]
    labels = [int(r["label"]) for r in records]
    return Dataset.from_dict({"text": texts, "label": labels})


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    train_file, val_file, test_file = resolve_data_files(args)

    print("========== A30 Text Scam Baseline: Chinese RoBERTa ==========")
    print(f"Using model: {args.model_name_or_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Train file: {train_file}")
    print(f"Validation file: {val_file}")
    print(f"Test file: {test_file}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    train_ds = build_dataset(train_file)
    val_ds = build_dataset(val_file)
    test_ds = build_dataset(test_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Tokenize raw text so the pretrained encoder can consume it.
    def preprocess_function(examples: Dict):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = train_ds.map(preprocess_function, batched=True)
    val_ds = val_ds.map(preprocess_function, batched=True)
    test_ds = test_ds.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics_from_predictions(labels, preds)

    best_model_dir = os.path.join(args.output_dir, "best_model")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # Trainer handles the train, eval, and checkpoint loop for this baseline.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nEvaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    pretty_print_metrics("Validation", {k: v for k, v in val_metrics.items() if isinstance(v, (float, int))})

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    filtered_test_metrics = {k: v for k, v in test_metrics.items() if isinstance(v, (float, int))}
    pretty_print_metrics("Test", filtered_test_metrics)

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    save_json(
        {
            "0": "normal",
            "1": "scam",
        },
        os.path.join(args.output_dir, "label_mapping.json"),
    )

    save_json(filtered_test_metrics, os.path.join(args.output_dir, "test_metrics.json"))
    print(f"\nBest model saved to: {best_model_dir}")
    print(f"Test metrics saved to: {os.path.join(args.output_dir, 'test_metrics.json')}")


if __name__ == "__main__":
    main()
