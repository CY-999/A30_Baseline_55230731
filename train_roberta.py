import argparse
import inspect
import os
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import Dataset, disable_progress_bar
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.utils import logging as hf_logging

from utils import (
    compute_metrics_from_predictions,
    load_records,
    pretty_print_metrics,
    save_json,
    set_seed,
)


class EpochMetricsCallback(TrainerCallback):
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.enabled = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.enabled or not metrics:
            return

        epoch = int(round(state.epoch or 0))
        summary = (
            f"Epoch {epoch}/{self.total_epochs} | "
            f"val_f1={metrics.get('eval_f1', 0.0):.4f} | "
            f"val_precision={metrics.get('eval_precision', 0.0):.4f} | "
            f"val_recall={metrics.get('eval_recall', 0.0):.4f} | "
            f"val_acc={metrics.get('eval_accuracy', 0.0):.4f}"
        )
        print(summary)


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


def configure_console_output() -> None:
    disable_progress_bar()
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    warnings.filterwarnings(
        "ignore",
        message=r"`tokenizer` is deprecated and will be removed in version 5\.0\.0 for `Trainer\.__init__`.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Was asked to gather along dimension 0, but all input tensors were scalars.*",
        category=UserWarning,
    )


def select_core_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    keys = ("eval_accuracy", "eval_precision", "eval_recall", "eval_f1")
    return {k: float(metrics[k]) for k in keys if k in metrics}


def build_training_arguments(args) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "no",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": args.seed,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "disable_tqdm": True,
        "dataloader_pin_memory": torch.cuda.is_available(),
    }

    if "overwrite_output_dir" in signature.parameters:
        kwargs["overwrite_output_dir"] = True

    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def main():
    args = parse_args()
    configure_console_output()
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

    training_args = build_training_arguments(args)

    # Trainer handles the train, eval, and checkpoint loop for this baseline.
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EpochMetricsCallback(total_epochs=args.epochs)],
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)

    epoch_metrics_callback = next(
        callback
        for callback in trainer.callback_handler.callbacks
        if isinstance(callback, EpochMetricsCallback)
    )

    trainer.train()
    epoch_metrics_callback.enabled = False

    print("\nEvaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    pretty_print_metrics("Validation", select_core_metrics(val_metrics))

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    filtered_test_metrics = select_core_metrics(test_metrics)
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
