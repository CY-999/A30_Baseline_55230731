import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    # 学号 55230731：固定随机种子，保证个人 baseline 复现实验结果更加稳定。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(file_path: str) -> List[Dict]:
    """
    Load records from .jsonl or .csv.
    Required columns/keys: text, label (for training/eval)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".jsonl"):
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")

    raise ValueError("Only .jsonl and .csv files are supported.")


def save_json(data: Dict, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_metrics_from_predictions(y_true, y_pred) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)

    # 自动兼容二分类/多分类
    unique_labels = sorted(set(list(y_true) + list(y_pred)))
    average = "binary" if len(unique_labels) == 2 else "macro"

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def pretty_print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    print(f"\n[{prefix}]")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
