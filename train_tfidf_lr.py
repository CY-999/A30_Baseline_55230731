import argparse
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import (
    compute_metrics_from_predictions,
    load_records,
    pretty_print_metrics,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline for A30 text scam detection.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/tfidf_lr_demo")
    parser.add_argument("--seed", type=int, default=55230731)
    return parser.parse_args()


def unpack(records):
    texts = [r["text"] for r in records]
    labels = [int(r["label"]) for r in records]
    return texts, labels


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_records = load_records(args.train_file)
    val_records = load_records(args.val_file)
    test_records = load_records(args.test_file)

    X_train, y_train = unpack(train_records)
    X_val, y_val = unpack(val_records)
    X_test, y_test = unpack(test_records)

    # 学号 55230731：经典 baseline，便于在无法下载预训练模型时快速兜底。
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, random_state=args.seed)),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)

    val_metrics = compute_metrics_from_predictions(y_val, val_pred)
    test_metrics = compute_metrics_from_predictions(y_test, test_pred)

    pretty_print_metrics("Validation", val_metrics)
    pretty_print_metrics("Test", test_metrics)

    joblib.dump(pipeline, os.path.join(args.output_dir, "tfidf_lr_pipeline.joblib"))
    save_json(val_metrics, os.path.join(args.output_dir, "val_metrics.json"))
    save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))

    print(f"\nModel saved to: {os.path.join(args.output_dir, 'tfidf_lr_pipeline.joblib')}")


if __name__ == "__main__":
    main()
