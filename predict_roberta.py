import argparse
import json
import os
import warnings
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from utils import load_records


LABEL_MAPPING = {
    0: "normal",
    1: "scam",
}


def configure_console_output() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with trained RoBERTa baseline.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--text", type=str, default=None, help="Single text to predict.")
    parser.add_argument("--input_file", type=str, default=None, help="Batch prediction file (.jsonl or .csv).")
    parser.add_argument("--output_file", type=str, default=None, help="Prediction output file for batch mode.")
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def predict_texts(model, tokenizer, texts: List[str], max_length: int):
    # 学号 55230731：推理阶段把输入文本编码后送入模型，并输出概率与最终标签。
    inputs = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    results = []
    for text, pred, prob in zip(texts, preds.cpu().tolist(), probs.cpu().tolist()):
        results.append(
            {
                "text": text,
                "pred_label": pred,
                "pred_name": LABEL_MAPPING.get(pred, str(pred)),
                "prob_normal": round(float(prob[0]), 6),
                "prob_scam": round(float(prob[1]), 6),
            }
        )
    return results


def main():
    configure_console_output()
    args = parse_args()

    if args.text is None and args.input_file is None:
        raise ValueError("You must provide either --text or --input_file.")

    if args.input_file is not None and args.output_file is None:
        raise ValueError("Batch mode requires --output_file.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    if args.text is not None:
        results = predict_texts(model, tokenizer, [args.text], args.max_length)
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
        return

    records = load_records(args.input_file)
    texts = [r["text"] for r in records]
    results = predict_texts(model, tokenizer, texts, args.max_length)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved batch predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
