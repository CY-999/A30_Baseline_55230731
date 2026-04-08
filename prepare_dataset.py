import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SEED = 55230731
LABEL_MAP = {
    "white": 0,
    "black": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test splits from total.json for RoBERTa training."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/demo/total.json",
        help="Source dataset file. Supports .json list or .jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/demo",
        help="Directory to save train.jsonl, val.jsonl, and test.jsonl.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_records(file_path: str) -> List[Dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of records.")
        return data

    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    raise ValueError("Only .json and .jsonl files are supported.")


def normalize_record(raw: Dict) -> Dict | None:
    text = str(raw.get("text", "")).strip()
    label_binary = str(raw.get("label_binary", "")).strip().lower()

    if not text or label_binary not in LABEL_MAP:
        return None

    return {
        "text": text,
        "label": LABEL_MAP[label_binary],
        "label_binary": label_binary,
        "index": raw.get("index"),
        "f_index": raw.get("f_index"),
        "riskType": raw.get("riskType"),
        "riskPoint": raw.get("riskPoint"),
        "risk_level": raw.get("risk_level"),
    }


def normalize_records(records: Iterable[Dict]) -> Tuple[List[Dict], int]:
    normalized = []
    skipped = 0
    for raw in records:
        item = normalize_record(raw)
        if item is None:
            skipped += 1
            continue
        normalized.append(item)
    return normalized, skipped


def validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")


def group_records(records: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for idx, record in enumerate(records):
        group_key = record.get("f_index")
        if group_key is None or group_key == "":
            group_key = f"row_{idx}"
        grouped[str(group_key)].append(record)
    return grouped


def split_group_ids(
    group_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = list(group_ids)
    rng.shuffle(shuffled)

    n_groups = len(shuffled)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]
    return train_ids, val_ids, test_ids


def flatten_groups(
    grouped_records: Dict[str, List[Dict]],
    group_ids: Iterable[str],
    seed: int,
) -> List[Dict]:
    records = []
    for group_id in group_ids:
        records.extend(grouped_records[group_id])

    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: List[Dict]) -> Dict[str, object]:
    label_counts = Counter(record["label"] for record in records)
    return {
        "size": len(records),
        "label_0_normal": int(label_counts.get(0, 0)),
        "label_1_scam": int(label_counts.get(1, 0)),
    }


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio)

    raw_records = load_records(args.input_file)
    records, skipped = normalize_records(raw_records)
    grouped = group_records(records)

    train_ids, val_ids, test_ids = split_group_ids(
        group_ids=list(grouped.keys()),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_records = flatten_groups(grouped, train_ids, args.seed)
    val_records = flatten_groups(grouped, val_ids, args.seed + 1)
    test_records = flatten_groups(grouped, test_ids, args.seed + 2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)
    write_jsonl(output_dir / "test.jsonl", test_records)

    summary = {
        "input_file": str(Path(args.input_file).resolve()),
        "total_raw_records": len(raw_records),
        "total_valid_records": len(records),
        "skipped_records": skipped,
        "group_count": len(grouped),
        "train": summarize(train_records),
        "val": summarize(val_records),
        "test": summarize(test_records),
    }

    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Prepared dataset splits successfully.")
    print(f"Input file: {args.input_file}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Skipped records: {skipped}")
    print(
        "Split sizes: "
        f"train={len(train_records)}, "
        f"val={len(val_records)}, "
        f"test={len(test_records)}"
    )
    print(
        "Label counts: "
        f"train(normal={summary['train']['label_0_normal']}, scam={summary['train']['label_1_scam']}), "
        f"val(normal={summary['val']['label_0_normal']}, scam={summary['val']['label_1_scam']}), "
        f"test(normal={summary['test']['label_0_normal']}, scam={summary['test']['label_1_scam']})"
    )
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
