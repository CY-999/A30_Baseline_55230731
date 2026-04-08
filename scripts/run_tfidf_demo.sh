#!/usr/bin/env bash
set -e

python prepare_demo_data.py

python train_tfidf_lr.py \
  --train_file data/demo/train.jsonl \
  --val_file data/demo/val.jsonl \
  --test_file data/demo/test.jsonl \
  --output_dir outputs/tfidf_lr_demo
