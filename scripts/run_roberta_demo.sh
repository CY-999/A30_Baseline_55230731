#!/usr/bin/env bash
set -e

python prepare_demo_data.py

python train_roberta.py \
  --train_file data/demo/train.jsonl \
  --val_file data/demo/val.jsonl \
  --test_file data/demo/test.jsonl \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --output_dir outputs/roberta_demo \
  --max_length 128 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --seed 55230731

python predict_roberta.py \
  --model_dir outputs/roberta_demo/best_model \
  --text "客服说我开通了百万保障，不关闭就会自动扣费，让我下载会议软件处理"
