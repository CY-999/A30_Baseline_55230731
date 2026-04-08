#!/usr/bin/env bash
set -e

python prepare_demo_data.py

python train_roberta.py \
  -o outputs/roberta_demo

python predict_roberta.py \
  --model_dir outputs/roberta_demo/best_model \
  --text "客服说我开通了百万保障，不关闭就会自动扣费，让我下载会议软件处理"
