# A30 文本反诈识别 Baseline（学号：55230731）

这是一个**面向 A30「多模态反诈智能助手」赛题方向**的**文本子任务 baseline**。  
它的目标不是一步做到完整多模态系统，而是先独立跑通一个**可训练、可评估、可推理**的基础版本，作为后续图片 OCR 文本、语音 ASR 转写文本接入大模型/风控系统的起点。

本项目提供两套 baseline：

1. **主 baseline：Chinese RoBERTa 文本二分类**（推荐，支持 GPU）
2. **备选 baseline：TF-IDF + Logistic Regression**（纯经典方法，便于兜底）

---

## 1. 项目目标

输入一段中文文本，输出：

- `label=1`：疑似诈骗
- `label=0`：正常文本

该 baseline 与 A30 的关系如下：

- A30 最终是**多模态反诈助手**
- 本项目先完成其中最基础、最容易独立验证的一条链路：**文本风险识别**
- 后续你们可以把 **OCR 提取文本**、**ASR 转写文本** 都接到这个 baseline 上，形成更完整的系统

---

## 2. 项目结构

```text
A30_text_baseline_55230731/
├── README.md
├── requirements.txt
├── prepare_demo_data.py
├── train_roberta.py
├── predict_roberta.py
├── train_tfidf_lr.py
├── utils.py
├── report_template.md
├── scripts/
│   ├── run_roberta_demo.sh
│   └── run_tfidf_demo.sh
├── data/
│   └── demo/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
└── outputs/
```

---

## 3. 环境要求

推荐环境：

- Python 3.10 或 3.11
- CUDA 可用的 Linux 服务器
- 一张普通消费级 GPU 即可完成 baseline（如 3090 / 4090 / A5000 / A6000 等）
- 若显存较小，可减小 `batch_size`

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 4. 数据格式

本项目默认使用 `jsonl`：

```json
{"text": "您好，您的账户异常，请点击链接完成认证。", "label": 1}
{"text": "明天下午三点开组会，请大家准时参加。", "label": 0}
```

字段说明：

- `text`：输入文本
- `label`：类别标签，`1=诈骗`，`0=正常`

### 如何替换成你自己的数据

你只需要把自己的数据整理成三个文件：

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

然后在训练命令里改路径即可。

---

## 5. Demo 数据说明

为了保证你能**开箱即用**，项目已经附带一份很小的 demo 数据集，位于：

```text
data/demo/train.jsonl
data/demo/val.jsonl
data/demo/test.jsonl
```

这些数据仅用于**跑通流程**。  
如果你想得到更可信的结果，建议后续自行扩充真实/半真实诈骗文本样本。

如果你希望重新生成 demo 数据，也可以运行：

```bash
python prepare_demo_data.py
```

---

## 6. 主 baseline：RoBERTa 文本分类（推荐）

默认使用中文开源基础模型：

```text
hfl/chinese-roberta-wwm-ext
```

你也可以改成：

- `bert-base-chinese`
- `hfl/chinese-macbert-base`
- 或者服务器上本地已下载好的模型目录

### 6.1 训练

```bash
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
```

### 6.2 推理（单条文本）

```bash
python predict_roberta.py \
  --model_dir outputs/roberta_demo/best_model \
  --text "客服说我开通了百万保障，不关闭就会自动扣费，让我下载会议软件处理"
```

### 6.3 推理（文件批量）

输入文件格式：

```json
{"text": "这是一条待预测文本"}
{"text": "这是另一条待预测文本"}
```

执行：

```bash
python predict_roberta.py \
  --model_dir outputs/roberta_demo/best_model \
  --input_file data/demo/test.jsonl \
  --output_file outputs/roberta_demo/test_predictions.jsonl
```

---

## 7. 备选 baseline：TF-IDF + Logistic Regression

如果你服务器暂时无法下载 HuggingFace 模型，或者你想快速生成一个经典算法 baseline，可以运行：

```bash
python train_tfidf_lr.py \
  --train_file data/demo/train.jsonl \
  --val_file data/demo/val.jsonl \
  --test_file data/demo/test.jsonl \
  --output_dir outputs/tfidf_lr_demo
```

它会输出：

- 训练后的 sklearn pipeline
- 验证集指标
- 测试集指标

---

## 8. 推荐的服务器运行方式

### 8.1 查看 GPU

```bash
nvidia-smi
```

### 8.2 后台训练并保存日志

```bash
nohup python train_roberta.py \
  --train_file data/demo/train.jsonl \
  --val_file data/demo/val.jsonl \
  --test_file data/demo/test.jsonl \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --output_dir outputs/roberta_demo \
  --max_length 128 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --seed 55230731 > outputs/roberta_demo/train.log 2>&1 &
```

### 8.3 查看训练日志

```bash
tail -f outputs/roberta_demo/train.log
```

---

## 9. 你个人任务一建议提交哪些材料

根据你给出的课程要求，建议你提交：

1. **运行日志截图**
   - `nvidia-smi`
   - 训练日志 `train.log`
   - 最终测试结果截图

2. **算力平台使用证明**
   - 服务器终端截图
   - GPU 占用截图
   - 训练过程截图

3. **关键代码片段**
   - `train_roberta.py`
   - `predict_roberta.py`
   - 标出带有学号 `55230731` 的关键注释

4. **结果文件**
   - `outputs/roberta_demo/test_metrics.json`
   - `outputs/roberta_demo/train.log`

---

## 10. 结果文件说明

训练完成后，主 baseline 会在输出目录生成：

- `best_model/`：最佳模型
- `test_metrics.json`：测试集指标
- `label_mapping.json`：标签说明
- `train.log`：训练日志（若你按上面的方式重定向）

---

## 11. 常见问题

### Q1：服务器无法联网，模型下载不了怎么办？
做法有两个：

- 先在可联网机器上下载 HuggingFace 模型，再拷到服务器
- 或者直接把 `--model_name_or_path` 改成服务器上的本地模型目录

例如：

```bash
python train_roberta.py \
  --model_name_or_path /path/to/local/chinese-roberta-wwm-ext \
  ...
```

### Q2：显存不够怎么办？
尝试：

- 减小 `--batch_size`
- 将 `--max_length` 从 128 调到 64
- 若仍不够，可先跑 `train_tfidf_lr.py`

### Q3：为什么 demo 数据效果不稳定？
因为 demo 数据量很小，只是用于**跑通 baseline**。  
正式实验请自行扩充数据。

---

## 12. 后续如何扩展到 A30 完整方案

你后续可以在这个 baseline 基础上继续做：

- 接入 OCR：图片 → 文本 → 本模型
- 接入 ASR：音频 → 文本 → 本模型
- 将本模型输出接入你负责的 Prompt / LLM 风险分析链路
- 从二分类扩展到：
  - 刷单返利
  - 冒充客服退款
  - 冒充公检法
  - 虚假贷款
  - 投资理财
  - 等多类别诈骗类型识别

---

## 13. 一键运行脚本

### RoBERTa demo

```bash
bash scripts/run_roberta_demo.sh
```

### TF-IDF + LR demo

```bash
bash scripts/run_tfidf_demo.sh
```

---

## 14. 提交建议

如果你要把这部分作为“个人任务一”提交，最稳的表述方式是：

> 我独立在服务器 GPU 环境下完成了一个与 A30 赛题方向一致的文本反诈识别 baseline。  
> 该 baseline 基于开源中文预训练模型 Chinese RoBERTa 构建，完整跑通了训练、评估与推理流程，并保留了带有本人学号 55230731 注释的关键代码与运行日志。  
> 同时，我提供了经典算法 TF-IDF + Logistic Regression 作为对照 baseline，以便后续开展更系统的实验对比。

