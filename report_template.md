# A30 个人任务一报告模板（学号：55230731）

## 1. 任务目标
我独立完成了一个与 A30 多模态反诈智能助手方向一致的文本反诈识别 baseline，
用于先跑通基础训练、评估与推理链路。

## 2. 算力环境
- 服务器名称：
- GPU 型号：
- CUDA 版本：
- Python 版本：
- 关键依赖版本：

> 在此处插入 `nvidia-smi` 截图

## 3. 采用方法
### 3.1 主 baseline
- 模型名称：Chinese RoBERTa (`hfl/chinese-roberta-wwm-ext`)
- 任务形式：中文文本二分类（诈骗 / 正常）

### 3.2 备选 baseline
- 方法名称：TF-IDF + Logistic Regression

## 4. 数据说明
- 训练集规模：
- 验证集规模：
- 测试集规模：
- 标签说明：0=正常，1=诈骗

## 5. 训练命令
```bash
python train_roberta.py \
  --train_file ... \
  --val_file ... \
  --test_file ... \
  --model_name_or_path ... \
  --output_dir ... \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --seed 55230731
```

## 6. 核心代码说明
请贴出以下关键片段并截图：
- 文本编码逻辑
- 模型训练逻辑
- 推理逻辑
- 带学号 `55230731` 的关键注释

## 7. 实验结果
- 验证集 Accuracy：
- 验证集 F1：
- 测试集 Accuracy：
- 测试集 F1：

> 在此处插入训练日志截图与测试结果截图

## 8. 结论
本次个人任务已在服务器 GPU 环境下独立跑通文本反诈识别 baseline，
完成了训练、评估、推理与结果保存流程，并保留了运行日志和关键代码注释。
