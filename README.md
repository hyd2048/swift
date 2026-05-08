# Swift Train

基于 [ms-swift](https://github.com/modelscope/ms-swift) 的 Qwen3-0.6B 训练项目，支持预训练、SFT 和 GRPO 强化学习。

## 环境准备

```bash
# Python 3.12 + uv
uv sync
```

依赖：PyTorch 2.8 (CUDA 12.9)、flash-attn、deepspeed、ms-swift。

## 使用

### 1. 初始化模型

从 Qwen3-0.6B 的 config 随机初始化权重：

```bash
python train/init_model.py
```

### 2. 预训练

```bash
bash train/pretrain.sh
```

- 8 卡 DeepSpeed ZeRO-2
- 数据集：`/data/heyudong/dataset/seq_monkey_datawhore.jsonl`

### 3. SFT 监督微调

```bash
bash train/sft.sh
```

- 单卡 LoRA 微调（rank=8, alpha=32）
- 数据集：`qwen3_32b_distill_1k_messages.jsonl` + `self_cognition_messages.jsonl`
- 输出：`model/qwen3-sft`

### 4. GRPO 强化学习

```bash
bash train/grpo.sh
```

- 8 卡 DeepSpeed ZeRO-3 + vLLM 共置
- 数据集：`/data/heyudong/dataset/rl/DAPO-Math-17k.jsonl`
- 奖励函数：accuracy
- 输出：`model/qwen3-grpo`
