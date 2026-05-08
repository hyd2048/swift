# Swift Pretrain

基于 [ms-swift](https://github.com/modelscope/ms-swift) 的 Qwen3-0.6B 训练项目。

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

### 2. 启动预训练

```bash
bash train/pretrain.sh
```

- 8 卡 DeepSpeed ZeRO-2
- 数据集：`/data/heyudong/dataset/seq_monkey_datawhore.jsonl`
