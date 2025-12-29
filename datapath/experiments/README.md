# 训练实验目录

本目录用于存放模型微调的输出结果。

## 目录结构

```
experiments/
└── flickr30k_finetune_pycharm/
    ├── checkpoints/
    │   ├── epoch_1.pt          # Epoch 1 权重（需训练生成）
    │   ├── epoch_2.pt          # Epoch 2 权重（需训练生成）
    │   └── epoch_latest.pt     # 最终权重（需训练生成）⭐
    ├── out_2025-11-27-*.log    # 训练日志
    └── params_2025-11-27-*.json # 训练参数

```

## 训练结果

- Epoch 1: Text→Image Recall@10 = 73.44%
- Epoch 2: Text→Image Recall@10 = 80.95%
- Epoch 3: Text→Image Recall@10 = 84.10% ✅

## 获取微调权重

由于模型权重文件较大，未包含在本仓库中。您可以：

1. **自行训练**：按照主 README.md 的说明进行微调
2. **联系作者**：通过 Issue 请求预训练权重

详细说明见主 README.md
