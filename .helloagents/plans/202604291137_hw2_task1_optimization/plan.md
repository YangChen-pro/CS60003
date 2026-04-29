# HW2 Task1 Flowers102 优化 — 实施规划

## 目标与范围
本轮目标是继续提高 Task1 Accuracy，不替换官方数据划分，不改变已有 Baseline 结论。优化重点放在可解释、可报告、可复现的训练配置和预训练模型对比。

## 架构与实现策略
- 数据增强：
  - 在现有 `basic` 增强之外新增 `mild`、`strong` 预设。
  - `strong` 使用 RandAugment、ColorJitter、RandomErasing。
- 模型：
  - 保留 ResNet-18 / SE-ResNet-18。
  - 新增 torchvision 预训练 ResNet-34、ResNet-50、EfficientNet-B0、ConvNeXt-Tiny。
- 训练策略：
  - 新增 AdamW 配置。
  - 新增 label smoothing 和 gradient clipping。
  - 保留 cosine scheduler。
- 评估策略：
  - 新增水平翻转 TTA，用于最终 test 评估。
- 子代理协作：
  - 错例分析代理：从 per-class accuracy 和 confusion matrix 找低分类别。
  - 策略代理：给出高收益实验优先级。
  - 审查代理：检查代码风险和可信度问题。

## 完成定义
- 新增优化配置可在远程启动并完成。
- 至少完成 3 组优化正式实验。
- 最佳 `test_acc` 高于当前 Baseline `0.6819`。
- `RESULTS.md` 记录优化实验、最佳模型和报告建议口径。
- GitHub、本机、远程仓库同步到同一 HEAD。

## 文件结构
```text
hw2/task1/
├── configs/
│   ├── opt_resnet18_adamw_strong.yaml
│   ├── opt_resnet34_adamw_strong.yaml
│   ├── opt_resnet50_adamw_mild.yaml
│   ├── opt_efficientnet_b0_strong.yaml
│   └── opt_convnext_tiny_strong.yaml
├── flowers102_task1/
│   ├── data.py
│   ├── engine.py
│   ├── models.py
│   └── config.py
└── RESULTS.md
```

## UI / 设计约束
不涉及 UI。

## 风险与验证
- 新模型预训练权重需要远程下载；若下载失败，优先继续运行已缓存 ResNet 系列。
- ConvNeXt / ResNet-50 显存占用更高，批量大小保守设置为 32。
- 强增强可能降低收敛速度，因此同时保留 ResNet-50 mild 配置。
- 如果 TTA 提升有限，报告中只作为评估增强说明，不夸大贡献。

## 决策记录
- [2026-04-29] 用户要求尽可能提升 Task1 实验分数，并允许使用 subagent 做错例分析后汇总优化。
- [2026-04-29] 优先实施低风险高收益项：AdamW、label smoothing、强增强、更强预训练 backbone、TTA。
