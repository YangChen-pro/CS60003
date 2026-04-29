# HW2 Task1 正式实验结果

实验时间：2026-04-29
远程主机：`135-3090-8`
远程仓库：`/data/yc/CS60003`
代码提交：`91b6b8beadeb8fc7f5e8e021b1cdd65e718af43e`
Python 环境：`/data/yc/miniconda/envs/llm-26-gpu`
主要依赖：PyTorch `2.11.0+cu130`，torchvision `0.26.0+cu130`
GPU：`CUDA_VISIBLE_DEVICES=5`，NVIDIA GeForce RTX 3090

## 数据校验

`hw2/Flowers102/` 官方划分校验通过：

| Split | 样本数 |
|---|---:|
| train | 1020 |
| val | 1020 |
| test | 6149 |

## 实验结果

| 实验 | 配置 | best epoch | best val acc | test acc | 结论 |
|---|---|---:|---:|---:|---|
| Baseline | `baseline_resnet18.yaml` | 33 | 0.7186 | 0.6819 | 当前最佳结果 |
| 低学习率 | `baseline_resnet18_low_lr.yaml` | 34 | 0.4873 | 0.4506 | 明显欠拟合 |
| 较短训练 | `baseline_resnet18_short.yaml` | 22 | 0.5725 | 0.5307 | 训练轮数不足 |
| 随机初始化 | `random_resnet18.yaml` | 40 | 0.1794 | 0.1571 | 预训练带来显著提升 |
| SE 注意力 | `se_resnet18.yaml` | 39 | 0.4500 | 0.4245 | 当前设置下不如 Baseline，需额外调参 |

## 远程产物路径

```text
hw2/task1/outputs/20260429_031731_baseline_resnet18/
hw2/task1/outputs/20260429_032048_baseline_resnet18_low_lr/
hw2/task1/outputs/20260429_032344_baseline_resnet18_short/
hw2/task1/outputs/20260429_032540_random_resnet18/
hw2/task1/outputs/20260429_032833_se_resnet18/
```

每个目录包含：

- `source_config.yaml`
- `config.json`
- `dataset_stats.json`
- `history.csv`
- `curves.png`
- `best.pt`
- `metrics.json`
- `test_details.json`

## 报告建议口径

- Baseline 使用 ImageNet 预训练 ResNet-18，分类头学习率 `1e-3`，backbone 学习率 `1e-4`，40 epochs。
- 低学习率和短训练轮数均明显降低性能，说明 Task1 对充分微调强依赖。
- 随机初始化 ResNet-18 在相同训练预算下远低于预训练 Baseline，能直接支撑预训练消融结论。
- SE 注意力模块虽然增加了通道注意力，但新增参数随机初始化，在当前学习率和训练轮数下未超过 Baseline；报告中可解释为“注意力结构本身不保证提升，需要配套调参或更长训练”。
