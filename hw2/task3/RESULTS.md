# HW2 Task3 实验结果

## 任务说明

Task3 使用 Stanford Background Dataset 从零训练手写 U-Net，并比较三种损失函数配置在验证集上的 mIoU 表现。训练全部在远程 `135-3090-8` 的 `/data/yc/miniconda/envs/llm-26-gpu` 环境完成。

## 数据与评估口径

- 数据集：`hw2/StanfordBackground/iccv09Data/`
- 样本数：715 张图像 / 715 个 `.regions.txt` 语义标签
- 划分：固定 seed=42，train 572 / val 143
- 输入尺寸：`240x320`
- 类别数：8 类（sky、tree、road、grass、water、building、mountain、foreground object）
- Ignore：标签 `-1` 转为 `255`，不参与 loss、pixel accuracy、mIoU 和 per-class IoU 统计
- 选模：以验证集 mIoU 最优 epoch 保存 `best.pt`

## 正式结果

| 实验 | 损失函数 | 最佳 epoch | Val mIoU | Val pixel acc | SwanLab |
|---|---|---:|---:|---:|---|
| `task3_unet_ce` | Cross-Entropy | 69 | 0.648151 | 0.833211 | <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/gky8mykesv1tyjf6hquq9> |
| `task3_unet_dice` | 手写 Dice Loss | 63 | 0.648211 | 0.829996 | <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/nawj01kfoerht6cp02hcn> |
| `task3_unet_ce_dice` | Cross-Entropy + 手写 Dice Loss | 52 | **0.648970** | **0.834739** | <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/fbpzyv27p65mc0xiqksk3> |

结论：三组实验 mIoU 非常接近，`CE + Dice` 组合损失略优，作为最终推荐模型。

## Per-class IoU

| 类别 | CE | Dice | CE + Dice |
|---|---:|---:|---:|
| sky | 0.883785 | 0.880028 | **0.884562** |
| tree | 0.659176 | 0.663875 | **0.668288** |
| road | **0.811282** | 0.784136 | 0.806054 |
| grass | 0.723195 | 0.726316 | **0.747842** |
| water | 0.640778 | 0.607719 | **0.645330** |
| building | 0.702307 | 0.706188 | **0.712474** |
| mountain | 0.187689 | **0.253740** | 0.158662 |
| foreground_object | **0.576994** | 0.563686 | 0.568549 |

观察：组合损失在 sky、tree、grass、water、building 等主要类别上更稳定；Dice-only 对 mountain 这类稀有类别更有利，但整体 pixel accuracy 略低。

## 远程产物

```text
hw2/task3/outputs/20260429_085730_task3_unet_ce/
hw2/task3/outputs/20260429_085801_task3_unet_dice/
hw2/task3/outputs/20260429_085805_task3_unet_ce_dice/
```

每个目录均包含：`source_config.yaml`、`config.json`、`dataset_stats.json`、`env.json`、`history.csv`、`curves.png`、`best.pt`、`metrics.json`、`val_samples.png`、`palette_legend.png`。

## ModelScope

最佳模型和关键产物已上传到：

- 仓库：<https://modelscope.cn/models/youngchen/CS60003/>
- 最佳模型路径：`hw2/task3/unet_ce_dice/best.pt`
- 指标文件路径：`hw2/task3/unet_ce_dice/metrics.json`
- 曲线图路径：`hw2/task3/unet_ce_dice/curves.png`
- 验证样例路径：`hw2/task3/unet_ce_dice/val_samples.png`
