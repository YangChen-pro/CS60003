# HW2 模块说明

## 任务范围

HW2 期中作业包含三个方向：

- 任务 1：在 102 Category Flower Dataset 上微调 ImageNet 预训练 CNN，并做超参数、预训练消融与注意力机制对比。
- 任务 2：在 Road Vehicle Images Dataset 上微调 YOLOv8 或同类模型，并完成视频多目标跟踪、遮挡分析和越线计数。
- 任务 3：从零实现 U-Net，在 Stanford Background Dataset 上比较 Cross-Entropy、Dice Loss 与组合损失的 mIoU。

## 数据集位置

```yaml
Flowers102: hw2/Flowers102/
RoadVehicleImages: hw2/RoadVehicleImages/
StanfordBackground: hw2/StanfordBackground/
数据说明: hw2/DATASETS.md
```

## 当前状态

- Flowers102 已下载：8189 张图片，含官方标签 `imagelabels.mat` 与划分 `setid.mat`。
- Road Vehicle Images 已下载：train 2704 张图片 / 2704 个标签，valid 300 张图片 / 300 个标签。
- Stanford Background 已下载：715 张图片，含 `labels/` 与 `horizons.txt`。
- Road Vehicle 已生成本地 YOLO 配置：`hw2/RoadVehicleImages/trafic_data/data_hw2.yaml`。

## 待补充

- 任务 2 仍需用户准备 10–30 秒测试视频。
- 后续实现阶段需决定是否统一使用 PyTorch + torchvision + ultralytics，并规划实验记录方式（wandb 或 swanlab）。
