# HW2 模块说明

## 任务范围

HW2 期中作业包含三个方向：

- 任务 1：在 102 Category Flower Dataset 上微调 ImageNet 预训练 CNN，并做超参数、预训练消融与注意力机制对比。
- 任务 2：在 Road Vehicle Images Dataset 上微调 YOLOv8 或同类模型，并完成视频多目标跟踪、遮挡分析和越线计数。
- 任务 3：从零实现 U-Net，在 Stanford Background Dataset 上比较 Cross-Entropy、Dice Loss 与组合损失的 mIoU。

## 数据集位置

本目录的数据集放置方式参考 `hw1/EuroSAT_RGB/`：每个数据集直接放在 `hw2/` 下的独立目录中。

```yaml
Flowers102: hw2/Flowers102/
RoadVehicleImages: hw2/RoadVehicleImages/
StanfordBackground: hw2/StanfordBackground/
数据说明: 本文件
```

## 当前状态

- `hw2/Flowers102/`
  - 来源：Oxford VGG 102 Category Flower Dataset
  - 内容：`jpg/` 图片目录、`imagelabels.mat` 标签、`setid.mat` 官方划分、`README.txt`
  - 当前图片数：8189
  - Task1 工程：`hw2/task1/`
  - Task1 方案包：`.helloagents/plans/202604291109_hw2_task1_flowers102/`
  - 执行约束：不在本机 smoke test；代码通过 Git 同步到远程 `135-3090-8` 后直接做正式实验；远程使用前检查 Git 身份为 `YangChen-pro <1369792882@qq.com>`。
  - 远程优先环境：`/data/yc/miniconda/envs/llm-26-gpu`，正式运行前需确认 PyTorch、torchvision 和 CUDA 可用。
  - Task1 正式结果：`hw2/task1/RESULTS.md`
  - Task1 当前最佳：Baseline ResNet-18，`best_val_acc=0.7186`，`test_acc=0.6819`
- `hw2/RoadVehicleImages/`
  - 来源：Kaggle Road Vehicle Images Dataset
  - 内容：`trafic_data/train/`、`trafic_data/valid/`、`trafic_data/data_1.yaml`
  - 当前划分：train 2704 张图片 / 2704 个标签，valid 300 张图片 / 300 个标签
  - 注意：原始 `data_1.yaml` 中的路径写法可能与本地目录层级不一致；已额外生成本地可用配置 `trafic_data/data_hw2.yaml`，训练 YOLOv8 时优先使用该文件。
- `hw2/StanfordBackground/`
  - 来源：Stanford Background Dataset
  - 内容：`iccv09Data/images/`、`iccv09Data/labels/`、`horizons.txt`、`README`
  - 当前图片数：715

## 待补充

- 任务 2 仍需用户准备 10–30 秒测试视频；题面允许使用手机拍摄校园或路口视频。
- 后续 Task2 / Task3 仍需决定是否统一使用 PyTorch + torchvision + ultralytics，并规划实验记录方式（wandb 或 swanlab）。
