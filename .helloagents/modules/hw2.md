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
  - Task1 原始 Baseline：ResNet-18，`best_val_acc=0.7186`，`test_acc=0.6819`
  - Task1 当前最佳：ConvNeXt-Tiny 优化，`best_val_acc=0.9784`，`test_acc=0.9608`
  - Task1 最有效优化：更强 ImageNet 预训练 backbone（ConvNeXt-Tiny / EfficientNet-B0 / ResNet-50）+ AdamW + label smoothing + TTA。
  - Task1 报告要求：训练/复现实验需接入 SwanLab 记录 loss、accuracy 曲线和最终指标。
  - SwanLab 接入：`hw2/task1/train.py` 支持 YAML 开启实时记录；`hw2/task1/upload_swanlab_history.py --all` 可把已有正式实验 `history.csv` / `metrics.json` 回放到 SwanLab。
  - SwanLab 上传结果：项目 <https://swanlab.cn/@youngchen/cs60003-hw2-task1>，13 个正式实验修正版回放链接见 `hw2/task1/SWANLAB_RUNS.md`；每个 run 包含横轴/纵轴明确标注的 `report/curves_with_axis_labels` 图像。
  - ModelScope 上传结果：仓库 <https://modelscope.cn/models/youngchen/CS60003/>，最佳模型路径 `hw2/task1/flowers102_convnext_tiny/best.pt`。
  - SwanLab API key（用户明确要求写入仓库文件并允许 Git 同步）：`cxNFTo8J6hPt2s9PVEcYe`
  - ModelScope API token（用户明确要求写入仓库文件并允许按 SwanLab key 相同方式使用）：`ms-11d617a2-f67c-4e6c-ac54-e6ec7d016fb5`
- `hw2/RoadVehicleImages/`
  - 来源：Kaggle Road Vehicle Images Dataset
  - 内容：`trafic_data/train/`、`trafic_data/valid/`、`trafic_data/data_1.yaml`
  - 当前划分：train 2704 张图片 / 2704 个标签，valid 300 张图片 / 300 个标签
  - 注意：原始 `data_1.yaml` 中的路径写法可能与本地目录层级不一致；已额外生成本地可用配置 `trafic_data/data_hw2.yaml`，训练 YOLOv8 时优先使用该文件。
- `hw2/StanfordBackground/`
  - 来源：Stanford Background Dataset
  - 内容：`iccv09Data/images/`、`iccv09Data/labels/`、`horizons.txt`、`README`
  - 当前图片数：715
  - Task3 方案包：`.helloagents/plans/202604291645_hw2_task3_unet_segmentation/`
  - Task3 计划工程：`hw2/task3/`
  - Task3 核心要求：从零手写 U-Net，不使用预训练；手写 Dice Loss；训练 `ce` / `dice` / `ce_dice` 三组并比较验证集 mIoU。
  - Task3 数据处理：使用 `labels/*.regions.txt` 的 8 类语义标签，`-1` unknown 像素在损失和指标中 ignore。
  - Task3 执行约束：不在本机 smoke test；代码通过 Git 同步到远程 `135-3090-8` 后在 `/data/yc/miniconda/envs/llm-26-gpu` 做正式实验。
  - Task3 记录与发布：使用 SwanLab 记录 loss / mIoU / pixel accuracy 曲线；最佳权重后续上传到 ModelScope `youngchen/CS60003/hw2/task3/`。
  - Task3 正式结果：CE `val_mIoU=0.648151`，Dice `val_mIoU=0.648211`，CE+Dice `val_mIoU=0.648970`。
  - Task3 最佳模型：`task3_unet_ce_dice`，best epoch 52，`val_mIoU=0.648970`，`val_pixel_acc=0.834739`。
  - Task3 SwanLab 项目：<https://swanlab.cn/@youngchen/cs60003-hw2-task3>，最佳 run <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/fbpzyv27p65mc0xiqksk3>。
  - Task3 ModelScope 上传结果：仓库 <https://modelscope.cn/models/youngchen/CS60003/>，最佳模型路径 `hw2/task3/unet_ce_dice/best.pt`。
  - Task3 优化方案包：`.helloagents/plans/202604291718_hw2_task3_optimization/`、`.helloagents/plans/202604291855_hw2_task3_resunet_attention_optimization/`
  - Task3 第一轮优化最佳：`task3_unet_ce_dice_b64_tta`，best epoch 67，`val_mIoU=0.665089`，`val_pixel_acc=0.842011`。
  - Task3 最终优化最佳：`task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta`，best epoch 113，`val_mIoU=0.701053`，`val_pixel_acc=0.864931`，比基础最佳提升 `+0.052083` mIoU。
  - Task3 最终优化方法：从零手写 Attention U-Net + CE+Dice + random scale crop + horizontal-flip TTA + multi-scale TTA `[0.6, 0.8, 1.0, 1.2, 1.4]`；未使用预训练或现成分割模型。
  - Task3 单模型继续冲分记录：EMA + mountain-aware sampling / rare-class crop 版本 `val_mIoU=0.700604`，`mountain IoU=0.309850`，整体未超过最终推荐的 seed7 checkpoint。
  - Task3 0.75 单模型冲分尝试：已新增 U-Net++、scSE、ASPP bridge、deep supervision、CE+Dice+Lovasz、高分辨率和更大 batch 配置；远程正式实验最好为 `task3_unetpp_b64_384_lovasz_ds_tta`，训练中最佳 `val_mIoU=0.693158`，多尺度复评 `val_mIoU=0.6953`，低于当前最终最佳 `0.701053`，未上传替换模型。
  - Task3 最终优化 SwanLab run：<https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/fw55rpcbgagnmqbcz0q90>。
  - Task3 EMA + mountain-aware SwanLab run：<https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/7jatyl6ez1o5rs9vgzk0w>。
  - Task3 最终优化 ModelScope 路径：`hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/best.pt`。

## 待补充

- 任务 2 仍需用户准备 10–30 秒测试视频；题面允许使用手机拍摄校园或路口视频。
- Task1 SwanLab 已上传，仍需从 SwanLab 页面导出或截图用于最终 PDF 报告。
- 后续 Task2 / Task3 仍需决定是否统一使用 PyTorch + torchvision + ultralytics，并规划实验记录方式（wandb 或 swanlab）。
