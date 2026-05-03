# 模块: HW2

## 用途
记录 CS60003 HW2 期中作业的数据集、任务范围、实验工程、关键指标、报告要求和后续待办。当前 Task1 与 Task3 已完成正式实验和报告内容，Task2 仍待补齐。

## 任务范围
- Task 1：在 Oxford 102 Category Flower Dataset 上微调 ImageNet 预训练 CNN，完成 baseline、学习率/epoch 超参数对比、随机初始化消融、注意力结构对比和额外高分模型。
- Task 2：在 Road Vehicle Images Dataset 上训练检测模型，并完成 10–30 秒视频多目标跟踪、遮挡/ID 跳变分析和越线计数。
- Task 3：从零手写 U-Net / Attention U-Net，在 Stanford Background Dataset 上比较 Cross-Entropy、Dice Loss 与 CE+Dice，并在合规范围内优化 mIoU。

## 关键文件
- `hw2/hw2.md`：HW2 题面要求。
- `hw2/task1/`：Flowers102 分类工程、配置、训练评估、SwanLab 曲线回放和结果文档。
- `hw2/task1/RESULTS.md`：Task1 正式实验表与最佳模型说明。
- `hw2/task1/SWANLAB_RUNS.md`：Task1 私有 SwanLab run 保留清单；报告中不直接暴露云端链接。
- `hw2/task3/`：Stanford Background 语义分割工程、从零模型、loss、metrics、训练评估和上传脚本。
- `hw2/task3/RESULTS.md`：Task3 loss 对比、优化链路、per-class IoU 和最终模型说明。
- `hw2/task3/SWANLAB_RUNS.md`：Task3 私有 SwanLab run 保留清单；报告中不直接暴露云端链接。
- `/Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report/src/hw2.tex`：HW2 报告源文件。
- `/Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report/out/hw2.pdf`：当前编译通过的 HW2 报告 PDF。

## 数据集
```yaml
Flowers102: hw2/Flowers102/，官方 split train 1020 / val 1020 / test 6149
RoadVehicleImages: hw2/RoadVehicleImages/，train 2704 / valid 300，Task2 待做
StanfordBackground: hw2/StanfordBackground/，715 张图像；固定 seed=42 划分 train 572 / val 143
```

## 依赖
- 远程主机：`135-3090-8`
- 远程仓库：`/data/yc/CS60003`
- 远程 Python 环境：`/data/yc/miniconda/envs/llm-26-gpu`
- 远程 Git 身份：`YangChen-pro <1369792882@qq.com>`
- 实验记录：SwanLab（私有记录，报告不公开链接）
- 权重发布：ModelScope 仓库，按 `hw2/task1/` 与 `hw2/task3/` 子目录组织

## 当前状态
### Task 1：Flowers102 分类
- 数据：Oxford 102 Category Flower Dataset，102 类，官方 split train 1020 / val 1020 / test 6149。
- Baseline：ImageNet 预训练 ResNet-18，backbone lr `1e-4`，classifier lr `1e-3`，40 epochs。
- 必要对比已完成：低学习率、短训练、随机初始化、SE-ResNet-18、ResNet-18 label smoothing、ResNet-34 强增强。
- 最佳模型：ConvNeXt-Tiny 优化，`best_val_acc=0.9784`，`test_acc=0.9608`。
- ModelScope 路径：`hw2/task1/flowers102_convnext_tiny/best.pt`。
- SwanLab：云端已清理为 8 个报告必要 run；报告只使用导出的本地图片，不提供云端链接。

### Task 2：目标检测与多目标跟踪
- 当前状态：报告中保留结构化占位。
- 待做内容：检测模型训练、mAP/precision/recall 曲线、10–30 秒视频流推理、多目标跟踪 ID 可视化、遮挡片段 3–4 帧分析、越线计数逻辑和结果。
- 数据：`hw2/RoadVehicleImages/`；训练 YOLOv8 时优先检查 `trafic_data/data_hw2.yaml` 或实际数据配置。

### Task 3：Stanford Background 语义分割
- 数据：715 张图像，固定 seed=42 划分 train 572 / val 143；8 个语义类，`-1` unknown 映射为 ignore index 255。
- 基础三组 loss：CE `val_mIoU=0.648151`，Dice `val_mIoU=0.648211`，CE+Dice `val_mIoU=0.648970`。
- 第一轮优化：`task3_unet_ce_dice_b64_tta`，`val_mIoU=0.665089`，`val_pixel_acc=0.842011`。
- 最终推荐：`task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta`，best epoch 113，`val_mIoU=0.701053`，`val_pixel_acc=0.864931`。
- 最终方法：从零手写 Attention U-Net + CE+Dice + random scale crop + horizontal-flip TTA + multi-scale TTA `[0.6, 0.8, 1.0, 1.2, 1.4]`；未使用预训练或现成分割模型。
- ModelScope 路径：`hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/best.pt`。
- SwanLab：云端已清理为 6 个报告必要 run；报告只使用导出的本地图片，不提供云端链接。

## HW2 报告状态
- 报告工程：`/Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report/`
- 源文件：`src/hw2.tex`
- PDF：`out/hw2.pdf`
- 样式：复用 HW1 报告的 `elegantpaper.cls`
- 当前覆盖：Task1 与 Task3 完整写入，Task2 占位。
- 已完成修正：
  - 删除 SwanLab 私有链接；正文不出现私有实验平台域名。
  - GitHub/ModelScope 使用“仓库”作为可点击链接，不明文显示仓库名。
  - 长路径表格改用短路径/省略号。
  - Task1 表格用粗体标最佳、下划线标次优。
  - 图 1/2/3 曲线放大；图 4 调整为单页大图并使用裁剪后的类别图例。
- 编译验证：`latexmk -xelatex -shell-escape -outdir=../out hw2.tex` 已通过；最终 PDF 7 页，日志无 Overfull / Underfull / Warning / Error / undefined reference。

## 经验
- HW2 训练类正式实验不需要本机 smoke test；用户允许直接在远程 135 正式实验环境运行。
- 代码同步优先使用 Git；远程使用前检查 Git 身份是否与本机一致。
- SwanLab SDK 当前不直接暴露已上传媒体图片下载接口；报告图片优先使用远程正式实验产物中的导出图，或从 SwanLab 页面手动/工具导出后放入报告 `pic/`。
- Task3 在固定验证集上多 seed 与多尺度 TTA 选优，报告中必须明确最终指标是 validation mIoU，不是独立 test 指标。
- 若以完整 HW2 冲 A+，Task2 缺失是最大扣分项，必须补完整检测、跟踪、遮挡分析、越线计数和图表。
- 临时凭据记录：用户曾明确要求将 SwanLab API key 与 ModelScope token 写入 `.helloagents` 并允许 Git 同步；作业结束后需要删除。
