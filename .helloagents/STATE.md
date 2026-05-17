# 恢复快照

## 主线目标
同步 CS60003 项目知识库，记录最新 GitHub 拉取后的 HW2 Task2 代码与实验状态。

## 正在做什么
空闲；已完成 Delivery Gate 修复，历史方案包元数据、验证证据和收尾证据已闭合。

## 关键上下文
- 本地仓库已在 2026-05-17 fast-forward 到 `origin/main` 的 `f9f2ff8e`。
- HW1 已完成 EuroSAT 三层 MLP，核心实现仍为 CuPy 手写前向/反向传播；权重通过 ModelScope 发布。
- HW2 Task1 已完成 Flowers102 分类：ConvNeXt-Tiny 最终 `test_acc=0.9608`，权重发布到 ModelScope 的 `hw2/task1/flowers102_convnext_tiny/best.pt`。
- HW2 Task2 已补入检测与跟踪工程：YOLOv8s baseline `best_mAP50=0.5150`、`best_mAP50_95=0.2854`、`best_epoch=38`；`track_video.py` 实现 ByteTrack 跟踪、稳定 display ID、遮挡重连、缺失预测越线计数和快照输出。
- HW2 Task3 已完成 Stanford Background 从零 U-Net/Attention U-Net 分割：最终 `val_mIoU=0.701053`，权重发布到 ModelScope 的 `hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/best.pt`。
- HW2 报告目录仍是 `/Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report`；是否已纳入最新 Task2 内容需要单独复核。
- SwanLab 实验记录用于内部追踪；报告中默认只嵌入导出的本地图片，不直接暴露私有云端链接。

## 下一步
无当前必需动作；若继续处理 HW2 提交，再复核并更新 `/Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report/src/hw2.tex` 的 Task2 检测、跟踪、遮挡分析和越线计数内容。

## 阻塞项
（无）

## 方案


## 已标记技能
helloagents:~wiki
