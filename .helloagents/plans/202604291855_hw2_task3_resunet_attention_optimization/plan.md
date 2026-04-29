# HW2 Task3 ResUNet / Attention U-Net 冲分 — 实施规划

## 目标与范围
继续在合规边界内提升 Task3 validation mIoU。当前基线最佳为 `task3_unet_ce_dice_b64_tta`，`val_mIoU=0.665089`。本轮比较手写 ResUNet、Attention U-Net、Attention ResUNet 三类 U-Net 家族变体。

## 架构与实现策略
- 在 `stanford_unet/models.py` 中加入可配置残差卷积块和 additive attention gate。
- `build_model` 仅接受从零初始化的 `unet`、`resunet`、`attention_unet`、`attention_resunet`。
- 新增四组正式配置：ResUNet b64、Attention U-Net b64、Attention ResUNet b48、Attention ResUNet b64。
- 训练策略沿用上一轮最优：`256x320` 输入、CE+Dice、轻量 dropout、horizontal-flip TTA。

## 完成定义
- 本机编译通过。
- Git 推送后远程 135 拉取到同一 HEAD。
- 远程模型 shape 检查通过。
- 至少完成 3 组正式训练；如任一结果超过 `0.665089`，上传新最佳到 ModelScope。
- 更新 `RESULTS.md`、`README.md`、知识库和状态文件。

## 文件结构
- `hw2/task3/stanford_unet/models.py`
- `hw2/task3/configs/opt_resunet_b64_tta.yaml`
- `hw2/task3/configs/opt_attention_unet_b64_tta.yaml`
- `hw2/task3/configs/opt_attention_resunet_b48_tta.yaml`
- `hw2/task3/configs/opt_attention_resunet_b64_tta.yaml`
- `.helloagents/plans/202604291855_hw2_task3_resunet_attention_optimization/`

## UI / 设计约束
本任务不涉及 UI。

## 风险与验证
- 风险：更强结构不一定提升 Stanford Background 验证 mIoU；可能因小数据集过拟合。
- 风险：Attention ResUNet b64 可能显存压力更高；必要时保留 b48 结果或停掉失败实验。
- 验证：远程 shape 检查、正式训练 metrics、SwanLab 曲线、ModelScope file_exists。

## 决策记录
- [2026-04-29] 为保持 hw2.md 合规，不使用 SAM 或现成预训练分割模型，仅实现手写 U-Net 家族结构。
- [2026-04-29] 固定验证集 143 张，仅改变模型结构与训练随机初始化，不改变数据划分。
