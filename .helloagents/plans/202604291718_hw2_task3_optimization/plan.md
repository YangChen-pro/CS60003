# HW2 Task3 U-Net 继续冲分优化 — 实施规划

## 目标与范围
以已完成的 `task3_unet_ce_dice` 为基准，在不违反题面约束的前提下测试更强 U-Net 训练配置，力争提升 validation mIoU。

## 架构与实现策略
- 模型容量：在手写 U-Net 内支持 `base_channels=48/64`，保持 encoder、decoder、skip connection 结构不变。
- 正则化：在 DoubleConv 中加入可选 `Dropout2d`，用于更大模型的轻量正则。
- Loss：保留 CE+Dice 主路线，新增可选 inverse-sqrt class weighting，改善类别不平衡。
- 输入与评估：尝试 `256x320` 输入；验证阶段支持 horizontal-flip TTA。
- 实验矩阵：
  - `opt_ce_dice_wide_tta.yaml`: base 48 + CE+Dice + TTA。
  - `opt_ce_dice_wide_weighted_tta.yaml`: base 48 + class-weighted CE+Dice + TTA。
  - `opt_ce_dice_b64_tta.yaml`: base 64 + CE+Dice + TTA。

## 完成定义
- 远程检查通过：compile、模型前向、loss、mIoU。
- 至少完成 2 组优化正式实验。
- 若最佳 mIoU 高于 `0.648970`，上传新最佳到 ModelScope 并更新结果文档。

## 风险与验证
- 更大模型可能过拟合或显存更高：通过 dropout、early stopping、较小 batch 控制。
- class weighting 可能牺牲大类：作为独立实验，不覆盖原结果。
- TTA 只用于验证/评估，不参与训练。

## 决策记录
- [2026-04-29] 用户选择全自动继续优化 Task3 得分。
