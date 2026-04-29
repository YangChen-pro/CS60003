# 恢复快照

## 主线目标
HW2 Task3 在 hw2.md 合规范围内继续冲分，尽量接近或超过 0.7 validation mIoU。

## 正在做什么
已实现手写 ResUNet / Attention U-Net 变体并新增正式实验配置；下一步提交推送并在远程 135 训练。

## 关键上下文
- 当前最佳：`task3_unet_ce_dice_b64_tta`，`val_mIoU=0.665089`，ModelScope 路径 `hw2/task3/unet_ce_dice_b64_tta/best.pt`。
- 本轮合规边界：不使用 SAM、DeepLab、SegFormer、torchvision segmentation models 或预训练权重。
- 固定数据划分：train 572 / val 143。
- 新增模型：`resunet`、`attention_unet`、`attention_resunet`，均从零初始化。
- 远程主机：`135-3090-8`；远程仓库：`/data/yc/CS60003`；远程 Python：`/data/yc/miniconda/envs/llm-26-gpu`。
- 使用 Git 同步本机和远程代码。

## 下一步
提交并推送 `hw2/task3/stanford_unet/models.py`、新增 `hw2/task3/configs/opt_*` 配置和方案包，然后远程拉取并启动正式实验。

## 阻塞项
（无）

## 方案
`.helloagents/plans/202604291855_hw2_task3_resunet_attention_optimization/`

## 已标记技能
helloagents
