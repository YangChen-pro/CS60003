# 恢复快照

## 主线目标
HW2 Task3 在 hw2.md 合规范围内继续冲分，尽量接近或超过 0.7 validation mIoU。

## 正在做什么
第一批 ResUNet / Attention U-Net 结构实验已完成，最佳提升到 `0.667801`；正在加入 random scale crop 几何增强做第二轮冲分。

## 关键上下文
- 旧最佳：`task3_unet_ce_dice_b64_tta`，`val_mIoU=0.665089`。
- 本轮结构优化新最佳：`task3_attention_unet_b64_tta`，`val_mIoU=0.667801`，SwanLab run `6s3s3o8thd5p4q6p2ju12`。
- 第一批结构实验结果：ResUNet b64 `0.639848`，Attention ResUNet b48 `0.645112`，Attention ResUNet b64 `0.638071`。
- 合规边界：不使用 SAM、DeepLab、SegFormer、torchvision segmentation models 或预训练权重。
- 固定数据划分：train 572 / val 143。
- 远程主机：`135-3090-8`；远程仓库：`/data/yc/CS60003`；远程 Python：`/data/yc/miniconda/envs/llm-26-gpu`。
- 已按用户要求将 PyTorch AMP deprecated API 更新为 `torch.amp` 写法；当前运行中的旧进程不受影响，后续实验使用新写法。

## 下一步
提交并同步 `hw2/task3/stanford_unet/data.py` 与 `hw2/task3/configs/opt_*aug*.yaml`，远程验证后启动第二轮增强实验。

## 阻塞项
（无）

## 方案
`.helloagents/plans/202604291855_hw2_task3_resunet_attention_optimization/`

## 已标记技能
helloagents
