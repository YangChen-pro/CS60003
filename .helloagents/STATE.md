# 恢复快照

## 主线目标
HW2 Task3 在 hw2.md 合规范围内继续单模型冲分已完成；最终超过 0.7 validation mIoU，但 0.75 未达到。

## 正在做什么
Task3 结构优化、几何增强、多 seed 实验、EMA/mountain-aware 消融、0.75 激进单模型尝试、多尺度 TTA 复评、ModelScope 上传和文档更新已完成；Task1 SwanLab 云端已清理为报告必要 run；等待用户决定是否整合 HW2 报告或继续 Task2。

## 关键上下文
- Task3 基础最佳：`task3_unet_ce_dice`，`val_mIoU=0.648970`，`val_pixel_acc=0.834739`。
- Task3 最终最佳：`task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta`，best epoch 113，`val_mIoU=0.701053`，`val_pixel_acc=0.864931`。
- 提升幅度：相对基础最佳 `+0.052083` mIoU；相对上一轮 U-Net b64 + TTA 最佳 `+0.035964` mIoU。
- 合规边界：从零训练手写 Attention U-Net；未使用 SAM、DeepLab、SegFormer、torchvision segmentation models、预训练权重或现成分割网络；固定 train 572 / val 143 split。
- 最终方法：CE+Dice，base_channels=64，256x320，dropout 0.05，random scale crop，horizontal-flip TTA，multi-scale TTA `[0.6, 0.8, 1.0, 1.2, 1.4]`。
- 单模型继续冲分消融：EMA + mountain-aware sampling / rare-class crop run `7jatyl6ez1o5rs9vgzk0w`，`val_mIoU=0.700604`，`mountain IoU=0.309850`，整体未超过最终 seed7 checkpoint。
- 0.75 单模型尝试：实现 U-Net++、scSE、ASPP bridge、deep supervision、CE+Dice+Lovasz、高分辨率和更大 batch；最好为 `task3_unetpp_b64_384_lovasz_ds_tta`，训练中最佳 `val_mIoU=0.693158`，多尺度复评 `val_mIoU=0.6953`，未超过最终最佳。
- SwanLab 最终训练 run：`https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/fw55rpcbgagnmqbcz0q90`。
- Task1 SwanLab 项目：`https://swanlab.cn/@youngchen/cs60003-hw2-task1`；云端保留 8 个报告必要 run，链接见 `hw2/task1/SWANLAB_RUNS.md`。
- ModelScope 仓库：`https://modelscope.cn/models/youngchen/CS60003/`。
- ModelScope 最终模型路径：`hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/best.pt`。
- 远程最终模型包：`/data/yc/CS60003/hw2/task3/outputs/final_task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/`。
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`。

## 下一步
如继续 HW2，优先整合 Task1 + Task3 到最终 PDF 报告；也可以开始 Task2。

## 阻塞项
（无）

## 方案
`.helloagents/plans/202604291855_hw2_task3_resunet_attention_optimization/`

## 已标记技能
helloagents
