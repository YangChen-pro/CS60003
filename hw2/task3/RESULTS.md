# HW2 Task3 实验结果

## 当前状态

Task3 工程代码已规划并进入实现阶段。正式远程实验完成后，本文件将汇总三组损失配置的验证集 mIoU、pixel accuracy、per-class IoU、SwanLab 链接和 ModelScope 权重路径。

## 实验配置

| 实验 | 损失函数 | 初始化 | 数据集 | 指标 |
|---|---|---|---|---|
| `task3_unet_ce` | Cross-Entropy | 随机初始化 | Stanford Background | val mIoU / pixel accuracy |
| `task3_unet_dice` | 手写 Dice Loss | 随机初始化 | Stanford Background | val mIoU / pixel accuracy |
| `task3_unet_ce_dice` | Cross-Entropy + 手写 Dice Loss | 随机初始化 | Stanford Background | val mIoU / pixel accuracy |

## 正式结果

待远程 135 正式实验完成后更新。
