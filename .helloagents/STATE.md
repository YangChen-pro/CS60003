# 项目状态

主线目标：HW2 Task3 Stanford Background U-Net 语义分割实验已完成。

正在做什么：Task3 三组正式训练、SwanLab 记录、ModelScope 最佳模型上传和仓库文档更新已完成；等待用户决定是否继续 HW2 报告整合或 Task2。

关键上下文：
- 题面文件：`hw2/hw2.md`
- Task3 数据集：`hw2/StanfordBackground/iccv09Data/`，715 对图像/语义标签。
- Task3 工程：`hw2/task3/`
- Task3 方案包：`.helloagents/plans/202604291645_hw2_task3_unet_segmentation/`
- 实现满足题面：从零手写 U-Net；不使用预训练；手写 Dice Loss；训练 CE / Dice / CE+Dice 三组。
- 划分：seed=42，train 572 / val 143；标签 `-1` unknown 在 loss 和指标中 ignore。
- 正式结果：CE `val_mIoU=0.648151`，Dice `val_mIoU=0.648211`，CE+Dice `val_mIoU=0.648970`。
- 最佳模型：`task3_unet_ce_dice`，best epoch 52，`val_mIoU=0.648970`，`val_pixel_acc=0.834739`。
- SwanLab 项目：`https://swanlab.cn/@youngchen/cs60003-hw2-task3`
- 最佳 SwanLab run：`https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/fbpzyv27p65mc0xiqksk3`
- ModelScope 仓库：`https://modelscope.cn/models/youngchen/CS60003/`
- ModelScope 最佳模型路径：`hw2/task3/unet_ce_dice/best.pt`
- 远程输出目录：`/data/yc/CS60003/hw2/task3/outputs/20260429_085805_task3_unet_ce_dice/`
- 远程主机：`ssh 135-3090-8`
- 远程环境：`/data/yc/miniconda/envs/llm-26-gpu`
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`

下一步：如继续 HW2，优先整合 Task1 + Task3 到最终 PDF 报告；也可以继续做 Task2。

阻塞项：
- 无

方案：
- `.helloagents/plans/202604291645_hw2_task3_unet_segmentation/`
