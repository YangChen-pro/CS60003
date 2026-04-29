# 项目状态

主线目标：HW2 Task3 Stanford Background U-Net 继续冲分优化已完成。

正在做什么：Task3 优化实验、SwanLab 记录、ModelScope 最佳模型上传和文档更新已完成；等待用户决定是否整合 HW2 报告或继续 Task2。

关键上下文：
- Task3 基础最佳：`task3_unet_ce_dice`，`val_mIoU=0.648970`，`val_pixel_acc=0.834739`。
- Task3 优化最佳：`task3_unet_ce_dice_b64_tta`，best epoch 67，`val_mIoU=0.665089`，`val_pixel_acc=0.842011`。
- 提升幅度：`+0.016118` validation mIoU。
- 合规边界：仍从零训练手写 U-Net；未使用预训练或现成分割网络；验证集只用于评估和选模。
- 优化策略：base_channels=64，输入 `256x320`，CE+Dice，轻量 dropout，horizontal-flip TTA。
- 方案包：`.helloagents/plans/202604291718_hw2_task3_optimization/`
- SwanLab 项目：`https://swanlab.cn/@youngchen/cs60003-hw2-task3`
- 优化最佳 SwanLab run：`https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/9odmxzsyxzeo44c8joex7`
- ModelScope 仓库：`https://modelscope.cn/models/youngchen/CS60003/`
- ModelScope 优化最佳模型路径：`hw2/task3/unet_ce_dice_b64_tta/best.pt`
- 远程输出目录：`/data/yc/CS60003/hw2/task3/outputs/20260429_092021_task3_unet_ce_dice_b64_tta/`
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`

下一步：如继续 HW2，优先整合 Task1 + Task3 到最终 PDF 报告；也可以继续 Task2。

阻塞项：
- 无

方案：
- `.helloagents/plans/202604291718_hw2_task3_optimization/`
