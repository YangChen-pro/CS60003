# 项目状态

主线目标：继续优化 HW2 Task3 Stanford Background U-Net 验证集 mIoU。

正在做什么：已实现 Task3 优化能力和三组优化配置；下一步提交并同步远程 135，运行正式优化实验。

关键上下文：
- 当前基准：Task3 CE+Dice `val_mIoU=0.648970`，`val_pixel_acc=0.834739`。
- 新方案包：`.helloagents/plans/202604291718_hw2_task3_optimization/`
- 优化实现：U-Net 支持可选 Dropout2d；loss 支持 class weights；评估支持 horizontal-flip TTA；新增 class weight 统计。
- 新配置：`opt_ce_dice_wide_tta.yaml`、`opt_ce_dice_wide_weighted_tta.yaml`、`opt_ce_dice_b64_tta.yaml`。
- 合规边界：仍从零训练手写 U-Net，不使用预训练和现成分割网络；验证集不参与训练。
- 远程主机：`ssh 135-3090-8`
- 远程环境：`/data/yc/miniconda/envs/llm-26-gpu`

下一步：提交代码，Git 同步到远程 135，运行远程检查与优化正式实验。

阻塞项：
- 无

方案：
- `.helloagents/plans/202604291718_hw2_task3_optimization/`
