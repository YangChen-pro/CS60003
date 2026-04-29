# 项目状态

主线目标：继续优化 HW2 Task1 Flowers102 分数，目标是在当前 Baseline `test_acc=0.6819` 上进一步提升。

正在做什么：已启动子代理分析错例、训练策略和代码风险；主代理正在实现优化训练能力与配置。

关键上下文：
- 题面文件：`hw2/hw2.md`
- Task1 数据集：`hw2/Flowers102/`，含 `jpg/`、`imagelabels.mat`、`setid.mat`、`README.txt`
- 用户确认：不在本机 smoke test；本机只做编辑和文档整理。
- 用户确认：代码通过 Git 同步到远程，远程机器使用前检查 Git 身份是否与本机一致。
- 本机轻量环境：`conda run -n nlp`
- 远程主机：`ssh 135-3090-8`
- 远程优先环境：`/data/yc/miniconda/envs/llm-26-gpu`，执行前需确认 PyTorch / torchvision / CUDA 可用。
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`
- 本次代码提交：`91b6b8beadeb8fc7f5e8e021b1cdd65e718af43e`
- Task1 正式结果：`hw2/task1/RESULTS.md`
- 当前最佳：Baseline ResNet-18，`best_val_acc=0.7186`，`test_acc=0.6819`
- 优化方案：`.helloagents/plans/202604291137_hw2_task1_optimization/`
- 本轮优化方向：AdamW、label smoothing、RandAugment/RandomErasing、ResNet-34/50、EfficientNet-B0、ConvNeXt-Tiny、TTA

下一步：提交优化代码，通过 Git 同步远程 `135-3090-8` 后运行正式冲分实验。

阻塞项：
- 无

方案：
- `.helloagents/plans/202604291137_hw2_task1_optimization/`
