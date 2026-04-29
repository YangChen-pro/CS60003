# 项目状态

主线目标：HW2 Task1 Flowers102 冲分优化已完成，最佳 `test_acc=0.9608`。

正在做什么：整理 Task1 优化完成状态；等待后续进入 Task2 或报告撰写。

关键上下文：
- 题面文件：`hw2/hw2.md`
- Task1 数据集：`hw2/Flowers102/`，含 `jpg/`、`imagelabels.mat`、`setid.mat`、`README.txt`
- 用户确认：不在本机 smoke test；本机只做编辑和文档整理。
- 用户确认：代码通过 Git 同步到远程，远程机器使用前检查 Git 身份是否与本机一致。
- 本机轻量环境：`conda run -n nlp`
- 远程主机：`ssh 135-3090-8`
- 远程优先环境：`/data/yc/miniconda/envs/llm-26-gpu`，执行前需确认 PyTorch / torchvision / CUDA 可用。
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`
- 当前同步状态：本机 / GitHub / 远程仓库以最新 `main` 为准，具体提交用 `git rev-parse HEAD` 核对。
- Task1 正式结果：`hw2/task1/RESULTS.md`
- 原始 Baseline：ResNet-18，`best_val_acc=0.7186`，`test_acc=0.6819`
- 优化方案：`.helloagents/plans/202604291137_hw2_task1_optimization/`
- 本轮优化方向：AdamW、label smoothing、RandAugment/RandomErasing、ResNet-34/50、EfficientNet-B0、ConvNeXt-Tiny、TTA
- 最佳优化结果：ConvNeXt-Tiny，`best_val_acc=0.9784`，`test_acc=0.9608`
- 相比原始 Baseline `test_acc=0.6819` 提升 `+27.89` 个百分点。
- SwanLab 要求：后续 Task1 复现实验或报告补充需接入 SwanLab；用户已明确要求将临时 API key 写入 `.helloagents/modules/hw2.md` 并允许 Git 同步，作业完成后删除。

下一步：若写报告，先补 SwanLab 可视化截图或链接，再使用 `hw2/task1/RESULTS.md` 和远程最佳产物 `hw2/task1/outputs/20260429_034831_opt_convnext_tiny_strong/`；若继续 HW2，进入 Task2 前先准备 10–30 秒测试视频。

阻塞项：
- 无

方案：
- `.helloagents/plans/202604291137_hw2_task1_optimization/`
