# 项目状态

主线目标：HW2 Task1 Flowers102 预训练 CNN 微调、消融、注意力对比和远程正式实验已完成。

正在做什么：整理 Task1 完成状态；等待后续进入 HW2 Task2 或报告撰写。

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

下一步：若继续做 HW2，进入 Task2 前先准备 10–30 秒测试视频；若写报告，优先使用 `hw2/task1/RESULTS.md` 与远程 `hw2/task1/outputs/*/curves.png`。

阻塞项：
- 无

方案：
- `.helloagents/plans/202604291109_hw2_task1_flowers102/`
