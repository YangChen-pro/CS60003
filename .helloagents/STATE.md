# 项目状态

主线目标：完成 HW2 Task1 Flowers102 预训练 CNN 微调、消融、注意力对比和远程正式实验。

正在做什么：已确认方案，正在实现 `hw2/task1/` 工程并准备通过 Git 同步到远程 `135-3090-8`。

关键上下文：
- 题面文件：`hw2/hw2.md`
- Task1 数据集：`hw2/Flowers102/`，含 `jpg/`、`imagelabels.mat`、`setid.mat`、`README.txt`
- 用户确认：不在本机 smoke test；本机只做编辑和文档整理。
- 用户确认：代码通过 Git 同步到远程，远程机器使用前检查 Git 身份是否与本机一致。
- 本机轻量环境：`conda run -n nlp`
- 远程主机：`ssh 135-3090-8`
- 远程优先环境：`/data/yc/miniconda/envs/llm-26-gpu`，执行前需确认 PyTorch / torchvision / CUDA 可用。
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`

下一步：实现 `hw2/task1/` 数据管线、模型、训练评估入口和配置，然后提交推送并在远程拉取运行正式实验。

阻塞项：
- 无

方案：
- `.helloagents/plans/202604291109_hw2_task1_flowers102/`
