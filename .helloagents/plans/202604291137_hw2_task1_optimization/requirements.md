# HW2 Task1 Flowers102 优化 — 需求

确认后冻结，执行阶段不可修改。如需变更必须回到设计阶段重新确认。

## 核心目标
在已完成 HW2 Task1 Baseline 的基础上，尽可能提升 Flowers102 分类分数，并保留可用于报告的优化路径、对比结果和错例分析依据。

## 功能边界
- 保留当前最佳 Baseline：ResNet-18，`test_acc=0.6819`。
- 使用子代理并行分析错例、训练策略和代码风险，再由主代理汇总执行。
- 从多个维度冲分：数据增强、优化器、label smoothing、TTA、更强 ImageNet 预训练模型。
- 代码仍通过 Git 同步到远程 `135-3090-8`，正式实验仍在远程执行。
- 输出新结果到 `hw2/task1/RESULTS.md`，并更新 README 与知识库。

## 非目标
- 不在本机做 smoke test 或训练。
- 不提交大型 `.pt` / `.pth` 权重文件。
- 不改变 Flowers102 官方 train / val / test 划分。
- 不将 Task2 / Task3 纳入本轮优化。

## 技术约束
- 远程主机：`135-3090-8:/data/yc/CS60003`
- 远程环境：`/data/yc/miniconda/envs/llm-26-gpu`
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`
- 远程训练优先使用空闲 GPU 5/6/7。
- 现有输出目录仍位于远程 `hw2/task1/outputs/`，不进入 Git。

## 质量要求
- 新增配置必须可复现，并写明优化变量。
- 所有正式优化实验必须保存 `metrics.json`、`history.csv`、`curves.png` 和 `best.pt`。
- 最终汇总必须明确最佳模型、相比 Baseline 的提升幅度、失败或低收益实验的解释。
