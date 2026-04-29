# HW2 Task1 Flowers102 — 需求

确认后冻结，执行阶段不可修改。如需变更必须回到设计阶段重新确认。

## 核心目标
完成 HW2 Task1：在 102 Category Flower Dataset 上微调 ImageNet 预训练 CNN，形成可复现实验代码、正式远端实验结果、训练曲线、Accuracy 对比表和报告素材。

## 功能边界
- 使用 `hw2/Flowers102/` 中的官方数据文件：`jpg/`、`imagelabels.mat`、`setid.mat`。
- 严格使用 `setid.mat` 的官方 train / val / test 划分。
- 实现 ResNet-18 ImageNet 预训练 Baseline：替换输出层为 102 类，分类头较大学习率，backbone 较小学习率微调。
- 完成超参数对比：至少覆盖 2–3 组学习率 / epoch / batch size 组合。
- 完成预训练消融：同架构 ResNet-18，从随机初始化开始训练，与预训练 Baseline 对比。
- 完成注意力对比：优先实现 SE-ResNet-18，与 Baseline 比较 Accuracy。
- 保存每组实验的配置、日志、指标、曲线和 best checkpoint。
- 代码通过 Git 同步到远程 `135-3090-8` 后，在远程直接做正式实验。

## 非目标
- 不在本机执行 smoke test 或正式训练。
- 不使用 `scp` 手动覆盖远程代码；远程仅通过 Git 同步代码。
- 不把大型模型权重直接纳入 Git。
- 不在 Task1 中实现 Task2 检测跟踪或 Task3 分割。
- 不把随机初始化模型调到与预训练模型同等训练预算之外，避免消融实验失去可比性。

## 技术约束
- 语言：Python。
- 深度学习框架：PyTorch + torchvision。
- 本机环境：`conda run -n nlp`，仅用于编辑、文档和轻量静态检查。
- 远程主机：`ssh 135-3090-8`。
- 远程优先环境：`/data/yc/miniconda/envs/llm-26-gpu`；执行前必须确认 PyTorch、torchvision 和 CUDA 可用性。
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`；远程执行前必须检查 `user.name` 与 `user.email`。
- 正式实验以远程日志和指标为准。

## 质量要求
- 数据读取必须校验图片数量、标签数量、类别范围和官方划分数量。
- 训练脚本必须支持可复现实验配置、固定随机种子、自动保存 best checkpoint。
- 结果文件必须包含 `metrics.json`、`history.csv`、训练曲线和最终 `test_acc`。
- 代码不能依赖硬编码绝对路径，默认从仓库根目录解析数据。
- 验证主路径为远程正式验证：先检查环境和 Git 状态，再运行训练入口。
