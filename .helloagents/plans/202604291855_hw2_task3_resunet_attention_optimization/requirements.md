# HW2 Task3 ResUNet / Attention U-Net 冲分 — 需求

确认后冻结，执行阶段不可修改。如需变更必须回到设计阶段重新确认。

## 核心目标
在 `hw2.md` 允许范围内继续优化 HW2 Task3 Stanford Background 语义分割验证 mIoU，尽量冲击 `0.7+`，不承诺 `0.8+`。

## 功能边界
- 必须使用从零训练的手写 U-Net 家族结构。
- 必须保留固定 train/val split：train 572、val 143。
- 必须使用远程 135 正式训练，不在本机 smoke test。
- 若刷新最佳结果，上传到 ModelScope `youngchen/CS60003` 下 `hw2/task3/...` 子目录。

## 非目标
- 不使用 SAM、DeepLab、SegFormer、torchvision segmentation models 或任何预训练权重。
- 不改变验证集划分，不用验证集参与训练。
- 不为了刷分隐瞒实验条件或选择性报告。

## 技术约束
- 远程主机：`135-3090-8`，仓库 `/data/yc/CS60003`。
- 远程 Python 环境：`/data/yc/miniconda/envs/llm-26-gpu`。
- 使用 Git 同步本机和远程代码，远程 Git 身份需与本机一致。
- SwanLab 和 ModelScope token 已按用户要求记录在 `.helloagents/modules/hw2.md`。

## 质量要求
- 本机至少通过 Python 编译检查。
- 远程必须通过模型 shape 检查后再正式训练。
- 每个正式实验必须产生 `metrics.json`、`history.csv`、`best.pt` 和 SwanLab 记录。
- 文档、知识库和 ModelScope 路径必须与最佳结果一致。
