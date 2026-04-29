# HW2 Task3 U-Net 继续冲分优化 — 需求

## 核心目标
在 `hw2.md` 允许范围内继续提高 Task3 Stanford Background 语义分割的验证集 mIoU，基准为当前最佳 CE+Dice `val_mIoU=0.648970`。

## 功能边界
- 仍然从零训练手写 U-Net，不使用预训练权重。
- 不使用现成分割网络或第三方 segmentation model。
- 保留原三组 CE / Dice / CE+Dice 正式结果，新增优化实验作为额外冲分结果。
- 正式实验仍在远程 `135-3090-8` 执行，并接入 SwanLab。
- 若刷新最优结果，更新 `RESULTS.md`、README、知识库，并上传最佳权重到 ModelScope。

## 非目标
- 不改变题面数据集和验证集口径。
- 不用验证集参与训练。
- 不进入 Task2。

## 技术约束
- 代码目录：`hw2/task3/`
- 远程环境：`/data/yc/miniconda/envs/llm-26-gpu`
- Git 身份：`YangChen-pro <1369792882@qq.com>`

## 质量要求
- 新增配置必须能复现。
- 指标仍以验证集 mIoU 为主。
- SwanLab 曲线和本地 `curves.png` 必须有明确坐标轴。
