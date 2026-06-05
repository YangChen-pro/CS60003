# HW3 Task1 3DGS/AIGC — 实施规划

## 目标与范围
先把 Task1 做到“能运行、能验证、能同步到 136”的起点：建立标准工程骨架，使用 AI 生成测试图完成 asset smoke test，并为后续 COLMAP/3DGS/AIGC 分阶段接入保留接口。

## 架构与实现策略
- `hw3/task1/train.py`：对齐 HW2 的训练入口，当前执行 `smoke_assets` 阶段。
- `hw3/task1/evaluate.py`：对齐 HW2 的评估入口，检查 smoke test 产物完整性。
- `hw3/task1/task1_3dgs_aigc/`：内部包，拆分 config、assets、smoke、utils。
- `configs/ai_generated_smoke.yaml`：当前 smoke test 配置；未来新增 COLMAP/3DGS/threestudio/Zero123 配置。

## 领域语言
- 物体 A：多视角重建物体，最终必须来自手机实拍。
- 物体 B：文本到 3D 生成资产，计划使用 threestudio / SDS 或等价路线。
- 物体 C：单图到 3D 物体，最终必须来自手机实拍单图。
- 背景：Mip-NeRF 360 场景的 3DGS 重建。
- smoke test：验证流程，不作为最终实验结果。

## 完成定义
- `hw3/task1/` 工程结构与 HW2 task 目录一致。
- smoke test 能生成 `manifest.json`、`image_stats.csv`、`pairwise_yaw_diffs.csv`、`contact_sheet.png`、`summary.json`。
- 本地与 136 均通过 `train.py` 和 `evaluate.py`。
- `qaMode=standard`，`qaFocus` 覆盖目录风格、输出边界、测试图语义和验证命令。

## 文件结构
```text
hw3/task1/
  README.md
  RESULTS.md
  configs/ai_generated_smoke.yaml
  requirements.txt
  train.py
  evaluate.py
  task1_3dgs_aigc/
    __init__.py
    config.py
    assets.py
    smoke.py
    utils.py
```

## UI / 设计约束
不涉及 UI。

## 风险与验证
- 风险：AI 生成多视角不具备真实几何一致性。处理：明确只用于 smoke test。
- 风险：136 网络不稳定。处理：使用本机代理通道和本地 bundle/rsync 同步。
- 风险：输出误进 Git。处理：`.gitignore` 忽略 `hw3/task1/outputs/`、data、external。

## 决策记录
- [2026-06-05] Task1 目录采用 `hw3/task1/`，对齐 HW2 的 task 目录命名与结构，而不是 `task1_3dgs_aigc/` 顶层目录。
- [2026-06-05] 当前只做 asset smoke test；正式 3DGS/AIGC 链路分阶段接入。
