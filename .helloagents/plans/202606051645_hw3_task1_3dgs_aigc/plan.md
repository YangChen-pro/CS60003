# HW3 Task1 3DGS/AIGC — 实施规划

## 目标与范围
先把 Task1 做到“能运行、能验证、能同步到 136”的正式链路起点：建立标准工程骨架，并按用户最新要求直接使用 AI 生成物体 A/C 图作为正式输入，补齐物体 B，生成 A/B/C/背景融合场景和多视角渲染素材。

## 架构与实现策略
- `hw3/task1/train.py`：对齐 HW2 的训练入口，当前执行 `smoke_assets` 阶段。
- `hw3/task1/evaluate.py`：对齐 HW2 的评估入口，检查 smoke test 产物完整性。
- `hw3/task1/task1_3dgs_aigc/`：内部包，拆分 config、assets、smoke、utils。
- `configs/ai_generated_smoke.yaml`：图片输入 smoke test 配置。
- `configs/formal_ai_chain.yaml`：正式 AI 素材链路配置，输出 PLY、metrics、preview、turntable GIF 和报告素材。

## 领域语言
- 物体 A：AI 多视角火箭图，formal 阶段采样为 Gaussian/point-cloud 代理。
- 物体 B：文本到 3D 代理资产，由 prompt 程序化生成紫色发光水晶蘑菇。
- 物体 C：AI 单图木偶图，formal 阶段做单图挤出式 3D 代理。
- 背景：Mip-NeRF 360 风格桌面/墙面 Gaussian 背景代理。
- formal AI chain：按用户授权将 AI 素材作为正式输入跑通端到端融合链路。

## 完成定义
- `hw3/task1/` 工程结构与 HW2 task 目录一致。
- smoke test 能生成 `manifest.json`、`image_stats.csv`、`pairwise_yaw_diffs.csv`、`contact_sheet.png`、`summary.json`。
- 本地与 136 均通过 `train.py` 和 `evaluate.py`。
- formal AI chain 能生成 `fused_scene.ply`、`metrics.csv`、`asset_manifest.json`、`renders/fused_scene_preview.png`、`renders/fused_scene_turntable.gif`。
- 报告可引用素材落到 `hw3/task1/report_assets/formal_ai_chain/`。
- `qaMode=standard`，`qaFocus` 覆盖目录风格、输出边界、测试图语义和验证命令。

## 文件结构
```text
hw3/task1/
  README.md
  RESULTS.md
  configs/ai_generated_smoke.yaml
  configs/formal_ai_chain.yaml
  requirements.txt
  train.py
  evaluate.py
  task1_3dgs_aigc/
    __init__.py
    config.py
    assets.py
    smoke.py
    formal_chain.py
    geometry.py
    utils.py
```

## UI / 设计约束
不涉及 UI。

## 风险与验证
- 风险：AI 生成多视角不具备真实 COLMAP 几何一致性。处理：在 formal AI chain 中明确这是用户授权的可复现代理链路，不伪称真实实拍 3DGS。
- 风险：136 网络不稳定。处理：使用本机代理通道和本地 bundle/rsync 同步。
- 风险：输出误进 Git。处理：`.gitignore` 忽略 `hw3/task1/outputs/`、data、external。

## 决策记录
- [2026-06-05] Task1 目录采用 `hw3/task1/`，对齐 HW2 的 task 目录命名与结构，而不是 `task1_3dgs_aigc/` 顶层目录。
- [2026-06-05] 当前只做 asset smoke test；正式 3DGS/AIGC 链路分阶段接入。
