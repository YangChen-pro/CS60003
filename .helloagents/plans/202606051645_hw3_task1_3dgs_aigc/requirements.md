# HW3 Task1 3DGS/AIGC — 需求

确认后冻结，执行阶段不可修改。如需变更必须回到设计阶段重新确认。

## 核心目标
为 HW3 题目一建立和 HW1/HW2 风格一致的工程骨架，并开始可验证的 Task1 pipeline：先用 AI 生成的物体 A/C 测试图跑 smoke test，后续替换为真实手机拍摄素材、文本到 3D 资产和 Mip-NeRF 360 背景。

## 功能边界
- 必须采用 `hw3/task1/` 目录，内部结构对齐 `hw2/task1`、`hw2/task2`、`hw2/task3`。
- 必须提供 `README.md`、`RESULTS.md`、`configs/`、`requirements.txt`、`train.py`、`evaluate.py` 和内部 Python 包。
- 当前阶段必须能验证 8 张物体 A 多视角图和 1 张物体 C 单图，并输出 manifest、图像统计、相邻视角差异和 contact sheet。
- 后续正式实验必须明确 AI 生成测试图不是最终素材。

## 非目标
- 当前阶段不直接完成 COLMAP、3DGS、threestudio、Zero123 或 Blender 正式重建。
- 不把 `hw3/task1/outputs/`、外部仓库、数据集和权重纳入 Git。
- 不把 SwanLab / ModelScope 真实 key 写入 Git 跟踪文件。

## 技术约束
- 本地代码风格遵循 HW1/HW2：配置驱动、顶层 CLI、内部包封装、输出目录统一。
- 远程训练/重建使用 136：`/home/dell/yc/CS60003`，conda 环境 `qwen14b`。
- 运行时凭据通过 `.helloagents/secrets/hw3.env` 提供：`SWANLAB_API_KEY`、`MODELSCOPE_API_TOKEN`。

## 质量要求
- 本地和 136 都能运行 smoke test。
- `python3 -m py_compile` 通过。
- `tools/verify_delivery_metadata.py` 通过。
- QA 需要检查 Task1 骨架是否和 HW1/HW2 风格一致、输出是否不进 Git、测试图是否只作为 smoke test 使用。
