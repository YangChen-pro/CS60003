# HW3 Task1 3DGS/AIGC — 需求

确认后冻结，执行阶段不可修改。如需变更必须回到设计阶段重新确认。

## 核心目标
为 HW3 题目一建立和 HW1/HW2 风格一致的工程骨架，并按用户最新要求使用现有 AI 生成物体 A/C 图片作为正式输入，补齐物体 B 的可复现文本到 3D 代理资产，跑通 A/B/C/背景统一表达、融合和多视角渲染链路。

## 功能边界
- 必须采用 `hw3/task1/` 目录，内部结构对齐 `hw2/task1`、`hw2/task2`、`hw2/task3`。
- 必须提供 `README.md`、`RESULTS.md`、`configs/`、`requirements.txt`、`train.py`、`evaluate.py` 和内部 Python 包。
- smoke 阶段必须能验证 8 张物体 A 多视角图和 1 张物体 C 单图，并输出 manifest、图像统计、相邻视角差异和 contact sheet。
- formal 阶段必须把 AI 生成物体 A/C 图片作为正式输入，生成物体 A 多视角 Gaussian/point-cloud 代理、物体 B 文本到 3D 代理、物体 C 单图到 3D 代理、背景 Gaussian 代理、融合场景 PLY 和多视角渲染 GIF。
- 必须保留报告可直接引用的轻量素材到 `hw3/task1/report_assets/formal_ai_chain/`。

## 非目标
- 不强行安装或运行重型 COLMAP、3DGS、threestudio、Zero123 或 Blender；当前正式链路用可复现代理跑通端到端集成。
- 不把 `hw3/task1/outputs/`、外部仓库、数据集和权重纳入 Git。
- 不把 SwanLab / ModelScope 真实 key 写入 Git 跟踪文件。

## 技术约束
- 本地代码风格遵循 HW1/HW2：配置驱动、顶层 CLI、内部包封装、输出目录统一。
- 远程训练/重建使用 136：`/home/dell/yc/CS60003`，conda 环境 `qwen14b`。
- 运行时凭据通过 `.helloagents/secrets/hw3.env` 提供：`SWANLAB_API_KEY`、`MODELSCOPE_API_TOKEN`。

## 质量要求
- 本地和 136 都能运行 smoke test 与 formal AI chain。
- `python3 -m py_compile` 通过。
- `tools/verify_delivery_metadata.py` 通过。
- QA 需要检查 Task1 骨架是否和 HW1/HW2 风格一致、输出是否不进 Git、AI 素材正式链路是否有清晰限制说明、报告素材是否可引用。
