# HW3 Task1 3DGS/AIGC — 需求

确认后冻结，执行阶段不可修改。如需变更必须回到设计阶段重新确认。

## 核心目标
为 HW3 题目一保留一条真实高质量链路：真实物体 A 多视角/视频、真实背景多视角/视频、物体 B 文本到 3D、物体 C 单图到 3D，最终融合渲染为可报告的场景结果。

## 功能边界
- 必须采用 `hw3/task1/` 目录，结构对齐 HW1/HW2 的配置驱动入口。
- 必须保留 `README.md`、`RESULTS.md`、`configs/real_high_quality.yaml`、`requirements.txt`、`train.py`、`evaluate.py` 和内部 Python 包。
- 必须删除早期 AI smoke、程序化 proxy、`formal_ai_chain` 和报告素材链路。
- `train.py` 只支持 `task1.stage=real_high_quality`。
- plan 模式必须生成 7 个真实外部工具脚本；run 模式必须执行真实工具，不允许静默降级。
- ModelScope 只允许上传训练好的模型权重，不上传 config、summary、metrics、图片、GIF、报告素材或 proxy 文件。

## 非目标
- 不把低质量代理结果包装为正式交付。
- 不把 `hw3/task1/outputs/`、外部仓库、真实素材、数据集和权重纳入 Git。
- 不把 SwanLab / ModelScope 真实 key 写入 Git 跟踪文件。

## 技术约束
- 本地代码风格遵循 HW1/HW2：配置驱动、顶层 CLI、内部包封装、输出目录统一。
- 远程训练/重建使用 136：`/home/dell/yc/CS60003`，conda 环境 `qwen14b`。
- 运行时凭据通过 `.helloagents/secrets/hw3.env` 提供：`SWANLAB_API_KEY`、`MODELSCOPE_API_TOKEN`。

## 质量要求
- 本地和 136 都能 py_compile。
- 本地和 136 都能以 plan 模式生成真实链路脚本并通过 evaluate。
- `tools/verify_delivery_metadata.py` 通过。
- ModelScope 远端此前误传的 `hw3/task1/formal_ai_chain/` 非权重文件必须删除。
