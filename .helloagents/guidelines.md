# 项目约定

<!-- 只记录从代码看不出来的约定。AI 能从代码推断的风格不需要写在这里。 -->

## 编码风格
- 作业实现必须遵守对应 `hw*.md` 题面约束；文档与代码不一致时，以代码和实际实验产物为准。
- HW1 主体训练逻辑保持手写，不回退到 PyTorch/TensorFlow/JAX 等现成训练框架。
- HW2 Task2 以 YOLOv8 训练和 ByteTrack 跟踪为当前实现主线；修改跟踪逻辑时必须同步维护 `hw2/task2/tests/test_track_video_logic.py`。
- HW2 Task3 仍以从零手写 U-Net / Dice Loss 为主线；不使用预训练权重、SAM、DeepLab、SegFormer 或 torchvision segmentation models 作为 Task3 主体。
- HW3 当前只有题面，不能把 3DGS/AIGC/LeRobot/ACT 等题目要求写成已完成实现；开始实现前先明确选题路径、数据来源、算力环境和交付结构。

## 命名规范
- HW2 实验目录和 run 名使用 `task3_attention_unet_b64_aug_seed7_tta`、`task2_yolov8s_baseline` 这类可复现实验名。
- HW3 后续若新增工程目录，优先按选题建立清晰入口，例如 `hw3/task1_3dgs_aigc/` 或 `hw3/task2_lerobot_act/`，不要把不同题目方向混在同一脚本目录。
- 报告中长路径不直接完整塞入表格；主表用短路径、省略号或可点击“仓库”链接，完整路径放文档或结果文件。
- Task2 跟踪输出中的 `raw_track_id` 指 ByteTrack 原始 ID，`display_id` 指稳定展示和计数 ID；文档和代码注释中不要混用。

## Git 工作流
- 本机与远程 `135-3090-8` 通过 Git 同步代码。
- 远程正式实验前检查 Git 身份应为 `YangChen-pro <1369792882@qq.com>`。
- 不自动远程 push，除非用户明确要求。
- 用户曾明确允许将临时 SwanLab / ModelScope key 写入 `.helloagents` 并 Git 同步；作业完成后需要删除这些临时 key。后续新增凭据时优先使用环境变量，不再写入仓库文件。
- 当前项目按用户要求只保留 Codex 规则载体 `AGENTS.md`；`CLAUDE.md` 与 `.gemini/GEMINI.md` 不维护，除非用户后续明确要求恢复。

## 测试
- 本机主要做轻量测试、语法检查、文档维护和 LaTeX 编译。
- 训练类正式实验默认在远程 `135-3090-8` 的 `/data/yc/miniconda/envs/llm-26-gpu` 中执行。
- 用户已明确：HW2 训练不需要本机 smoke test，可以直接在远程正式实验环境运行。
- Task2 跟踪逻辑可本地运行单元测试：`python3 -m pytest hw2/task2/tests` 或 `python3 -m unittest discover -s hw2/task2/tests`。
- 当前 HelloAGENTS 验证入口见 `.helloagents/verify.yaml`，以方案包元数据检查与 Python 语法编译为主。

## 报告规范
- SwanLab 实验记录为私有时，报告正文不提供 SwanLab 云端链接；需要图时使用从 SwanLab 或远程正式实验产物导出的本地图片。
- ModelScope / GitHub 仓库可在报告中用“仓库”文字做可点击链接，避免直接明文显示仓库名。
- 生成 PDF 后应渲染页面图片做视觉检查，尤其检查图表大小、路径溢出、长表格和图例可读性。
- HW2 最终提交前必须确认 Task2 检测指标、跟踪可视化、遮挡分析和越线计数已经写入报告，而不是只存在于代码目录。
- HW3 期末提交必须覆盖 PDF 实验报告、Public GitHub Repository 和模型权重下载链接；报告首页必须包含成员姓名、学号和分工。
