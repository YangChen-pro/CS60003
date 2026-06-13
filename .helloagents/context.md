# 项目上下文

## 概述
CS60003 是“深度学习与空间智能”课程作业仓库，当前包含 HW1、HW2 与 HW3。HW1 已完成 EuroSAT 三层 MLP；HW2 已包含 Flowers102 分类、Road Vehicle 检测/跟踪、Stanford Background 语义分割三条工程线；HW3 已包含 Task1 真实高质量 3DGS/AIGC 链路和 Task2 LeRobot ACT 跨环境泛化实验。

## 技术栈
- 主要语言：Python。
- HW1：CuPy 作为远端 GPU 数组后端；前向传播、交叉熵、反向传播、SGD、学习率衰减、L2 正则和 dropout 均为手写实现。
- HW2 Task1：PyTorch / torchvision，使用 ResNet、SE-ResNet、EfficientNet、ConvNeXt 等分类模型做 Flowers102 微调与消融。
- HW2 Task2：Ultralytics YOLOv8、ByteTrack、OpenCV、PyTorch；完成 Road Vehicle Images 检测训练、评估、视频跟踪、遮挡重连和越线计数。
- HW2 Task3：PyTorch 基础 API 手写 U-Net / Attention U-Net、Dice Loss、mIoU 评估、多尺度 TTA 和可视化。
- HW3 Task1：3DGS / COLMAP / Nerfstudio / threestudio / Zero123 / Blender 资产融合。
- HW3 Task2：LeRobot `ACTPolicy` / CALVIN LeRobot 数据 / PyTorch DDP / SwanLab / ModelScope。
- 配置与实验：YAML 配置、Matplotlib 曲线、SwanLab 实验记录、ModelScope 权重发布。
- 报告：LaTeX 报告工程位于 `/Users/yangchen/Documents/Latex_Project`。

## 架构
- `hw1/`：EuroSAT 三层 MLP 作业实现、训练/评估/搜索脚本、测试与最终报告资料。
- `hw2/task1/`：Flowers102 分类工程，包含 YAML 配置、数据加载、模型构建、训练/评估、SwanLab 回放和结果文档。
- `hw2/task2/`：Road Vehicle Images 检测与跟踪工程，包含 YOLOv8 训练/评估、ByteTrack 跟踪、稳定 display ID、遮挡重连、越线计数、快照/摘要输出和逻辑测试。
- `hw2/task3/`：Stanford Background 语义分割工程，包含从零 U-Net 家族模型、loss、metrics、训练/评估、ModelScope 上传与结果文档。
- `hw3/`：期末作业目录，包含 `hw3.md` / `hw3.pdf` 题面、`task1/` 3DGS/AIGC 工程和 `task2/` ACT/CALVIN 工程。
- `.helloagents/`：项目知识库、模块说明、方案包、状态快照和变更历史。

## 领域语言
- **官方划分**：Flowers102 的 `setid.mat` 中 train/val/test 划分；Task1 使用 train 1020 / val 1020 / test 6149。
- **Road Vehicle Images**：Task2 的 21 类车辆检测数据集，当前 `train=2704`、`valid=300`。
- **mAP50 / mAP50-95**：Task2 检测模型的主要指标；`mAP50` 按 IoU 0.5 统计，`mAP50-95` 为 COCO 风格多阈值平均。
- **raw track ID / display ID**：ByteTrack 原始 ID 可能因遮挡跳变；`track_video.py` 用 display ID 稳定显示和计数。
- **越线计数**：Task2 跟踪脚本支持横线或竖线，记录已越线 display ID，并用短时运动预测处理遮挡期间跨线。
- **validation mIoU**：Task3 在固定 train/val split 上报告的验证集 mIoU，不等同于独立 test set 指标。
- **3DGS**：HW3 题目一中的 3D Gaussian Splatting，用于真实物体与背景场景重建；工程位于 `hw3/task1/`。
- **ACT**：HW3 题目二中的 Action Chunking Transformer，用于 LeRobot / CALVIN 跨环境策略泛化；工程位于 `hw3/task2/`。
- **CALVIN splitA/B/C/D**：Task2 使用老师划分好的 LeRobot 数据；`splitA=A`、`splitB=B`、`splitC=C`、`splitD=D`，正式评估在未见过的 `splitD`。
- **SwanLab 记录**：用于内部实验追踪和曲线导出；报告正文默认不放私有云端 URL。
- **ModelScope 仓库**：统一公网权重发布位置；报告中用“仓库”作为可点击链接，不明文展示仓库名。

## 目录结构
- `README.md`：仓库总说明。
- `hw1/`：HW1 代码、题面、报告 PDF、测试与模型下载脚本。
- `hw2/hw2.md`：HW2 题面要求。
- `hw2/Flowers102/`：Flowers102 数据集。
- `hw2/RoadVehicleImages/`：Road Vehicle Images 数据集。
- `hw2/StanfordBackground/`：Stanford Background 数据集。
- `hw2/task1/`：HW2 Task1 分类实验工程与结果。
- `hw2/task2/`：HW2 Task2 检测、跟踪、越线计数工程与结果。
- `hw2/task3/`：HW2 Task3 分割实验工程与结果。
- `hw3/hw3.md`：HW3 期末作业题面。
- `hw3/task1/`：HW3 Task1 真实高质量 3DGS/AIGC 工程。
- `hw3/task2/`：HW3 Task2 LeRobot ACT 跨环境泛化工程；正式运行在 135 的 `/data/yc/miniconda/envs/llm-26-gpu`。
- `tools/verify_delivery_metadata.py`：历史方案包元数据验证脚本。
- `/Users/yangchen/Documents/Latex_Project/CS60003_HW1_Report/`：HW1 正式报告工程。
- `/Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report/`：HW2 报告工程；最新 Task2 内容是否写入需单独复核。

## 模块文档
- [HW1 模块](modules/hw1.md)
- [HW2 模块](modules/hw2.md)
- [HW3 模块](modules/hw3.md)

## 最近变更
见 [CHANGELOG.md](CHANGELOG.md)
