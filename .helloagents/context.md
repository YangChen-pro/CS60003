# 项目上下文

## 概述
CS60003 是课程作业仓库，当前包含 HW1 与 HW2 两次作业的代码、实验记录、结果文档和项目知识库。HW1 已完成并收口；HW2 目前已完成 Task1 图像分类与 Task3 语义分割，Task2 目标检测与多目标跟踪仍待补。

## 技术栈
- 主要语言：Python。
- HW1：手写三层 MLP，CuPy 作为远端 GPU 数组后端；本地只做轻量编辑与测试。
- HW2 Task1：PyTorch / torchvision / timm 风格模型接口，完成 Flowers102 图像分类微调实验。
- HW2 Task2：计划使用 YOLOv8 或同类目标检测模型，并结合跟踪算法完成视频多目标跟踪和越线计数。
- HW2 Task3：PyTorch 基础 API 手写 U-Net / Attention U-Net、Dice Loss、mIoU 评估与可视化。
- 实验记录：SwanLab 记录训练曲线；SwanLab 页面为私有，报告中只使用导出的本地图片，不提供云端链接。
- 模型发布：训练权重统一发布到 ModelScope 仓库，按 `hw1/`、`hw2/task1/`、`hw2/task3/` 子目录组织。
- 报告：LaTeX，HW1/HW2 报告均使用 `/Users/yangchen/Documents/Latex_Project` 下的 `elegantpaper.cls` 风格。

## 架构
- `hw1/`：EuroSAT 三层 MLP 作业实现、训练/评估/搜索脚本、测试与最终报告资料。
- `hw2/task1/`：Flowers102 分类工程，包含 YAML 配置、训练/评估、SwanLab 历史曲线回放和结果文档。
- `hw2/task2/`：待建设，目标是 Road Vehicle Images 检测训练、视频跟踪、遮挡分析和越线计数。
- `hw2/task3/`：Stanford Background 语义分割工程，包含从零 U-Net 家族模型、loss、metrics、训练/评估、ModelScope 上传与结果文档。
- `.helloagents/`：项目知识库、模块说明、方案包、状态快照和变更历史。

## 领域语言
- **官方划分**：Flowers102 的 `setid.mat` 中 train/val/test 划分；Task1 使用 train 1020 / val 1020 / test 6149。
- **validation mIoU**：Task3 在固定 train/val split 上报告的验证集 mIoU，不等同于独立 test set 指标。
- **SwanLab 私有记录**：只用于内部实验追踪和导出报告图片；报告正文不放 SwanLab 云端 URL。
- **ModelScope 仓库**：统一公网权重发布位置；报告中用“仓库”作为可点击链接，不明文展示仓库名。

## 目录结构
- `README.md`：仓库总说明。
- `hw1/`：HW1 代码、题面、报告 PDF、测试与模型下载脚本。
- `hw2/hw2.md`：HW2 题面要求。
- `hw2/Flowers102/`：Flowers102 数据集。
- `hw2/RoadVehicleImages/`：Road Vehicle Images 数据集。
- `hw2/StanfordBackground/`：Stanford Background 数据集。
- `hw2/task1/`：HW2 Task1 分类实验工程与结果。
- `hw2/task3/`：HW2 Task3 分割实验工程与结果。
- `/Users/yangchen/Documents/Latex_Project/CS60003_HW1_Report/`：HW1 正式报告工程。
- `/Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report/`：HW2 当前报告工程；`out/hw2.pdf` 已编译通过。

## 模块文档
- [HW1 模块](modules/hw1.md)
- [HW2 模块](modules/hw2.md)

## 最近变更
见 [CHANGELOG.md](CHANGELOG.md)
