# 模块索引

> 通过此文件快速定位模块文档。

## 模块清单

| 模块 | 职责 | 状态 | 文档 |
|------|------|------|------|
| hw1 | 记录 HW1 的作业要求、环境选择、后端决策、最终交付约定与证据链 | ✅ | [hw1.md](./hw1.md) |
| hw2 | 记录 HW2 三个任务的数据集、实验工程、关键指标、SwanLab/ModelScope/报告约束 | 🚧 | [hw2.md](./hw2.md) |
| hw3 | 记录期末作业 Task1 3DGS/AIGC 与 Task2 LeRobot ACT/CALVIN 的工程、远程环境、实验结果和发布位置 | 🚧 | [hw3.md](./hw3.md) |

## 模块依赖关系

```text
hw1 → 远端主机 135-3090-8
hw1 → llm-26-gpu 环境
hw1 → CuPy 唯一训练后端
hw1 → README / outputs / PDF 报告一致性

hw2 → Flowers102 / RoadVehicleImages / StanfordBackground 本地数据集
hw2-task1 → PyTorch / torchvision / SwanLab 私有记录 / ModelScope 权重
hw2-task2 → Ultralytics YOLOv8 / ByteTrack / OpenCV / SwanLab / 越线计数测试
hw2-task3 → PyTorch 基础 API / 手写 U-Net / 手写 Dice Loss / SwanLab 私有记录 / ModelScope 权重
hw2-report → /Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report

hw3 → hw3/hw3.md 期末题面
hw3-task1 → 3DGS / COLMAP / Nerfstudio / threestudio / Zero123 / Blender / ModelScope 权重
hw3-task2 → 135-3090-8 / llm-26-gpu / LeRobot ACT / CALVIN splitA+B+C→D / SwanLab / ModelScope hw3/task2
```

## 状态说明
- ✅ 稳定
- 🚧 开发中
- 📝 规划中
