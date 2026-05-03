# 模块索引

> 通过此文件快速定位模块文档。

## 模块清单

| 模块 | 职责 | 状态 | 文档 |
|------|------|------|------|
| hw1 | 记录 HW1 的作业要求、环境选择、后端决策、最终交付约定与证据链 | ✅ | [hw1.md](./hw1.md) |
| hw2 | 记录 HW2 期中作业的数据集、Task1/Task3 实验、Task2 待办、SwanLab/ModelScope/报告约束 | 🚧 | [hw2.md](./hw2.md) |

## 模块依赖关系

```text
hw1 → 远端主机 135-3090-8
hw1 → llm-26-gpu 环境
hw1 → CuPy 唯一训练后端
hw1 → README / outputs / PDF 报告一致性

hw2 → Flowers102 / RoadVehicleImages / StanfordBackground 本地数据集
hw2-task1 → PyTorch / torchvision / SwanLab 私有记录 / ModelScope 权重
hw2-task2 → RoadVehicleImages / YOLOv8 或同类检测模型 / 视频跟踪 / 越线计数（待做）
hw2-task3 → PyTorch 基础 API / 手写 U-Net / 手写 Dice Loss / SwanLab 私有记录 / ModelScope 权重
hw2-report → /Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report
```

## 状态说明
- ✅ 稳定
- 🚧 开发中
- 📝 规划中
