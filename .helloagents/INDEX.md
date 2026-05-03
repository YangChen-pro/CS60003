# CS60003 知识库

> 本文件是知识库的入口点。

## 快速导航

| 需要了解 | 读取文件 |
|---------|---------|
| 项目概况、训练环境、执行偏好 | [context.md](context.md) |
| 项目约定、Git/远程/报告规范 | [guidelines.md](guidelines.md) |
| 验证命令 | [verify.yaml](verify.yaml) |
| 模块索引 | [modules/_index.md](modules/_index.md) |
| HW1 的环境决策与运行约定 | [modules/hw1.md](modules/hw1.md) |
| HW2 的任务状态、实验指标、报告约束 | [modules/hw2.md](modules/hw2.md) |
| 项目变更历史 | [CHANGELOG.md](CHANGELOG.md) |

## 模块关键词索引

| 模块 | 关键词 | 摘要 |
|------|--------|------|
| hw1 | EuroSAT, MLP, CuPy, 135-3090-8, llm-26-gpu, final_p, final_o, A+ | 记录 HW1 的默认远端执行环境、最终提交流程、证据链与最新性能结论。 |
| hw2 | Flowers102, ConvNeXt-Tiny, RoadVehicleImages, YOLOv8, StanfordBackground, U-Net, Dice Loss, SwanLab, ModelScope, LaTeX report | 记录 HW2 Task1/Task3 已完成实验、Task2 待补、私有实验记录和 HW2 报告状态。 |

## 知识库状态

```yaml
kb_version: unknown
最后更新: 2026-05-03
模块数量: 2
待执行方案: 0
当前重点: HW2 Task2 待补；HW2 Task1/Task3 报告已生成并视觉检查
```

## 读取指引

```yaml
涉及 HW1 训练/benchmark:
  1. 先读取 context.md
  2. 再读取 modules/hw1.md
  3. 默认优先使用远端 ssh 135-3090-8 的 llm-26-gpu 环境
  4. 最终提交时仅以 GitHub 链接和最终 PDF 报告为准

涉及 HW2 Task1/Task3 或报告:
  1. 先读取 context.md
  2. 再读取 modules/hw2.md
  3. 训练类正式实验默认在远程 135-3090-8 的 llm-26-gpu 环境执行
  4. SwanLab 记录为私有，报告只嵌入导出的本地图片，不提供云端链接

涉及 HW2 Task2:
  1. 读取 hw2/hw2.md 和 modules/hw2.md
  2. 检查 hw2/RoadVehicleImages/ 数据配置
  3. 补检测训练、视频跟踪、遮挡分析、越线计数和报告内容
```
