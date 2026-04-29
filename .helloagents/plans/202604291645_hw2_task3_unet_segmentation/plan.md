# HW2 Task3 Stanford Background U-Net 分割 — 实施规划

## 目标与范围
本轮目标是把 Task3 做成一个可复现、可报告、可上传模型权重的分割实验工程。重点不是堆复杂模型，而是严格满足题面：随机初始化、手写 U-Net、手写 Dice Loss、三种 loss 对比、验证集 mIoU 与 SwanLab 曲线证据完整。

## 架构与实现策略
- 数据层：
  - 新增 `stanford_unet/data.py`，读取 `images/*.jpg` 与 `labels/*.regions.txt`。
  - `-1` 标签转换为 `ignore_index=255`；其余标签保持 `0..7`。
  - 固定 seed 生成 `splits/train.txt` 与 `splits/val.txt`，默认 80% / 20%。
  - 训练增强只采用低风险方案：horizontal flip、轻量 color jitter；验证只 resize + normalize。
- 模型层：
  - 新增 `stanford_unet/models.py`，实现轻量 U-Net：DoubleConv、Down、Up、OutConv。
  - 默认 `base_channels=32`，4 次下采样，输出 logits shape 为 `[B, 8, H, W]`。
  - 禁止加载 pretrained weights；配置中显式记录 `pretrained: false`。
- 损失层：
  - 新增 `stanford_unet/losses.py`。
  - `ce` 使用 `CrossEntropyLoss(ignore_index=255)`。
  - `dice` 手写 softmax Dice Loss，按 ignore mask 过滤 unknown 像素。
  - `ce_dice` 使用 `CE + Dice`，默认等权。
- 训练与评估：
  - `train.py` 读取 YAML 配置，执行 train/val 循环，按 val mIoU 保存 `best.pt`。
  - `evaluate.py` 支持从 `best.pt` 重新评估，输出 `metrics.json` 与可视化样例。
  - `metrics.py` 实现 pixel accuracy、confusion matrix、per-class IoU、mIoU，全部忽略 `255`。
- 实验记录：
  - 三个配置文件：`ce.yaml`、`dice.yaml`、`ce_dice.yaml`。
  - 输出目录：`hw2/task3/outputs/{timestamp}_{experiment}/`。
  - SwanLab 项目建议：`cs60003-hw2-task3`；记录 loss、mIoU、pixel accuracy、per-class IoU 和最终报告曲线图。
- 结果发布：
  - `RESULTS.md` 汇总三组 loss 对比与推荐报告口径。
  - 最佳权重上传到 ModelScope `youngchen/CS60003/hw2/task3/{best_experiment}/best.pt`。

## 完成定义
- 代码中存在完整手写 U-Net 与手写 Dice Loss，且无预训练加载路径。
- 远程 135 上完成三组正式实验：`ce`、`dice`、`ce_dice`。
- 每组实验产生可复现输出：配置副本、history、metrics、curves、best 权重、验证样例可视化。
- `RESULTS.md` 明确三组验证集 mIoU 排名、per-class IoU 差异和推荐结论。
- SwanLab 中能看到横纵轴明确的曲线；报告可直接截图引用。
- 最佳模型权重上传到 ModelScope，并在 README / RESULTS 中写入公网链接。
- 本机、GitHub、远程 135 最终同步到同一 HEAD。

## 文件结构
```text
hw2/task3/
├── README.md
├── RESULTS.md
├── requirements.txt
├── train.py
├── evaluate.py
├── configs/
│   ├── ce.yaml
│   ├── dice.yaml
│   └── ce_dice.yaml
├── splits/
│   ├── train.txt
│   └── val.txt
└── stanford_unet/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── models.py
    ├── losses.py
    ├── metrics.py
    ├── engine.py
    ├── visualization.py
    ├── swanlab_utils.py
    └── utils.py
```

## UI / 设计约束
不涉及应用 UI。报告可视化属于实验图表，要求坐标轴、图例、标题与单位明确；分割样例需同时展示原图、GT mask、Prediction mask，并使用固定类别颜色表。

## 风险与验证
- 小数据集从零训练可能过拟合：使用轻量增强、weight decay、early stopping，并按验证 mIoU 选模。
- Dice-only 可能早期不稳定：保留梯度裁剪、较小学习率和 smooth 项；若收敛失败，在报告中作为 loss 对比现象说明，不篡改实验口径。
- 标签 `-1` 若未正确 ignore 会污染 mIoU：单元检查和评估脚本必须验证 ignored pixels 不参与统计。
- U-Net 输入尺寸需与下采样倍率兼容：统一 `240x320`，避免 skip concat shape mismatch。
- SwanLab 图表必须避免无轴标注：训练脚本实时记录标量，报告曲线图本地生成时强制设置 x/y label。

## 决策记录
- [2026-04-29] 用户在 Task2 / Task3 对比后选择 Task3，目标是更稳妥完成一个可控的 HW2 后续任务。
- [2026-04-29] 继续沿用 HW2 已确认约束：本机不做训练或 smoke test，正式实验在远程 `135-3090-8` 执行，并通过 Git 同步代码。
- [2026-04-29] Task3 采用 PyTorch 基础 API 实现，U-Net 与 Dice Loss 必须由项目代码手写，不使用预训练。
