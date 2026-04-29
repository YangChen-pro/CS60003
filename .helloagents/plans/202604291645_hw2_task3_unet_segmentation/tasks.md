# HW2 Task3 Stanford Background U-Net 分割 — 任务分解

## 任务列表
- [ ] 任务1：建立 Task3 工程骨架与可复现配置（涉及文件：`hw2/task3/README.md`、`requirements.txt`、`configs/*.yaml`、`splits/*.txt`；完成标准：目录、配置、split 固定且可追踪；验证方式：远程读取配置与 split 文件）。
- [ ] 任务2：实现 Stanford Background 数据集读取与可视化（涉及文件：`stanford_unet/data.py`、`visualization.py`；完成标准：图像和 `.regions.txt` mask 对齐，`-1` 正确映射为 ignore；验证方式：远程抽样保存 GT 可视化图）。
- [ ] 任务3：手写 U-Net 模型（涉及文件：`stanford_unet/models.py`；完成标准：包含 encoder、decoder、skip concat，输出 8 类 logits；验证方式：远程前向 shape 检查与 `compileall`）。
- [ ] 任务4：手写 Dice Loss 与分割指标（涉及文件：`stanford_unet/losses.py`、`metrics.py`；完成标准：CE、Dice、CE+Dice 三种 loss 可选，mIoU 忽略 unknown；验证方式：远程构造小张量单元检查）。
- [ ] 任务5：实现训练、评估、SwanLab 记录（涉及文件：`train.py`、`evaluate.py`、`engine.py`、`swanlab_utils.py`、`utils.py`；完成标准：按 val mIoU 保存 best，输出 history/metrics/curves/可视化；验证方式：远程正式运行时生成完整产物）。
- [ ] 任务6：Git 同步并在远程 135 执行三组正式实验（涉及文件：Git 本机/GitHub/远程仓库、远程 `hw2/task3/outputs/`；完成标准：`ce`、`dice`、`ce_dice` 均完成；验证方式：汇总三个 `metrics.json`）。
- [ ] 任务7：整理结果、上传最佳模型到 ModelScope、更新知识库（涉及文件：`hw2/task3/RESULTS.md`、`README.md`、`.helloagents/modules/hw2.md`、ModelScope `hw2/task3/`；完成标准：报告口径、SwanLab 链接、ModelScope 链接完整；验证方式：ModelScope 文件存在且 Git 三端同 HEAD）。

## 进度
- [√] 已完成 Task3 方案包规划。
- [ ] 待进入实现阶段。
