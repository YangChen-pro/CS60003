# HW2 Task3 U-Net 继续冲分优化 — 任务分解

## 任务列表
- [√] 任务1：实现优化能力（涉及文件：models.py, losses.py, data.py, engine.py, train.py, evaluate.py；完成标准：支持 dropout、class weights、TTA；验证方式：远程 compile 和小张量检查）。
- [√] 任务2：新增优化配置（涉及文件：configs/opt_*.yaml；完成标准：三组配置可复现；验证方式：远程读取配置）。
- [√] 任务3：Git 同步远程并运行正式实验（涉及远程 outputs；完成标准：至少 2 组完成；验证方式：metrics.json）（涉及文件：hw2/task3/, hw2/task3/configs/, hw2/task3/RESULTS.md, .helloagents/modules/hw2.md；完成标准：Task3 第一轮优化最佳 val_mIoU=0.665089 已记录并发布；验证方式：远程 metrics.json、ModelScope file_exists、RESULTS.md 和知识库核对）。
- [√] 任务4：汇总并上传最佳模型（涉及 RESULTS.md、ModelScope；完成标准：新最佳路径可访问；验证方式：ModelScope file_exists）（涉及文件：hw2/task3/, hw2/task3/configs/, hw2/task3/RESULTS.md, .helloagents/modules/hw2.md；完成标准：Task3 第一轮优化最佳 val_mIoU=0.665089 已记录并发布；验证方式：远程 metrics.json、ModelScope file_exists、RESULTS.md 和知识库核对）。
- [√] 任务5：更新知识库并收尾（涉及 .helloagents；完成标准：当前状态、模块文档和 changelog 一致）（涉及文件：hw2/task3/, hw2/task3/configs/, hw2/task3/RESULTS.md, .helloagents/modules/hw2.md；完成标准：Task3 第一轮优化最佳 val_mIoU=0.665089 已记录并发布；验证方式：远程 metrics.json、ModelScope file_exists、RESULTS.md 和知识库核对）。

## 进度
- [√] 本机已完成代码与配置修改（涉及文件：hw2/task3/, hw2/task3/configs/, hw2/task3/RESULTS.md, .helloagents/modules/hw2.md；完成标准：Task3 第一轮优化最佳 val_mIoU=0.665089 已记录并发布；验证方式：远程 metrics.json、ModelScope file_exists、RESULTS.md 和知识库核对）。
- [√] 已完成远程正式优化实验，最佳 `task3_unet_ce_dice_b64_tta`，`val_mIoU=0.665089`（涉及文件：hw2/task3/, hw2/task3/configs/, hw2/task3/RESULTS.md, .helloagents/modules/hw2.md；完成标准：Task3 第一轮优化最佳 val_mIoU=0.665089 已记录并发布；验证方式：远程 metrics.json、ModelScope file_exists、RESULTS.md 和知识库核对）。

- [√] 优化后最佳模型已上传 ModelScope：`hw2/task3/unet_ce_dice_b64_tta/best.pt`（涉及文件：hw2/task3/, hw2/task3/configs/, hw2/task3/RESULTS.md, .helloagents/modules/hw2.md；完成标准：Task3 第一轮优化最佳 val_mIoU=0.665089 已记录并发布；验证方式：远程 metrics.json、ModelScope file_exists、RESULTS.md 和知识库核对）。
