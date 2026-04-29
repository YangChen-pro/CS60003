# HW2 Task3 ResUNet / Attention U-Net 冲分 — 任务分解

## 任务列表
- [√] 任务1：实现手写 ResUNet / Attention U-Net 结构（涉及文件：`hw2/task3/stanford_unet/models.py`；完成标准：`build_model` 支持四种 U-Net 家族变体；验证方式：本机 `py_compile`）。
- [√] 任务2：新增冲分配置（涉及文件：`hw2/task3/configs/opt_*resunet*.yaml`, `opt_attention_unet_b64_tta.yaml`；完成标准：配置可加载；验证方式：远程配置读取和小张量检查）。
- [√] 任务3：Git 同步并远程正式训练（涉及远程 `/data/yc/CS60003`；完成标准：远程 HEAD 与本机一致，至少 3 组训练完成；验证方式：`metrics.json`）。
- [√] 任务4：第二轮几何增强冲分（涉及 `data.py`、`configs/opt_*aug*.yaml`；完成标准：至少 2 组增强实验完成；验证方式：远程 `metrics.json`）。
- [√] 任务5：汇总最佳并上传 ModelScope（涉及 `hw2/task3/outputs`、ModelScope `youngchen/CS60003`；完成标准：若刷新最佳则新路径存在；验证方式：ModelScope file_exists）。
- [√] 任务6：更新文档与知识库（涉及 `README.md`、`RESULTS.md`、`.helloagents`；完成标准：最佳结果、SwanLab、ModelScope 路径一致）。

## 进度
- [√] 本机已实现模型变体并新增配置。
- [√] 已完成提交、推送、远程训练和结果汇总。

- [√] 第一批 ResUNet / Attention U-Net 正式实验完成，当前新最佳 `task3_attention_unet_b64_tta`，`val_mIoU=0.667801`。
- [√] 第二轮 random scale crop 几何增强实验完成。

- [√] 第三/四轮多 seed 与多尺度 TTA 复评完成，阶段最佳 `task3_attention_unet_b64_aug_seed7_ms_tta` 达到 `val_mIoU=0.700608`。
- [√] 单模型继续冲分完成：EMA + mountain-aware 版本整体未刷新最佳，但 TTA 尺度扫描将最终最佳刷新为 `task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta`，`val_mIoU=0.701053`。
- [√] 最终最佳模型已上传 ModelScope：`hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/best.pt`。
