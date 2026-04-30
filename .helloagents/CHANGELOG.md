# 变更日志

## [Unreleased]

### 快速修改
- **[hw1]**: 同步 HW1 正式报告源文件的 ModelScope 新权重路径状态到知识库，记录 PDF 需按需重新编译 — by YangChen-pro
  - 类型: 知识库同步
  - 文件: .helloagents/STATE.md, .helloagents/context.md, .helloagents/modules/hw1.md, .helloagents/CHANGELOG.md

- **[hw1/hw2]**: 整理 ModelScope `youngchen/CS60003` 仓库结构，将 HW1 权重迁移到 `hw1/` 目录并上传 HW2 Task1 最佳模型，同时更新所有仓库内引用 — by YangChen-pro
  - 类型: 公网模型发布与引用迁移
  - 结果: HW1 `hw1/final_p/best_model.npz` / `hw1/final_o/best_model.npz`，HW2 Task1 `hw2/task1/flowers102_convnext_tiny/best.pt`
  - 文件: hw1/download_weights.py, hw1/README.md, hw2/task1/README.md, hw2/task1/RESULTS.md, .helloagents/context.md, .helloagents/modules/hw1.md, .helloagents/modules/hw2.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 按用户明确要求将 ModelScope API token 写入 `.helloagents/modules/hw2.md`，用于后续上传 Task1 模型权重到公网 — by YangChen-pro
  - 类型: 项目要求记录（含临时凭据）
  - 文件: .helloagents/modules/hw2.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 接入 SwanLab 训练记录与历史实验回放工具，支持将已有正式实验曲线上传用于报告截图 — by YangChen-pro
  - 类型: 实验记录补齐
  - 文件: hw2/task1/train.py, hw2/task1/upload_swanlab_history.py, hw2/task1/flowers102_task1/swanlab_utils.py, hw2/task1/flowers102_task1/engine.py, hw2/task1/flowers102_task1/config.py, hw2/task1/README.md, hw2/task1/RESULTS.md, hw2/task1/requirements.txt, .helloagents/modules/hw2.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 在远程 135 将 13 个 Task1 正式实验历史记录回放上传到 SwanLab，并记录项目与 run 链接 — by YangChen-pro
  - 类型: 远程实验记录同步
  - 结果: SwanLab 项目 `https://swanlab.cn/@youngchen/cs60003-hw2-task1`
  - 文件: hw2/task1/SWANLAB_RUNS.md, hw2/task1/RESULTS.md, hw2/task1/README.md, .helloagents/modules/hw2.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 修正 SwanLab 报告曲线展示，补充横轴 `Epoch` 与纵轴 `Loss` / `Accuracy` 明确标注，并重新上传 `task1-report-curves` 分组 — by YangChen-pro
  - 类型: 可视化质量修正
  - 结果: 修正版 run 链接已更新到 `hw2/task1/SWANLAB_RUNS.md`
  - 文件: hw2/task1/upload_swanlab_history.py, hw2/task1/flowers102_task1/swanlab_utils.py, hw2/task1/flowers102_task1/utils.py, hw2/task1/SWANLAB_RUNS.md, hw2/task1/RESULTS.md, .helloagents/modules/hw2.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 记录 Task1 后续报告需接入 SwanLab，可视化凭据通过 `SWANLAB_API_KEY` 运行时环境变量提供，不写入 Git 跟踪文件 — by YangChen-pro
  - 类型: 项目要求记录
  - 文件: .helloagents/modules/hw2.md, .helloagents/STATE.md, .gitignore, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 按用户明确要求将 SwanLab 临时 API key 写入 `.helloagents/modules/hw2.md` 并允许 Git 同步，作业完成后需删除 — by YangChen-pro
  - 类型: 项目要求记录（含临时凭据）
  - 文件: .helloagents/modules/hw2.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

### 方案开发

- **[hw2-task3]**: 尝试在单模型与 `hw2.md` 合规约束下冲击 `0.75`，新增 U-Net++、scSE、ASPP、deep supervision、Lovasz 和高分辨率配置；实验未超过当前最终最佳 — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604291855_hw2_task3_resunet_attention_optimization）
  - 结果: 最好 `task3_unetpp_b64_384_lovasz_ds_tta`，训练中 `val_mIoU=0.693158`，多尺度复评 `val_mIoU=0.6953`；最终模型仍保持 `task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta`
  - 文件: hw2/task3/, .helloagents/STATE.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md

- **[hw2-task3]**: 完成 Task3 单模型继续冲分，新增 EMA、mountain-aware sampling/crop 消融与 TTA 尺度扫描，将最佳 `val_mIoU` 提升到 `0.701053` 并上传 ModelScope — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604291855_hw2_task3_resunet_attention_optimization）
  - 结果: `task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta`，best epoch 113，`val_mIoU=0.701053`，`val_pixel_acc=0.864931`
  - 文件: hw2/task3/, .helloagents/plans/202604291855_hw2_task3_resunet_attention_optimization/*, .helloagents/STATE.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md
- **[hw2-task3]**: 继续优化手写 U-Net 分割实验，新增 TTA、宽模型和 class weighting 对比，将最佳 `val_mIoU` 提升到 `0.665089` — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604291718_hw2_task3_optimization）
  - 结果: `task3_unet_ce_dice_b64_tta`，best epoch 67，`val_mIoU=0.665089`，`val_pixel_acc=0.842011`
  - 文件: hw2/task3/, .helloagents/plans/202604291718_hw2_task3_optimization/*, .helloagents/STATE.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md

- **[hw2-task3]**: 完成 Stanford Background 手写 U-Net 三组 loss 正式实验，CE+Dice 最佳 `val_mIoU=0.648970`，并上传最佳模型到 ModelScope — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604291645_hw2_task3_unet_segmentation）
  - 结果: CE `0.648151`，Dice `0.648211`，CE+Dice `0.648970`；最佳模型 `hw2/task3/unet_ce_dice/best.pt`
  - 文件: hw2/task3/README.md, hw2/task3/RESULTS.md, .helloagents/plans/202604291645_hw2_task3_unet_segmentation/*, .helloagents/STATE.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md

- **[hw2-task3]**: 新增 Stanford Background U-Net 语义分割方案包，规划手写 U-Net、手写 Dice Loss、三组 loss 对比、SwanLab 记录与 ModelScope 发布路径 — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604291645_hw2_task3_unet_segmentation）
  - 文件: .helloagents/plans/202604291645_hw2_task3_unet_segmentation/*, .helloagents/STATE.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 多维优化 Flowers102 分类实验，新增 ConvNeXt / EfficientNet / ResNet-50 等配置并将最佳 `test_acc` 提升到 `0.9608` — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604291137_hw2_task1_optimization）
  - 结果: ConvNeXt-Tiny `best_val_acc=0.9784`, `test_acc=0.9608`
  - 文件: hw2/task1/, .helloagents/plans/202604291137_hw2_task1_optimization/*, .helloagents/STATE.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 新增 Flowers102 ResNet 微调方案包与 Task1 实验工程，覆盖 Baseline、超参数、随机初始化消融和 SE 注意力对比 — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604291109_hw2_task1_flowers102）
  - 结果: Baseline ResNet-18 `test_acc=0.6819`，随机初始化消融 `test_acc=0.1571`，SE-ResNet-18 `test_acc=0.4245`
  - 文件: hw2/task1/, .gitignore, .helloagents/plans/202604291109_hw2_task1_flowers102/*, .helloagents/STATE.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md

### 快速修改
- **[hw2-task3]**: 清理 SwanLab Task3 云端实验，仅保留 6 个报告必要 run，并同步更新结果文档与保留清单 — by YangChen-pro
  - 类型: 云端实验记录整理
  - 文件: hw2/task3/SWANLAB_RUNS.md, hw2/task3/README.md, hw2/task3/RESULTS.md, .helloagents/modules/hw2.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 清理 SwanLab Task1 云端实验，仅保留报告必要的 8 个 `task1-report-curves` run，并同步更新 run 链接文档 — by YangChen-pro
  - 类型: 云端实验记录整理
  - 文件: hw2/task1/SWANLAB_RUNS.md, .helloagents/modules/hw2.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

- **[hw2]**: 删除已同步到知识库的 `hw2/DATASETS.md` 原始说明文件，并移除对应 `.gitignore` 规则 — by YangChen-pro
  - 类型: 文档收口
  - 文件: hw2/DATASETS.md, .gitignore, .helloagents/context.md, .helloagents/modules/hw2.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

- **[hw2]**: 将 `hw2/DATASETS.md` 中的数据集说明同步到 HW2 知识库模块 — by YangChen-pro
  - 类型: 知识库同步
  - 文件: .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md

- **[hw2]**: 下载并整理期中作业所需的 Flowers102、Road Vehicle Images 与 Stanford Background 数据集，补充本地数据说明与忽略规则 — by YangChen-pro
  - 类型: 数据集准备
  - 文件: hw2/Flowers102/, hw2/RoadVehicleImages/, hw2/StanfordBackground/, hw2/DATASETS.md, .gitignore, .helloagents/context.md, .helloagents/modules/_index.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md, .helloagents/STATE.md

- **[hw2]**: 按用户确认移除 HW2 数据集目录的 `.gitignore` 忽略规则，并同步数据说明与状态记录 — by YangChen-pro
  - 类型: 配置调整
  - 文件: .gitignore, hw2/DATASETS.md, .helloagents/STATE.md, .helloagents/CHANGELOG.md

- **[kb]**: 同步 HW1 的最终提交状态到知识库，补齐 full search 最优 trial 证据链、最终提交口径和 A+ 复审结论 — by YangChen-pro
  - 类型: 知识库同步
  - 文件: .helloagents/context.md, .helloagents/INDEX.md, .helloagents/modules/_index.md, .helloagents/modules/hw1.md, .helloagents/CHANGELOG.md, .helloagents/STATE.md

### 历史记录
- **[hw1]**: 初始化项目知识库，并记录 HW1 的默认远端执行环境与 CuPy 方案选择 — by YangChen-pro
  - 类型: 快速修改（无方案包）
  - 文件: .helloagents/INDEX.md:1-35, .helloagents/context.md:1-55, .helloagents/modules/_index.md:1-18, .helloagents/modules/hw1.md:1-54, .helloagents/CHANGELOG.md:1-8

### 历史方案开发
- **[hw1]**: 新增 EuroSAT 三层 MLP 作业实现，包含数据处理、手写反向传播、训练/评估/搜索脚本与可视化产物说明 — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604091720_hw1_mlp）
  - 文件: hw1/mlp_hw1/backend.py, hw1/mlp_hw1/config.py, hw1/mlp_hw1/data.py, hw1/mlp_hw1/metrics.py, hw1/mlp_hw1/model.py, hw1/mlp_hw1/trainer.py, hw1/mlp_hw1/visualization.py, hw1/train.py, hw1/evaluate.py, hw1/search.py, hw1/tests/test_core.py, hw1/requirements.txt, hw1/README.md, .gitignore, .helloagents/STATE.md, .helloagents/plans/202604091720_hw1_mlp/*

- **[hw1]**: 将 HW1 运行约束收紧为“远端 CuPy 唯一后端”，删除 CLI 中的 NumPy 回退入口，并同步更新知识库约束 — by YangChen-pro
  - 类型: 快速修改（约束收紧）
  - 文件: hw1/mlp_hw1/backend.py, hw1/mlp_hw1/trainer.py, hw1/train.py, hw1/evaluate.py, hw1/search.py, hw1/tests/test_core.py, hw1/README.md, .helloagents/context.md, .helloagents/modules/hw1.md, .helloagents/CHANGELOG.md

- **[hw1]**: 完成正式搜索、最终训练与简写实验报告，新增 `best` 预设并沉淀最终结论到 README / REPORT / 知识库 — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604091720_hw1_mlp）
  - 文件: hw1/mlp_hw1/config.py, hw1/train.py, hw1/evaluate.py, hw1/README.md, hw1/REPORT.md, hw1/outputs/runs/final_a/*, hw1/outputs/runs/final_c/*, .helloagents/context.md, .helloagents/modules/hw1.md, .helloagents/CHANGELOG.md, .helloagents/STATE.md, .helloagents/plans/202604091720_hw1_mlp/*

- **[hw1]**: 继续冲三层 MLP 性能，引入可选 dropout，并将正式提交模型刷新为 `final_p`（`val_acc=0.6901`） — by YangChen-pro
  - 类型: 标准流程（持续优化）
  - 文件: hw1/mlp_hw1/config.py, hw1/mlp_hw1/model.py, hw1/mlp_hw1/trainer.py, hw1/tests/test_core.py, hw1/README.md, hw1/REPORT.md, hw1/outputs/runs/final_p/*, .gitignore, .helloagents/context.md, .helloagents/modules/hw1.md, .helloagents/CHANGELOG.md, .helloagents/STATE.md

- **[kb]**: 同步 `final_p / final_o` 到知识库入口、模块文档与状态快照，并完成一次手工 validatekb / upgradekb 收口 — by YangChen-pro
  - 类型: 快速修改（知识库维护）
  - 文件: .helloagents/INDEX.md, .helloagents/context.md, .helloagents/modules/hw1.md, .helloagents/CHANGELOG.md, .helloagents/STATE.md
