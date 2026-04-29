# 变更日志

## [Unreleased]

### 快速修改
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
- **[hw2-task1]**: 多维优化 Flowers102 分类实验，新增 ConvNeXt / EfficientNet / ResNet-50 等配置并将最佳 `test_acc` 提升到 `0.9608` — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604291137_hw2_task1_optimization）
  - 结果: ConvNeXt-Tiny `best_val_acc=0.9784`, `test_acc=0.9608`
  - 文件: hw2/task1/, .helloagents/plans/202604291137_hw2_task1_optimization/*, .helloagents/STATE.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md

- **[hw2-task1]**: 新增 Flowers102 ResNet 微调方案包与 Task1 实验工程，覆盖 Baseline、超参数、随机初始化消融和 SE 注意力对比 — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604291109_hw2_task1_flowers102）
  - 结果: Baseline ResNet-18 `test_acc=0.6819`，随机初始化消融 `test_acc=0.1571`，SE-ResNet-18 `test_acc=0.4245`
  - 文件: hw2/task1/, .gitignore, .helloagents/plans/202604291109_hw2_task1_flowers102/*, .helloagents/STATE.md, .helloagents/modules/hw2.md, .helloagents/CHANGELOG.md

### 快速修改
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
