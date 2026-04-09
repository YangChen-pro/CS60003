# 变更日志

## [Unreleased]

### 快速修改
- **[hw1]**: 初始化项目知识库，并记录 HW1 的默认远端执行环境与 CuPy 方案选择 — by YangChen-pro
  - 类型: 快速修改（无方案包）
  - 文件: .helloagents/INDEX.md:1-35, .helloagents/context.md:1-55, .helloagents/modules/_index.md:1-18, .helloagents/modules/hw1.md:1-54, .helloagents/CHANGELOG.md:1-8

### 方案开发
- **[hw1]**: 新增 EuroSAT 三层 MLP 作业实现，包含数据处理、手写反向传播、训练/评估/搜索脚本与可视化产物说明 — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604091720_hw1_mlp）
  - 文件: hw1/mlp_hw1/backend.py, hw1/mlp_hw1/config.py, hw1/mlp_hw1/data.py, hw1/mlp_hw1/metrics.py, hw1/mlp_hw1/model.py, hw1/mlp_hw1/trainer.py, hw1/mlp_hw1/visualization.py, hw1/train.py, hw1/evaluate.py, hw1/search.py, hw1/tests/test_core.py, hw1/requirements.txt, hw1/README.md, .gitignore, .helloagents/STATE.md, .helloagents/plans/202604091720_hw1_mlp/*

- **[hw1]**: 将 HW1 运行约束收紧为“远端 CuPy 唯一后端”，删除 CLI 中的 NumPy 回退入口，并同步更新知识库约束 — by YangChen-pro
  - 类型: 快速修改（约束收紧）
  - 文件: hw1/mlp_hw1/backend.py, hw1/mlp_hw1/trainer.py, hw1/train.py, hw1/evaluate.py, hw1/search.py, hw1/tests/test_core.py, hw1/README.md, .helloagents/context.md, .helloagents/modules/hw1.md, .helloagents/CHANGELOG.md

- **[hw1]**: 完成正式搜索、最终训练与简写实验报告，新增 `best` 预设并沉淀最终结论到 README / REPORT / 知识库 — by YangChen-pro
  - 类型: 标准流程（方案包：.helloagents/plans/202604091720_hw1_mlp）
  - 文件: hw1/mlp_hw1/config.py, hw1/train.py, hw1/evaluate.py, hw1/README.md, hw1/REPORT.md, hw1/outputs/runs/final_a/*, hw1/outputs/runs/final_c/*, .helloagents/context.md, .helloagents/modules/hw1.md, .helloagents/CHANGELOG.md, .helloagents/STATE.md, .helloagents/plans/202604091720_hw1_mlp/*
