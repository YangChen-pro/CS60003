# 方案设计

## 总体方案

采用“轻量模块化”结构：

- `mlp_hw1/data.py` 负责 EuroSAT 读取、缓存、归一化与分层划分
- `mlp_hw1/model.py` 负责三层 MLP、交叉熵损失与手写反向传播
- `mlp_hw1/trainer.py` 负责训练循环、评估和超参数搜索
- `mlp_hw1/visualization.py` 负责训练曲线、混淆矩阵、权重和错例可视化
- `train.py` / `evaluate.py` / `search.py` 提供少参数入口脚本

## 关键决策

- 训练后端固定为远端 `CuPy`
- 为减少重复 I/O，首次读取后将数据缓存到 `hw1/outputs/cache/`
- 为避免过度工程化，不构建通用自动微分图系统，而是围绕 HW1 的三层 MLP 手写链式求导
- 通过 `quick/default/full` 预设控制训练规模，减少命令行参数数量
- 正式实验围绕三层 MLP 做宽度、激活函数、学习率、衰减、L2、梯度裁剪等系统化搜索
- 报告先以 Markdown 简写形式沉淀，保留必要图表与数据，后续可再转 PDF

## 验证计划

- 运行 `hw1/tests/test_core.py`
- 推送后在远端环境使用 `cupy` 跑 smoke test
- 在远端运行正式搜索与最终训练
- 生成报告与结果汇总后再同步 GitHub/远端
