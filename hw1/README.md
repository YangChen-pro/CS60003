# HW1：EuroSAT 三层 MLP 分类器

本目录给出一个不依赖 PyTorch / TensorFlow / JAX 自动微分能力的课程作业实现。运行前提已经固定为远端 GPU 环境，核心思路是：

- 使用 `CuPy` 作为唯一底层数组后端
- 手写三层 MLP 的前向传播、交叉熵损失和反向传播
- 使用 SGD、学习率衰减和 L2 正则完成训练
- 自动保存验证集最优权重
- 生成训练曲线、混淆矩阵、第一层权重可视化和错例图

## 目录结构

```text
hw1/
├── EuroSAT_RGB/              # 数据集
├── mlp_hw1/
│   ├── backend.py            # CuPy 后端封装
│   ├── config.py             # 训练与搜索配置
│   ├── data.py               # 数据加载、缓存、归一化、划分
│   ├── metrics.py            # 准确率与混淆矩阵
│   ├── model.py              # 手写三层 MLP 与反向传播
│   ├── trainer.py            # 训练、评估、超参数搜索
│   └── visualization.py      # 曲线、权重、错例可视化
├── train.py                  # 训练入口
├── evaluate.py               # 测试集评估入口
├── search.py                 # 超参数搜索入口
├── tests/
│   └── test_core.py          # 不依赖数据集的基础单元测试
└── requirements.txt
```

## 环境依赖

基础依赖写在 `requirements.txt` 中，其中已经显式包含 `CuPy`：

```bash
python -m pip install -r "hw1/requirements.txt"
```

当前项目默认直接运行在远端 GPU 环境，已经提供可用的 `CuPy`：

- 远端主机：`135-3090-8`
- conda 环境：`/data/yc/miniconda/envs/llm-26-gpu`
- CuPy 版本：`14.0.1`

## 运行方式

以下命令默认在仓库根目录执行。

### 1. 快速检查训练流程

```bash
python -X utf8 "hw1/train.py" --preset quick
```

`quick` 预设只抽取少量样本并训练 2 个 epoch，适合先在远端验证代码链路。

### 2. 常规训练

```bash
python -X utf8 "hw1/train.py" --preset default
```

可选地覆盖少量关键参数：

```bash
python -X utf8 "hw1/train.py" --preset default --activation tanh --hidden-dim 768 --epochs 16
```

### 3. 超参数搜索

```bash
python -X utf8 "hw1/search.py" --preset quick --strategy grid --max-trials 4
```

搜索结果会保存在 `hw1/outputs/search/...` 下，包括：

- `results.csv`
- `results.json`
- `best_result.json`

### 4. 评估最优模型

```bash
python -X utf8 "hw1/evaluate.py" --preset default --checkpoint "hw1/outputs/runs/<run_name>/best_model.npz"
```

## 输出产物

每次训练会在 `hw1/outputs/runs/<run_name>/` 下生成：

- `best_model.npz`：验证集最优权重
- `history.json`：训练和验证曲线对应的数值
- `summary.json`：训练配置与核心指标摘要
- `confusion_matrix.json`：测试集混淆矩阵
- `training_curves.png`：训练集/验证集 loss 曲线与验证准确率曲线
- `confusion_matrix.png`：测试集混淆矩阵可视化
- `first_layer_weights.png`：第一层隐藏层权重可视化
- `misclassified_examples.png`：测试集错例分析图

## 实现说明

### 模型

- 网络结构：`input -> hidden -> hidden -> output`
- 默认两层隐藏层共用同一个 `hidden_dim`
- 支持 `relu`、`tanh`、`sigmoid`
- 反向传播由 `mlp_hw1/model.py` 中的手写链式法则完成

### 训练

- 优化器：纯手写 SGD
- 损失函数：Softmax Cross-Entropy
- 正则化：L2 Weight Decay
- 学习率策略：`lr / (1 + decay * epoch)` 的逆时衰减
- 选模标准：验证集准确率最高时自动保存权重

### 数据处理

- 从 `EuroSAT_RGB` 文件夹读取 64×64 RGB 图像
- 按类别分层划分 train / val / test
- 使用训练集均值和标准差做特征归一化
- 首次运行会在 `hw1/outputs/cache/` 下生成缓存，减少重复读取图片的开销

## 测试

不依赖真实数据集的基础回归测试：

```bash
python -X utf8 -m unittest discover -s "hw1/tests"
```

## 说明

- 本实现把 `CuPy` 作为唯一数组计算后端，不保留 `NumPy` 训练回退
- 代码刻意保持“作业级工程化”：模块清晰、注释适量，但不过度抽象
- 所有 HW1 相关代码都放在 `hw1/` 目录中，便于后续新增 `hw2/`
