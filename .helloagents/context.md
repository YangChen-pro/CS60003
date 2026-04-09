# 项目上下文

## 1. 基本信息

```yaml
名称: CS60003
描述: 课程作业工作区，当前重点是 HW1 的 EuroSAT 三层 MLP 分类任务
类型: 课程作业仓库
状态: 开发中
```

## 2. 技术上下文

```yaml
语言: Python
框架: 无深度学习框架作为主实现，训练逻辑需手写
包管理器: conda / pip / uv
构建工具: 无
```

### 主要依赖
| 依赖 | 版本 | 用途 |
|------|------|------|
| CuPy | 14.0.1 | 远端 GPU array backend，作为 HW1 的首选计算后端 |
| NumPy | 本机 2.2.6，远端 2.4.4 | CPU array backend，对照实现与回退方案 |
| MLX | 本机可用 | 本机 Apple Silicon array backend，作为次优本地方案 |

## 3. 项目概述

### 核心功能
- 完成 HW1：在 EuroSAT 上手写三层 MLP 分类器
- 在满足作业要求前提下选择高效的数组计算后端
- 记录训练环境、后端决策与性能结论，便于后续直接复用

### 项目边界
```yaml
范围内:
  - 手写前向传播、损失、梯度、反向传播、SGD、学习率衰减、L2 正则
  - 使用 CuPy/NumPy/MLX 作为底层数组计算后端
范围外:
  - 使用现成 autograd / nn / optimizer / trainer
  - 依赖 PyTorch、TensorFlow、JAX 的训练能力完成作业主体
```

## 4. 开发约定

### 训练环境约定
```yaml
默认远端主机: ssh 135-3090-8
默认远端环境: /data/yc/miniconda/envs/llm-26-gpu
默认远端后端: CuPy + 空闲 GPU
本机回退环境: conda env nlp
本机回退后端: MLX
```

### 执行约定
```yaml
涉及 HW1 训练或 benchmark:
  - 默认先在远端 135-3090-8 上执行
  - 先检查空闲 GPU（优先 5/6/7 这类低占用卡）
  - 再在 llm-26-gpu 环境中运行代码
```

## 5. 当前约束（源自已确认决策）

| 约束 | 原因 | 决策来源 |
|------|------|---------|
| HW1 主实现必须手写训练逻辑，不可使用现成 autograd/nn/optimizer | 作业要求明确限制现成深度学习框架能力 | 当前会话 |
| HW1 首选后端为远端 CuPy，而非本机 MLX 或远端 NumPy | 实测远端 CuPy 明显更快，且符合“仅作 array backend”约束 | 当前会话 |
| 下次涉及 HW1 训练/benchmark 默认优先 ssh 到 135-3090-8 并使用 llm-26-gpu | 该环境已验证包含可用的 CuPy 与 GPU | 当前会话 |
