# 项目上下文

## 1. 基本信息

```yaml
名称: CS60003
描述: 课程作业工作区，HW1 已完成，当前开始准备 HW2 期中作业数据集
类型: 课程作业仓库
状态: HW1 已完成并达到最终提交状态；HW2 数据集已下载到本地
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
| CuPy | 14.0.1 | HW1 的唯一数组计算后端，运行于远端 GPU 环境 |
| NumPy | 本机 2.2.6，远端 2.4.4 | 仅用于数据读取、缓存、可视化和序列化等 CPU 侧辅助处理，不作为训练后端 |

## 3. 项目概述

### 核心功能
- 完成 HW1：在 EuroSAT 上手写三层 MLP 分类器
- 在满足作业要求前提下使用远端 CuPy 作为唯一训练后端
- 为 HW2 期中作业准备 Flowers102、Road Vehicle Images、Stanford Background 三个数据集
- 保留 README、实验输出、报告 PDF 与知识库之间的一致证据链
- 记录训练环境、后端决策与性能结论，便于后续直接复用

### 项目边界
```yaml
范围内:
  - 手写前向传播、损失、梯度、反向传播、SGD、学习率衰减、L2 正则
  - 使用 CuPy 作为唯一训练数组后端
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
本地环境角色: 仅用于轻量编辑、文档维护与单元测试，不作为 HW1 训练回退
本地测试环境: conda run -n nlp
```

### 执行约定
```yaml
涉及 HW1 训练或 benchmark:
  - 默认先在远端 135-3090-8 上执行
  - 先检查空闲 GPU（优先 5/6/7 这类低占用卡）
  - 再在 llm-26-gpu 环境中运行代码
涉及 HW1 的本机/远端结果同步:
  - 优先在远端仓库直接 git commit + git push
  - 再在本机仓库 git pull 同步
  - 尽量不使用 scp 传输实验产物
最终提交同步状态:
  - GitHub / 本机 / 135 当前对齐到同一 HEAD
  - 最终提交身份固定为 YangChen-pro <1369792882@qq.com>
```

### 已确认正式结果
```yaml
正式提交模型: best 预设（1280 -> 768, relu, dropout=0.15, 44 epochs）
正式提交指标: val_acc=0.6901, test_acc=0.6758
扩展上限实验: final_o（1280 -> 768, dropout=0.18）test_acc=0.6810
选模原则: 以验证集最优为正式提交结果，不按测试集反选
最终报告路径: /Users/yangchen/Documents/Latex_Project/CS60003_HW1_Report/out/elegantpaper-cn.pdf
模型权重发布: https://modelscope.cn/models/youngchen/CS60003/
```

## 5. 当前约束（源自已确认决策）

| 约束 | 原因 | 决策来源 |
|------|------|---------|
| HW1 主实现必须手写训练逻辑，不可使用现成 autograd/nn/optimizer | 作业要求明确限制现成深度学习框架能力 | 当前会话 |
| HW1 唯一训练后端为远端 CuPy，不再保留 NumPy/MLX 训练回退 | 用户已明确运行环境始终为远端，且希望代码层面固定到 CuPy | 当前会话 |
| 下次涉及 HW1 训练/benchmark 默认优先 ssh 到 135-3090-8 并使用 llm-26-gpu | 该环境已验证包含可用的 CuPy 与 GPU | 当前会话 |
| 最终提交物只看 GitHub 链接与最终 PDF 报告 | 用户已明确提交方式，仓库内题面 PDF 不作为评分材料 | 当前会话 |

## 6. 当前交付状态

```yaml
HW2 数据集:
  Flowers102: hw2/Flowers102/
  RoadVehicleImages: hw2/RoadVehicleImages/
  StanfordBackground: hw2/StanfordBackground/
  数据说明: hw2/DATASETS.md
GitHub 仓库: https://github.com/YangChen-pro/CS60003
最终报告 PDF: /Users/yangchen/Documents/Latex_Project/CS60003_HW1_Report/out/elegantpaper-cn.pdf
README 状态: 已与最终代码、输出证据和报告对齐
135 同步状态: 已与本机和 GitHub 同步
严格复审结论: A+
```
