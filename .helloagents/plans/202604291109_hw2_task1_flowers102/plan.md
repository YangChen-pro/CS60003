# HW2 Task1 Flowers102 — 实施规划

## 目标与范围
本次只处理 HW2 Task1。目标是在 Flowers102 数据集上建立一套可复现实验工程，覆盖 Baseline、超参数分析、预训练消融和注意力机制对比，并将正式实验放在远程 `135-3090-8` 上执行。

## 架构与实现策略
- 在 `hw2/task1/` 下建立独立实验包，避免污染 HW1 代码。
- 数据层读取 Oxford Flowers102 官方文件：
  - `imagelabels.mat` 提供 1–102 类标签，代码内部转换为 0–101。
  - `setid.mat` 提供官方 train / val / test 图片编号，转换为对应文件名。
- 模型层以 torchvision ResNet-18 为核心：
  - `pretrained_resnet18`：ImageNet 预训练 Baseline。
  - `random_resnet18`：不加载 ImageNet 权重的消融模型。
  - `se_resnet18`：在 ResNet BasicBlock 输出后加入 SE 模块的注意力版本。
- 训练层使用统一入口 `train.py`：
  - 读取 YAML 配置。
  - 自动区分 backbone / classifier 参数组。
  - 保存 best checkpoint、训练历史、最终指标和曲线。
- 评估层使用 `evaluate.py`：
  - 加载指定 checkpoint。
  - 在 test 划分上输出 Accuracy 和 per-class accuracy。
- 实验配置放在 `hw2/task1/configs/`，每个配置对应一个可复现实验。

## 完成定义
- `hw2/task1/` 可在远程环境中直接运行。
- 数据检查能确认 Flowers102 官方划分数量正确：train 1020、val 1020、test 6149。
- Baseline、随机初始化、SE 注意力和至少两组超参数实验均可通过配置启动。
- 每次实验产出目录包含：
  - `config.json`
  - `history.csv`
  - `metrics.json`
  - `best.pt`
  - `curves.png`
- README 或 Task1 文档包含训练命令、远程 Git 同步流程、结果文件说明。
- 验证主路径：`test-first`。tester 重点验证数据划分、训练入口、远程环境、Git 身份和产物完整性。

## 文件结构
```text
hw2/task1/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline_resnet18.yaml
│   ├── baseline_resnet18_low_lr.yaml
│   ├── baseline_resnet18_short.yaml
│   ├── random_resnet18.yaml
│   └── se_resnet18.yaml
├── flowers102_task1/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── metrics.py
│   ├── models.py
│   └── utils.py
├── train.py
└── evaluate.py
```

## UI / 设计约束
不涉及 UI。训练曲线使用 Matplotlib 生成报告可用图像即可。

## 风险与验证
- 远程环境可能缺少 PyTorch / torchvision：先检查环境，若缺依赖则停止并说明缺口，不擅自换环境。
- 远程仓库可能不是最新 HEAD：先检查 `git status` 和 `git pull --ff-only`。
- Git 身份可能不一致：先检查并按授权目标配置为 `YangChen-pro <1369792882@qq.com>`。
- 模型权重体积大：输出目录默认加入 `.gitignore`，正式权重后续上传网盘或模型托管，不直接提交。
- Flowers102 训练集小，随机初始化消融可能表现明显差于预训练模型：保留相同训练预算，报告中解释可比性。

## 决策记录
- [2026-04-29] 用户确认不在本机 smoke test，所有正式验证和训练直接在远程 `135-3090-8` 执行。
- [2026-04-29] 用户确认代码通过 Git 同步到远程，远程使用前需检查 Git 身份与本机最终提交身份一致。
- [2026-04-29] Task1 注意力机制优先选择 SE-block，因实现稳定、解释清晰、训练成本可控。
