# hw1

## 职责

记录 HW1 的作业约束、默认运行环境、后端选型、最终交付约定和已完成的性能结论，确保后续再做训练、benchmark、复现或答辩时可以直接复用同一套判断。

## 行为规范

### 训练环境选择
**条件**: 涉及 HW1 的训练、benchmark、后端比较或超参数搜索。  
**行为**: 固定连接 `ssh 135-3090-8`，进入 `/data/yc/miniconda/envs/llm-26-gpu` 环境，使用 `CuPy` 和空闲 GPU。  
**结果**: 统一 HW1 的真实运行环境，不再维护本地训练回退路径。

### 同步策略
**条件**: 涉及 HW1 的实验结果、图表或报告同步。  
**行为**: 优先在远端仓库直接 `git commit` 与 `git push`，随后在本机 `git pull`；尽量不使用 `scp` 传输大批实验产物。  
**结果**: 利用 Git 作为本机和远端之间的主同步通道，减少大文件拷贝开销。

### 最终提交口径
**条件**: 涉及 HW1 的最终提交、评分复核或结果说明。  
**行为**: 只把 GitHub 仓库链接和最终 PDF 报告作为正式提交物；权重不进 Git，通过 ModelScope 提供。  
**结果**: 提交口径与课程要求、仓库结构和报告表述保持一致。

### 证据链优先级
**条件**: 涉及 full search、选模证据或助教复核。  
**行为**: 当前 full search 以 `hw1/outputs/search/20260419_045438/results.csv`、`results.json` 与 `best_result.json` 为主证据；`hw1/outputs/runs/trial_04/` 下的 4 个 JSON 用于闭合最优 trial 的可核验证据链；`manifest.csv` 仅作为历史实验台账使用。  
**结果**: 避免把历史台账与当前 full search 的 trial 命名混为一谈。

### 后端边界
**条件**: 实现 HW1 的核心训练逻辑。  
**行为**: 将 `CuPy` 作为唯一 array backend 使用，前向传播、损失函数、梯度计算、反向传播、SGD、学习率衰减、L2 正则全部手写。  
**结果**: 满足作业对“自主实现自动微分与反向传播”的要求，同时保留 GPU 加速。

### 禁止回退
**条件**: 涉及 HW1 的训练、评估或搜索脚本。  
**行为**: 不保留 `NumPy` / `MLX` 的训练回退；脚本接口和训练逻辑都固定到 `CuPy`。  
**结果**: 代码与真实运行环境保持一致，避免出现未使用的回退分支。

## 依赖关系

```yaml
依赖:
  - 远端主机别名: 135-3090-8
  - 远端环境: /data/yc/miniconda/envs/llm-26-gpu
  - 远端 GPU 后端: CuPy 14.0.1
被依赖:
  - 后续 HW1 的训练脚本
  - 后续 HW1 的 benchmark 和超参数搜索
  - 最终提交与复现说明
```

## 已确认事实

### 环境信息
- 远端主机 `135-3090-8` 可用，硬件包含 `8 x RTX 3090 24GB`
- 远端环境 `/data/yc/miniconda/envs/llm-26-gpu` 已验证可用
- 远端 `CuPy` 版本为 `14.0.1`
- 远端 `CuPy` 运行时为 CUDA `13.0`，driver 口径为 `13.2`

### 后端决策
- `CuPy` 方案是当前已选方案
- 对 HW1 而言，`CuPy` 是唯一训练 array backend
- 不再保留 `NumPy` / `MLX` 训练回退
- 不使用现成 `autograd`、`nn`、`optimizer`、`trainer`
- 当前实现采用轻量模块化结构，训练入口位于 `hw1/train.py`
- 超参数搜索入口位于 `hw1/search.py`
- 独立评估入口位于 `hw1/evaluate.py`
- 当前知识库、README、`outputs/` 证据和最终 PDF 报告已完成一致性收口

## 当前实现状态

- 已实现 EuroSAT 数据读取、缓存、分层划分和归一化
- 已实现三层 MLP、交叉熵损失与手写反向传播
- 已实现 SGD、学习率衰减、L2 正则和最佳权重保存
- 已实现训练曲线、混淆矩阵、第一层权重和错例可视化
- 已完成 GitHub 推送和远端运行验证
- 已将脚本接口收紧为远端 CuPy 唯一路径
- 已新增 `best` 预设，用于直接复现实验报告中的正式提交模型
- 已验证轻量 dropout 仍能保持“三层 MLP”边界内的合规优化
- 已补齐 `trial_04` 轻量留档，闭合当前 full search 最优 trial 的仓库内证据链
- 已完成 README、`manifest.csv` 与最终 PDF 的交叉核对
- 已通过最严格复现实验复审，最终结论为 A+

## 当前最优结论

- 正式提交模型按验证集选择为 `best`/`final_p`
  - 结构: `1280 -> 768`
  - 激活: `relu`
  - 训练: `44 epochs`, `lr=0.012`, `lr_decay=0.01`, `weight_decay=2e-4`, `grad_clip=3.0`, `dropout=0.15`
  - 结果: `val_acc=0.6901`, `test_acc=0.6758`
- 扩展实验中的最高测试精度为 `final_o`
  - 结构: `1280 -> 768`
  - 训练: `42 epochs`, `dropout=0.18`
  - 结果: `val_acc=0.6877`, `test_acc=0.6810`
- 当前经验结论:
  - 扩大第一隐藏层宽度比单纯延长训练更有效
  - `ReLU` 明显优于本轮搜索中的 `tanh`
  - 轻量 `dropout` 能继续提升验证集泛化
  - 难分类别主要集中在 `Highway` 与 `PermanentCrop`

## 当前提交证据

- 最终 GitHub 仓库: `https://github.com/YangChen-pro/CS60003`
- 最终报告 PDF: `/Users/yangchen/Documents/Latex_Project/CS60003_HW1_Report/out/elegantpaper-cn.pdf`
- ModelScope 权重: `https://modelscope.cn/models/youngchen/CS60003/`
- 当前 full search 主证据:
  - `hw1/outputs/search/20260419_045438/results.csv`
  - `hw1/outputs/search/20260419_045438/results.json`
  - `hw1/outputs/search/20260419_045438/best_result.json`
- 当前 full search 最优 trial 留档:
  - `hw1/outputs/runs/trial_04/config.json`
  - `hw1/outputs/runs/trial_04/history.json`
  - `hw1/outputs/runs/trial_04/summary.json`
  - `hw1/outputs/runs/trial_04/confusion_matrix.json`
- 历史实验台账:
  - `hw1/outputs/runs/manifest.csv`
  - 其中 `source_group=full_search_20260419_045438` 的 `trial_04` 才与当前 full search 一一对应

### 历史性能对比（仅保留背景）
- 远端 `CuPy` on `GPU 5`
  - `H=256`: `0.059s / epoch`
  - `H=512`: `0.069s / epoch`
  - `H=1024`: `0.086s / epoch`
- 本机 `MLX`
  - `H=256`: `0.184s / epoch`
  - `H=512`: `0.313s / epoch`
- 远端 `NumPy`
  - `H=256`: `0.369s / epoch`
  - `H=512`: `0.598s / epoch`
- 结论: 这些数据仅用于说明为什么最终收敛到“远端 `CuPy` 唯一后端”；当前实现不再保留 `MLX/NumPy` 训练路径
