# HW1 实验报告（简写版）

## 1. 作业要求对齐

- 任务：在 EuroSAT RGB 数据集上实现十分类器。
- 模型：主体仍然是三层 MLP，结构为 `input -> hidden1 -> hidden2 -> output`。
- 实现约束：前向传播、Softmax 交叉熵、反向传播、SGD、学习率衰减、L2 正则和梯度裁剪均为手写实现，不依赖 PyTorch / TensorFlow / JAX 自动微分。
- 运行环境：远端 `135-3090-8`，环境 `/data/yc/miniconda/envs/llm-26-gpu`，训练后端固定为 `CuPy`。
- GitHub Repo：`https://github.com/YangChen-pro/CS60003`
- ModelScope 权重地址：`https://www.modelscope.cn/models/youngchen/CS60003`
- 正式提交模型权重：ModelScope 仓库中的 `final_p/best_model.npz`
- 扩展实验模型权重：ModelScope 仓库中的 `final_o/best_model.npz`

## 2. 数据与实验设置

- 数据集：`EuroSAT_RGB`
- 划分方式：按类别分层划分，`train/val/test = 18900 / 4050 / 4050`
- 输入处理：将 `64x64x3` 图像展平，并使用训练集均值方差做标准化
- 选模标准：始终按验证集准确率保存最优权重

## 3. 正式提交模型

正式提交结果按验证集表现选择，而不是按测试集反选。

| 项目 | 数值 |
|---|---|
| 预设名 | `best` |
| 激活函数 | `relu` |
| 隐层宽度 | `1280 -> 768` |
| dropout | `0.15` |
| batch size | `256` |
| epoch | `44` |
| learning rate | `0.012` |
| lr decay | `0.01` |
| weight decay | `2e-4` |
| grad clip | `3.0` |
| 最佳 epoch | `42` |
| 最佳验证集准确率 | `0.6901` |
| 测试集准确率 | `0.6758` |

复现命令：

```bash
python -X utf8 "hw1/train.py" --preset best
```

若直接使用已上传权重做评估，可先从 ModelScope 下载 `final_p/best_model.npz`，再运行：

```bash
python -X utf8 "hw1/evaluate.py" --preset best --checkpoint "/path/to/final_p/best_model.npz"
```

## 4. 搜索与优化结果

| 实验 | hidden1 -> hidden2 | epoch | lr | decay | wd | clip | val acc | test acc | 说明 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `trial_01` | `768 -> 512` | 28 | 0.010 | 0.03 | 5e-5 | 2.5 | 0.6659 | 0.6588 | 正式搜索中的基线 |
| `trial_02` | `1280 -> 768` | 28 | 0.012 | 0.01 | 2e-4 | 3.0 | 0.6837 | 0.6635 | 扩宽双隐层后明显提升 |
| `final_a` | `1280 -> 768` | 36 | 0.012 | 0.01 | 2e-4 | 3.0 | 0.6849 | 0.6669 | 第一版正式提交模型 |
| `final_b` | `1280 -> 1024` | 36 | 0.010 | 0.01 | 1e-4 | 3.0 | 0.6790 | 0.6704 | 更宽第二层提升了测试集，但验证集略退 |
| `final_c` | `1536 -> 768` | 32 | 0.010 | 0.01 | 1e-4 | 2.5 | 0.6815 | **0.6748** | 扩展实验中的最高测试精度 |
| `final_d` | `1536 -> 768` | 36 | 0.010 | 0.01 | 2e-4 | 2.5 | 0.6765 | 0.6684 | 继续训练后未继续提升 |
| `final_l` | `1280 -> 768` + `dropout 0.15` | 40 | 0.012 | 0.01 | 2e-4 | 3.0 | 0.6886 | 0.6714 | dropout 明显提升了验证集 |
| `final_o` | `1280 -> 768` + `dropout 0.18` | 42 | 0.012 | 0.01 | 2e-4 | 3.0 | 0.6877 | **0.6810** | 当前最高测试集精度 |
| `final_p` | `1280 -> 768` + `dropout 0.15` | 44 | 0.012 | 0.01 | 2e-4 | 3.0 | **0.6901** | 0.6758 | 当前正式提交模型 |

从这些结果可以看到：

- 只把 `768 -> 512` 扩到 `1280 -> 768`，验证集精度就从 `0.6659` 提升到 `0.6837`
- 在三层 MLP 不变的前提下，扩大第一隐藏层比盲目延长训练更有效
- `ReLU + 较小 lr 衰减 + 中等 weight decay + 轻量梯度裁剪` 是当前最稳定的基础组合
- 在此基础上加入轻量 dropout 后，验证集还能继续提升，说明先前模型已经进入轻微过拟合区间
- `final_o` 的测试集更高，但因为验证集低于 `final_p`，所以不能把它当作正式提交模型，只能作为扩展实验说明“三层 MLP 的上限还可继续逼近”

## 5. 必要图表

### 5.1 训练曲线

![训练曲线](outputs/runs/final_p/training_curves.png)

### 5.2 混淆矩阵

![混淆矩阵](outputs/runs/final_p/confusion_matrix.png)

### 5.3 第一层权重可视化

![第一层权重](outputs/runs/final_p/first_layer_weights.png)

### 5.4 错例分析

![错例分析](outputs/runs/final_p/misclassified_examples.png)

## 6. 结果分析

正式提交模型 `final_p` 的分类别准确率如下：

| 类别 | acc |
|---|---:|
| AnnualCrop | 0.7111 |
| Forest | 0.8467 |
| HerbaceousVegetation | 0.5733 |
| Highway | 0.3973 |
| Industrial | 0.8347 |
| Pasture | 0.7900 |
| PermanentCrop | 0.4667 |
| Residential | 0.6533 |
| River | 0.6480 |
| SeaLake | 0.8156 |

现象总结：

- 最容易分类的是 `Forest`、`Industrial`、`SeaLake`，说明这几类在颜色纹理上更稳定
- 最难的仍然是 `Highway` 和 `PermanentCrop`，这两类会和 `River`、`Residential`、`AnnualCrop`、`HerbaceousVegetation` 发生明显混淆
- `dropout=0.15` 后，验证集最优值进一步提升，说明适度随机失活确实改善了泛化
- `final_o` 把测试集推到 `0.6810`，说明若以测试集上限为目标，稍强的 dropout 也值得继续深挖

## 7. 结论

- 在满足作业要求的前提下，当前这套三层 MLP 已经把正式验证集结果提高到 `69.01%`
- 扩展实验中，测试集最高已达到 `68.10%`
- 继续优化时，优先方向仍然是“三层 MLP 内部”的 dropout、宽度和训练轮数组合，而不是跳到明显超出作业边界的新结构
- 因此，本次提交采用 `final_p` 作为正式模型，同时将 `final_o` 作为扩展实验展示测试集上限
