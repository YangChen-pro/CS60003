# HW2 Task3 Stanford Background U-Net 分割 — 需求

确认后冻结，执行阶段不可修改。如需变更必须回到设计阶段重新确认。

## 核心目标
完成 HW2 Task3：从零搭建经典 U-Net，在 Stanford Background Dataset 上随机初始化训练语义分割模型，并比较 Cross-Entropy、手写 Dice Loss、Cross-Entropy + Dice 三种损失配置的验证集 mIoU。

## 功能边界
- 使用 `hw2/StanfordBackground/iccv09Data/` 数据集，图像与 `labels/*.regions.txt` 一一对应。
- 将语义类别按题面数据集说明处理为 8 类：sky、tree、road、grass、water、building、mountain、foreground object。
- 对标签中的 `-1` unknown 像素使用 ignore mask，不纳入 CE、Dice、mIoU、pixel accuracy 统计。
- 从随机初始化开始训练，不使用任何预训练权重。
- 手写搭建 U-Net，必须包含完整 encoder/downsampling、decoder/upsampling、skip connection 拼接逻辑。
- 手动实现 Dice Loss，并分别运行三组正式实验：`ce`、`dice`、`ce_dice`。
- 接入 SwanLab 记录训练/验证 loss、验证集 mIoU、验证集 pixel accuracy 和报告曲线图。
- 最佳权重后续上传到 ModelScope `youngchen/CS60003` 的 `hw2/task3/` 子目录。

## 非目标
- 不使用 torchvision / segmentation_models_pytorch 等现成分割网络或预训练 backbone。
- 不在本机做训练或 smoke test；本机只做代码与文档整理。
- 不把大型权重、输出目录、SwanLab 缓存写入 Git。
- 不扩展到 Task2；不把 Task1 结果重新训练纳入本轮。

## 技术约束
- 代码位置：`hw2/task3/`。
- 训练框架：PyTorch 基础 API，可使用 `Dataset`、`DataLoader`、tensor 运算、优化器和基础 loss，但 U-Net 与 Dice Loss 需项目代码手写。
- 数据划分：固定随机种子生成 train / val，优先使用 80% / 20% 分割，并保存 split 文件保证可复现。
- 输入尺寸：统一 resize 到 `240x320`，mask 用 nearest，image 用 bilinear；该尺寸与 U-Net 下采样倍率兼容。
- 远程主机：`135-3090-8:/data/yc/CS60003`。
- 远程环境：`/data/yc/miniconda/envs/llm-26-gpu`，执行前检查 PyTorch、torchvision、CUDA、SwanLab 可用。
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`。

## 质量要求
- 三组正式实验均需保存 `metrics.json`、`history.csv`、`curves.png`、`best.pt`、代表性验证集可视化图。
- `RESULTS.md` 必须汇总三组 loss 的 best val mIoU、pixel accuracy、收敛曲线和结论。
- 验证指标必须排除 ignored pixels；mIoU 必须按类别 IoU 求均值，并记录 per-class IoU。
- SwanLab 图像横轴、纵轴必须有明确标注，避免 Task1 曾出现的曲线标注问题。
- README 必须说明环境、训练命令、评估命令、SwanLab/ModelScope 产物链接和报告口径。
