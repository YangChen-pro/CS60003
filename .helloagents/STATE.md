# 项目状态

主线目标：HW2 Task1 Flowers102 冲分优化已完成，最佳 `test_acc=0.9608`。

正在做什么：ModelScope 权重目录迁移和 HW1 报告源文件引用更新已完成；等待按需重新编译 HW1 PDF 或继续 HW2。

关键上下文：
- 题面文件：`hw2/hw2.md`
- Task1 数据集：`hw2/Flowers102/`，含 `jpg/`、`imagelabels.mat`、`setid.mat`、`README.txt`
- 用户确认：不在本机 smoke test；本机只做编辑和文档整理。
- 用户确认：代码通过 Git 同步到远程，远程机器使用前检查 Git 身份是否与本机一致。
- 本机轻量环境：`conda run -n nlp`
- 远程主机：`ssh 135-3090-8`
- 远程优先环境：`/data/yc/miniconda/envs/llm-26-gpu`，执行前需确认 PyTorch / torchvision / CUDA 可用。
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`
- 当前同步状态：本机 / GitHub / 远程仓库以最新 `main` 为准，具体提交用 `git rev-parse HEAD` 核对。
- Task1 正式结果：`hw2/task1/RESULTS.md`
- 原始 Baseline：ResNet-18，`best_val_acc=0.7186`，`test_acc=0.6819`
- 优化方案：`.helloagents/plans/202604291137_hw2_task1_optimization/`
- 本轮优化方向：AdamW、label smoothing、RandAugment/RandomErasing、ResNet-34/50、EfficientNet-B0、ConvNeXt-Tiny、TTA
- 最佳优化结果：ConvNeXt-Tiny，`best_val_acc=0.9784`，`test_acc=0.9608`
- 相比原始 Baseline `test_acc=0.6819` 提升 `+27.89` 个百分点。
- SwanLab 要求：Task1 复现实验或报告补充需接入 SwanLab；用户已明确要求将临时 API key 写入 `.helloagents/modules/hw2.md` 并允许 Git 同步，作业完成后删除。
- SwanLab 接入实现：`hw2/task1/train.py` 可实时记录；`hw2/task1/upload_swanlab_history.py --all` 可回放已有正式实验曲线到 SwanLab。
- SwanLab 上传结果：项目 <https://swanlab.cn/@youngchen/cs60003-hw2-task1>，13 个正式实验修正版回放链接记录在 `hw2/task1/SWANLAB_RUNS.md`；每个 run 包含 `report/curves_with_axis_labels` 图像，横轴 `Epoch`，纵轴 `Loss` / `Accuracy`。
- ModelScope 要求：Task1 训练好的模型需要上传到公网，用户偏好 ModelScope；用户已明确要求将 ModelScope API token 写入 `.helloagents/modules/hw2.md`，按 SwanLab key 相同方式使用。
- ModelScope 上传结果：仓库 <https://modelscope.cn/models/youngchen/CS60003/> 已按作业目录整理；HW1 新路径为 `hw1/final_p/best_model.npz`、`hw1/final_o/best_model.npz`，HW2 Task1 最佳模型为 `hw2/task1/flowers102_convnext_tiny/best.pt`。
- HW1 报告源文件：`/Users/yangchen/Documents/Latex_Project/CS60003_HW1_Report/src/hw1.tex` 已更新为 ModelScope 新路径；尚未在本轮重新编译 PDF。

下一步：如需收口 HW1，重新编译 `/Users/yangchen/Documents/Latex_Project/CS60003_HW1_Report/src/hw1.tex` 生成 PDF；如继续 HW2，从 SwanLab 页面导出或截图，并在报告中引用 ModelScope 的 HW2 Task1 权重路径。

阻塞项：
- 无

方案：
- `.helloagents/plans/202604291137_hw2_task1_optimization/`
