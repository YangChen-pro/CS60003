# 项目状态

主线目标：完成 HW2 Task3 Stanford Background U-Net 语义分割实验，并比较 CE / Dice / CE+Dice 的验证集 mIoU。

正在做什么：Task3 工程代码已在本机创建；下一步通过 Git 同步到远程 135，执行远程 compile / 小张量检查和三组正式训练。

关键上下文：
- 题面文件：`hw2/hw2.md`
- Task3 数据集：`hw2/StanfordBackground/iccv09Data/`，含 `images/` 与 `labels/*.regions.txt`，共 715 对图像/语义标签。
- 语义类别：8 类（sky、tree、road、grass、water、building、mountain、foreground object），标签 `-1` 表示 unknown，训练和指标中需 ignore。
- 方案包：`.helloagents/plans/202604291645_hw2_task3_unet_segmentation/`
- 实现路径：`hw2/task3/`
- 已创建内容：手写 U-Net、手写 Dice Loss、CE/Dice/CE+Dice 配置、数据读取、mIoU 指标、SwanLab 记录、ModelScope 上传脚本、README/RESULTS 占位。
- 实验要求：从零手写 U-Net，不使用预训练；手写 Dice Loss；训练 `ce`、`dice`、`ce_dice` 三组并比较验证集 mIoU。
- 记录要求：接入 SwanLab，曲线必须有明确横纵轴；最佳模型后续上传到 ModelScope `youngchen/CS60003` 的 `hw2/task3/` 子目录。
- 用户确认：不在本机 smoke test；本机只做编辑和文档整理。
- 用户确认：代码通过 Git 同步到远程，远程机器使用前检查 Git 身份是否与本机一致。
- 远程主机：`ssh 135-3090-8`
- 远程优先环境：`/data/yc/miniconda/envs/llm-26-gpu`，执行前需确认 PyTorch / torchvision / CUDA / SwanLab 可用。
- Git 身份目标：`YangChen-pro <1369792882@qq.com>`

下一步：提交并推送当前 Task3 工程代码，再在远程 135 拉取后运行检查与三组正式实验。

阻塞项：
- 无

方案：
- `.helloagents/plans/202604291645_hw2_task3_unet_segmentation/`
