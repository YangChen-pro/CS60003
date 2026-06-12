# 恢复快照

## 主线目标
推进 HW3 Task2 CALVIN 数据准备：在不下载完整 task_ABC_D.zip 的前提下，准备足够完成 A-only、ABC 联合训练和 D zero-shot 验证的可控子集。

## 正在做什么
已完成脚本增强与远程下载：`hw3/task2/scripts/extract_calvin_zip_subset.py` 支持 tqdm 进度条、并行 Range 抽取、紧凑 manifest 输出；135 上已下载 Task2 子集。

## 关键上下文
- 远程机器：`135-3090-8`；项目路径：`/data/yc/CS60003`；Python 环境：`/data/yc/miniconda/envs/llm-26-gpu`。
- 官方 `task_ABC_D.zip` 约 517GB；当前方案不下载完整 ZIP，只缓存约 219MB 中央目录。
- 训练子集：`/data/yc/CS60003/hw3/task2/data/calvin_task2_subset/training_abc_2k`，包含 A/B/C 各 2000 个 training npz，可同时用于 A-only 与 ABC 联合训练。
- D 验证子集：`/data/yc/CS60003/hw3/task2/data/calvin_task2_subset/validation_d_1k`，包含 validation/D 1000 个 npz。
- 当前子集总占用约 1.9GB，中央目录缓存约 219MB；`/data` 剩余约 500GB。
- 已验证样本 shape：`rgb_static=(200,200,3)`，`rel_actions=(7,)`，`robot_obs=(15,)`。

## 下一步
基于 `calvin_task2_subset` 接 Task2 数据加载、A-only ACT 训练、ABC ACT 训练和 D zero-shot 评估。

## 阻塞项
无。
