# 恢复快照

## 主线目标
推进 HW3 Task2 CALVIN 数据准备：在不下载完整 task_ABC_D.zip 的前提下，验证能否从官方大 ZIP 中按 Range 抽取 A/B/C 子集 episode。

## 正在做什么
已完成脚本实现与远程验证：`hw3/task2/scripts/extract_calvin_zip_subset.py` 可读取官方 `task_ABC_D.zip` ZIP64 中央目录，按 `scene_info.npy` 选择 A/B/C episode，并只下载选中成员。

## 关键上下文
- 远程机器：`135-3090-8`；项目路径：`/data/yc/CS60003`；Python 环境：`/data/yc/miniconda/envs/llm-26-gpu`。
- 官方 `task_ABC_D.zip` 大小约 555,309,812,705 bytes，中央目录约 229,211,511 bytes；服务器支持 HTTP Range。
- 官方没有单独 A/B/C 小 ZIP；脚本缓存中央目录到 `hw3/task2/cache/calvin_remote_zip`，避免完整下载 517GB 大包。
- 语言标注数组默认不抽取；如需抽取可加 `--include-lang-annotations`。本轮验证只证明 A/B/C episode npz 与基础元数据可抽取。
- 远程验证输出目录：`/data/yc/CS60003/hw3/task2/data/calvin_abc_subset_probe`。
- 已验证抽取 A/B/C 各 1 个 training episode：B=`episode_0000000.npz`，C=`episode_0598910.npz`，A=`episode_1191339.npz`；每个 npz 包含 `rgb_static`、`rgb_gripper`、`depth_static`、`robot_obs`、`actions`、`rel_actions` 等字段。
- 本地脚本已通过 `python3 -m py_compile`；远程脚本已同步。

## 下一步
基于该脚本扩展 Task2 数据加载/训练流程；需要更大样本时增加 `--episodes-per-scene`，仍避免完整下载大 ZIP。

## 阻塞项
无。
