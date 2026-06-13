# 恢复快照

## 主线目标
完成 HW3 Task1 与 Task2 的工程、实验、权重发布和报告准备，并保持知识库、GitHub、本机与远端 135/136 状态可恢复。

## 正在做什么
已按用户要求同步 `.helloagents` 知识库，重点补齐 HW3 Task2 的 LeRobot ACT 正式实验、135 运行环境、数据划分、SwanLab 记录、ModelScope 最终目录和上传配置。

## 关键上下文
- 训练与远程运行通过 `.helloagents/secrets/hw3.env` 注入 SwanLab/ModelScope/代理环境变量；不得在状态、文档、命令输出中打印 key 值。
- Task1 远程主机：`136-3090-4`；项目路径：`/home/dell/yc/CS60003`；conda 环境：`qwen14b`。
- Task1 strict 统一 3D 视频：`hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene.mp4`；视觉预览视频 `final_3dgs_backplate/fused_scene.mp4` 只能作为预览，不能作为 strict 交付。
- Task1 ModelScope 仓库：`https://www.modelscope.cn/models/youngchen/CS60003`；远程子目录：`hw3/task1/real_high_quality/`；只上传最终训练权重。
- Task2 远程主机：`135-3090-8`；项目路径：`/data/yc/CS60003`；Python 环境：`/data/yc/miniconda/envs/llm-26-gpu`。
- Task2 数据目录：`/data/yc/CS60003/hw3/task2/data/calvin_lerobot`；split 映射：`splitA=A`、`splitB=B`、`splitC=C`、`splitD=D`。
- Task2 正式模型：`act_splitA` 用 `splitA` 训练，`act_splitABC` 用 `splitA+splitB+splitC` 训练，均在未见过的 `splitD` 评估。
- Task2 正式结果：`act_splitA` 显式评估 Action L1 `0.1731817282`；`act_splitABC` 显式评估 Action L1 `0.1464656246`。
- Task2 SwanLab 项目：`CS60003_HW3_Task2`；run id：`act_splitA=05kubpls24j5jrp2wbl1t`，`act_splitABC=4si6dcrut2krorrbalfkn`。
- Task2 ModelScope 最终位置：`youngchen/CS60003/hw3/task2/`，包含 `act_splitA/`、`act_splitABC/`、`eval/`；旧独立模型仓不是交付位置。
- 本机、GitHub、135 最新同步提交：`b767eb11 docs(hw3): 更新 task2 ModelScope 位置`。

## 下一步
继续 HW3 最终报告整合；如需要云端清理，旧 Task2 独立 ModelScope 模型仓需在 ModelScope 网页端手动删除，因为当前 token 不支持删除仓库。

## 阻塞项
无代码阻塞；旧独立 ModelScope 模型仓删除受平台限制，需要网页端操作。

## 方案
.helloagents/plans/202606051645_hw3_task1_3dgs_aigc/

## 已标记技能
helloagents
