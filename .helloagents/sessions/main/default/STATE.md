# 恢复快照

## 主线目标
开始做 HW3 Task1：3DGS 与 AIGC 多源资产融合。

## 正在做什么
已建立 `hw3/task1/` 工程骨架，并完成本地与 136 `qwen14b` 的 AI 生成素材 smoke test。

## 关键上下文
- Task1 结构对齐 HW2：README、RESULTS、configs、requirements、train/evaluate、内部包。
- 当前 smoke test 使用 `hw3/assets/ai_generated_test/` 的 8 张物体 A 多视角图和 1 张物体 C 单图；这些不是最终实拍素材。
- 方案包：`.helloagents/plans/202606051645_hw3_task1_3dgs_aigc/`。
- 本地验证已通过：`python3 -X utf8 hw3/task1/train.py --config hw3/task1/configs/ai_generated_smoke.yaml` 与 `python3 -X utf8 hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_ai_generated_smoke`。
- 136 验证已通过：`python hw3/task1/train.py --config hw3/task1/configs/ai_generated_smoke.yaml` 与 `python hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_ai_generated_smoke`。

## 下一步
等待用户提供真实手机拍摄的物体 A 多视角素材和物体 C 单图；拿到素材后放入 `hw3/task1/data/` 或远程数据目录，再进入正式 3DGS/SDS 链路。

## 阻塞项
（无）

## 方案
.helloagents/plans/202606051645_hw3_task1_3dgs_aigc/

## 已标记技能
helloagents
