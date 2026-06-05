# HW3 Task1 3DGS/AIGC — 任务分解

## 拆分原则
- 默认按端到端垂直切片拆分：每个任务交付一个可验证行为，而不是单独交付某一层。
- `AFK` 表示代理可独立完成；`HITL` 表示需要用户决策、外部凭据、人工视觉确认或手动验收。
- 厚任务必须继续拆小；横向前置任务只在确有技术依赖时保留。

## 任务列表
- [√] 任务1（AFK）：建立 Task1 标准工程骨架（依赖：无；涉及文件：`hw3/task1/`、`.gitignore`；预期变更：README、RESULTS、configs、requirements、train/evaluate、内部包；完成标准：结构对齐 HW2 task 目录；验证方式：文件检查与 py_compile）。
- [√] 任务2（AFK）：实现 AI 生成测试图 smoke test（依赖：任务1；涉及文件：`hw3/task1/task1_3dgs_aigc/`；预期变更：manifest、图像统计、相邻视角差异、contact sheet；完成标准：9 张图片验证通过；验证方式：`python3 hw3/task1/train.py --config hw3/task1/configs/ai_generated_smoke.yaml`）。
- [ ] 任务3（AFK）：在 136 `qwen14b` 运行 smoke test（依赖：任务2；涉及文件：远程 `/home/dell/yc/CS60003`；预期变更：远程输出目录；完成标准：远程 train/evaluate 均通过；验证方式：远程命令输出）。
- [ ] 任务4（HITL）：替换为真实手机拍摄物体 A/C 素材（依赖：用户拍摄；涉及文件：`hw3/task1/data/` 或远程数据目录；预期变更：最终素材集；完成标准：真实素材通过 smoke test；验证方式：图片/视频检查与 COLMAP 输入检查）。
- [ ] 任务5（AFK）：接入 COLMAP/3DGS 背景与物体 A 正式重建（依赖：任务4、背景数据；涉及文件：后续 configs/scripts；完成标准：产生可报告重建结果；验证方式：重建日志、输出模型和渲染图）。

## Codex /goal 执行入口
/goal 按 `.helloagents/plans/202606051645_hw3_task1_3dgs_aigc/tasks.md` 执行本方案；遵守 `requirements.md`、`plan.md`、`contract.json`。默认主执行命令是 `~auto`；按顺序完成所有 AFK 任务；HITL 仅在缺少真实素材或人工验收时暂停。全部 AFK 任务完成后必须进入 `~qa`，写最新质量证据并完成 HelloAGENTS 收尾，再标记 goal complete。

## 进度
- [√] 2026-06-05：已建立 `hw3/task1/` 工程骨架，并在本地完成 `ai_generated_smoke` train/evaluate（涉及文件：`hw3/task1/`；完成标准：本地 smoke 产物完整；验证方式：本地 `train.py` / `evaluate.py` 命令输出）。
- [ ] 下一步：同步到 136 并在 `qwen14b` 运行同一 smoke test（涉及文件：远程 `/home/dell/yc/CS60003/hw3/task1/`；完成标准：远程 smoke 产物完整；验证方式：136 `train.py` / `evaluate.py` 命令输出）。
