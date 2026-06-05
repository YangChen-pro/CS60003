# HW3 Task1 3DGS/AIGC — 任务分解

## 拆分原则
- 默认按端到端垂直切片拆分：每个任务交付一个可验证行为，而不是单独交付某一层。
- `AFK` 表示代理可独立完成；`HITL` 表示需要用户决策、外部凭据、人工视觉确认或手动验收。
- 厚任务必须继续拆小；横向前置任务只在确有技术依赖时保留。

## 任务列表
- [√] 任务1（AFK）：建立 Task1 标准工程骨架（依赖：无；涉及文件：`hw3/task1/`、`.gitignore`；预期变更：README、RESULTS、configs、requirements、train/evaluate、内部包；完成标准：结构对齐 HW2 task 目录；验证方式：文件检查与 py_compile）。
- [√] 任务2（AFK）：实现 AI 生成测试图 smoke test（依赖：任务1；涉及文件：`hw3/task1/task1_3dgs_aigc/`；预期变更：manifest、图像统计、相邻视角差异、contact sheet；完成标准：9 张图片验证通过；验证方式：`python3 hw3/task1/train.py --config hw3/task1/configs/ai_generated_smoke.yaml`）。
- [√] 任务3（AFK）：在 136 `qwen14b` 运行 smoke test（依赖：任务2；涉及文件：远程 `/home/dell/yc/CS60003`；预期变更：远程输出目录；完成标准：远程 train/evaluate 均通过；验证方式：远程命令输出）。
- [√] 任务4（AFK）：按用户最新要求将 AI 素材作为正式输入（依赖：用户授权；涉及文件：`hw3/task1/configs/formal_ai_chain.yaml`；预期变更：formal 配置；完成标准：`task1.stage=formal_ai_chain` 且 `final_submission_assets=true`；验证方式：配置检查）。
- [√] 任务5（AFK）：实现物体 B 文本到 3D 代理、背景代理和 A/B/C 融合渲染链路（依赖：任务4；涉及文件：`hw3/task1/task1_3dgs_aigc/formal_chain.py`、`geometry.py`；预期变更：PLY、metrics、preview、GIF；完成标准：本地 formal train/evaluate 通过；验证方式：`python3 hw3/task1/train.py --config hw3/task1/configs/formal_ai_chain.yaml` 与 evaluate）。
- [√] 任务6（AFK）：在 136 `qwen14b` 运行 formal AI chain 并上传/记录结果（依赖：任务5；涉及文件：远程 `/home/dell/yc/CS60003`；预期变更：远程输出目录与 ModelScope 记录；完成标准：远程 formal train/evaluate 通过，必要素材可引用；验证方式：远程命令输出与 `RESULTS.md`）。
- [√] 任务7（AFK）：增加真实高质量链路入口（依赖：任务6；涉及文件：`hw3/assets/README.md`、`hw3/task1/configs/real_high_quality.yaml`、`hw3/task1/task1_3dgs_aigc/real_chain.py`、`hw3/task1/scripts/`；预期变更：真实素材路径约定、外部工具脚本、136 安装脚本；完成标准：替换真实 A/C/背景素材后可切换 `mode: run` 执行真实链路；验证方式：plan 模式生成 7 个脚本并 evaluate 通过）。

## Codex /goal 执行入口
/goal 按 `.helloagents/plans/202606051645_hw3_task1_3dgs_aigc/tasks.md` 执行本方案；遵守 `requirements.md`、`plan.md`、`contract.json`。默认主执行命令是 `~auto`；按顺序完成所有 AFK 任务；HITL 仅在缺少真实素材或人工验收时暂停。全部 AFK 任务完成后必须进入 `~qa`，写最新质量证据并完成 HelloAGENTS 收尾，再标记 goal complete。

## 进度
- [√] 2026-06-05：已建立 `hw3/task1/` 工程骨架，并在本地完成 `ai_generated_smoke` train/evaluate（涉及文件：`hw3/task1/`；完成标准：本地 smoke 产物完整；验证方式：本地 `train.py` / `evaluate.py` 命令输出）。
- [√] 2026-06-05：已同步到 136 并在 `qwen14b` 运行同一 smoke test（涉及文件：远程 `/home/dell/yc/CS60003/hw3/task1/`；完成标准：远程 smoke 产物完整；验证方式：136 `train.py` / `evaluate.py` 命令输出）。
- [√] 2026-06-05：用户改为使用 AI 素材作为正式链路输入；已在本地跑通 `formal_ai_chain`（涉及文件：`hw3/task1/configs/formal_ai_chain.yaml`、`hw3/task1/task1_3dgs_aigc/formal_chain.py`、`hw3/task1/report_assets/formal_ai_chain/`；完成标准：本地 formal 输出完整；验证方式：本地 `train.py` / `evaluate.py` 命令输出）。
- [√] 2026-06-05：已同步到 136 并在 `qwen14b` 运行 `formal_ai_chain`（涉及文件：远程 `/home/dell/yc/CS60003/hw3/task1/`；完成标准：远程 formal 输出完整；验证方式：136 `train.py` / `evaluate.py` 命令输出）。
- [√] 2026-06-05：已将 formal AI chain 关键产物上传到 ModelScope `youngchen/CS60003` 的 `hw3/task1/formal_ai_chain/`（涉及文件：`upload_modelscope.py`、ModelScope 仓库；完成标准：summary、metrics、PLY、preview、GIF 上传成功；验证方式：ModelScope upload 命令逐项输出 `uploaded ...`）。
- [√] 2026-06-05：已补真实高质量链路 plan 模式（涉及文件：`real_high_quality.yaml`、`real_chain.py`、`render_real_chain_blender.py`、`setup_real_chain_136.sh`；完成标准：生成 7 个真实外部工具脚本；验证方式：`python3 hw3/task1/train.py --config hw3/task1/configs/real_high_quality.yaml` 与 evaluate）。
