# HW3 Task1 3DGS/AIGC — 任务分解

## 拆分原则
- 默认按端到端垂直切片拆分：每个任务交付一个可验证行为，而不是单独交付某一层。
- `AFK` 表示代理可独立完成；`HITL` 表示需要用户决策、外部凭据、人工视觉确认或手动验收。

## 任务列表
- [√] 任务1（AFK）：建立 Task1 标准工程骨架（涉及文件：`hw3/task1/`、`.gitignore`；完成标准：结构对齐 HW1/HW2；验证方式：文件检查与 py_compile）。
- [√] 任务2（AFK）：增加真实高质量链路入口（涉及文件：`real_high_quality.yaml`、`real_chain.py`、`render_real_chain_blender.py`、`setup_real_chain_136.sh`；完成标准：plan 模式生成 7 个脚本；验证方式：train/evaluate）。
- [√] 任务3（AFK）：删除旧低质量链路（涉及文件：`ai_generated_smoke.yaml`、`formal_ai_chain.yaml`、`smoke.py`、`formal_chain.py`、`geometry.py`、`report_assets/formal_ai_chain/`、`hw3/assets/ai_generated_test/`；完成标准：Git 中只保留真实链路；验证方式：`git status` 与 `rg`）。
- [√] 任务4（AFK）：收敛 ModelScope 上传策略（涉及文件：`upload_modelscope.py`；完成标准：只允许训练权重上传；验证方式：py_compile 和代码审查）。
- [√] 任务5（AFK）：清理 ModelScope 远端非权重文件（涉及文件：ModelScope Git 仓库 `youngchen/CS60003`；完成标准：删除 `hw3/task1/formal_ai_chain/`；验证方式：ModelScope Git push 成功）。
- [√] 任务6（AFK）：本地验证、提交并同步 136（涉及文件：本地与远程 `/home/dell/yc/CS60003`；完成标准：本地和 136 验证通过且远程代码同步；验证方式：py_compile、train/evaluate）。
- [√] 任务7（AFK）：QA / closeout（涉及文件：`.helloagents` 证据；完成标准：写入最新 QA 与收尾证据；验证方式：HelloAGENTS gate 通过）。

## Codex /goal 执行入口
/goal 按 `.helloagents/plans/202606051645_hw3_task1_3dgs_aigc/tasks.md` 执行本方案；遵守 `requirements.md`、`plan.md`、`contract.json`。默认主执行命令是 `~auto`；全部 AFK 任务完成后必须进入 `~qa`，写最新质量证据并完成 HelloAGENTS 收尾。

## 进度
- [√] 2026-06-05：已建立 `hw3/task1/` 工程骨架（涉及文件：`hw3/task1/`；完成标准：基础结构存在；验证方式：文件检查）。
- [√] 2026-06-05：已补真实高质量链路 plan 模式（涉及文件：`real_high_quality.yaml`、`real_chain.py`；完成标准：生成 7 个脚本；验证方式：train/evaluate）。
- [√] 2026-06-05：已删除旧低质量链路文件（涉及文件：`hw3/assets/ai_generated_test/`、`hw3/task1/configs/ai_generated_smoke.yaml`、`hw3/task1/configs/formal_ai_chain.yaml`、旧 Python 模块与报告素材；完成标准：Git 删除完成；验证方式：`git status`）。
- [√] 2026-06-05：已把 ModelScope 上传脚本收敛为权重白名单（涉及文件：`hw3/task1/upload_modelscope.py`；完成标准：只上传训练权重；验证方式：py_compile）。
- [√] 2026-06-05：已通过 ModelScope Git 删除 `hw3/task1/formal_ai_chain/` 非权重文件，提交 `a430be1`（涉及文件：ModelScope Git 仓库 `youngchen/CS60003`；完成标准：远端删除提交成功；验证方式：`git push` 输出）。
- [√] 2026-06-05：本地已通过 metadata、py_compile、real_high_quality plan train/evaluate（涉及文件：本地 `hw3/task1/`；完成标准：本地验证通过；验证方式：`tools/verify_delivery_metadata.py`、py_compile、train/evaluate）。
- [√] 2026-06-05：已提交 `b1a805e` 并同步 136，136 `qwen14b` 已通过 py_compile、real_high_quality plan train/evaluate（涉及文件：远程 `/home/dell/yc/CS60003`；完成标准：远程验证通过且工作区干净；验证方式：ssh 远程命令）。
