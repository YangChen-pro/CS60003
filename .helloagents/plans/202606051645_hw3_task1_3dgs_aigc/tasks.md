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
- [√] 任务7（AFK）：清理预览配置和旧运行产物（涉及文件：`ai_assets_high_quality_preview.yaml`、`preview_chain.py`、`render_ai_assets_preview_blender.py`、`outputs/`；完成标准：只保留正式 `real_high_quality` 链路；验证方式：文件检查与 py_compile）。
- [√] 任务8（AFK）：把正式链路从 TripoSR 口径修正为 Zero123，并增加 A 的前景保真预处理（涉及文件：`real_high_quality.yaml`、`real_chain.py`、`preprocess_object_foreground.py`、`evaluate.py`、`render_real_chain_blender.py`、文档；完成标准：生成 `04_object_c_zero123.sh`，C 使用 threestudio `zero123.yaml`，A 的 `ns-process-data` 使用前景图与 CPU COLMAP；验证方式：py_compile、plan train/evaluate）。
- [√] 任务9（AFK）：增加 Mip-NeRF 360 `counter` 背景准备入口（涉及文件：`prepare_mipnerf360_counter_136.sh`、文档；完成标准：脚本能下载官方数据并链接 `hw3/assets/background_scene/images`；验证方式：脚本审查和 plan input issue 定位）。
- [√] 任务10（AFK）：接入 SwanLab 训练记录（涉及文件：`run_nerfstudio_swanlab.py`、`run_threestudio_swanlab.py`、`real_chain.py`、`real_high_quality.yaml`、`requirements.txt`；完成标准：A/background 训练脚本走 TensorBoard scalar sync，B/C 训练脚本走 WandB scalar sync 到 SwanLab；验证方式：plan 脚本审查、py_compile）。
- [√] 任务11（AFK）：同步 136 并检查 Zero123 / 背景依赖 / SwanLab 环境（涉及文件：远程 `/home/dell/yc/CS60003`；完成标准：136 plan train/evaluate 通过，`00_check_tools.sh` 明确通过或准确报告缺失权重/背景/SwanLab 依赖；验证方式：ssh 远程命令）。
- [√] 任务12（AFK/HITL）：在 136 执行 run 模式真实训练、导出与融合渲染（涉及文件：`hw3/task1/outputs/task1_real_high_quality/`；完成标准：A/B/C/background 真实输出与 1080p `fused_scene.mp4` 存在，SwanLab 有训练曲线；验证方式：运行日志、输出 manifest、视频文件、SwanLab run 链接）。
- [√] 任务13（AFK）：评估与报告素材整理（涉及文件：`RESULTS.md`、`SWANLAB_RUNS.md`、报告素材目录；完成标准：几何/纹理/耗时对比、SwanLab 曲线、ModelScope 权重路径记录；验证方式：evaluate 与文件检查）。
- [√] 任务14（AFK）：QA / closeout（涉及文件：`.helloagents` 证据；完成标准：全链路真实完成后写入最新 QA 与收尾证据；验证方式：HelloAGENTS gate 通过）。

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
- [√] 2026-06-05：已按用户要求删除 `ai_assets_high_quality_preview` 配置、预览代码和本地/136 旧运行产物（涉及文件：`hw3/task1/`、远程 `/home/dell/yc/CS60003/hw3/task1/outputs/`；完成标准：当前只保留正式链路；验证方式：文件检查、py_compile）。
- [√] 2026-06-05：已把物体 C 正式链路从 TripoSR 改为 threestudio Zero123，并增加 Mip-NeRF 360 `counter` 背景准备脚本（涉及文件：`hw3/task1/`、`hw3/assets/README.md`；完成标准：本地 plan 生成 `04_object_c_zero123.sh`；验证方式：py_compile、train/evaluate）。
- [√] 2026-06-05：已按用户要求把正式训练脚本接入 SwanLab（涉及文件：`run_nerfstudio_swanlab.py`、`run_threestudio_swanlab.py`、`real_chain.py`；完成标准：生成脚本引用 `.helloagents/secrets/hw3.env` 且不打印 key；验证方式：plan 脚本审查、py_compile）。
- [√] 2026-06-06：136 `qwen14b` 已有 A/background 3DGS splat、B/C OBJ + ckpt、SwanLab run 与 ModelScope 权重记录；远程 strict evaluate 已通过（涉及文件：远程 `hw3/task1/outputs/task1_real_high_quality/`；完成标准：真实训练产物存在；验证方式：ssh strict evaluate）。
- [√] 2026-06-06：已新增 strict 统一 3D 融合视频 `renders/fused_splats/fused_scene.mp4`，由 `render_fused_splats.py` 加载 A/background splat 与 B/C OBJ 在同一 gsplat renderer 下渲染；保留 `final_3dgs_backplate/` 仅作为视觉预览（涉及文件：`evaluate.py`、`RESULTS.md`、`EVALUATION.md`；完成标准：strict 不再接受 2D/2.5D 后合成；验证方式：远程 strict evaluate）。
- [√] 2026-06-06：已完成 QA 复验：本地 py_compile 通过、本地非 strict evaluate 通过、密钥扫描通过、136 远程 strict evaluate 通过；已记录 strict 视频视觉质量限制（涉及文件：`evaluate.py`、`README.md`、`RESULTS.md`、`EVALUATION.md`、`.helloagents` 状态；完成标准：QA 证据更新；验证方式：命令输出）。
