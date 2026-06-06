# 恢复快照

## 主线目标
完成 HW3 Task1 真实高质量 3DGS 与 AIGC 多源资产融合链路，并按 hw3.md 输出可用于报告的结果、视频、评估和训练权重托管记录。

## 正在做什么
Task1 本轮训练、SwanLab 记录、渲染、评估文档、ModelScope 权重上传和本地/远程严格检查已闭合；下一步是撰写 HW3 报告或继续 Task2。

## 关键上下文
- 训练与远程运行通过 `.helloagents/secrets/hw3.env` 注入 SwanLab/ModelScope/代理环境变量；不得在状态、文档、命令输出中打印 key 值。
- 136 机器：`136-3090-4`；项目路径：`/home/dell/yc/CS60003`；conda 环境：`qwen14b`。
- A：真实多视角素材 + COLMAP + Nerfstudio `splatfacto` 30k steps；导出 `exports/object_a/splat/splat.ply`；SwanLab run 已写入 `hw3/task1/SWANLAB_RUNS.md`。
- B：threestudio SDS 15k steps；导出 `exports/object_b/mesh/model.obj` 和 `object_b_threestudio/**/ckpts/last.ckpt`；SwanLab run 已记录。
- C：Zero123XL 1200 steps；导出 `exports/object_c/mesh/model.obj` 和 `object_c_zero123/**/ckpts/last.ckpt`；SwanLab run 已记录。
- 背景：Mip-NeRF 360 `counter` + Nerfstudio `splatfacto` 30k steps；导出 `exports/background/splat/splat.ply`；SwanLab run 已记录。
- 最终本地视频：`hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/fused_scene.mp4`；关键帧：同目录 `frame_0072.png`。
- 最终远程视频：`/home/dell/yc/CS60003/hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/fused_scene.mp4`。
- 评估文档：`hw3/task1/EVALUATION.md`；结果说明：`hw3/task1/RESULTS.md`；SwanLab 链接：`hw3/task1/SWANLAB_RUNS.md`；ModelScope 权重清单：`hw3/task1/MODELSCOPE_WEIGHTS.md`。
- ModelScope 仓库：`https://www.modelscope.cn/models/youngchen/CS60003`；远程子目录：`hw3/task1/real_high_quality/`；只上传最终训练权重，不上传 config、metrics、图片、视频、报告素材或 COLMAP 中间文件。
- 视频策略：背景为训练完成 3DGS backplate；A 使用真实前景 mask/cutout；B/C 使用训练后的测试渲染序列并保留 mesh/ckpt 权重。报告需如实说明这不是低质量 proxy，也不是严格单渲染器原生融合。
- 不得恢复 `ai_assets_high_quality_preview`、`preview_chain`、`formal_ai_chain` 或程序化预览渲染作为正式结果。
- 最新验证：本地 py_compile 通过；本地 `evaluate.py` 通过；远程 `evaluate.py --strict-real-outputs` 通过；密钥泄露扫描通过。

## 下一步
撰写 HW3 Task1 报告分析，或进入 HW3 Task2。

## 阻塞项
无当前执行阻塞。限制：严格单渲染器 splat/mesh 原生融合质量仍低于最终 backplate 合成视频，报告中需说明取舍。

## 方案
.helloagents/plans/202606051645_hw3_task1_3dgs_aigc/

## 已标记技能
helloagents
