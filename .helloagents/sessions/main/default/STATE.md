# 恢复快照

## 主线目标
完成 HW3 Task1 真实高质量 3DGS 与 AIGC 多源资产融合链路，并按 hw3.md 输出可用于报告的结果、视频、评估和训练权重托管记录。

## 正在做什么
Task1 已重新打开并修正最终渲染口径：后合成 backplate 视频不再作为 strict 交付；136 上已生成 strict 统一 3D `fused_splats` 视频并通过远程 strict evaluate。当前 Task1 build/QA 已闭合为 PASS_WITH_LIMITATIONS：strict 统一 3D 视频通过远程 strict evaluate，但视觉质量弱于 backplate 预览；HW3 整体仍需继续 Task2 和最终报告。

## 关键上下文
- 训练与远程运行通过 `.helloagents/secrets/hw3.env` 注入 SwanLab/ModelScope/代理环境变量；不得在状态、文档、命令输出中打印 key 值。
- 136 机器：`136-3090-4`；项目路径：`/home/dell/yc/CS60003`；conda 环境：`qwen14b`。
- A：真实多视角素材 + COLMAP + Nerfstudio `splatfacto` 30k steps；导出 `exports/object_a/splat/splat.ply`；SwanLab run 已写入 `hw3/task1/SWANLAB_RUNS.md`。
- B：threestudio SDS 15k steps；导出 `exports/object_b/mesh/model.obj` 和 `object_b_threestudio/**/ckpts/last.ckpt`；SwanLab run 已记录。
- C：Zero123XL 1200 steps；导出 `exports/object_c/mesh/model.obj` 和 `object_c_zero123/**/ckpts/last.ckpt`；SwanLab run 已记录。
- 背景：Mip-NeRF 360 `counter` + Nerfstudio `splatfacto` 30k steps；导出 `exports/background/splat/splat.ply`；SwanLab run 已记录。
- strict 统一 3D 视频：`hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene.mp4`；manifest 记录 renderer=`gsplat fused splat renderer`，四个资产源分别为 A/background splat 与 B/C OBJ。
- 高质量视觉预览视频：`hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/fused_scene.mp4`；只能作为预览，不能作为 strict 交付。
- 评估文档：`hw3/task1/EVALUATION.md`；结果说明：`hw3/task1/RESULTS.md`；SwanLab 链接：`hw3/task1/SWANLAB_RUNS.md`；ModelScope 权重清单：`hw3/task1/MODELSCOPE_WEIGHTS.md`。
- ModelScope 仓库：`https://www.modelscope.cn/models/youngchen/CS60003`；远程子目录：`hw3/task1/real_high_quality/`；只上传最终训练权重，不上传 config、metrics、图片、视频、报告素材或 COLMAP 中间文件。
- 最新验证：本地 `evaluate.py` 非 strict 通过；本地 strict 会因本地未拉全权重而失败；136 strict evaluate 已通过。密钥泄露扫描仍需在提交前重跑。
- 当前限制：strict 统一 3D 视频避免了 2D/2.5D 偷懒，但背景 splat 相机/尺度视觉质量仍差；报告中必须如实说明。

## 下一步
继续 HW3 Task2 和最终报告；如时间允许，进一步提升 strict fused_splats 背景相机/尺度视觉质量。

## 阻塞项
无当前执行阻塞；136 连接偶发超时，必要时重试。

## 方案
.helloagents/plans/202606051645_hw3_task1_3dgs_aigc/

## 已标记技能
helloagents
