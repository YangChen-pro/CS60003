# HW3 Task1 Results

## 当前状态

Task1 已建立工程骨架，当前处于 smoke test 阶段。

## Smoke test 数据

- 物体 A：8 张 AI 生成多视角 PNG。
- 物体 C：1 张 AI 生成单图 PNG。
- 这些图片只用于早期流程验证，不作为最终实验素材。

## 待完成正式结果

- 物体 A 真实手机多视角重建。
- 物体 B 文本到 3D 生成。
- 物体 C 真实手机单图到 3D 生成。
- Mip-NeRF 360 背景场景 3DGS 重建。
- 三类资产与背景场景融合渲染。
- 几何准确度、纹理细节和计算耗时对比表。

## 最新验证

### 2026-06-05 本地 `ai_generated_smoke`

命令：

```bash
python3 -X utf8 hw3/task1/train.py --config hw3/task1/configs/ai_generated_smoke.yaml
python3 -X utf8 hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_ai_generated_smoke
```

结果：

- 状态：`PASS`
- 图片总数：9
- 物体 A 多视角图：8
- 物体 C 单图：1
- 最小尺寸：`1254x1254`
- 相邻 yaw 缩略图平均 RMS 差异：`10.731`
- 运行目录：`hw3/task1/outputs/task1_ai_generated_smoke`

本地未检测到 `colmap`、`ffmpeg`、`blender`、`nvidia-smi`，这不影响当前图片 smoke test；后续 136 / 专用环境再验证正式 3DGS 工具链。

### 2026-06-05 136 `qwen14b` 远程 `ai_generated_smoke`

命令：

```bash
python hw3/task1/train.py --config hw3/task1/configs/ai_generated_smoke.yaml
python hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_ai_generated_smoke
```

结果：

- 状态：`PASS`
- 图片总数：9
- 物体 A 多视角图：8
- 物体 C 单图：1
- 最小尺寸：`1254x1254`
- 相邻 yaw 缩略图平均 RMS 差异：`10.731`
- 运行目录：`/home/dell/yc/CS60003/hw3/task1/outputs/task1_ai_generated_smoke`
- 远程环境：Python `3.10.19`，`nvidia-smi` 位于 `/usr/bin/nvidia-smi`

当前结论：Task1 工程骨架和 AI 生成图 smoke test 已经在本地与 136 上通过；正式 3DGS/SDS 实验还需要真实手机素材和后续外部工具链。
