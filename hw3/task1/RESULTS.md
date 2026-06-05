# HW3 Task1 Results

## 当前状态

Task1 已建立工程骨架，并按用户最新要求使用 AI 素材跑通本地 `formal_ai_chain`：物体 A 多视角代理、物体 B 文本到 3D 代理、物体 C 单图代理、背景代理和融合 turntable 渲染均已生成。

## 当前正式输入

- 物体 A：8 张 AI 生成多视角 PNG，作为多视角 Gaussian/point-cloud 代理输入。
- 物体 B：文本 prompt `a small bioluminescent purple crystal mushroom with a green stem`，由程序生成 3D 代理资产。
- 物体 C：1 张 AI 生成单图 PNG，作为单图挤出式 3D 代理输入。
- 背景：Mip-NeRF 360 风格桌面/墙面 Gaussian 背景代理。

## 当前正式结果

- 物体 A 多视角代理：`object_a_ai_multiview_gaussians.ply`。
- 物体 B 文本到 3D 代理：`object_b_text_to_3d_proxy.ply`。
- 物体 C 单图代理：`object_c_single_image_proxy.ply`。
- 背景代理：`background_proxy_gaussians.ply`。
- 融合场景：`fused_scene.ply`。
- 多视角渲染：`renders/fused_scene_turntable.gif`。
- 报告素材：`hw3/task1/report_assets/formal_ai_chain/`。

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

当前结论：Task1 工程骨架和 AI 生成图 smoke test 已经在本地与 136 上通过。


### 2026-06-05 本地 `formal_ai_chain`

命令：

```bash
python3 -X utf8 hw3/task1/train.py --config hw3/task1/configs/formal_ai_chain.yaml
python3 -X utf8 hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_formal_ai_chain
```

结果：

- 状态：`PASS`
- 资产数量：4（物体 A、物体 B、物体 C、背景）
- 融合点数：`58013`
- 物体 A 点数：`44077`
- 物体 B 点数：`2758`
- 物体 C 点数：`3828`
- 背景点数：`7350`
- 运行目录：`hw3/task1/outputs/task1_formal_ai_chain`
- 报告素材目录：`hw3/task1/report_assets/formal_ai_chain/`
- 预览图：`hw3/task1/report_assets/formal_ai_chain/fused_scene_preview.png`
- 多视角 GIF：`hw3/task1/report_assets/formal_ai_chain/fused_scene_turntable.gif`

说明：按用户要求，当前 formal 链路把 AI 图片当作正式输入；重型 COLMAP/3DGS/threestudio/Zero123 步骤用可复现的 Gaussian/point-cloud 代理表达跑通。报告中需要如实写明这是代理链路，不伪称为真实手机实拍或原版 threestudio 训练。
