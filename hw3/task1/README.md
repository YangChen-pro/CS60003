# HW3 Task1：3DGS 与 AIGC 多源资产融合

本目录用于完成 HW3 题目一：物体 A/B/C 资产准备、背景重建代理、场景融合与多视角渲染。代码骨架保持和 HW1/HW2 类似：`configs/` 管实验配置、`task1_3dgs_aigc/` 放任务代码、`train.py` / `evaluate.py` 作为固定入口、`outputs/` 不进 Git，SwanLab / ModelScope 保留独立记录和上传入口。

按用户本轮要求，当前正式链路直接使用 `hw3/assets/ai_generated_test/` 中的 AI 生成物体 A/C 图片作为正式输入；物体 B 由代码根据文本 prompt 生成一个可复现的 3D 代理资产。重型 COLMAP/3DGS/threestudio/Zero123 步骤用可复现的 Gaussian/point-cloud 代理实现，目的是先把端到端融合与渲染链路跑通。

## 目录结构

```text
hw3/task1/
├── configs/                 # 每个 YAML 对应一组实验或阶段
├── task1_3dgs_aigc/         # 数据检查、几何代理、渲染与工具封装
├── train.py                 # 阶段执行入口，风格对齐 HW1/HW2
├── evaluate.py              # 输出验证入口
├── upload_modelscope.py     # ModelScope 上传入口
├── SWANLAB_RUNS.md          # SwanLab 云端 run 记录
├── RESULTS.md               # 结果汇总
├── report_assets/           # 报告可直接引用的小型结果素材
├── requirements.txt
└── outputs/                 # 运行输出，默认不进入 Git
```

## 当前正式输入

- 物体 A：`hw3/assets/ai_generated_test/object_a_multiview/`，8 张 AI 多视角火箭图片。
- 物体 B：文本 prompt `a small bioluminescent purple crystal mushroom with a green stem`，程序生成 3D 点云代理。
- 物体 C：`hw3/assets/ai_generated_test/object_c_single/object_c_single_front.png`，1 张 AI 单图木偶图片。
- 背景：程序生成 Mip-NeRF 360 风格的桌面/墙面 Gaussian 背景代理。

## 136 环境

远程机器：

```bash
ssh 136-3090-4
cd /home/dell/yc/CS60003
source /home/dell/miniconda3/etc/profile.d/conda.sh
conda activate qwen14b
set -a
source /home/dell/yc/CS60003/.helloagents/secrets/hw3.env
set +a
```

## Smoke test

```bash
python hw3/task1/train.py --config hw3/task1/configs/ai_generated_smoke.yaml
python hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_ai_generated_smoke
```

## 正式 AI 素材链路

```bash
python hw3/task1/train.py --config hw3/task1/configs/formal_ai_chain.yaml
python hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_formal_ai_chain
```

该链路会生成：

- `assets/object_a_ai_multiview_gaussians.ply`
- `assets/object_b_text_to_3d_proxy.ply`
- `assets/object_c_single_image_proxy.ply`
- `assets/background_proxy_gaussians.ply`
- `fused_scene.ply`
- `renders/fused_scene_preview.png`
- `renders/fused_scene_turntable.gif`
- `metrics.csv`
- `asset_manifest.json`
- `summary.json`

报告可引用素材会同步到：

```text
hw3/task1/report_assets/formal_ai_chain/
```

## SwanLab / ModelScope

需要打开 SwanLab 时，在 YAML 里设置：

```yaml
logging:
  swanlab:
    enabled: true
    project: cs60003-hw3-task1
    mode: cloud
    group: formal-ai-chain
```

上传 ModelScope：

```bash
python hw3/task1/upload_modelscope.py \
  --run-dir hw3/task1/outputs/task1_formal_ai_chain \
  --remote-subdir formal_ai_chain
```

运行前必须先通过环境变量提供用户本人的 key：

```bash
set -a
source /home/dell/yc/CS60003/.helloagents/secrets/hw3.env
set +a
```

## 当前链路解释

1. 物体 A：AI 多视角图 → 前景采样 → 多 yaw Gaussian/point-cloud 代理。
2. 物体 B：文本 prompt → 程序化 text-to-3D 代理点云。
3. 物体 C：AI 单图 → 单图挤出式 3D 代理。
4. 背景：桌面/墙面 Gaussian 背景代理。
5. 融合：统一为 PLY 点云/高斯半径表达，按空间位置合并并生成多视角 turntable GIF。

## 真实高质量链路（后续替换真实 A/C 后使用）

这条链路不再使用早期 proxy 结果，而是调用真实外部工具：

- 物体 A / 背景：Nerfstudio `splatfacto-big` + COLMAP，导出 Gaussian splat 与 TSDF mesh。
- 物体 B：threestudio 文本到 3D，默认使用 DreamFusion/SDS 配置并启用 Perp-Neg / D-SDS 相关质量参数。
- 物体 C：TripoSR 单图到 3D，输出带纹理 mesh。
- 融合渲染：Blender 导入 A/B/C/背景 mesh，输出 1280×720 H.264 漫游视频。

素材放置位置见 `hw3/assets/README.md`。先生成运行脚本：

```bash
python hw3/task1/train.py --config hw3/task1/configs/real_high_quality.yaml
python hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_real_high_quality
```

当前 YAML 默认是 `real_chain.execution.mode: plan`，只校验素材并生成 7 个脚本。真实素材与依赖安装好后，把它改成 `run`，再执行同一条 `train.py` 命令即可跑完整高质量链路。

136 环境安装入口：

```bash
bash hw3/task1/scripts/setup_real_chain_136.sh
```

注意：高质量链路依赖 COLMAP/FFmpeg/Blender/Nerfstudio/threestudio/TripoSR。缺少这些工具时不做静默降级，`00_check_tools.sh` 会直接失败。
