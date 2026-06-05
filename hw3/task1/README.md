# HW3 Task1：真实高质量 3DGS 与 AIGC 融合链路

本目录只保留高质量正式链路。早期 AI 图片 smoke test、程序化 proxy 点云、伪正式 `formal_ai_chain` 和报告素材已经移除，避免把低质量中间结果误当成作业交付。

代码骨架继续对齐 HW1/HW2：

- `configs/`：实验 YAML。
- `task1_3dgs_aigc/`：配置解析、真实工具链编排、SwanLab 工具。
- `train.py`：固定训练/执行入口。
- `evaluate.py`：固定验证入口。
- `outputs/`：运行输出，不进入 Git。
- `upload_modelscope.py`：只上传训练好的模型权重。

## 当前维护的链路

配置文件：

```bash
hw3/task1/configs/real_high_quality.yaml
```

链路内容：

1. 物体 A：真实多视角图片或视频 → COLMAP / Nerfstudio `splatfacto-big` → Gaussian splat 与 TSDF mesh。
2. 背景：真实场景图片或视频 → COLMAP / Nerfstudio `splatfacto-big` → Gaussian splat 与 TSDF mesh。
3. 物体 B：文本 prompt → threestudio / SDS 训练 → 3D mesh。
4. 物体 C：真实单图 → TripoSR → 带纹理 3D mesh。
5. 融合渲染：Blender 融合 A/B/C/背景，输出漫游视频和报告可引用画面。

## 真实素材放置

素材目录见 `hw3/assets/README.md`。默认路径如下：

```text
hw3/assets/object_a_multiview/
hw3/assets/object_c_single/object_c_single_front.png
hw3/assets/background_scene/images/
```

这些真实素材默认不进入 Git。之后把 A/C 换成手机实拍图，不需要改代码，只要保持路径结构一致。背景路径用于放开源 3D 数据集场景图片；如果使用视频或其他本地数据路径，直接改 `real_high_quality.yaml` 中的 `object_a_video` / `background_video` / `background_images`，不额外限制图片数量。

## 136 环境

```bash
ssh 136-3090-4
cd /home/dell/yc/CS60003
source /home/dell/miniconda3/etc/profile.d/conda.sh
conda activate qwen14b
set -a
source /home/dell/yc/CS60003/.helloagents/secrets/hw3.env
set +a
```

安装/检查外部工具入口：

```bash
bash hw3/task1/scripts/setup_real_chain_136.sh
```

## 运行方式

默认 `real_chain.execution.mode: plan`，只验证真实素材是否齐全并生成 7 个可审查脚本：

```bash
python hw3/task1/train.py --config hw3/task1/configs/real_high_quality.yaml
python hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_real_high_quality
```

真实素材和外部依赖都准备好后，把 YAML 改成：

```yaml
real_chain:
  execution:
    mode: run
```

再执行同一条 `train.py` 命令。`00_check_tools.sh` 会先检查 COLMAP、FFmpeg、Blender、Nerfstudio、threestudio、TripoSR；缺少依赖时直接失败，不做静默降级。

## SwanLab / ModelScope

SwanLab 继续通过环境变量里的用户 key 记录曲线：

```yaml
logging:
  swanlab:
    enabled: true
    project: cs60003-hw3-task1
    mode: cloud
    group: real-high-quality
```

ModelScope 只放训练好的模型权重。`upload_modelscope.py` 会递归筛选 `.pt`、`.pth`、`.ckpt`、`.safetensors`、`.bin`、`.onnx`、`.npz`，以及真实 3DGS 训练导出的 `point_cloud.ply` / `splat.ply` / `gaussian_splat.ply`；不会上传 config、summary、metrics、图片、GIF、报告素材或 proxy 文件。

```bash
python hw3/task1/upload_modelscope.py \
  --run-dir hw3/task1/outputs/task1_real_high_quality \
  --remote-subdir real_high_quality
```

如果当前 run 目录没有训练权重，上传脚本会直接报错并跳过，不会把杂项文件塞进 ModelScope。
