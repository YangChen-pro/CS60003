# HW3 Task1：真实高质量 3DGS 与 AIGC 融合链路

本目录只保留 task1 高质量正式链路。早期非正式中间产物和报告草稿不再保留。

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

1. 物体 A：真实多视角图片或视频 → 前景保真预处理去除静态背景干扰 → COLMAP / Nerfstudio `splatfacto-big` → Gaussian splat 与 TSDF mesh；训练曲线通过 SwanLab 记录。
2. 背景：真实场景图片或视频 → COLMAP / Nerfstudio `splatfacto-big` → Gaussian splat 与 TSDF mesh；训练曲线通过 SwanLab 记录。
3. 物体 B：文本 prompt → threestudio / SDS 训练 → 3D mesh；WandB 标量由 SwanLab 同步，不上传 WandB 云端。
4. 物体 C：真实单图前景 → threestudio `zero123.yaml` / Zero123 → 3D mesh；WandB 标量由 SwanLab 同步，不上传 WandB 云端。
5. 融合渲染：使用统一 3D 表达（object A 的 splat、object B/C 的 mesh）在同一相机路径下输出一段 1080p 漫游视频，用于报告与评估。

## 真实素材放置

素材目录见 `hw3/assets/README.md`。默认路径如下：

```text
hw3/assets/object_a_multiview/
hw3/assets/object_c_single/object_c_single_front.png
hw3/assets/background_scene/images/
```

这些真实素材默认不进入 Git。之后把 A/C 换成手机实拍图，不需要改代码，只要保持路径结构一致。背景路径用于放开源 3D 数据集场景图片；如果使用视频或其他本地数据路径，直接改 `real_high_quality.yaml` 中的 `object_a_video` / `background_video` / `background_images`，不额外限制图片数量。

背景默认选择 Mip-NeRF 360 `counter`，原因是它是题目举例中的开源室内桌面场景，适合插入小物体 A/B/C。136 上可执行：

```bash
bash hw3/task1/scripts/prepare_mipnerf360_counter_136.sh
```

脚本会下载官方 `360_v2.zip`，解压 `counter`，并把 `hw3/assets/background_scene/images` 链接到 `counter/images_2`；如需更省显存，可用 `RESOLUTION_DIR=images_4` 覆盖。

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

`setup_real_chain_136.sh` 会安装 threestudio 并准备 Zero123XL 所需的 `zero123-xl.ckpt` 与模型配置文件；不会安装或使用 TripoSR。

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

再执行同一条 `train.py` 命令。`00_check_tools.sh` 会先检查 COLMAP、FFmpeg、Nerfstudio、threestudio 和 Zero123 权重；缺少依赖时直接失败，不做静默降级。

## 当前真实运行结果

当前 136-3090-4 / `qwen14b` 已完成一次真实 Task1 链路：

- object A：COLMAP + Nerfstudio `splatfacto`，30k steps，SwanLab 已记录。
- object B：threestudio SDS，15k steps，SwanLab 已记录。
- object C：Zero123XL，1200 steps，SwanLab 已记录。
- background：Mip-NeRF 360 `counter` + Nerfstudio `splatfacto`，30k steps，SwanLab 已记录。

严格统一 3D 本地视频：

```text
hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene.mp4
```

远程视频：

```text
/home/dell/yc/CS60003/hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene.mp4
```

关键帧和 manifest：

 ```text
hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/frame_0072.png
hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene_manifest.json
```

严格统一 3D 视频参数：1920×1080、144 帧、24 fps，由 `render_fused_splats.py` 直接加载 A/background 3DGS splat 与 B/C OBJ mesh 并用同一 gsplat camera path 渲染。
结果细节、限制、SwanLab 链接和 ModelScope 权重路径见 `RESULTS.md`、`EVALUATION.md`、`SWANLAB_RUNS.md` 与 `MODELSCOPE_WEIGHTS.md`。

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

`real_high_quality.yaml` 默认启用 SwanLab，并通过 `.helloagents/secrets/hw3.env` 加载 `SWANLAB_API_KEY`。这个文件不进入 Git；运行脚本只引用路径，不打印 key。Nerfstudio 训练使用 TensorBoard scalar sync，threestudio / Zero123 训练使用 SwanLab 的 WandB scalar sync。

ModelScope 只放训练好的模型权重。`upload_modelscope.py` 会筛选当前最终权重：A/background 的 3DGS splat、A/background 的 Nerfstudio checkpoint，以及 B/C 的最终 `last.ckpt`；不会上传 config、summary、metrics、图片、GIF、报告素材、COLMAP 中间文件、旧试跑 checkpoint 或 proxy 文件。

```bash
python hw3/task1/upload_modelscope.py \
  --run-dir hw3/task1/outputs/task1_real_high_quality \
  --remote-subdir real_high_quality
```

如果当前 run 目录没有训练权重，上传脚本会直接报错并跳过，不会把杂项文件塞进 ModelScope。
