# HW3 Task1 ModelScope Weights

ModelScope 仓库：<https://www.modelscope.cn/models/youngchen/CS60003>

远程目录：

```text
hw3/task1/real_high_quality/
```

本次只上传训练好的模型权重，不上传 config、summary、metrics、图片、GIF、视频、报告素材、COLMAP 中间文件或旧试跑 checkpoint。上传前 dry-run 白名单与实际上传文件一致。

| 类型 | ModelScope 路径 | 说明 |
|---|---|---|
| background 3DGS splat | `hw3/task1/real_high_quality/exports/background/splat/splat.ply` | 背景 Mip-NeRF 360 `counter` 的 3DGS 权重 |
| object A 3DGS splat | `hw3/task1/real_high_quality/exports/object_a/splat/splat.ply` | 物体 A 的 3DGS splat 权重 |
| background Nerfstudio ckpt | `hw3/task1/real_high_quality/nerfstudio/background/background/splatfacto/2026-06-06_044859/nerfstudio_models/step-000029999.ckpt` | 背景 30k step checkpoint |
| object A Nerfstudio ckpt | `hw3/task1/real_high_quality/nerfstudio/object_a/object_a/splatfacto/2026-06-05_235402/nerfstudio_models/step-000029999.ckpt` | 物体 A 30k step checkpoint |
| object B SDS ckpt | `hw3/task1/real_high_quality/object_b_threestudio/object_b/sds@20260606-032908/ckpts/last.ckpt` | 物体 B SDS 最终 checkpoint |
| object C Zero123 ckpt | `hw3/task1/real_high_quality/object_c_zero123/object_c/zero123@20260606-041853/ckpts/last.ckpt` | 物体 C Zero123 最终 checkpoint |

上传命令：

```bash
set -a
source /home/dell/yc/CS60003/.helloagents/secrets/hw3.env
set +a
python hw3/task1/upload_modelscope.py \
  --run-dir hw3/task1/outputs/task1_real_high_quality \
  --remote-subdir real_high_quality
```
