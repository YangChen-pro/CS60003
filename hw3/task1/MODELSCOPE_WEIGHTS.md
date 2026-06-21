# Task 1 ModelScope 权重

仓库：<https://www.modelscope.cn/models/youngchen/CS60003>

目录：

```text
hw3/task1/real_high_quality/
```

该目录包含 `task1_real_quality_v2` 的六个模型文件：

| 类型 | 路径 |
|---|---|
| background 3DGS splat | `exports/background/splat/splat.ply` |
| object A 3DGS splat | `exports/object_a/splat/splat.ply` |
| background Nerfstudio checkpoint | `nerfstudio/background/background/splatfacto/2026-06-10_223702/nerfstudio_models/step-000029999.ckpt` |
| object A Nerfstudio checkpoint | `nerfstudio/object_a/object_a/splatfacto/2026-06-11_141848/nerfstudio_models/step-000029999.ckpt` |
| object B SDS checkpoint | `object_b_threestudio/object_b/sds@20260608-021639/ckpts/last.ckpt` |
| object C Zero123 checkpoint | `object_c_zero123/object_c/zero123@20260608-022916/ckpts/last.ckpt` |

视频、图片、日志、配置、COLMAP 中间文件和 mesh 导出不上传到模型仓库。

```bash
python hw3/task1/upload_modelscope.py \
  --run-dir hw3/task1/outputs/task1_real_quality_v2 \
  --remote-subdir real_high_quality \
  --replace-remote-subdir
```
