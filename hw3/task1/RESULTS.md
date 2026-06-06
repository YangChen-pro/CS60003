# HW3 Task1 Results

## 当前结论

Task1 当前只保留真实高质量链路。早期 AI smoke test、程序化 proxy 点云、`formal_ai_chain` 伪正式结果、报告素材和 ModelScope 非权重上传都已经判定为不合格交付，不再作为结果记录。

本轮在 136-3090-4 的 `qwen14b` 环境完成 A/B/C/背景训练记录和 1080p 多视角展示视频生成。训练和远程运行前通过 `.helloagents/secrets/hw3.env` 注入 SwanLab / ModelScope / 代理环境变量，文档和代码不保存 key 明文。

## 训练与资产结果

| 资产 | 要求 | 当前实现 | 主要产物 | SwanLab |
|---|---|---|---|---|
| 物体 A | 多视角真实物体 + COLMAP + 3DGS | 用户真实多视角图片；Nerfstudio `splatfacto` 30k steps | `outputs/task1_real_high_quality/exports/object_a/splat/splat.ply` | <https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/dcbclc90gpdtpsn69brv3> |
| 物体 B | 文本 prompt + threestudio + SDS | Prompt 生成水晶蘑菇；threestudio SDS 15k steps | `outputs/task1_real_high_quality/exports/object_b/mesh/model.obj`、`object_b_threestudio/.../ckpts/last.ckpt` | <https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/d2ebu67uf76o4cceq7avb> |
| 物体 C | 单图真实物体 + Zero123 | 用户真实单图；Zero123XL 1200 steps | `outputs/task1_real_high_quality/exports/object_c/mesh/model.obj`、`object_c_zero123/.../ckpts/last.ckpt` | <https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/5yhocvbs6adfpiuo0f1jg> |
| 背景 | 开源 3D 数据集 + 3DGS | Mip-NeRF 360 `counter`；Nerfstudio `splatfacto` 30k steps | `outputs/task1_real_high_quality/exports/background/splat/splat.ply` | <https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/7r8mxm6zzz7kjblrtbhm8> |

本地最终展示视频：

```text
hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene.mp4
hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene_manifest.json
hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/fused_scene.mp4
hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/fused_scene_manifest.json
hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/frame_0072.png
```

远程对应路径：

```text
/home/dell/yc/CS60003/hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene.mp4
```

视频编码参数：1920×1080、144 帧、24 fps、CRF 16，远程 ffmpeg 编码日志已确认。

## 融合渲染策略

当前保留两类视频：

- strict 统一 3D 视频使用 `render_fused_splats.py` 生成，直接加载 A/background 的 `splat.ply` 与 B/C 的 `model.obj`，在同一 gsplat renderer 与同一 camera path 下渲染。该视频用于拦截 2D/2.5D 后合成冒充 3D 融合。
- 高质量视觉预览视频使用 `compose_3dgs_backplate_video.py` 生成，只作为报告辅助预览，不作为 strict 交付。

后合成预览视频的生成方式如下：

1. 背景来自训练完成的 Mip-NeRF 360 `counter` 3DGS，并先用 Nerfstudio 渲染 215 张高质量 backplate。
2. 物体 A 保留已训练 3DGS splat 权重，同时在最终展示视频中使用真实前景 mask/cutout 插入，以保证当前手机素材下的视觉质量。
3. 物体 B 使用训练完成的 SDS 测试渲染序列和导出的 mesh/ckpt 作为资产来源。
4. 物体 C 使用训练完成的 Zero123 测试渲染序列和导出的 mesh/ckpt 作为资产来源；最终脚本过滤不可见角度，并使用 C 专用 mask，避免空帧或背景色块进入视频。

strict 统一 3D 视频已经生成在 `renders/fused_splats/fused_scene.mp4`，manifest 中记录四个真实 3D 资产源和 splat 数量；该视频满足“不是 2D 贴图合成”的验证口径，但背景视觉质量仍弱于 Nerfstudio backplate 预览。报告中必须据实区分：`fused_splats/` 是统一 3D 渲染证据，`final_3dgs_backplate/` 是视觉预览，不可把后者冒充 strict 结果。

## 质量与限制

三种资产生成方式的几何准确度、纹理细节和计算耗时对比见 `hw3/task1/EVALUATION.md`。关键结论如下：

- A 的原始 3DGS 已完成并导出 splat；后续基于前景 mask 的重跑只匹配到 2/8 张图，不作为正式重建结果。
- A/background 的 TSDF mesh 导出因 `splatfacto` 输出不含 TSDF 所需 RGB 字段而跳过；Gaussian splat 权重是正式 3DGS 产物。
- B 的视觉结果偏绿色蘑菇，和原 prompt 中的 violet crystal 有偏差；训练链路、SDS 结果和 SwanLab 曲线是真实的。
- C 的 Zero123 结果可用，但部分角度不可见；后合成预览只使用可见帧，strict 统一 3D 视频使用导出的 OBJ surface splats。
- 当前 strict 统一 3D 视频已避免 2D/2.5D 后合成，但背景 splat 的相机/尺度视觉质量仍不理想；若追求最终展示质量，还需要继续改进背景 3DGS 相机轨迹和 B/C mesh-splat 转换质量。

## SwanLab 策略

正式训练必须加载 `.helloagents/secrets/hw3.env` 中的 `SWANLAB_API_KEY`。Nerfstudio A/background 训练通过 TensorBoard scalar sync 写入 SwanLab；threestudio B/C 训练通过 SwanLab 的 WandB scalar sync 写入 SwanLab，并强制 `wandb_run=False`，不向 WandB 云端上传。真实 run 链接见 `hw3/task1/SWANLAB_RUNS.md`。

## ModelScope 策略

ModelScope 仓库只保存训练好的模型权重。`upload_modelscope.py` 已收敛为权重白名单上传：

- 常规权重：`.pt`、`.pth`、`.ckpt`、`.safetensors`、`.bin`、`.onnx`、`.npz`
- 真实 3DGS 权重：`point_cloud.ply`、`splat.ply`、`gaussian_splat.ply`

以下内容不会再上传 ModelScope：

- `source_config.yaml`
- `config.json`
- `summary.json`
- `metrics.csv` / `metrics.json`
- 图片、GIF、视频预览
- 报告素材
- 程序化 proxy / fused scene 杂项文件

此前误传到 `youngchen/CS60003/hw3/task1/formal_ai_chain/` 的非权重文件已通过 ModelScope Git 仓库提交删除。
