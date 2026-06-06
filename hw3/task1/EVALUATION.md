# HW3 Task1 Evaluation Notes

本文件记录题目一报告需要的评估对比素材，重点覆盖 hw3.md 要求的三项：几何准确度、纹理细节、计算耗时。所有结论基于当前 136-3090-4 / `qwen14b` 真实运行产物；SwanLab run 链接见 `SWANLAB_RUNS.md`。

## 运行环境与记录口径

- GPU 机器：`136-3090-4`
- 远程项目路径：`/home/dell/yc/CS60003`
- Conda 环境：`qwen14b`
- SwanLab：训练前通过 `.helloagents/secrets/hw3.env` 注入 `SWANLAB_API_KEY`，不在 Git 文件中保存 key。
- ModelScope：只允许上传训练权重，不上传 config、metrics、图片、视频或报告素材。

## 产物与指标

| 项目 | 技术路径 | 训练/优化步数 | 主要产物 | 产物大小 | 定量/状态 |
|---|---|---:|---|---:|---|
| 物体 A | COLMAP + Nerfstudio `splatfacto` | 30000 | `exports/object_a/splat/splat.ply` | 549.4 MB | `ns-eval` 因无 eval split 显式跳过；保留 3DGS splat |
| 物体 B | threestudio + SDS | 15000 | `exports/object_b/mesh/model.obj` / `ckpts/last.ckpt` | OBJ 41.9 MB / CKPT 151.5 MB | 文本到 3D 成功；视觉偏绿，和 violet prompt 有偏差 |
| 物体 C | Zero123XL | 1200 | `exports/object_c/mesh/model.obj` / `ckpts/last.ckpt` | OBJ 33.6 MB / CKPT 151.5 MB | 单图到 3D 成功；部分角度不可见，后合成预览会过滤不可见帧 |
| 背景 | Mip-NeRF 360 `counter` + Nerfstudio `splatfacto` | 30000 | `exports/background/splat/splat.ply` | 288.6 MB | PSNR 29.05 / SSIM 0.9186 / LPIPS 0.1101 |
| strict 统一 3D 视频 | A/background 3DGS splat + B/C OBJ surface splats | - | `renders/fused_splats/fused_scene.mp4` | 13.9 MB | 1920×1080 / 144 frames / 24 fps；同一 gsplat renderer |
| 视觉预览视频 | 3DGS backplate + 训练对象渲染结果后合成 | - | `renders/final_3dgs_backplate/fused_scene.mp4` | 18.3 MB | 1920×1080 / 144 frames / 24 fps / CRF 16；不作为 strict 交付 |

## 三种资产生成方式对比

| 维度 | A：多视角重建 | B：文本到 3D / SDS | C：单图到 3D / Zero123 |
|---|---|---|---|
| 输入信息 | 用户真实多视角照片，几何约束最强，但依赖 COLMAP 匹配质量 | 单段 prompt，无真实几何观测，主要依赖扩散先验 | 单张真实照片，保留正面外观，但背面和侧面由模型补全 |
| 几何准确度 | 对已观测表面最可信；当前 A 的正式 3DGS 可导出 splat，但前景重跑只匹配 2/8 张，说明少视角/弱纹理会限制几何稳定性 | 形状整体可用，但几何由 SDS 优化和 2D 先验驱动，细节容易偏离 prompt | 正面几何与输入一致性较好，未观测侧/背面存在生成不稳定，部分角度不可见 |
| 纹理细节 | 来自真实照片，局部材质和颜色可信；依赖拍摄清晰度、曝光和 COLMAP 注册质量 | 有明显生成风格，当前结果偏绿色蘑菇，与“violet crystal”不完全一致 | 保留 C 单图的主体外观和卡通木质质感，但多视角纹理连续性弱于 A |
| 计算耗时 | 30k 3DGS 训练 + COLMAP/导出，GPU 时间较长；artifact mtime 口径包含等待/导出，不能当作纯训练时长 | 15k SDS 优化，迭代多，单资产成本较高 | 1200 Zero123 优化，步数最少，但质量依赖输入抠图和 Zero123 权重 |
| 适合场景 | 真实物体复现、报告中体现 3DGS 重建能力 | 创造不存在的虚拟资产，适合补充想象物体 | 从少量真实素材快速扩展为 3D 资产 |
| 当前限制 | strict 统一 3D 视频中 A 作为 splat 参与渲染，但视觉质量弱于真实 cutout 预览 | 语义满足“蘑菇”，颜色偏离 prompt；需在报告中说明 | 可见角度不稳定；strict 视频使用 OBJ surface splats，预览视频才会过滤不可见帧 |

## 融合表达说明

题目要求说明如何统一 AIGC mesh 与 3DGS 背景。当前实现保留两条链路：

1. **strict 统一 3D 链路**：`render_fused_splats.py` 直接加载 A/background 的 Gaussian splat 权重，并把 B/C 的 OBJ 顶点采样为 surface splats，在同一个 `gsplat` renderer、同一相机路径下输出 `renders/fused_splats/fused_scene.mp4`。这个视频用于证明最终结果不是 2D/2.5D 图像贴片。
2. **视觉预览链路**：先用背景 3DGS 渲染多视角 backplate，再合成 A 的真实前景 cutout、B/C 的训练后测试渲染结果，输出稳定 1080p 预览视频。

当前必须如实说明两者差异：`fused_splats/` 是符合统一 3D 表达的 strict 证据，但背景相机/尺度视觉质量仍弱；`final_3dgs_backplate/` 是视觉质量更好的后合成预览，不可把它包装成“同一个 3D 场景原生融合”。

## 可直接引用的结果路径

```text
hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene.mp4
hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/frame_0072.png
hw3/task1/outputs/task1_real_high_quality/renders/fused_splats/fused_scene_manifest.json
hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/fused_scene.mp4
hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/frame_0072.png
hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/fused_scene_manifest.json
```

远程完整产物仍在：

```text
/home/dell/yc/CS60003/hw3/task1/outputs/task1_real_high_quality/
```
