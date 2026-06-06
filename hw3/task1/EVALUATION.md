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
| 物体 C | Zero123XL | 1200 | `exports/object_c/mesh/model.obj` / `ckpts/last.ckpt` | OBJ 33.6 MB / CKPT 151.5 MB | 单图到 3D 成功；部分角度不可见，最终视频过滤不可见帧 |
| 背景 | Mip-NeRF 360 `counter` + Nerfstudio `splatfacto` | 30000 | `exports/background/splat/splat.ply` | 288.6 MB | PSNR 29.05 / SSIM 0.9186 / LPIPS 0.1101 |
| 最终视频 | 3DGS backplate + 训练对象渲染结果合成 | - | `renders/final_3dgs_backplate/fused_scene.mp4` | 18.3 MB | 1920×1080 / 144 frames / 24 fps / CRF 16 |

## 三种资产生成方式对比

| 维度 | A：多视角重建 | B：文本到 3D / SDS | C：单图到 3D / Zero123 |
|---|---|---|---|
| 输入信息 | 用户真实多视角照片，几何约束最强，但依赖 COLMAP 匹配质量 | 单段 prompt，无真实几何观测，主要依赖扩散先验 | 单张真实照片，保留正面外观，但背面和侧面由模型补全 |
| 几何准确度 | 对已观测表面最可信；当前 A 的正式 3DGS 可导出 splat，但前景重跑只匹配 2/8 张，说明少视角/弱纹理会限制几何稳定性 | 形状整体可用，但几何由 SDS 优化和 2D 先验驱动，细节容易偏离 prompt | 正面几何与输入一致性较好，未观测侧/背面存在生成不稳定，部分角度不可见 |
| 纹理细节 | 来自真实照片，局部材质和颜色可信；依赖拍摄清晰度、曝光和 COLMAP 注册质量 | 有明显生成风格，当前结果偏绿色蘑菇，与“violet crystal”不完全一致 | 保留 C 单图的主体外观和卡通木质质感，但多视角纹理连续性弱于 A |
| 计算耗时 | 30k 3DGS 训练 + COLMAP/导出，GPU 时间较长；artifact mtime 口径包含等待/导出，不能当作纯训练时长 | 15k SDS 优化，迭代多，单资产成本较高 | 1200 Zero123 优化，步数最少，但质量依赖输入抠图和 Zero123 权重 |
| 适合场景 | 真实物体复现、报告中体现 3DGS 重建能力 | 创造不存在的虚拟资产，适合补充想象物体 | 从少量真实素材快速扩展为 3D 资产 |
| 当前限制 | 严格单渲染器融合时 splat/mesh 统一渲染质量不足；最终视频用真实 cutout 保证视觉质量 | 语义满足“蘑菇”，颜色偏离 prompt；需在报告中说明 | 可见帧需要过滤；最终视频使用 C 专用 mask 避免背景块 |

## 融合表达说明

题目要求说明如何统一 AIGC mesh 与 3DGS 背景。当前实现保留两条链路：

1. **模型产物链路**：A/background 以 Gaussian splat 权重保存，B/C 以 OBJ mesh 和 checkpoint 保存，满足不同技术路径的真实训练产物留存。
2. **报告展示链路**：先用背景 3DGS 渲染多视角 backplate，再合成 A 的真实前景 cutout、B/C 的训练后测试渲染结果，输出稳定 1080p 视频。

选择 backplate 合成的原因是当前单渲染器 splat/mesh 融合脚本只能得到低质量点云/proxy 画面，不符合“高质量渲染视频”的目标。报告中应如实说明该取舍：训练产物是真实的 3DGS/SDS/Zero123，最终展示视频是基于训练产物的高质量后合成，而不是把 proxy 结果冒充原生融合。

## 可直接引用的结果路径

```text
hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/fused_scene.mp4
hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/frame_0072.png
hw3/task1/outputs/task1_real_high_quality/renders/final_3dgs_backplate/fused_scene_manifest.json
```

远程完整产物仍在：

```text
/home/dell/yc/CS60003/hw3/task1/outputs/task1_real_high_quality/
```
