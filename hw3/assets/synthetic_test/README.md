# HW3 Synthetic Test Images

这些图片只用于 HW3 早期 smoke test，不作为最终提交素材。

当前生成内容：

- `object_a_multiview/`：物体 A 的 8 张合成多视角占位图，用于先跑数据目录、图像读取、COLMAP/3DGS 输入链路。
- `object_c_single/object_c_single_front.png`：物体 C 的 1 张合成单图占位图，用于先跑去背景、单图到 3D 或相关预处理链路。
- `metadata.json`：文件清单和用途说明。

限制：

- 这些图不是手机实拍，也不保证满足真实 photogrammetry 的几何一致性。
- 题目一最终仍需要替换为真实手机拍摄素材：物体 A 使用环绕视频或多视角照片，物体 C 使用一张清晰单物体照片。

重新生成：

```bash
python3 -X utf8 hw3/scripts/generate_synthetic_assets.py
```
