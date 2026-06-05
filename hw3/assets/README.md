# HW3 Task1 真实高质量输入目录

把真实素材放到以下位置后，`hw3/task1/configs/real_high_quality.yaml` 可以直接读取，不需要改代码。

```text
hw3/assets/
├── object_a_multiview/                 # 物体 A：真实环绕照片，建议 40-120 张
├── object_c_single/
│   └── object_c_single_front.png       # 物体 C：真实单图，建议先去背景
└── background_scene/
    └── images/                         # 背景：真实场景或 Mip-NeRF 360 场景图片，建议 80+ 张
```

拍摄要求：

- 物体 A：绕物体一圈，保证相邻照片有明显重叠，避免运动模糊和强反光。
- 背景：沿稳定路径拍摄，保证墙面、桌面、地面等纹理有足够重叠。
- 物体 C：单图尽量正面、清晰、干净背景；最好先去背景后保存为 PNG。

真实高质量链路入口：

```bash
python hw3/task1/train.py --config hw3/task1/configs/real_high_quality.yaml
```

默认 `real_chain.execution.mode=plan`，会先校验素材并生成实际运行脚本。确认 136 已安装 Nerfstudio、COLMAP、threestudio、TripoSR、Blender 后，把 YAML 改成 `mode: run` 即可执行完整链路。
