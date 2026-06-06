# HW3 Task1 3DGS/AIGC — 实施规划

## 目标与范围
严格完成 `hw3/hw3.md` 题目一真实高质量链路。A/C 使用当前已有真实拍摄素材，B 使用 threestudio + SDS，C 使用 Zero123，背景使用开源 3D 数据集 3DGS 重建，最终融合 A/B/C 到统一背景并输出高质量多视角漫游视频。

## 架构与实现策略
- `hw3/task1/train.py`：只执行 `real_high_quality`。
- `hw3/task1/evaluate.py`：只验证真实链路 plan/run 输出结构。
- `hw3/task1/task1_3dgs_aigc/config.py`：只保留真实链路默认配置与校验。
- `hw3/task1/task1_3dgs_aigc/real_chain.py`：生成或执行 7 个真实外部工具脚本。
- `hw3/task1/scripts/prepare_mipnerf360_counter_136.sh`：在 136 准备 Mip-NeRF 360 `counter` 背景数据。
- `hw3/task1/scripts/run_nerfstudio_swanlab.py`：Nerfstudio 训练时把 TensorBoard 标量同步到 SwanLab。
- `hw3/task1/scripts/run_threestudio_swanlab.py`：threestudio / Zero123 训练时把 WandB 标量同步到 SwanLab，且不上传 WandB 云端。
- `hw3/task1/upload_modelscope.py`：按权重白名单上传，拒绝非权重杂项。
- `hw3/assets/README.md`：定义真实素材替换路径。

## 链路定义
1. 物体 A：真实多视角或视频 → 前景保真预处理去除静态背景干扰 → COLMAP / Nerfstudio `splatfacto-big`。
2. 背景：Mip-NeRF 360 `counter` 或等价开源 3D 数据集场景 → COLMAP / Nerfstudio `splatfacto-big`。
3. 物体 B：用户指定文本 prompt → threestudio / SDS。
4. 物体 C：真实单图前景 → threestudio `zero123.yaml` / Zero123。
5. 融合渲染：Blender 导入真实训练/重建导出 mesh，按合理尺度摆放并输出 1080p 漫游视频。

## 完成定义
- 旧低质量链路文件从 Git 删除。
- README / RESULTS 只记录真实高质量链路和 ModelScope 权重策略。
- 本地与 136 的 plan 模式 train/evaluate 通过。
- 136 准备好 Mip-NeRF 360 `counter` 背景数据和 Zero123XL 权重后，进入 run 模式完成真实训练、导出、融合渲染和评估。
- ModelScope `youngchen/CS60003` 中此前误传的 `hw3/task1/formal_ai_chain/` 已删除。
- `qaMode=standard`，`qaFocus` 覆盖删除边界、真实链路、ModelScope 权重策略、验证命令和远程同步。

## 文件结构
```text
hw3/task1/
  README.md
  RESULTS.md
  configs/real_high_quality.yaml
  requirements.txt
  train.py
  evaluate.py
  upload_modelscope.py
  scripts/render_real_chain_blender.py
  scripts/setup_real_chain_136.sh
  task1_3dgs_aigc/
    __init__.py
    config.py
    real_chain.py
    swanlab_utils.py
    utils.py
```

## 风险与验证
- 风险：背景数据或 Zero123 权重下载失败。处理：脚本明确失败命令与缺失路径；不替换为 proxy。
- 风险：ModelScope 再次上传杂项。处理：上传脚本只筛选训练权重，找不到权重直接失败。
- 风险：136 与本地代码不一致。处理：本地提交后用 bundle 同步，并在 136 执行同一组验证命令。

## 决策记录
- [2026-06-05] 用户明确要求只要高质量链路，旧低质量链路删除。
- [2026-06-05] 用户明确指出 ModelScope 只放训练好的模型权重，非权重 HW3 文件必须清理。
- [2026-06-05] 用户确认 A/C 当前素材是真实拍摄正式素材；正式链路不能再以自设数量阈值阻塞。
- [2026-06-05] 用户要求 C 按 `hw3.md` 使用 Zero123，不允许把 TripoSR 冒充 Zero123。
- [2026-06-05] 用户要求训练阶段必须使用 SwanLab 记录，环境变量从 `.helloagents/secrets/hw3.env` 读取，不写入 Git。
