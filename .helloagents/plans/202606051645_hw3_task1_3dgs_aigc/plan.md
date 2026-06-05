# HW3 Task1 3DGS/AIGC — 实施规划

## 目标与范围
只保留真实高质量链路。删除早期 AI smoke、程序化 proxy、`formal_ai_chain`、预览渲染和报告素材结果，让后续把 `hw3/assets` 换成真实拍摄 A/C 后可以直接运行可信链路。

## 架构与实现策略
- `hw3/task1/train.py`：只执行 `real_high_quality`。
- `hw3/task1/evaluate.py`：只验证真实链路 plan/run 输出结构。
- `hw3/task1/task1_3dgs_aigc/config.py`：只保留真实链路默认配置与校验。
- `hw3/task1/task1_3dgs_aigc/real_chain.py`：生成或执行 7 个真实外部工具脚本。
- `hw3/task1/upload_modelscope.py`：按权重白名单上传，拒绝非权重杂项。
- `hw3/assets/README.md`：定义真实素材替换路径。

## 链路定义
1. 物体 A：真实多视角或视频 → COLMAP / Nerfstudio `splatfacto-big`。
2. 背景：真实多视角或视频 → COLMAP / Nerfstudio `splatfacto-big`。
3. 物体 B：文本 prompt → threestudio / SDS。
4. 物体 C：真实单图 → TripoSR。
5. 融合渲染：Blender 导入真实训练/重建产物并输出视频。

## 完成定义
- 旧低质量链路文件从 Git 删除。
- README / RESULTS 只记录真实高质量链路和 ModelScope 权重策略。
- 本地与 136 的 plan 模式 train/evaluate 通过。
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
- 风险：真实素材暂未到位。处理：plan 模式允许 `NEEDS_INPUTS`，但仍生成真实工具脚本；不生成 proxy。
- 风险：ModelScope 再次上传杂项。处理：上传脚本只筛选训练权重，找不到权重直接失败。
- 风险：136 与本地代码不一致。处理：本地提交后用 bundle 同步，并在 136 执行同一组验证命令。

## 决策记录
- [2026-06-05] 用户明确要求只要高质量链路，旧低质量链路删除。
- [2026-06-05] 用户明确指出 ModelScope 只放训练好的模型权重，非权重 HW3 文件必须清理。
