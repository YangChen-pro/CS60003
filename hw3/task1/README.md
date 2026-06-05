# HW3 Task1：3DGS 与 AIGC 多源资产融合

本目录用于完成 HW3 题目一：真实物体 3DGS、多源 AIGC 资产生成、背景 3DGS 重建，以及最终场景融合渲染。

当前阶段先建立和 HW1/HW2 一致的工程骨架：`configs/` 管实验配置、`task1_3dgs_aigc/` 放任务代码、`train.py` / `evaluate.py` 作为固定入口、`outputs/` 不进 Git，SwanLab / ModelScope 也保留独立记录和上传入口。当前使用 `hw3/assets/ai_generated_test/` 中的 9 张 AI 生成图片做 pipeline smoke test；它们只用于验证数据读取、路径、图像质量统计和后续 COLMAP/3DGS 输入口径，不作为最终提交素材。

## 目录结构

```text
hw3/task1/
├── configs/                 # 每个 YAML 对应一组实验或阶段
├── task1_3dgs_aigc/         # 数据检查、工具封装、后续 3DGS/SDS 代码
├── train.py                 # 阶段执行入口，风格对齐 HW1/HW2
├── evaluate.py              # 输出验证入口
├── upload_modelscope.py     # ModelScope 上传入口
├── SWANLAB_RUNS.md          # SwanLab 云端 run 记录
├── RESULTS.md               # 结果汇总
├── requirements.txt
└── outputs/                 # 运行输出，默认不进入 Git
```

## 当前数据

- 物体 A 测试图：`hw3/assets/ai_generated_test/object_a_multiview/`
- 物体 C 测试图：`hw3/assets/ai_generated_test/object_c_single/object_c_single_front.png`

最终实验仍需替换为：

- 物体 A：真实手机环绕视频或多视角照片。
- 物体 C：真实手机单图。
- 物体 B：threestudio / SDS 文本到 3D 生成资产。
- 背景：Mip-NeRF 360 场景的 3DGS 重建结果。

## 136 环境

远程机器：

```bash
ssh 136-3090-4
cd /home/dell/yc/CS60003
source /home/dell/miniconda3/etc/profile.d/conda.sh
conda activate qwen14b
set -a
source /home/dell/yc/CS60003/.helloagents/secrets/hw3.env
set +a
```

## Smoke test

```bash
python hw3/task1/train.py --config hw3/task1/configs/ai_generated_smoke.yaml
python hw3/task1/evaluate.py --run-dir hw3/task1/outputs/task1_ai_generated_smoke
```

需要打开 SwanLab 时，在 YAML 里设置：

```yaml
logging:
  swanlab:
    enabled: true
    project: cs60003-hw3-task1
    mode: cloud
    group: smoke
```

运行前必须先通过环境变量提供用户本人的 key：

```bash
set -a
source /home/dell/yc/CS60003/.helloagents/secrets/hw3.env
set +a
```

当前 smoke test 产物包括：

- `source_config.yaml`
- `config.json`
- `manifest.json`
- `image_stats.csv`
- `pairwise_yaw_diffs.csv`
- `contact_sheet.png`
- `summary.json`

上传 ModelScope：

```bash
python hw3/task1/upload_modelscope.py \
  --run-dir hw3/task1/outputs/task1_ai_generated_smoke
```

## 后续正式链路

1. 真实物体 A：手机多视角素材 → COLMAP → 3DGS。
2. 物体 B：文本 prompt → threestudio / SDS → mesh 或 radiance field。
3. 物体 C：手机单图 → 去背景 → Zero123 或等价单图到 3D 方法。
4. 背景：Mip-NeRF 360 场景 → 3DGS。
5. 融合：统一 mesh / point cloud / Gaussian 表达，生成多视角漫游视频。
