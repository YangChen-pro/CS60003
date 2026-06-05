# HW3 Task1 Results

## 当前结论

Task1 只保留真实高质量链路。早期 AI smoke test、程序化 proxy 点云、`formal_ai_chain` 伪正式结果、报告素材和 ModelScope 非权重上传都已经判定为不合格交付，不再作为结果记录。

## 当前维护结果

- 配置：`hw3/task1/configs/real_high_quality.yaml`
- 入口：`hw3/task1/train.py`
- 验证：`hw3/task1/evaluate.py`
- 编排模块：`hw3/task1/task1_3dgs_aigc/real_chain.py`
- Blender 渲染脚本：`hw3/task1/scripts/render_real_chain_blender.py`
- 136 安装入口：`hw3/task1/scripts/setup_real_chain_136.sh`
- 真实素材说明：`hw3/assets/README.md`

## 验证状态

当前 YAML 默认是 `real_chain.execution.mode: plan`。在没有真实手机拍摄 A/C 和背景素材时，计划模式会生成 7 个真实外部工具脚本，并返回 `NEEDS_INPUTS`，这是正确状态，不代表低质量降级。

待真实素材放入 `hw3/assets/` 后，把配置切到 `run`，再执行完整训练和渲染。

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
