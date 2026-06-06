# HW3 Task1 SwanLab 记录

项目链接：<https://swanlab.cn/@youngchen/cs60003-hw3-task1>

训练命令在 136 的 `qwen14b` 环境中执行，运行前通过 `.helloagents/secrets/hw3.env` 注入 `SWANLAB_API_KEY`；本文档只记录 run 链接，不记录 key。

| 阶段 | 技术路径 | SwanLab run | 状态 |
|---|---|---|---|
| object_a_3dgs | COLMAP + Nerfstudio `splatfacto` | <https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/dcbclc90gpdtpsn69brv3> | 已完成 30k steps |
| object_b_sds | threestudio + Stable Diffusion SDS | <https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/d2ebu67uf76o4cceq7avb> | 已完成 15k steps |
| object_c_zero123 | threestudio + Zero123XL | <https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/5yhocvbs6adfpiuo0f1jg> | 已完成 1200 steps |
| background_3dgs | Mip-NeRF 360 `counter` + Nerfstudio `splatfacto` | <https://swanlab.cn/@youngchen/cs60003-hw3-task1/runs/7r8mxm6zzz7kjblrtbhm8> | 已完成 30k steps |

说明：Nerfstudio 训练曲线通过 TensorBoard scalar sync 写入 SwanLab；threestudio / Zero123 训练曲线通过 SwanLab 的 WandB scalar sync 写入 SwanLab，并关闭 WandB 云端上传。
