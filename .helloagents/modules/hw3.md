# 模块: HW3

## 用途
记录 CS60003 HW3 题面、技术路线、提交要求、实现边界与远程实验状态。当前已决定 HW3 两个题目都做；Task1 保留真实高质量 3DGS/AIGC 链路，Task2 已完成 LeRobot ACT 跨环境泛化实验。

## 关键文件
- `hw3/hw3.md`：期末作业 Markdown 题面。
- `hw3/hw3.pdf`：期末作业 PDF 题面。
- `hw3/assets/README.md`：真实素材路径约定；A 多视角、C 单图和背景素材默认不进 Git。
- `hw3/task1/`：题目一工程目录，结构对齐 HW1/HW2，包含 `configs/real_high_quality.yaml`、README、RESULTS、train/evaluate 和内部包。
- `hw3/task2/`：题目二 LeRobot ACT 工程目录，包含 `configs/act_splitA.yaml`、`configs/act_splitABC.yaml`、README、训练/评估/ModelScope 上传脚本和 `src/hw3_task2/` 内部包。
- `.helloagents/secrets/hw3.env`：HW3 临时凭据 env 文件，Git 忽略，不提交；本机、135 和 136 按需保留各自路径副本。

## 依赖
- 题目一候选技术：3D Gaussian Splatting、COLMAP、Nerfstudio、threestudio、2D diffusion + SDS Loss、TripoSR、Blender 或等价融合渲染链路。
- 题目二技术：LeRobot `ACTPolicy`、CALVIN LeRobot 数据集、PyTorch DDP/torchrun、SwanLab 曲线记录、ModelScope 权重发布。
- 提交依赖：PDF 实验报告、Public GitHub Repository、模型权重云盘或 ModelScope 链接。

## 经验
- 组队人数要求少于或等于 3 人；报告首页必须标明所有组员姓名、学号和具体分工。
- 截止时间为 2026-06-24 23:59（北京时间）。
- 题目一要求同时完成真实多视角重建、文本到 3D、单图到 3D、背景场景重建、三类资产融合渲染和技术报告对比。
- 题目二要求对比环境 A 单环境 ACT 策略与 A/B/C 多环境联合策略，并在未见过的环境 D 做 zero-shot 泛化测试。
- 已决定 HW3 同时完成题目一和题目二，不按二选一处理；后续计划、报告和 README 需要分别覆盖两部分实验。
- 题目一现在只保留高质量真实链路：真实 A 多视角/视频、真实背景多视角/视频、B 文本到 3D、C 单图到 3D、Blender 融合渲染。
- `real_high_quality` 的 plan 模式允许在真实素材缺失时返回 `NEEDS_INPUTS`，但仍生成真实外部工具脚本；不再生成低质量 proxy 结果。
- ModelScope 只放训练好的模型权重。非权重文件、summary、metrics、图片、GIF、报告素材和 proxy 文件不上传；此前误传的 `hw3/task1/formal_ai_chain/` 已从 `youngchen/CS60003` 删除。
- HW3 运行时凭据通过环境变量提供：`SWANLAB_API_KEY` 用于 SwanLab，`MODELSCOPE_API_TOKEN` 用于 ModelScope；从 `.helloagents/secrets/hw3.env` 读取，不在 Git 跟踪文件中写入具体值。
- 136 的 `qwen14b` 已安装 `swanlab==0.7.19` 和 `modelscope==1.37.1`；SwanLab 已用用户 HW3 临时 key 登录为 `youngchen`。
- 开始正式训练前需要先放入真实 A/C/背景素材、确认 136 外部工具安装、再把 `real_chain.execution.mode` 从 `plan` 切到 `run`。
- Task2 数据准备已验证官方 `task_ABC_D.zip` 支持 HTTP Range：无需下载完整 517GB 大包，可用 `hw3/task2/scripts/extract_calvin_zip_subset.py` 缓存约 219MB ZIP 中央目录后按 `scene_info.npy` 抽取 A/B/C episode；135 验证目录为 `/data/yc/CS60003/hw3/task2/data/calvin_abc_subset_probe`。
- Task2 正式环境固定为 135：远程主机 `135-3090-8`，项目路径 `/data/yc/CS60003`，Python 环境 `/data/yc/miniconda/envs/llm-26-gpu`；本机、GitHub、135 需保持 `main` 同步。
- Task2 数据采用老师划分后的 `xiaoma26/calvin-lerobot` 路线，落地目录为 `/data/yc/CS60003/hw3/task2/data/calvin_lerobot`；split 映射为 `splitA=A`、`splitB=B`、`splitC=C`、`splitD=D`。
- Task2 训练两个 ACT 模型：`act_splitA` 用 `splitA` 训练、`act_splitABC` 用 `splitA+splitB+splitC` 训练，二者都在未见过的 `splitD` 上做离线 Action L1 评估。
- Task2 正式结果：`act_splitA` 显式评估 Action L1 为 `0.1731817282`；`act_splitABC` 显式评估 Action L1 为 `0.1464656246`。结论是多环境联合训练在 D 环境上误差更低。
- Task2 SwanLab 项目为 `CS60003_HW3_Task2`；正式 run：`act_splitA=05kubpls24j5jrp2wbl1t`，`act_splitABC=4si6dcrut2krorrbalfkn`。
- Task2 ModelScope 最终位置是统一项目 `youngchen/CS60003` 下的 `hw3/task2/`，包含 `act_splitA/`、`act_splitABC/` 与 `eval/`；此前两个独立模型仓 `youngchen/CS60003-HW3-Task2-ACT-splitA` 和 `youngchen/CS60003-HW3-Task2-ACT-splitABC` 是误放位置，不作为交付链接。
- Task2 上传脚本 `hw3/task2/scripts/upload_modelscope.sh` 第 4 个参数是 ModelScope 仓内前缀，正式值为 `hw3/task2`；配置中的 `upload.repo_id` 固定为 `youngchen/CS60003`，`upload.path_prefix` 固定为 `hw3/task2`。

- Task2 已补齐 A+ 报告所需小体积证据：`hw3/task2/results/` 保存 best/final-only D 评估表、SwanLab 曲线源数据、曲线图、任务/episode/动作维度分解和 CALVIN simulator 探测记录。
- Task2 final-only 结果：`act_splitA` final splitD Action L1 为 `0.1886211640`，`act_splitABC` final splitD Action L1 为 `0.1549013643`；二者均弱于 best，但多环境训练优势保持，报告中应说明 best 使用 D 离线指标选 checkpoint。
- Task2 官方 CALVIN simulator 探测：135 已克隆 `/data/yc/tools/calvin_sim/calvin` 和 `calvin_env`，补装轻量依赖后官方评估入口可导入；当前 `xiaoma26/calvin-lerobot` splitD 缺少 `validation/.hydra/merged_config.yaml`，不能直接跑真实 Success Rate。官方 `task_D_D.zip` 约 165GiB，若要真实 Success Rate 需另行准备原始 CALVIN validation 数据并实现 ACT 的 `CustomModel.step(obs, goal)` 适配。
- Task2 精简依赖版本写入 `hw3/task2/requirements.txt`；CALVIN simulator 额外探测依赖不放入作业精简 requirements。

- Task2 A+ 补强新增 paired 统计与复现检查：`hw3/task2/results/STATISTICAL_SUMMARY.md` 记录 final.pt 主口径下 episode/task/action-dim 配对优势；`hw3/task2/scripts/check_reproducibility.sh --strict-data` 在 135 上验证 Python 包、GPU、数据 split、结果文件和 final.pt ABC 优势均 PASS。
- Task2 final.pt 统计主结论：ABC 相对 A 的 Action L1 提升为 `17.88%`；episode 级 `4041/5124` 更优，task 级 `355/389` 更优，action-dim 级 `7/7` 更优。报告和 README 应以 final.pt 为主结论，best.pt 只作为参考。
