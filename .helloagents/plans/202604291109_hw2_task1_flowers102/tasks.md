# HW2 Task1 Flowers102 — 任务分解

## 任务列表
- [ ] 任务1：建立 Task1 工程结构（涉及文件：`hw2/task1/`；完成标准：目录、包入口、配置目录和 README 存在；验证方式：远程 `python -m compileall hw2/task1`）。
- [ ] 任务2：实现 Flowers102 官方划分数据管线（涉及文件：`hw2/task1/flowers102_task1/data.py`；完成标准：读取 `imagelabels.mat`、`setid.mat`，校验 train 1020 / val 1020 / test 6149；验证方式：远程运行数据检查命令）。
- [ ] 任务3：实现 ResNet-18 Baseline、随机初始化和 SE 注意力模型（涉及文件：`hw2/task1/flowers102_task1/models.py`；完成标准：三类模型均能创建 102 类输出；验证方式：远程导入模型并打印参数组）。
- [ ] 任务4：实现训练、评估、指标和曲线保存（涉及文件：`hw2/task1/train.py`、`hw2/task1/evaluate.py`、`hw2/task1/flowers102_task1/engine.py`、`metrics.py`、`utils.py`；完成标准：每次实验保存 `history.csv`、`metrics.json`、`best.pt`、`curves.png`；验证方式：远程运行至少一个正式实验）。
- [ ] 任务5：编写实验配置与运行说明（涉及文件：`hw2/task1/configs/*.yaml`、`hw2/task1/README.md`、`.gitignore`；完成标准：Baseline、低学习率、随机初始化、SE 配置可复用；验证方式：远程按 README 命令启动）。
- [ ] 任务6：Git 同步并远程执行正式实验（涉及文件：Git 本地/远程仓库；完成标准：远程 HEAD 与本机一致，Git 身份为 `YangChen-pro <1369792882@qq.com>`，至少 Baseline 正式实验完成；验证方式：远程日志、`metrics.json` 和 `git rev-parse HEAD`）。
- [ ] 任务7：整理结果与知识库（涉及文件：`hw2/task1/README.md`、`.helloagents/modules/hw2.md`、`.helloagents/CHANGELOG.md`；完成标准：记录已完成实验、指标位置和后续报告素材；验证方式：核对文件内容与远程产物一致）。

## 进度
- [√] 已确认用户约束：不在本机 smoke test，远程直接正式实验，代码通过 Git 同步。
- [ ] 待实现代码。
- [ ] 待 Git 同步到远程。
- [ ] 待远程正式训练与结果整理。
