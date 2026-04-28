# 项目状态

主线目标：为 HW2 期中作业准备全部题面数据集，沿用 HW1 的“数据集直接放在作业目录下”组织方式。

正在做什么：HW2 数据集已下载并完成基础完整性检查。

关键上下文：
- 题面文件：`hw2/hw2.md`
- HW1 参考布局：`hw1/EuroSAT_RGB/` 直接位于作业目录下
- HW2 数据集目录：
  - `hw2/Flowers102/`：Oxford 102 Category Flower Dataset，含 `jpg/`、`imagelabels.mat`、`setid.mat`、`README.txt`
  - `hw2/RoadVehicleImages/`：Kaggle Road Vehicle Images Dataset，含 `trafic_data/train/`、`trafic_data/valid/`
  - `hw2/StanfordBackground/`：Stanford Background Dataset，含 `iccv09Data/images/`、`iccv09Data/labels/`
- Road Vehicle 原始 `data_1.yaml` 路径可能不适合当前目录层级，已生成本地配置 `hw2/RoadVehicleImages/trafic_data/data_hw2.yaml`
- 数据说明已写入 `hw2/DATASETS.md`
- 用户已确认删除 HW2 数据集目录的 `.gitignore` 规则；数据集可被 Git 发现，但会显著增大仓库体积。

下一步：后续进入 HW2 实现阶段时，先基于 `hw2/DATASETS.md` 读取数据位置；任务 2 仍需准备 10–30 秒测试视频。

阻塞项：
- 无
