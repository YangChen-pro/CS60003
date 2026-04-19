# 项目状态

主线目标：收口 HW1 的最终提交版本，使代码、README、输出证据与 PDF 报告完全一致，并达到 A+ 水准。

正在做什么：知识库已同步到 HW1 最终提交状态，最新 KB 更新已提交、推送并同步到 135；严格复现复审已给出 A+。

关键上下文：
- 主实现使用手写反向传播，`CuPy` 仅作为数组后端
- 所有 HW1 相关代码均保留在 `hw1/` 目录
- 正式提交模型仍为 `final_p`：`1280 -> 768`、`dropout=0.15`、`44 epochs`、`val_acc=0.6901`、`test_acc=0.6758`
- 扩展实验 `final_o` 的测试集准确率更高（`0.6810`），但按课程要求仍以验证集最优的 `final_p` 作为正式提交
- 当前 full search 主证据为 `hw1/outputs/search/20260419_045438/results.csv`、`results.json` 与 `best_result.json`
- 为闭合证据链，`hw1/outputs/runs/trial_04/` 已保留 `config.json`、`history.json`、`summary.json` 与 `confusion_matrix.json`
- `hw1/outputs/runs/manifest.csv` 是历史实验台账，`source_group` 区分实验来源，`tracked_in_repo` 标明是否保留可直接核验产物
- 本机、GitHub 与 135 已同步到当前 `main` 最新提交；本机测试命令此前通过：`conda run -n nlp python -X utf8 -m unittest discover -s "hw1/tests"`
- 最终报告 PDF 位于 `/Users/yangchen/Documents/Latex_Project/CS60003_HW1_Report/out/elegantpaper-cn.pdf`

下一步：如需最终提交，提交 GitHub 链接 `https://github.com/YangChen-pro/CS60003` 与最终报告 PDF。

阻塞项：
- 无
