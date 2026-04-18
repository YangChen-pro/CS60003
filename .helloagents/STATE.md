# 项目状态

主线目标：收口 HW1 的最终提交版本，使代码、README、输出证据与 PDF 报告完全一致，并达到 A+ 水准。

正在做什么：本轮已完成 README、代码 preset、测试与报告的联动修订；当前处于可提交状态。

关键上下文：
- 主实现使用手写反向传播，`CuPy` 仅作为数组后端
- 所有 HW1 相关文件均保留在 `hw1/` 目录
- 正式提交模型仍为 `final_p`：`1280 -> 768`、`dropout=0.15`、`44 epochs`、`val_acc=0.6901`、`test_acc=0.6758`
- 扩展实验 `final_o` 的测试集准确率更高（`0.6810`），但按课程要求仍以验证集最优的 `final_p` 作为正式提交
- 为补齐证据链，`final_k`、`final_l`、`final_n` 已加入正式 preset，README 和报告已同步说明 6 组邻域实验与两份 `search_config.json` 的角色
- 本机验证命令已通过：`conda run -n nlp python -X utf8 -m unittest discover -s "hw1/tests"`
- 报告最新 PDF 位于 `/Users/yangchen/Documents/Latex_Project/CS60003_HW1_Report/out/elegantpaper-cn.pdf`
- 多个子代理复审的最终结论已达到 `A+`

下一步：如需提交，执行 `git add` / `git commit`；如需同步远端 135，则推送当前仓库后在 `/data/yc/CS60003` 执行 `git pull`。

阻塞项：
- 无
