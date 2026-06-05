# 恢复快照

## 主线目标
补齐活跃方案包 `202604091720_hw1_mlp` 的结构化 QA 契约，并完成 `~auto -> ~plan -> ~qa` 收尾。

## 正在做什么
空闲；HW1 活跃方案包 QA 契约与质量闭环已完成。

## 关键上下文
- 当前 HelloAGENTS 版本为 3.1.1，`project_store_mode=local`，Codex 安装模式为 standby。
- 运行时要求当前显式 `~auto` 不停在只读 HW3 GPU 判断，需先补齐活跃方案包 `202604091720_hw1_mlp` 的 `contract.json`，再进入 `~qa`。
- 已将 `.helloagents/plans/202604091720_hw1_mlp/contract.json` 改为兼容契约：保留 `verifyMode` / `reviewerFocus` / `testerFocus`，新增 `qaMode=standard` 与 `qaFocus`。
- QA 已通过：`python3 -X utf8 -m unittest discover -s hw1/tests`、`python3 -X utf8 tools/verify_delivery_metadata.py`、入口 `py_compile`。
- 已写入当前会话结构化证据：`artifacts/qa-review.json` 与 `artifacts/closeout.json`。

## 下一步
提交 `contract.json` 与本状态快照变更；随后写 turn-state complete。

## 阻塞项
（无）

## 方案


## 已标记技能
helloagents
