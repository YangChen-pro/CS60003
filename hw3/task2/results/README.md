# HW3 Task2 Results

该目录只保存可提交的小体积结果证据；完整 checkpoint、日志缓存和数据集仍保留在 135 远程机器与 ModelScope。

## 核心结果

- 主结果表：`task2_results_table.csv` / `task2_results_table.json`。
- final-only 评估表：`final_only_eval_table.csv` / `final_only_eval_table.json`。
- 清洗后的曲线源数据：`curves/*_metrics_clean.csv`。
- 曲线图：`figures/*.png`。

## Final-only D 评估

- `act_splitA` final Action L1 = `0.1886211640`, best Action L1 = `0.1731817282`。
- `act_splitABC` final Action L1 = `0.1549013643`, best Action L1 = `0.1464656246`。

## 曲线数据摘要

```json
{
  "copy": {
    "copied_files": [
      "task2_results_table.csv",
      "task2_results_table.json",
      "act_splitA_train_summary.json",
      "act_splitA_dataset_summary.json",
      "act_splitA_results_summary.json",
      "act_splitA_splitD.json",
      "act_splitA_splitD.csv",
      "act_splitA_final_splitD.json",
      "act_splitA_final_splitD.csv",
      "act_splitABC_train_summary.json",
      "act_splitABC_dataset_summary.json",
      "act_splitABC_results_summary.json",
      "act_splitABC_splitD.json",
      "act_splitABC_splitD.csv",
      "act_splitABC_final_splitD.json",
      "act_splitABC_final_splitD.csv"
    ]
  },
  "curves": {
    "act_splitA": {
      "source": "swanlab",
      "train_points": 114,
      "eval_points": 8,
      "last_train_action_l1": 0.0721546784043312,
      "best_eval_action_l1": 0.173181727528572,
      "final_eval_action_l1_from_training": 0.1886211633682251
    },
    "act_splitABC": {
      "source": "swanlab",
      "train_points": 335,
      "eval_points": 8,
      "last_train_action_l1": 0.0621607340872287,
      "best_eval_action_l1": 0.1464656293392181,
      "final_eval_action_l1_from_training": 0.1549013704061508
    }
  }
}
```
