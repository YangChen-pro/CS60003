"""Experiment summary and report generation."""

from __future__ import annotations

import json
from pathlib import Path

from .config import REPORT_PATH


def build_report(
    best_run_dir: Path,
    best_summary: dict,
    search_dir: Path | None,
    top_trials: list[dict],
    report_path: Path = REPORT_PATH,
) -> Path:
    """Generate a concise markdown report for HW1."""
    lines = [
        "# HW1 实验简报",
        "",
        "## 1. 作业目标",
        "",
        "- 使用手写反向传播的三层 MLP 完成 EuroSAT 十分类任务。",
        "- 训练逻辑不依赖 PyTorch / TensorFlow / JAX 自动微分。",
        "- 在满足作业要求前提下尽量逼近三层 MLP 的性能上限。",
        "",
        "## 2. 最终模型",
        "",
        f"- 激活函数：`{best_summary['config']['activation']}`",
        f"- 隐层宽度：`{best_summary['config']['hidden_dim']} -> {best_summary['config']['hidden_dim2']}`",
        f"- epoch：`{best_summary['config']['epochs']}`",
        f"- 学习率：`{best_summary['config']['learning_rate']}`",
        f"- 学习率衰减：`{best_summary['config']['lr_decay']}`",
        f"- L2 正则：`{best_summary['config']['weight_decay']}`",
        f"- 梯度裁剪：`{best_summary['config']['grad_clip']}`",
        "",
        "## 3. 核心结果",
        "",
        f"- 最佳验证集准确率：`{best_summary['best_val_accuracy']:.4f}`",
        f"- 测试集准确率：`{best_summary['test_accuracy']:.4f}`",
        f"- 最佳 epoch：`{best_summary['best_epoch']}`",
        "",
        "## 4. 必要图表",
        "",
        f"- 训练曲线：`{(best_run_dir / 'training_curves.png').relative_to(report_path.parent.parent)}`",
        f"- 混淆矩阵：`{(best_run_dir / 'confusion_matrix.png').relative_to(report_path.parent.parent)}`",
        f"- 第一层权重可视化：`{(best_run_dir / 'first_layer_weights.png').relative_to(report_path.parent.parent)}`",
        f"- 错例分析图：`{(best_run_dir / 'misclassified_examples.png').relative_to(report_path.parent.parent)}`",
        "",
        "## 5. 搜索结果摘要",
        "",
    ]
    if search_dir is not None:
        lines.append(f"- 搜索目录：`{search_dir.relative_to(report_path.parent.parent)}`")
    for index, row in enumerate(top_trials, start=1):
        lines.append(
            f"- Top {index}: act={row['activation']}, hidden=({row['hidden_dim']},{row['hidden_dim2']}), "
            f"lr={row['learning_rate']}, decay={row['lr_decay']}, wd={row['weight_decay']}, "
            f"clip={row['grad_clip']}, val_acc={row['best_val_accuracy']:.4f}, test_acc={row['test_accuracy']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 6. 结论",
            "",
            "- 在当前作业约束下，三层 MLP 主要受模型表达能力限制，增大双隐层宽度和适度拉长训练轮数有明显帮助。",
            "- `ReLU` 往往比 `tanh` 更稳定，配合较小初始学习率、较弱权重衰减和梯度裁剪更容易得到更高验证准确率。",
            "- 后续若还要继续冲性能，优先继续扩大双隐层宽度并延长训练，而不是引入明显超出作业边界的结构。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
