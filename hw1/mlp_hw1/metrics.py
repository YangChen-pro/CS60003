"""Metric helpers."""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute a confusion matrix without sklearn."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def per_class_accuracy(matrix: np.ndarray) -> np.ndarray:
    """Compute per-class accuracy from a confusion matrix."""
    totals = matrix.sum(axis=1)
    scores = np.zeros(matrix.shape[0], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(matrix.diagonal(), totals, out=scores, where=totals > 0)
    return scores
