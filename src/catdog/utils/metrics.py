"""评估指标：准确率、精确率、召回率、F1、ROC-AUC、混淆矩阵。

全部使用纯 NumPy 实现，不依赖 scikit-learn，保证项目可独立运行。
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """计算混淆矩阵，cm[i, j] = 真实为 i、预测为 j 的样本数。"""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """二分类 ROC-AUC。使用 Mann-Whitney U 统计量的秩和公式计算。

    AUC = (sum(rank(positive)) - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sorted_scores = y_score[order]

    # 处理并列分数（平均秩）
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
    positive_label: int = 1,
) -> Dict[str, float]:
    """计算完整分类指标。

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_score: 正类概率（用于 ROC-AUC），可为 None
        positive_label: 正类标签值

    Returns:
        包含 accuracy / precision / recall / f1 / roc_auc 的字典
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    n = len(y_true)

    accuracy = float((y_true == y_pred).sum() / n) if n else 0.0

    tp = int(((y_pred == positive_label) & (y_true == positive_label)).sum())
    fp = int(((y_pred == positive_label) & (y_true != positive_label)).sum())
    fn = int(((y_pred != positive_label) & (y_true == positive_label)).sum())
    tn = int(((y_pred != positive_label) & (y_true != positive_label)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    if y_score is not None:
        result["roc_auc"] = roc_auc_binary(y_true, np.asarray(y_score, dtype=float))

    return result


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """按类别计算 precision / recall / f1。"""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    out: Dict[str, Dict[str, float]] = {}
    for cls in np.unique(np.concatenate([y_true, y_pred])):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out[str(int(cls))] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int((y_true == cls).sum()),
        }
    return out
