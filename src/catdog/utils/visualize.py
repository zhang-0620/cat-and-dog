"""可视化：混淆矩阵、ROC 曲线、预测样例网格、指标柱状图、灰度图集。

使用 Matplotlib 输出 PNG 图片（Agg 后端，无需 GUI）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体回退（Windows 常见字体）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 配色：中国股市习惯 —— 涨为红。此处用于强调「正确/错误」而非涨跌
COLOR_CORRECT = "#1f9d55"   # 绿色：正确
COLOR_WRONG = "#d13438"     # 红色：错误
COLOR_CAT = "#f0a020"
COLOR_DOG = "#2f7bd8"


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: str | Path,
    normalize: bool = False,
    title: str = "混淆矩阵 / Confusion Matrix",
) -> Path:
    """绘制混淆矩阵热力图。"""
    cm_show = cm.astype(float)
    if normalize:
        row_sum = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, row_sum, out=np.zeros_like(cm_show), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=130)
    im = ax.imshow(cm_show, cmap="Blues", vmin=0)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("预测标签 / Predicted", fontsize=12)
    ax.set_ylabel("真实标签 / Actual", fontsize=12)
    ax.set_title(title, fontsize=13, pad=14, fontweight="bold")

    thresh = cm_show.max() / 2.0 if cm_show.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_show[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
            ax.text(
                j, i, text,
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if cm_show[i, j] > thresh else "#1a1a1a",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str | Path,
    auc: float | None = None,
) -> Path:
    """绘制 ROC 曲线 + AUC 面积。"""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    thresholds = np.unique(np.concatenate([[0.0, 1.0], np.sort(np.unique(y_score))[::-1]]))
    tprs, fprs = [], []
    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)

    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=130)
    ax.plot(fprs, tprs, color="#2f7bd8", lw=2.2, label=f"ROC (AUC = {auc:.4f})" if auc else "ROC")
    ax.fill_between(fprs, tprs, alpha=0.16, color="#2f7bd8")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.2, label="随机猜测 / Random")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("假正率 / False Positive Rate", fontsize=12)
    ax.set_ylabel("真正率 / True Positive Rate", fontsize=12)
    ax.set_title("ROC 曲线 / ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.28)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def plot_sample_predictions(
    images: List[np.ndarray],
    filenames: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    save_path: str | Path,
    grid: tuple[int, int] = (3, 6),
) -> Path:
    """绘制预测样例网格：绿框=预测正确，红框=预测错误。"""
    rows, cols = grid
    n = min(len(images), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.25 * rows), dpi=130)
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        if idx >= n:
            continue
        img = np.asarray(images[idx])
        # 归一化到 [0,1] 以便正确显示（float 数组直接 imshow 会全黑）
        if img.dtype != np.uint8:
            lo, hi = float(img.min()), float(img.max())
            img = (img - lo) / (hi - lo + 1e-8)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
        else:
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ok = int(y_true[idx]) == int(y_pred[idx])
        color = COLOR_CORRECT if ok else COLOR_WRONG
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.6)
            spine.set_visible(True)
        ax.set_title(
            f"{class_names[int(y_pred[idx])]}",
            fontsize=9, color=color, pad=2.5,
        )
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "测试集预测样例（绿=正确 / 红=错误）  Sample Predictions",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def plot_metric_bars(
    metrics_by_model: Dict[str, Dict[str, float]],
    save_path: str | Path,
    metric_keys: Sequence[str] = ("accuracy", "precision", "recall", "f1", "roc_auc"),
) -> Path:
    """对比多个模型的指标柱状图（分组）。"""
    models = list(metrics_by_model.keys())
    keys = [k for k in metric_keys if any(k in m for m in metrics_by_model.values())]

    x = np.arange(len(keys))
    width = 0.8 / max(len(models), 1)
    palette = ["#2f7bd8", "#f0a020", "#1f9d55", "#8a5cd6", "#d13438"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=130)
    for i, m in enumerate(models):
        vals = [metrics_by_model[m].get(k, 0.0) for k in keys]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=m, color=palette[i % len(palette)])
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in keys], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("得分 / Score", fontsize=12)
    ax.set_title("模型指标对比 / Model Metrics Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=min(len(models), 4))
    ax.grid(axis="y", alpha=0.28)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def save_grayscale_grid(
    images: List[np.ndarray],
    save_path: str | Path,
    grid: tuple[int, int] = (4, 8),
    title: str = "灰度化 + 尺寸归一化后的图像 / Grayscale & Resized",
) -> Path:
    """输出灰度预处理结果网格，用于展示预处理效果。"""
    rows, cols = grid
    n = min(len(images), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.5 * rows), dpi=130)
    axes = np.atleast_2d(axes)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        if idx >= n:
            continue
        img = np.asarray(images[idx])
        is_gray = img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        if img.dtype != np.uint8:
            lo, hi = float(img.min()), float(img.max())
            img = (img - lo) / (hi - lo + 1e-8)
            ax.imshow(img, cmap="gray" if is_gray else None, vmin=0, vmax=1)
        else:
            ax.imshow(img, cmap="gray" if is_gray else None)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path
