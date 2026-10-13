"""颜色直方图特征：描述猫狗的毛色分布差异（如橘猫 vs 黑狗）。"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """向量化 RGB -> HSV 转换。输入 (..., 3) ∈ [0,1]，输出 H∈[0,1], S,V∈[0,1]。"""
    arr = rgb.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    delta = maxc - minc + 1e-8

    h = np.zeros_like(maxc)
    mask = maxc == r
    h[mask] = ((g - b) / delta)[mask] % 6
    mask = maxc == g
    h[mask] = ((b - r) / delta)[mask] + 2
    mask = maxc == b
    h[mask] = ((r - g) / delta)[mask] + 4
    h = h / 6.0

    s = np.where(maxc > 1e-8, delta / (maxc + 1e-8), 0.0)
    v = maxc
    return np.stack([h, s, v], axis=-1)


def color_histogram(
    img: np.ndarray,
    bins: int = 16,
    space: str = "HSV",
    normalize: bool = True,
) -> np.ndarray:
    """计算单张图像的颜色直方图特征。

    Args:
        img: (H, W, 3) RGB 图像
        bins: 每个通道的 bin 数，总维度 = bins * 3
        space: 'HSV' 或 'RGB'
        normalize: 是否做 L1 归一化（消除图像大小影响）

    Returns:
        (bins*3,) 特征向量
    """
    arr = img.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.max() > 1.5:
        arr = arr / 255.0

    if space.upper() == "HSV":
        arr = _rgb_to_hsv(arr)

    feats = []
    for c in range(3):
        hist, _ = np.histogram(arr[..., c], bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        if normalize:
            hist = hist / (hist.sum() + 1e-8)
        feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


def color_hist_features(
    images: np.ndarray,
    bins: int = 16,
    space: str = "HSV",
) -> np.ndarray:
    """批量计算颜色直方图特征，返回 (N, bins*3)。"""
    return np.stack([color_histogram(img, bins=bins, space=space) for img in images], axis=0)
