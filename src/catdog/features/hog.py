"""HOG (方向梯度直方图) 特征提取，纯 NumPy 实现。

HOG 通过统计局部区域的梯度方向分布来描述物体形状，
对猫狗的轮廓、耳朵、五官等形状差异具有较好区分力。
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.shape[-1] == 1:
        return img[..., 0].astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def _gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """使用中心差分计算梯度幅值和方向（角度制，0~180）。"""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    return magnitude, angle


def hog_descriptor(
    img: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Sequence[int] = (8, 8),
    cells_per_block: Sequence[int] = (2, 2),
    block_norm: str = "L2-Hys",
    eps: float = 1e-6,
) -> np.ndarray:
    """计算单张图像的 HOG 特征向量。

    Args:
        img: (H, W, C) 或 (H, W)，float 或 uint8
        orientations: 方向 bin 数量（180° 均分）
        pixels_per_cell: (cell_h, cell_w) 每个 cell 的像素数
        cells_per_block: (block_h, block_w) 每个 block 包含的 cell 数
        block_norm: 归一化方式，'L2-Hys' | 'L2' | 'L1' | 'none'

    Returns:
        一维特征向量
    """
    gray = _to_gray(img)
    # 归一化到 [0,1] 以降低光照影响
    if gray.max() > 1.5:
        gray = gray / 255.0

    mag, ang = _gradients(gray)

    ph, pw = int(pixels_per_cell[0]), int(pixels_per_cell[1])
    cbh, cbw = int(cells_per_block[0]), int(cells_per_block[1])

    H, W = gray.shape
    n_cells_y = H // ph
    n_cells_x = W // pw
    if n_cells_y < cbh or n_cells_x < cbw:
        raise ValueError(
            f"图像过小 ({H}x{W})，无法形成 {cbh}x{cbw} 的 block。"
            f"请增大图像尺寸或减小 cell/block 配置。"
        )

    bin_width = 180.0 / orientations
    # 每个 cell 的方向直方图
    cell_hist = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)

    bin_idx = (ang / bin_width).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, orientations - 1)

    ys = (np.arange(n_cells_y)[:, None] * ph)
    xs = (np.arange(n_cells_x)[None, :] * pw)

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            m = mag[cy * ph : (cy + 1) * ph, cx * pw : (cx + 1) * pw].ravel()
            b = bin_idx[cy * ph : (cy + 1) * ph, cx * pw : (cx + 1) * pw].ravel()
            np.add.at(cell_hist[cy, cx], b, m)

    # block 级归一化并拼接
    features: list[np.ndarray] = []
    for by in range(n_cells_y - cbh + 1):
        for bx in range(n_cells_x - cbw + 1):
            block = cell_hist[by : by + cbh, bx : bx + cbw, :].ravel().astype(np.float32)
            features.append(_normalize_block(block, block_norm, eps))

    if not features:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(features).astype(np.float32)


def _normalize_block(block: np.ndarray, mode: str, eps: float) -> np.ndarray:
    mode = mode.lower()
    if mode == "l1":
        return block / (np.abs(block).sum() + eps)
    if mode == "l2":
        return block / np.sqrt((block**2).sum() + eps**2)
    if mode == "l2-hys":
        block = block / np.sqrt((block**2).sum() + eps**2)
        block = np.clip(block, 0, 0.2)
        block = block / np.sqrt((block**2).sum() + eps**2)
        return block
    return block


def hog_features(
    images: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Sequence[int] = (8, 8),
    cells_per_block: Sequence[int] = (2, 2),
    block_norm: str = "L2-Hys",
    verbose: bool = False,
) -> np.ndarray:
    """批量计算 HOG 特征。

    Args:
        images: (N, H, W, C) 图像批

    Returns:
        (N, D) 特征矩阵
    """
    feats = []
    for i, img in enumerate(images):
        feats.append(
            hog_descriptor(
                img,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm=block_norm,
            )
        )
        if verbose and (i + 1) % 20 == 0:
            print(f"  HOG 进度 {i + 1}/{len(images)}")
    return np.stack(feats, axis=0)
