"""主成分分析 (PCA) —— 用于特征降维，降低 SVM 训练开销并抑制噪声。

纯 NumPy 实现，基于 SVD 分解。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class PCA:
    """标准 PCA：中心化 -> SVD -> 投影到前 k 个主成分。"""

    def __init__(self, n_components: int = 120, whiten: bool = False) -> None:
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None      # (k, D)
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        k = min(self.n_components, n - 1, d)
        self.mean_ = X.mean(axis=0)

        Xc = X - self.mean_
        # economy SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.singular_values_ = S[:k]
        self.components_ = Vt[:k]
        var = (S**2) / max(n - 1, 1)
        self.explained_variance_ = var[:k]
        total = var.sum() + 1e-12
        self.explained_variance_ratio_ = var[:k] / total
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.components_ is not None, "请先调用 fit()"
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) @ self.components_.T
        if self.whiten:
            Z = Z / (np.sqrt(self.explained_variance_) + 1e-8)
        return Z.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        assert self.components_ is not None, "请先调用 fit()"
        X = np.asarray(Z, dtype=np.float64) @ self.components_ + self.mean_
        return X.astype(np.float32)

    @property
    def n_features_out(self) -> int:
        return 0 if self.components_ is None else self.components_.shape[0]
