"""逻辑回归基线模型（作为最简对照，便于体现特征工程的增益）。"""
from __future__ import annotations

from typing import Optional

import numpy as np


class LogisticRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 300, l2: float = 1e-4,
                 batch_size: int = 64, seed: int = 42) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.seed = seed
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "LogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        # 特征标准化，避免量纲差异导致不收敛
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0) + 1e-8
        Xn = (X - self.mu_) / self.sigma_

        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(d)
        self.b = 0.0

        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                xb = Xn[idx[s : s + self.batch_size]]
                yb = y[idx[s : s + self.batch_size]]
                p = self._sigmoid(xb @ self.w + self.b)
                g = (p - yb) / len(xb)
                self.w -= self.lr * (xb.T @ g + self.l2 * self.w)
                self.b -= self.lr * g.sum()
            if ep % 50 == 0:
                p_all = self._sigmoid(Xn @ self.w + self.b)
                eps = 1e-9
                loss = -np.mean(y * np.log(p_all + eps) + (1 - y) * np.log(1 - p_all + eps))
                self.loss_history.append(float(loss))
                if verbose:
                    print(f"  [LR] epoch {ep:>3}  loss={loss:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xn = (X - self.mu_) / self.sigma_
        p = self._sigmoid(Xn @ self.w + self.b)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.stack([1 - p, p], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
