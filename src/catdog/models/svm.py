"""支持向量机 (SVM) —— 纯 NumPy 实现，含 RBF 核 + SMO 简化版求解。

设计说明：
  * 二分类（cat=0 / dog=1）
  * 采用核技巧，支持 linear / rbf / poly 三种核
  * 使用序列最小优化 (SMO) 的轻量实现求解对偶问题
  * 通过 Platt 缩放 (sigmoid fitting) 输出概率，用于 ROC-AUC
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X @ Y.T


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """RBF 核。||x-y||^2 = |x|^2 + |y|^2 - 2<x,y>"""
    x2 = np.sum(X**2, axis=1)[:, None]
    y2 = np.sum(Y**2, axis=1)[None, :]
    d2 = np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-gamma * d2)


def poly_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0, coef0: float = 1.0, degree: int = 3) -> np.ndarray:
    return (gamma * (X @ Y.T) + coef0) ** degree


class SVMClassifier:
    """SMO 训练的核 SVM 二分类器。"""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Any = "scale",
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 500,
        probability: bool = True,
        random_state: int = 42,
    ) -> None:
        self.kernel_name = kernel
        self.C = float(C)
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.probability = bool(probability)
        self.random_state = int(random_state)

        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.alphas_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_alphas_: Optional[np.ndarray] = None
        self.support_y_: Optional[np.ndarray] = None
        self.gamma_: float = 1.0
        self._platt_A = -1.0
        self._platt_B = 0.0

    # ---------------- 核函数 ----------------
    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.kernel_name == "linear":
            return linear_kernel(X, Y)
        if self.kernel_name == "rbf":
            return rbf_kernel(X, Y, self.gamma_)
        if self.kernel_name == "poly":
            return poly_kernel(X, Y, self.gamma_, self.coef0, self.degree)
        raise ValueError(f"不支持的核函数: {self.kernel_name}")

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                return 1.0 / (X.shape[1] * X.var() + 1e-8)
            if self.gamma == "auto":
                return 1.0 / X.shape[1]
        return float(self.gamma)

    # ---------------- 训练（SMO） ----------------
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "SVMClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        # 标签映射到 {-1, +1}
        y = np.where(y > 0, 1.0, -1.0)

        n = X.shape[0]
        self.X_, self.y_ = X, y
        self.gamma_ = self._resolve_gamma(X)

        K = self._kernel(X, X)
        alphas = np.zeros(n)
        b = 0.0
        rng = np.random.default_rng(self.random_state)

        # 预先缓存误差：alphas 全为 0 时 f(x)=b=0，故 E = -y
        E = -y.copy()

        def _update_E(i: int) -> None:
            """用当前 alphas / b 重算第 i 个样本的误差。"""
            E[i] = float(np.dot(alphas * y, K[:, i]) + b - y[i])

        passes_no_change = 0
        for it in range(self.max_iter):
            num_changed = 0
            for i in range(n):
                Ei = E[i]
                # KKT 违背检查
                if (y[i] * Ei < -self.tol and alphas[i] < self.C) or (
                    y[i] * Ei > self.tol and alphas[i] > 0
                ):
                    # 第二变量选择：优先选 |Ei - Ej| 最大的 j（启发式加速收敛）
                    diff = np.abs(E - Ei)
                    diff[i] = -1.0
                    j = int(np.argmax(diff))
                    if diff[j] <= 0:
                        j = int(rng.integers(0, n - 1))
                        if j >= i:
                            j += 1
                    Ej = E[j]

                    ai_old, aj_old = alphas[i], alphas[j]
                    if y[i] != y[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)
                    if H <= L:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    aj_new = aj_old - y[j] * (Ei - Ej) / eta
                    aj_new = float(np.clip(aj_new, L, H))
                    if abs(aj_new - aj_old) < 1e-9:
                        continue

                    ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)
                    alphas[i], alphas[j] = ai_new, aj_new

                    b1 = (
                        b - Ei
                        - y[i] * (ai_new - ai_old) * K[i, i]
                        - y[j] * (aj_new - aj_old) * K[i, j]
                    )
                    b2 = (
                        b - Ej
                        - y[i] * (ai_new - ai_old) * K[i, j]
                        - y[j] * (aj_new - aj_old) * K[j, j]
                    )
                    if 0 < ai_new < self.C:
                        b = float(b1)
                    elif 0 < aj_new < self.C:
                        b = float(b2)
                    else:
                        b = float((b1 + b2) / 2.0)

                    _update_E(i)
                    _update_E(j)
                    num_changed += 1

            if num_changed == 0:
                passes_no_change += 1
                if passes_no_change >= 3:
                    break
            else:
                passes_no_change = 0
                # tol 逐步收紧，先粗后细，加速收敛
                self.tol = max(self.tol * 0.5, 1e-6)

            if verbose and (it + 1) % 20 == 0:
                print(f"  [SVM] SMO 迭代 {it + 1}/{self.max_iter}, 本轮更新 {num_changed} 个乘子")

        self.alphas_ = alphas
        self.b_ = b
        sv = alphas > 1e-6
        self.support_vectors_ = X[sv]
        self.support_alphas_ = alphas[sv]
        self.support_y_ = y[sv]

        if self.probability:
            self._fit_platt(X, y, K)
        return self

    # ---------------- 决策函数 ----------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.support_vectors_ is not None, "请先调用 fit()"
        X = np.asarray(X, dtype=np.float64)
        K = self._kernel(self.support_vectors_, X)
        return (self.support_alphas_ * self.support_y_) @ K + self.b_

    def _decision_scaled(self, X: np.ndarray) -> np.ndarray:
        """Platt 缩放所用的归一化决策值。"""
        return self.decision_function(X) / getattr(self, "_platt_scale", 1.0)

    def _fit_platt(self, X: np.ndarray, y: np.ndarray, K_train: np.ndarray) -> None:
        """Platt 缩放：拟合 p(dog|x) = sigmoid(A*f + B)，其中 f 为归一化决策值。

        对负对数似然做牛顿迭代：
            NLL = -Σ [t·log p + (1-t)·log(1-p)]
            ∇  = [ Σ f(p-t), Σ (p-t) ]
            H  = [[Σ pq f², Σ pq f], [Σ pq f, Σ pq]]   （半正定）
            更新: θ ← θ - H⁻¹ ∇
        由于 f>0 时样本更可能是 dog(t=1)，收敛结果 A>0。
        """
        f_raw = (self.alphas_ * self.y_) @ K_train + self.b_
        # 归一化尺度：决策值过小时 sigmoid 无法区分两类
        scale = float(np.std(f_raw)) if len(f_raw) > 1 else 1.0
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        f = f_raw / scale
        self._platt_scale = scale

        t = (y > 0).astype(np.float64)          # dog -> 1
        n_pos = float(t.sum())
        n_neg = float(len(t) - n_pos)
        # Platt 平滑目标，避免 0/1 导致发散
        t = np.where(t > 0, (n_pos + 1.0) / (n_pos + 2.0),
                     1.0 / (n_neg + 2.0))

        A, B = 1.0, 0.0    # f>0 => dog，故 A 初值为正
        for _ in range(200):
            z = np.clip(A * f + B, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            q = 1.0 - p
            gA = float(np.sum(f * (p - t)))
            gB = float(np.sum(p - t))
            h11 = float(np.sum(p * q * f * f)) + 1e-10
            h12 = float(np.sum(p * q * f))
            h22 = float(np.sum(p * q)) + 1e-10
            det = h11 * h22 - h12 * h12
            if det <= 1e-14:
                break
            stepA = (h22 * gA - h12 * gB) / det
            stepB = (h11 * gB - h12 * gA) / det
            # 限幅，避免早期迭代迈步过大
            stepA = float(np.clip(stepA, -2.0, 2.0))
            stepB = float(np.clip(stepB, -2.0, 2.0))
            A -= stepA
            B -= stepB
            A = float(np.clip(A, 0.05, 25.0))   # 保证 A>0（方向与决策值一致）
            B = float(np.clip(B, -25.0, 25.0))
            if abs(stepA) < 1e-8 and abs(stepB) < 1e-8:
                break

        if not np.isfinite(A) or A <= 0:
            A = 1.0
        if not np.isfinite(B):
            B = 0.0
        # 统一以 p(dog) = sigmoid(A*f + B) 的约定保存，f>0 对应 dog => A > 0
        self._platt_A, self._platt_B = float(A), float(B)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回 (N, 2) 概率矩阵 [P(cat), P(dog)]。

        采用 p(dog) = sigmoid(_platt_A * f + _platt_B) 的约定，
        f 为决策值（>0 倾向于 dog），故 _platt_A > 0。
        """
        f = self._decision_scaled(X)
        p_dog = 1.0 / (1.0 + np.exp(np.clip(-(self._platt_A * f + self._platt_B), -30, 30)))
        p_dog = np.clip(p_dog, 1e-6, 1 - 1e-6)
        return np.stack([1 - p_dog, p_dog], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(np.int64)

    # ---------------- 持久化 ----------------
    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            support_vectors=self.support_vectors_,
            support_alphas=self.support_alphas_,
            support_y=self.support_y_,
            b=np.array([self.b_]),
            gamma=np.array([self.gamma_]),
            platt=np.array([self._platt_A, self._platt_B]),
            platt_scale=np.array([getattr(self, "_platt_scale", 1.0)]),
            cfg=np.array(
                [self.C, self.degree, self.coef0, float(self.kernel_name == "rbf")],
            ),
        )

    @classmethod
    def load(cls, path: str) -> "SVMClassifier":
        data = np.load(path, allow_pickle=False)
        obj = cls()
        obj.support_vectors_ = data["support_vectors"]
        obj.support_alphas_ = data["support_alphas"]
        obj.support_y_ = data["support_y"]
        obj.b_ = float(data["b"][0])
        obj.gamma_ = float(data["gamma"][0])
        obj._platt_A, obj._platt_B = float(data["platt"][0]), float(data["platt"][1])
        if "platt_scale" in data:
            obj._platt_scale = float(data["platt_scale"][0])
        obj.C = float(data["cfg"][0])
        obj.degree = int(data["cfg"][1])
        obj.coef0 = float(data["cfg"][2])
        obj.kernel_name = "rbf" if data["cfg"][3] > 0.5 else "linear"
        return obj


# 兼容别名：线性 SVM 即 linear 核的 SVMClassifier
def LinearSVM(C: float = 1.0, max_iter: int = 500, **kw) -> SVMClassifier:
    return SVMClassifier(kernel="linear", C=C, max_iter=max_iter, probability=True, **kw)


def svm_config_to_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kernel": cfg.get("kernel", "rbf"),
        "C": cfg.get("C", 1.0),
        "gamma": cfg.get("gamma", "scale"),
        "max_iter": cfg.get("max_iter", 500),
        "probability": cfg.get("probability", True),
    }
