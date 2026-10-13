"""特征提取流水线：串联 HOG + 颜色直方图 + 标准化 + PCA。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .hog import hog_features
from .color_hist import color_hist_features
from .pca import PCA


class StandardScaler:
    """按特征维度做 z-score 标准化。

    核 SVM / 逻辑回归对特征尺度非常敏感：HOG 与颜色直方图量级差异巨大，
    不做标准化会导致 SMO 数值不稳定、决策值爆炸。
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class FeatureExtractor:
    """把原始图像批转换为可直接送入分类器的特征矩阵。

    流程：
        images (N,H,W,3)
          -> [HOG]  +  [Color Hist]
          -> 拼接
          -> [StandardScaler] 标准化
          -> [PCA 可选降维]
          -> features (N, D)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        feats = (config or {}).get("features", config or {})
        self.hog_cfg = feats.get("hog", {}) or {}
        self.color_cfg = feats.get("color_hist", {}) or {}
        self.pca_cfg = feats.get("pca", {}) or {}
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self._feature_dim: int | None = None

    # -------------------- 单模块 --------------------
    def extract_hog(self, images: np.ndarray) -> np.ndarray:
        cfg = self.hog_cfg
        return hog_features(
            images,
            orientations=int(cfg.get("orientations", 9)),
            pixels_per_cell=cfg.get("pixels_per_cell", (8, 8)),
            cells_per_block=cfg.get("cells_per_block", (2, 2)),
            block_norm=cfg.get("block_norm", "L2-Hys"),
        )

    def extract_color(self, images: np.ndarray) -> np.ndarray:
        cfg = self.color_cfg
        return color_hist_features(
            images,
            bins=int(cfg.get("bins", 16)),
            space=cfg.get("space", "HSV"),
        )

    # -------------------- 组合 --------------------
    def extract(self, images: np.ndarray) -> np.ndarray:
        """提取拼接后的原始特征（未 PCA）。"""
        parts = []
        if self.hog_cfg.get("enabled", True):
            parts.append(self.extract_hog(images))
        if self.color_cfg.get("enabled", True):
            parts.append(self.extract_color(images))
        if not parts:
            raise RuntimeError("未启用任何特征模块，请检查 config.features")
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit_transform(
        self,
        images: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        """在训练集上拟合标准化 + PCA 并转换（必须只在训练集上 fit，避免数据泄漏）。"""
        raw = self.extract(images)
        self._feature_dim = raw.shape[1]
        if verbose:
            print(f"[features] 原始特征维度 = {raw.shape[1]}")

        scaled = self.scaler.fit_transform(raw)

        if self.pca_cfg.get("enabled", False):
            self.pca = PCA(
                n_components=int(self.pca_cfg.get("n_components", 120)),
                whiten=bool(self.pca_cfg.get("whiten", False)),
            )
            out = self.pca.fit_transform(scaled)
            if verbose:
                ratio = float(self.pca.explained_variance_ratio_.sum())
                print(f"[features] PCA 降维 -> {out.shape[1]} 维，累计方差解释率 = {ratio:.2%}")
            return out
        return scaled

    def transform(self, images: np.ndarray, verbose: bool = False) -> np.ndarray:
        """使用已拟合的标准化/PCA 转换新数据（测试集）。"""
        raw = self.extract(images)
        scaled = self.scaler.transform(raw)
        if self.pca is not None:
            return self.pca.transform(scaled)
        return scaled

    @property
    def feature_dim(self) -> int | None:
        return self._feature_dim

    # -------------------- 持久化 --------------------
    def save(self, path: str | Path) -> None:
        """保存标准化器与 PCA 参数，便于推理时复用同一变换。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.scaler.mean_,
            "std": self.scaler.std_,
        }
        if self.pca is not None:
            payload["pca_mean"] = self.pca.mean_
            payload["pca_components"] = self.pca.components_
            payload["pca_evr"] = self.pca.explained_variance_
            payload["pca_whiten"] = np.array([float(self.pca.whiten)])
        np.savez_compressed(path, **payload)

    def load(self, path: str | Path) -> "FeatureExtractor":
        """恢复标准化器与 PCA 参数。"""
        data = np.load(path, allow_pickle=False)
        self.scaler.mean_ = data["mean"]
        self.scaler.std_ = data["std"]
        if "pca_components" in data:
            self.pca = PCA(n_components=int(data["pca_components"].shape[0]),
                           whiten=bool(data["pca_whiten"][0] > 0.5))
            self.pca.mean_ = data["pca_mean"]
            self.pca.components_ = data["pca_components"]
            self.pca.explained_variance_ = data["pca_evr"]
        return self
