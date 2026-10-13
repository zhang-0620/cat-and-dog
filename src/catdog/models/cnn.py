"""简易卷积神经网络 (LeNet 风格) —— 纯 NumPy 前向 + 反向传播实现。

网络结构：
    Input (96,96,3)
      -> Conv(16, 3x3) + ReLU -> MaxPool(2)  -> (48,48,16)
      -> Conv(32, 3x3) + ReLU -> MaxPool(2)  -> (24,24,32)
      -> Conv(64, 3x3) + ReLU -> MaxPool(2)  -> (12,12,64)
      -> Flatten -> Dense(64) + ReLU -> Dropout
      -> Dense(1) + Sigmoid  (二分类)

说明：为便于教学与无框架环境运行，采用 im2col 加速卷积。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


# ------------------------------------------------------------------
# 基础算子
# ------------------------------------------------------------------
def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """把 (N,C,H,W) 转为 (N*OH*OW, C*kh*kw) 的列矩阵。"""
    N, C, H, W = x.shape
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    Hp, Wp = x.shape[2], x.shape[3]
    OH = (Hp - kh) // stride + 1
    OW = (Wp - kw) // stride + 1
    cols = np.zeros((N, C, kh, kw, OH, OW), dtype=np.float32)
    for i in range(kh):
        i_max = i + stride * OH
        for j in range(kw):
            j_max = j + stride * OW
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
    return cols


def col2im(cols: np.ndarray, x_shape: Tuple[int, ...], kh: int, kw: int,
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """im2col 的逆操作，用于反传梯度。"""
    N, C, H, W = x_shape
    Hp, Wp = H + 2 * pad, W + 2 * pad
    OH = (Hp - kh) // stride + 1
    OW = (Wp - kw) // stride + 1
    cols = cols.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    x = np.zeros((N, C, Hp, Wp), dtype=np.float32)
    for i in range(kh):
        i_max = i + stride * OH
        for j in range(kw):
            j_max = j + stride * OW
            x[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]
    if pad > 0:
        x = x[:, :, pad:-pad, pad:-pad]
    return x


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ------------------------------------------------------------------
# 层定义
# ------------------------------------------------------------------
class Conv2D:
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, lr: float = 1e-3, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # He 初始化
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = (rng.standard_normal((out_ch, in_ch, k, k)) * scale).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.k, self.lr = k, lr
        self.x_shape: Tuple[int, ...] | None = None
        self.cols: np.ndarray | None = None
        self.relu_mask: np.ndarray | None = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        N, C, H, W = x.shape
        self.cols = im2col(x, self.k, self.k, 1, 0)          # (N*OH*OW, C*k*k)
        Wf = self.W.reshape(self.W.shape[0], -1)              # (OC, C*k*k)
        out = self.cols @ Wf.T + self.b                       # (N*OH*OW, OC)
        OH, OW = H - self.k + 1, W - self.k + 1
        out = out.reshape(N, OH, OW, self.W.shape[0]).transpose(0, 3, 1, 2)
        # 缓存 ReLU 掩码：只有正激活的梯度回传（替代单独上一层 ReLU 的导数）
        self.relu_mask = (out > 0).astype(np.float32)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, OC, OH, OW = dout.shape
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, OC)   # (N*OH*OW, OC)
        Wf = self.W.reshape(OC, -1)
        self.dW = (dout_r.T @ self.cols).reshape(self.W.shape)
        self.db = dout_r.sum(axis=0)
        dcols = dout_r @ Wf
        dx = col2im(dcols, self.x_shape, self.k, self.k, 1, 0)
        return dx

    def step(self, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.W -= lr * self.dW
        self.b -= lr * self.db


class MaxPool2D:
    def __init__(self, size: int = 2, stride: int = 2) -> None:
        self.size, self.stride = size, stride
        self.x_shape: Tuple[int, ...] | None = None
        self.argmax: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """使用 im2col + argmax 实现最大池化，天然支持非整除尺寸。"""
        self.x_shape = x.shape
        N, C, H, W = x.shape
        s, k = self.stride, self.size
        OH, OW = (H - k) // s + 1, (W - k) // s + 1
        # (N*OH*OW, C*k*k)
        cols = im2col(x, k, k, s, 0)
        cols = cols.reshape(N, OH, OW, C, k * k)
        idx = np.argmax(cols, axis=-1)                       # (N, OH, OW, C)
        self.argmax = idx
        out = np.take_along_axis(cols, idx[..., None], axis=-1)[..., 0]
        return out.transpose(0, 3, 1, 2).astype(np.float32)  # (N, C, OH, OW)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, OH, OW = dout.shape
        s, k = self.stride, self.size
        dx_cols = np.zeros((N, OH, OW, C, k * k), dtype=np.float32)
        np.put_along_axis(dx_cols, self.argmax[..., None], dout.transpose(0, 2, 3, 1)[..., None], axis=-1)
        dx_cols = dx_cols.reshape(N * OH * OW, C * k * k)
        return col2im(dx_cols, self.x_shape, k, k, s, 0)


class Dense:
    def __init__(self, in_dim: int, out_dim: int, lr: float = 1e-3,
                 activation: str = "relu", seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim) if activation == "relu" else np.sqrt(1.0 / in_dim)
        self.W = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.activation = activation
        self.lr = lr
        self.x: np.ndarray | None = None
        self.z: np.ndarray | None = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = x @ self.W + self.b
        if self.activation == "relu":
            return relu(self.z)
        if self.activation == "sigmoid":
            return sigmoid(self.z)
        return self.z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            dout = dout * (self.z > 0)
        elif self.activation == "sigmoid":
            s = sigmoid(self.z)
            dout = dout * s * (1 - s)
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def step(self, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Dropout:
    def __init__(self, p: float = 0.3, seed: int = 0) -> None:
        self.p = p
        self.training = True
        self.mask: np.ndarray | None = None
        self.rng = np.random.default_rng(seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            return x
        self.mask = (self.rng.random(x.shape) > self.p).astype(np.float32) / (1 - self.p + 1e-8)
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if not self.training or self.mask is None:
            return dout
        return dout * self.mask


class BatchNorm2D:
    """按通道做批归一化（NCHW），加速收敛并缓解内部协变量偏移。"""

    def __init__(self, channels: int, momentum: float = 0.9, eps: float = 1e-5) -> None:
        self.gamma = np.ones(channels, dtype=np.float32)
        self.beta = np.zeros(channels, dtype=np.float32)
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros(channels, dtype=np.float32)
        self.running_var = np.ones(channels, dtype=np.float32)
        self.training = True
        self.xhat: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        if self.training:
            mean = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean, var = self.running_mean, self.running_var

        self.var = var
        self.xhat = (x - mean.reshape(1, C, 1, 1)) / np.sqrt(var.reshape(1, C, 1, 1) + self.eps)
        g = self.gamma.reshape(1, C, 1, 1)
        b = self.beta.reshape(1, C, 1, 1)
        return (g * self.xhat + b).astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, H, W = dout.shape
        m = N * H * W
        g = self.gamma.reshape(1, C, 1, 1)
        self.dgamma = np.sum(dout * self.xhat, axis=(0, 2, 3))
        self.dbeta = np.sum(dout, axis=(0, 2, 3))
        dxhat = dout * g
        std_inv = 1.0 / np.sqrt(self.var.reshape(1, C, 1, 1) + self.eps)
        dvar = np.sum(dxhat * (self.xhat * -0.5 * std_inv), axis=(0, 2, 3))
        dmean = np.sum(dxhat * -std_inv, axis=(0, 2, 3)) + dvar * np.mean(-2.0 * self.xhat, axis=(0, 2, 3))
        return (dxhat * std_inv
                + dvar.reshape(1, C, 1, 1) * 2.0 * self.xhat / m
                + dmean.reshape(1, C, 1, 1) / m).astype(np.float32)

    def step(self, lr: float) -> None:
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


# ------------------------------------------------------------------
# 网络
# ------------------------------------------------------------------
class SimpleCNN:
    """面向猫狗二分类的紧凑 CNN（Conv-BN-ReLU-Pool 结构）。"""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("models", {}).get("cnn", {}) or config or {}
        self.cfg = cfg
        self.lr = float(cfg.get("learning_rate", 1e-3))
        self.epochs = int(cfg.get("epochs", 30))
        self.batch_size = int(cfg.get("batch_size", 32))
        filters: List[int] = list(cfg.get("filters", [16, 32, 64]))
        k = int(cfg.get("kernel_size", 3))
        dropout = float(cfg.get("dropout", 0.35))
        dense_units = int(cfg.get("dense_units", 64))
        in_shape = tuple(cfg.get("input_shape", (96, 96, 3)))
        self.input_shape = in_shape

        seed = 0
        self.convs: List[Conv2D] = []
        self.bns: List[BatchNorm2D] = []
        self.pools: List[MaxPool2D] = []
        ch_in = in_shape[2]
        for i, ch_out in enumerate(filters):
            self.convs.append(Conv2D(ch_in, ch_out, k, self.lr, seed=seed + i * 2 + 1))
            self.bns.append(BatchNorm2D(ch_out))
            self.pools.append(MaxPool2D(2, 2))
            ch_in = ch_out
        self.n_blocks = len(self.convs)

        # 推导扁平化维度（每个 block: conv 不变尺寸，pool 减半）
        h = in_shape[0]
        for _ in range(self.n_blocks):
            h = (h - k + 1) // 2
        self.flat_dim = h * h * filters[-1]

        self.fc1 = Dense(self.flat_dim, dense_units, self.lr, "relu", seed=seed + 4)
        self.drop = Dropout(dropout, seed=seed + 5)
        self.fc2 = Dense(dense_units, 1, self.lr, "sigmoid", seed=seed + 6)

        self.loss_history: List[float] = []
        self.acc_history: List[float] = []

    # -------------------- 前向 --------------------
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """x: (N, H, W, C) float32 in [0,1] -> (N,1) 概率。"""
        x = x.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC -> NCHW
        h = x
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            h = conv.forward(h)          # 卷积
            bn.training = training
            h = bn.forward(h)            # 批归一化
            h = relu(h)                  # 激活
            h = pool.forward(h)          # 池化
        flat = h.transpose(0, 2, 3, 1).reshape(x.shape[0], -1)
        h = self.fc1.forward(flat)
        self.drop.training = training
        h = self.drop.forward(h)
        return self.fc2.forward(h)

    # -------------------- 反向 --------------------
    def backward(self, dout: np.ndarray) -> None:
        d = self.fc2.backward(dout)
        d = self.drop.backward(d)
        d = self.fc1.backward(d)

        N = d.shape[0]
        c_out = self.convs[-1].W.shape[0]
        h = int(np.sqrt(d.shape[1] // c_out))
        d = d.reshape(N, h, h, c_out).transpose(0, 3, 1, 2)

        for conv, bn, pool in zip(reversed(self.convs), reversed(self.bns), reversed(self.pools)):
            d = pool.backward(d)
            d = d * (d > 0)          # ReLU 导数
            d = bn.backward(d)
            d = conv.backward(d)

    def _step(self, lr: float) -> None:
        for layer in list(self.convs) + list(self.bns) + [self.fc1, self.fc2]:
            layer.step(lr)

    # -------------------- 训练 --------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
        augment: bool = True,
    ) -> "SimpleCNN":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n = X.shape[0]
        rng = np.random.default_rng(42)
        lr0 = self.lr

        for ep in range(self.epochs):
            idx = rng.permutation(n)
            Xs, ys = X[idx], y[idx]
            # 学习率余弦衰减
            lr = lr0 * 0.5 * (1 + np.cos(np.pi * ep / max(self.epochs - 1, 1)))
            losses = []

            for start in range(0, n, self.batch_size):
                xb = Xs[start : start + self.batch_size]
                yb = ys[start : start + self.batch_size]
                if augment:
                    xb = _random_flip(xb, rng)

                p = self.forward(xb, training=True)
                # BCE 损失（仅作日志记录）
                eps = 1e-7
                loss = -np.mean(yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps))
                losses.append(float(loss))

                # BCE + Sigmoid 的梯度：dL/dz = (p - y) / N
                dout = (p - yb) / xb.shape[0]
                self.backward(dout)
                self._step(lr)

            train_loss = float(np.mean(losses))
            self.loss_history.append(train_loss)

            if X_val is not None:
                val_pred = (self.predict_proba(X_val) >= 0.5).astype(int)
                val_acc = float((val_pred == y_val.reshape(-1)).mean())
                self.acc_history.append(val_acc)
                if verbose and ((ep + 1) % 5 == 0 or ep == 0 or ep == self.epochs - 1):
                    print(f"  [CNN] epoch {ep + 1:>3}/{self.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")
            elif verbose and ((ep + 1) % 5 == 0 or ep == 0):
                print(f"  [CNN] epoch {ep + 1:>3}/{self.epochs}  loss={train_loss:.4f}")

        return self

    # -------------------- 推理 --------------------
    def predict_proba(self, X: np.ndarray, batch: int = 64) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        probs = []
        for i in range(0, len(X), batch):
            probs.append(self.forward(X[i : i + batch], training=False).ravel())
        return np.clip(np.concatenate(probs), 1e-6, 1 - 1e-6)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int64)


def _random_flip(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """随机水平翻转（数据增强）。"""
    out = x.copy()
    flip = rng.random(len(x)) < 0.5
    out[flip] = out[flip][:, :, ::-1, :]
    return out
