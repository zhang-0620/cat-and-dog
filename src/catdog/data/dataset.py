"""数据集加载与预处理。

约定：图片文件名形如 ``cat.1.jpg`` / ``dog.51.jpg``，
      前缀即为类别标签，无需额外的标注文件。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

CLASS_NAMES: Tuple[str, str] = ("cat", "dog")
CLASS_TO_IDX: Dict[str, int] = {"cat": 0, "dog": 1}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


@dataclass
class Sample:
    """单条样本记录。"""

    path: Path
    label: str
    index: int

    @property
    def label_id(self) -> int:
        return CLASS_TO_IDX[self.label]


@dataclass
class CatDogDataset:
    """猫狗数据集容器。"""

    samples: List[Sample] = field(default_factory=list)
    images: np.ndarray | None = None       # (N, H, W, C) uint8
    labels: np.ndarray | None = None       # (N,) int
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def class_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for s in self.samples:
            dist[s.label] = dist.get(s.label, 0) + 1
        return dist


def discover_samples(root: str | Path) -> List[Sample]:
    """扫描目录，按文件名前缀识别类别，返回样本列表。

    支持两种目录布局：
      1. 扁平布局： root/cat.1.jpg, root/dog.1.jpg
      2. 子目录布局：root/cat/cat.1.jpg, root/dog/dog.1.jpg
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")

    samples: List[Sample] = []

    def _label_from_name(name: str) -> str | None:
        stem = Path(name).stem.lower()
        for cls in CLASS_NAMES:
            if stem.startswith(cls):
                return cls
        return None

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        dir_name = Path(dirpath).name.lower()
        dir_label = dir_name if dir_name in CLASS_TO_IDX else None

        for fn in sorted(filenames):
            if Path(fn).suffix.lower() not in IMG_EXTS:
                continue
            label = _label_from_name(fn) or dir_label
            if label is None:
                continue
            stem = Path(fn).stem
            num_part = "".join(ch for ch in stem if ch.isdigit())
            idx = int(num_part) if num_part else len(samples)
            samples.append(Sample(path=Path(dirpath) / fn, label=label, index=idx))

    samples.sort(key=lambda s: (s.label_id, s.index))
    return samples


def load_image(path: str | Path, color_mode: str = "RGB") -> np.ndarray:
    """读取图片并转换为指定色彩模式，返回 uint8 数组。"""
    img = Image.open(path)
    if color_mode.upper() == "GRAY":
        img = img.convert("L")
        arr = np.asarray(img, dtype=np.uint8)
        return arr[..., None]  # 统一为 (H, W, 1)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def preprocess_image(
    img: np.ndarray,
    size: Sequence[int] = (96, 96),
    normalize: str = "minmax",
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """尺寸归一化 + 数值归一化。

    Args:
        img: (H, W, C) uint8
        size: (W, H) 目标尺寸
        normalize: minmax -> [0,1]; standard -> 标准化; none -> 原值
        mean/std: standard 模式使用的均值方差（留空则按自身统计）

    Returns:
        float32 数组 (H, W, C)
    """
    pil = Image.fromarray(img[..., 0] if img.shape[-1] == 1 else img)
    pil = pil.resize((int(size[0]), int(size[1])), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]

    mode = str(normalize).lower()
    if mode == "minmax":
        arr = arr / 255.0
    elif mode == "standard":
        arr = arr / 255.0
        m = mean if mean is not None else arr.mean(axis=(0, 1), keepdims=True)
        s = std if std is not None else arr.std(axis=(0, 1), keepdims=True) + 1e-8
        arr = (arr - m) / s
    return arr


def load_dataset(
    root: str | Path,
    size: Sequence[int] = (96, 96),
    color_mode: str = "RGB",
    normalize: str = "minmax",
    preprocess: bool = True,
) -> CatDogDataset:
    """批量加载一个目录下的全部图片。

    Returns:
        CatDogDataset，images 形状 (N, H, W, C) float32（已归一化时）
    """
    samples = discover_samples(root)
    if not samples:
        raise RuntimeError(f"在 {root} 下未找到任何有效图片（文件名需以 cat/dog 开头）")

    raw_imgs: List[np.ndarray] = []
    kept: List[Sample] = []
    for s in samples:
        try:
            img = load_image(s.path, color_mode=color_mode)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 跳过无法读取的图片 {s.path.name}: {exc}")
            continue
        if preprocess:
            img = preprocess_image(img, size=size, normalize=normalize)
        raw_imgs.append(img)
        kept.append(s)

    images = np.stack(raw_imgs, axis=0)
    labels = np.array([s.label_id for s in kept], dtype=np.int64)

    ds = CatDogDataset(samples=kept, images=images, labels=labels)
    ds.meta = {
        "root": str(root),
        "num_samples": len(kept),
        "shape": list(images.shape),
        "class_distribution": ds.class_distribution,
        "color_mode": color_mode,
        "normalize": normalize,
        "size": list(size),
    }
    return ds


def train_val_split(
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """按类别分层切分训练/验证索引。"""
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def describe_dataset(ds: CatDogDataset) -> str:
    """生成数据集描述文本。"""
    dist = ds.class_distribution
    lines = [
        f"数据集路径 : {ds.meta.get('root', '-')}",
        f"样本总数   : {len(ds)}",
        f"类别分布   : cat={dist.get('cat', 0)}, dog={dist.get('dog', 0)}",
        f"张量形状   : {ds.meta.get('shape', '-')}",
        f"色彩模式   : {ds.meta.get('color_mode', '-')}",
        f"归一化方式 : {ds.meta.get('normalize', '-')}",
    ]
    return "\n".join(lines)
