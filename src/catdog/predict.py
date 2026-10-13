"""单图预测脚本：用已训练的模型对任意图片做猫狗分类。

用法:
    python -m catdog.predict --image "ceshi/cat.1.jpg"
    python -m catdog.predict --image-dir ceshi --model svm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catdog.data.dataset import discover_samples, load_image, preprocess_image  # noqa: E402
from catdog.features.pipeline import FeatureExtractor  # noqa: E402
from catdog.models.svm import SVMClassifier  # noqa: E402
from catdog.models.logistic import LogisticRegression  # noqa: E402
from catdog.utils.io import load_config, project_root  # noqa: E402

CLASS_NAMES = ["cat", "dog"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="猫狗识别 - 单图/批量预测")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--image", default=None, help="单张图片路径")
    p.add_argument("--image-dir", default=None, help="批量预测目录")
    p.add_argument("--model", default="svm", choices=["svm", "logreg", "cnn"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    cfg = load_config(root / args.config)
    size = tuple(cfg["dataset"]["preprocess"]["resize"])

    # 收集待预测文件
    if args.image:
        files = [Path(args.image)]
    elif args.image_dir:
        d = Path(args.image_dir)
        if not d.is_absolute():
            d = root / d
        files = [s.path for s in discover_samples(d)]
    else:
        raise SystemExit("请提供 --image 或 --image-dir")

    print(f"[模型] {args.model.upper()}  待预测 {len(files)} 张")

    images = np.stack([
        preprocess_image(load_image(f, "RGB"), size=size, normalize="minmax")
        for f in files
    ])

    if args.model in ("svm", "logreg"):
        extractor = FeatureExtractor(cfg)
        svm_path = root / cfg["output"]["model_dir"] / "svm_model.npz"
        pipe_path = root / cfg["output"]["model_dir"] / "feature_pipeline.npz"

        if args.model == "svm" and svm_path.exists() and pipe_path.exists():
            # 推荐路径：复用训练时的特征变换 + 已保存的 SVM 权重
            extractor.load(str(pipe_path))
            model = SVMClassifier.load(str(svm_path))
            features = extractor.transform(images)
            probs = model.predict_proba(features)[:, 1]
            print(f"[加载] 模型权重 {svm_path.name} + 特征流水线 {pipe_path.name}")
        elif len(images) >= 2:
            # 无保存权重时，退化为在输入数据上拟合（仅演示）
            features = extractor.fit_transform(images, verbose=False)
            model = (SVMClassifier(**_svm_kwargs(cfg)) if args.model == "svm"
                     else LogisticRegression())
            model.fit(features, np.zeros(len(features), dtype=int))
            probs = model.predict_proba(features)[:, 1]
            print("[提示] 未找到已保存权重，已在输入数据上就地拟合（演示模式）")
        else:
            raise SystemExit(
                "仅 1 张图片且未找到已保存的模型权重。\n"
                "请先运行 'python -m catdog.evaluate --models svm' 生成 "
                "models/svm_model.npz 与 models/feature_pipeline.npz"
            )
    else:
        from catdog.models.cnn import SimpleCNN

        net = SimpleCNN(cfg)
        print("[提示] CNN 未加载预训练权重时会输出随机结果，请先用 evaluate 训练。")
        probs = net.predict_proba(images)

    print("-" * 56)
    for f, p in zip(files, probs):
        label = CLASS_NAMES[int(p >= 0.5)]
        conf = p if p >= 0.5 else 1 - p
        print(f"{f.name:<20} -> {label:<4} (置信度 {conf:.3f})")


def _svm_kwargs(cfg):
    s = cfg["models"]["svm"]
    return {"kernel": s.get("kernel", "rbf"), "C": s.get("C", 1.0),
            "gamma": s.get("gamma", "scale"), "max_iter": s.get("max_iter", 500)}


if __name__ == "__main__":
    main()
