"""训练入口：特征提取 + SVM / 逻辑回归 训练，并保存模型与评估结果。

用法:
    python -m catdog.train --config configs/config.yaml
    python -m catdog.train --data-dir "E:/cat and dog/ceshi"   # 单目录时按比例切分
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 允许直接以脚本方式运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catdog.data.dataset import load_dataset, train_val_split, describe_dataset  # noqa: E402
from catdog.features.pipeline import FeatureExtractor  # noqa: E402
from catdog.models.svm import SVMClassifier, svm_config_to_kwargs  # noqa: E402
from catdog.models.logistic import LogisticRegression  # noqa: E402
from catdog.utils.io import ensure_dir, load_config, project_root, save_json  # noqa: E402
from catdog.utils.metrics import compute_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="猫狗识别 - 训练")
    p.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    p.add_argument("--data-dir", default=None, help="训练数据目录（覆盖配置）")
    p.add_argument("--model", default="svm", choices=["svm", "logreg"], help="要训练的模型")
    p.add_argument("--no-pca", action="store_true", help="禁用 PCA")
    p.add_argument("--quiet", action="store_true", help="减少日志输出")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)

    data_cfg = cfg["dataset"]
    train_dir = args.data_dir or data_cfg.get("train_dir") or data_cfg.get("test_dir") or "ceshi"
    train_path = Path(train_dir)
    if not train_path.is_absolute():
        train_path = root / train_path

    print("=" * 68)
    print(" 猫狗识别 · 训练阶段 / Cat-Dog Recognition · Training")
    print("=" * 68)
    print(f"[数据] 训练目录: {train_path}")

    t0 = time.time()
    ds = load_dataset(
        train_path,
        size=tuple(data_cfg["preprocess"]["resize"]),
        color_mode=data_cfg["preprocess"]["color_mode"],
        normalize=data_cfg["preprocess"]["normalize"],
        preprocess=True,
    )
    print(describe_dataset(ds))

    val_ratio = float(data_cfg.get("split", {}).get("val_ratio", 0.2))
    seed = int(data_cfg.get("split", {}).get("random_seed", 42))
    tr_idx, val_idx = train_val_split(ds.labels, val_ratio=val_ratio, seed=seed)
    print(f"[划分] 训练 {len(tr_idx)} / 验证 {len(val_idx)}")

    # 特征提取
    fcfg = dict(cfg)
    if args.no_pca:
        fcfg["features"] = dict(cfg["features"])
        fcfg["features"]["pca"] = {"enabled": False}
    extractor = FeatureExtractor(fcfg)
    Xtr = extractor.fit_transform(ds.images[tr_idx], verbose=not args.quiet)
    Xval = extractor.transform(ds.images[val_idx])
    ytr, yval = ds.labels[tr_idx], ds.labels[val_idx]
    print(f"[特征] 最终维度 = {Xtr.shape[1]}")

    # 训练
    if args.model == "svm":
        kwargs = svm_config_to_kwargs(cfg["models"]["svm"])
        print(f"[模型] SVM kernel={kwargs['kernel']} C={kwargs['C']} gamma={kwargs['gamma']}")
        clf = SVMClassifier(**kwargs)
    else:
        print("[模型] 逻辑回归基线")
        clf = LogisticRegression(epochs=400, lr=0.15)

    t1 = time.time()
    clf.fit(Xtr, ytr, verbose=not args.quiet)
    train_time = time.time() - t1
    print(f"[训练] 完成，耗时 {train_time:.1f}s")

    # 验证
    yval_pred = clf.predict(Xval)
    yval_score = clf.predict_proba(Xval)[:, 1]
    metrics = compute_metrics(yval, yval_pred, yval_score, positive_label=1)
    print(
        f"[验证] acc={metrics['accuracy']:.4f}  P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics.get('roc_auc', float('nan')):.4f}"
    )

    # 保存
    model_dir = ensure_dir(root / cfg["output"]["model_dir"])
    out_dir = ensure_dir(root / cfg["output"]["output_dir"])
    print(f"[保存] 耗时合计 {time.time() - t0:.1f}s")

    metrics["train_time_sec"] = round(train_time, 2)
    metrics["feature_dim"] = int(Xtr.shape[1])
    metrics["train_size"] = int(len(tr_idx))
    metrics["val_size"] = int(len(val_idx))
    save_json(metrics, out_dir / f"{args.model}_val_metrics.json")
    print(f"[保存] 验证指标 -> {out_dir / f'{args.model}_val_metrics.json'}")
    print(f"[提示] 模型目录: {model_dir}")


if __name__ == "__main__":
    main()
