"""测试集评估入口：在指定测试集文件夹上评估模型表现。

★ 默认测试集 = E:/cat and dog/ceshi

用法:
    python -m catdog.evaluate                      # 用配置中的 test_dir 评估
    python -m catdog.evaluate --test-dir ceshi
    python -m catdog.evaluate --models svm cnn logreg
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catdog.data.dataset import load_dataset, train_val_split, describe_dataset  # noqa: E402
from catdog.features.pipeline import FeatureExtractor  # noqa: E402
from catdog.models.svm import SVMClassifier, svm_config_to_kwargs  # noqa: E402
from catdog.models.logistic import LogisticRegression  # noqa: E402
from catdog.models.cnn import SimpleCNN  # noqa: E402
from catdog.utils.io import ensure_dir, load_config, project_root, save_json  # noqa: E402
from catdog.utils.metrics import compute_metrics, confusion_matrix, per_class_metrics  # noqa: E402
from catdog.utils.visualize import (  # noqa: E402
    plot_confusion_matrix,
    plot_roc_curve,
    plot_sample_predictions,
    plot_metric_bars,
    save_grayscale_grid,
)

CLASS_NAMES = ["cat", "dog"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="猫狗识别 - 测试集评估")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--test-dir", default=None, help="测试集目录（默认为配置中的 test_dir = ceshi）")
    p.add_argument("--train-dir", default=None, help="训练集目录（默认使用 test_dir 分层切分）")
    p.add_argument("--models", nargs="+", default=["svm", "cnn", "logreg"],
                   choices=["svm", "cnn", "logreg"])
    p.add_argument("--epochs", type=int, default=None, help="CNN 训练轮数上限（快速演示可设小值）")
    p.add_argument("--max-samples", type=int, default=None, help="限制测试集样本数（调试用）")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def evaluate_test_set(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    out_dir: Path,
    filenames: List[str],
    images: np.ndarray,
) -> Dict:
    """对一个模型的测试集结果做完整评估并落盘。"""
    cm = confusion_matrix(y_true, y_pred, n_classes=2)
    metrics = compute_metrics(y_true, y_pred, y_score, positive_label=1)
    per_cls = per_class_metrics(y_true, y_pred)

    plot_confusion_matrix(cm, CLASS_NAMES, out_dir / f"cm_{name}.png",
                          title=f"混淆矩阵 - {name.upper()} / Confusion Matrix")
    plot_confusion_matrix(cm, CLASS_NAMES, out_dir / f"cm_{name}_norm.png", normalize=True,
                          title=f"混淆矩阵(归一化) - {name.upper()}")
    plot_roc_curve(y_true, y_score, out_dir / f"roc_{name}.png", auc=metrics.get("roc_auc"))

    # 预测样例网格
    order = np.argsort(-((y_true != y_pred).astype(int)))
    sel = order[:18]
    plot_sample_predictions(
        [images[i] for i in sel],
        [filenames[i] for i in sel],
        y_true[sel], y_pred[sel], CLASS_NAMES,
        out_dir / f"samples_{name}.png",
        grid=(3, 6),
    )

    # 预测明细 CSV
    csv_path = out_dir / f"predictions_{name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "pred_label", "prob_cat", "prob_dog", "correct"])
        for fn, t, pr, sc in zip(filenames, y_true, y_pred, y_score):
            w.writerow([
                fn, CLASS_NAMES[int(t)], CLASS_NAMES[int(pr)],
                f"{1 - sc:.6f}", f"{sc:.6f}", int(t == pr),
            ])

    metrics["per_class"] = per_cls
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def main() -> None:
    args = parse_args()
    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)

    data_cfg = cfg["dataset"]
    test_dir = args.test_dir or data_cfg.get("test_dir") or "ceshi"
    test_path = Path(test_dir)
    if not test_path.is_absolute():
        test_path = root / test_path

    train_dir = args.train_dir or data_cfg.get("train_dir") or None
    train_path = None
    if train_dir:
        train_path = Path(train_dir)
        if not train_path.is_absolute():
            train_path = root / train_path

    out_dir = ensure_dir(root / cfg["output"]["output_dir"])
    model_dir = ensure_dir(root / cfg["output"]["model_dir"])

    print("=" * 70)
    print(" 猫狗识别 · 测试集评估 / Cat-Dog Recognition · Test Evaluation")
    print("=" * 70)
    print(f"[测试集] {test_path}")

    size = tuple(data_cfg["preprocess"]["resize"])
    color_mode = data_cfg["preprocess"]["color_mode"]
    norm = data_cfg["preprocess"]["normalize"]

    # ---------- 加载数据 ----------
    test_ds = load_dataset(test_path, size=size, color_mode=color_mode, normalize=norm)
    print(describe_dataset(test_ds))

    test_images = test_ds.images
    test_labels = test_ds.labels
    test_files = [s.path.name for s in test_ds.samples]

    if args.max_samples:
        test_images = test_images[: args.max_samples]
        test_labels = test_labels[: args.max_samples]
        test_files = test_files[: args.max_samples]

    # 训练集：优先独立目录，否则从测试集分层切分（演示模式）
    if train_path and train_path.exists():
        train_ds = load_dataset(train_path, size=size, color_mode=color_mode, normalize=norm)
        tr_idx = np.arange(len(train_ds))
        tr_images, tr_labels = train_ds.images, train_ds.labels
        print(f"[训练集] {train_path} ({len(tr_idx)} 张)")
    else:
        tr_idx, _ = train_val_split(test_labels, val_ratio=0.2,
                                    seed=int(data_cfg.get("split", {}).get("random_seed", 42)))
        tr_images, tr_labels = test_images[tr_idx], test_labels[tr_idx]
        print(f"[训练集] 由测试集分层切分而来 ({len(tr_idx)} 张) —— 演示模式")
        print("         ⚠ 严格实验中训练/测试应分离，此处为单文件夹演示")

    # ---------- 预处理可视化 ----------
    gray_imgs = load_dataset(test_path, size=size, color_mode="GRAY",
                             normalize="none").images[:32]
    save_grayscale_grid(gray_imgs, out_dir / "preprocess_grayscale.png")
    print(f"[可视化] 预处理灰度图 -> {out_dir / 'preprocess_grayscale.png'}")

    results: Dict[str, Dict] = {}

    # ---------- 传统特征模型 ----------
    need_features = any(m in args.models for m in ("svm", "logreg"))
    if need_features:
        t0 = time.time()
        extractor = FeatureExtractor(cfg)
        Xtr = extractor.fit_transform(tr_images, verbose=not args.quiet)
        Xte = extractor.transform(test_images)
        print(f"[特征] 维度 {Xtr.shape[1]}，耗时 {time.time() - t0:.1f}s")

        if "logreg" in args.models:
            print("\n--- 逻辑回归基线 ---")
            lr = LogisticRegression(epochs=400, lr=0.15)
            lr.fit(Xtr, tr_labels, verbose=not args.quiet)
            pred = lr.predict(Xte)
            score = lr.predict_proba(Xte)[:, 1]
            results["logreg"] = evaluate_test_set(
                "logreg", test_labels, pred, score, out_dir, test_files, test_images)

        if "svm" in args.models:
            print("\n--- SVM 分类器 ---")
            kwargs = svm_config_to_kwargs(cfg["models"]["svm"])
            print(f"    kernel={kwargs['kernel']} C={kwargs['C']} gamma={kwargs['gamma']}")
            t0 = time.time()
            svm = SVMClassifier(**kwargs)
            svm.fit(Xtr, tr_labels, verbose=not args.quiet)
            print(f"    训练耗时 {time.time() - t0:.1f}s，支持向量数 {len(svm.support_alphas_)}")
            svm.save(str(model_dir / "svm_model.npz"))
            extractor.save(str(model_dir / "feature_pipeline.npz"))
            pred = svm.predict(Xte)
            score = svm.predict_proba(Xte)[:, 1]
            results["svm"] = evaluate_test_set(
                "svm", test_labels, pred, score, out_dir, test_files, test_images)

    # ---------- CNN ----------
    if "cnn" in args.models:
        print("\n--- 卷积神经网络 (LeNet 风格) ---")
        cnn_cfg = dict(cfg)
        cnn_cfg["models"] = dict(cfg["models"])
        cnn_cfg["models"]["cnn"] = dict(cfg["models"]["cnn"])
        if args.epochs:
            cnn_cfg["models"]["cnn"]["epochs"] = args.epochs
        net = SimpleCNN(cnn_cfg)

        tr_i, val_i = train_val_split(tr_labels, val_ratio=0.15, seed=7)
        t0 = time.time()
        net.fit(
            tr_images[tr_i], tr_labels[tr_i],
            X_val=tr_images[val_i], y_val=tr_labels[val_i],
            verbose=not args.quiet,
            augment=bool(data_cfg["preprocess"].get("augment", True)),
        )
        print(f"    CNN 训练耗时 {time.time() - t0:.1f}s")

        pred = net.predict(test_images)
        score = net.predict_proba(test_images)
        results["cnn"] = evaluate_test_set(
            "cnn", test_labels, pred, score, out_dir, test_files, test_images)

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    print(" 测试集结果汇总 / Test Set Summary")
    print("=" * 70)
    header = f"{'Model':<10}{'Acc':>9}{'Prec':>9}{'Recall':>9}{'F1':>9}{'AUC':>9}"
    print(header)
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name.upper():<10}{m['accuracy']:>9.4f}{m['precision']:>9.4f}"
            f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m.get('roc_auc', float('nan')):>9.4f}"
        )

    plot_metric_bars(results, out_dir / "metrics_comparison.png")
    save_json(
        {
            "test_dir": str(test_path),
            "num_test_samples": int(len(test_labels)),
            "class_distribution": test_ds.class_distribution,
            "models": results,
        },
        out_dir / "test_results.json",
    )
    print(f"\n[输出] 全部结果已保存至: {out_dir}")


if __name__ == "__main__":
    main()
