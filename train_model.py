"""Train the FBCSP+SVM model from recorded MAT datasets."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from algorithms.fbcsp import FBCSPConfig, FilterBankCSPClassifier
from config import mi_config as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an FBCSP+SVM decoder from MAT files")
    parser.add_argument("mat_files", nargs="+", help="Paths to MAT dataset files")
    parser.add_argument(
        "--output-dir",
        default=str(cfg.MODEL_OUTPUT_DIR),
        help="Directory to store trained model artifacts",
    )
    parser.add_argument(
        "--tag",
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        help="Tag appended to the saved model folder",
    )
    return parser.parse_args()


def load_datasets(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    trials, labels = [], []
    for path in paths:
        mat = loadmat(path, squeeze_me=True)
        trials.append(np.asarray(mat["data"], dtype=np.float32))
        labels.append(np.asarray(mat["labels"], dtype=np.int32))
    X = np.concatenate(trials, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def main() -> None:
    args = parse_args()
    X, y = load_datasets(args.mat_files)
    print(f"Loaded {X.shape[0]} trials from {len(args.mat_files)} files")

    fb_config = FBCSPConfig(
        sample_rate=cfg.SAMPLE_RATE_HZ,
        filter_banks=cfg.FILTER_BANKS,
        components_per_band=cfg.CSP_COMPONENTS_PER_BAND,
        svm_kernel=cfg.SVM_KERNEL,
        svm_c=cfg.SVM_C,
        svm_class_weight=cfg.SVM_CLASS_WEIGHT,
    )
    model = FilterBankCSPClassifier(fb_config)

    class_counts = np.bincount(y)
    class_counts = class_counts[class_counts > 0]
    max_valid_folds = int(class_counts.min()) if class_counts.size else 0
    scores: List[float] = []
    if max_valid_folds >= 2:
        n_splits = min(cfg.CROSS_VALIDATION_FOLDS, max_valid_folds)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            acc = accuracy_score(y[test_idx], preds)
            scores.append(acc)
            print(f"Fold {fold}: {acc:.3f}")
    else:
        print("Skipping cross-validation (not enough samples per class).")

    model.fit(X, y)  # retrain on full data
    train_preds = model.predict(X)
    train_accuracy = accuracy_score(y, train_preds)
    print(f"Training accuracy on full dataset: {train_accuracy:.3f}")

    out_dir = Path(args.output_dir) / f"{cfg.MODEL_ARTIFACT_BASENAME}_{args.tag}"
    model.save(out_dir)
    metrics = {
        "fold_scores": scores,
        "mean_accuracy": float(np.mean(scores)) if scores else None,
        "train_accuracy": float(train_accuracy),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "class_map": cfg.EVENT_LABELS,
        "mat_files": args.mat_files,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()
