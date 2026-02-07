"""Train ML probability model and save bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    import lightgbm as lgb
except Exception as e:  # pragma: no cover
    lgb = None

from .dataset import Dataset
from .feature_builder import FeatureBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--depth-levels", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if lgb is None:
        raise RuntimeError("lightgbm is required")

    builder = FeatureBuilder(depth_levels=args.depth_levels, vol_window=args.vol_window)
    dataset = Dataset(builder)
    X, y, entry_yes, entry_no, timestamps = dataset.load(args.data)
    train, val, test = dataset.split_time(X, y, entry_yes, entry_no, timestamps, args.train_frac, args.val_frac)

    class_weight = _class_weight(train.y)

    feature_names = builder.schema.names()

    models = []
    calibrators = []
    for seed in range(args.ensemble_size):
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            random_state=seed,
        )
        model.fit(
            train.X, train.y,
            sample_weight=_weights(train.y, class_weight),
            feature_name=feature_names,
        )
        val_proba = model.predict_proba(val.X)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_proba, val.y)
        models.append(model)
        calibrators.append(calibrator)

    test_proba = _predict_ensemble(models, calibrators, test.X, feature_names=feature_names)
    metrics = _evaluate(test.y, test_proba, test.entry_yes)

    bundle_dir = Path(args.bundle)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(models, bundle_dir / "model.bin")
    joblib.dump(calibrators, bundle_dir / "calibrator.bin")
    joblib.dump(None, bundle_dir / "encoder.bin")

    with open(bundle_dir / "features_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(builder.manifest(), fh, indent=2)

    with open(bundle_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    config = {
        "depth_levels": args.depth_levels,
        "vol_window": args.vol_window,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "ensemble_size": args.ensemble_size,
        "class_weight": class_weight,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "git_commit": _git_hash(),
    }
    with open(bundle_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


def _class_weight(y: np.ndarray) -> Dict[int, float]:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    total = pos + neg
    return {0: total / (2 * neg), 1: total / (2 * pos)}


def _weights(y: np.ndarray, class_weight: Dict[int, float]) -> np.ndarray:
    return np.asarray([class_weight[int(v)] for v in y], dtype=np.float32)


def _predict_ensemble(models, calibrators, X: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
    try:
        import pandas as _pd
        if feature_names:
            X = _pd.DataFrame(X, columns=feature_names)
    except ImportError:
        pass
    preds = []
    for model, calib in zip(models, calibrators):
        proba = model.predict_proba(X)[:, 1]
        preds.append(calib.transform(proba))
    return np.mean(np.vstack(preds), axis=0)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, entry_yes: np.ndarray) -> Dict[str, Any]:
    roc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_pred)
    baseline_brier = brier_score_loss(y_true, entry_yes)
    reliability = _reliability_curve(y_true, y_pred, bins=10)

    return {
        "roc_auc": roc,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "lift_brier": baseline_brier - brier,
        "reliability": reliability,
    }


def _reliability_curve(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> List[Dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    results = []
    for i in range(bins):
        mask = (y_pred >= edges[i]) & (y_pred < edges[i + 1])
        if not np.any(mask):
            continue
        avg_pred = float(np.mean(y_pred[mask]))
        avg_true = float(np.mean(y_true[mask]))
        results.append({"bin": i, "avg_pred": avg_pred, "avg_true": avg_true})
    return results


def _git_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2])
        return out.decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
