"""Metrics for imbalanced binary classification.

All functions are pure (no I/O) except the `save_metrics_json` writer.
Primary metric is PR-AUC (average precision) given the ~11% positive rate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full metric suite for a binary classifier.

    Args:
        y_true: ground truth labels (0/1).
        y_proba: predicted probabilities for the positive class.
        threshold: decision threshold for hard labels.

    Returns dict with keys: pr_auc, roc_auc, brier, log_loss, accuracy,
    precision, recall, f1, tp, fp, tn, fn.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape != y_proba.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape} y_proba={y_proba.shape}")
    if not np.all((y_proba >= 0) & (y_proba <= 1)):
        raise ValueError("y_proba must be in [0, 1]")

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-15, 1 - 1e-15))),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def precision_at_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.80,
) -> float:
    """Return the maximum precision achievable while keeping recall >= target.

    Clinical meaning: "at 80% recall, how precise can the model be?"

    sklearn's precision_recall_curve returns arrays ordered by increasing
    threshold, so `recall` is non-increasing. To get the best precision that
    still meets the recall constraint, we pick the **last** index where
    `recall >= target_recall` (highest threshold -> highest precision).

    Returns 0.0 if the target recall is unreachable.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape != y_proba.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape} y_proba={y_proba.shape}")
    if len(np.unique(y_true)) < 2:
        return 0.0

    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    mask = recalls >= target_recall
    if not mask.any():
        return 0.0
    # Last True -> highest threshold that still meets recall -> best precision.
    idx = int(np.flatnonzero(mask)[-1])
    return float(precisions[idx])


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    """Pretty-print a metrics dict (single line per metric group)."""
    print(f"--- {name} ---")
    print(f"  PR-AUC   : {metrics['pr_auc']:.4f}    ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  Brier    : {metrics['brier']:.4f}    LogLoss : {metrics['log_loss']:.4f}")
    print(
        f"  Precision: {metrics['precision']:.4f}    "
        f"Recall  : {metrics['recall']:.4f}    F1 : {metrics['f1']:.4f}"
    )
    print(f"  TP={metrics['tp']:<6} FP={metrics['fp']:<6} TN={metrics['tn']:<6} FN={metrics['fn']}")


def save_metrics_json(payload: dict[str, Any], path: Path) -> Path:
    """Write a metrics payload as JSON (creates parent dirs if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote %s", path)
    return path


__all__ = [
    "compute_metrics",
    "precision_at_recall",
    "print_metrics",
    "save_metrics_json",
]
