"""Unit tests for src.models.evaluate (offline, no model files)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.models.evaluate import (
    compute_metrics,
    precision_at_recall,
    print_metrics,
    save_metrics_json,
)

# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_predictions() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_metrics(y, p)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["tp"] == 2 and m["tn"] == 2
    assert m["fp"] == 0 and m["fn"] == 0
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_compute_metrics_all_wrong() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.8, 0.1, 0.2])
    m = compute_metrics(y, p)
    assert m["tp"] == 0 and m["tn"] == 0
    assert m["fp"] == 2 and m["fn"] == 2
    assert m["precision"] == 0.0 and m["recall"] == 0.0
    # With 2 positives ranked below 2 negatives, the exact AP is 5/12.
    assert m["pr_auc"] == pytest.approx(5.0 / 12.0)


def test_compute_metrics_raises_on_shape_mismatch() -> None:
    y = np.array([0, 1, 0])
    p = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match=r"Shape mismatch"):
        compute_metrics(y, p)


def test_compute_metrics_raises_on_out_of_range_proba() -> None:
    y = np.array([0, 1])
    p = np.array([0.1, 1.5])
    with pytest.raises(ValueError, match=r"y_proba must be in"):
        compute_metrics(y, p)


def test_compute_metrics_threshold_effect() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.4, 0.45, 0.55, 0.6])
    # With threshold=0.5 -> 2 TP, 2 TN
    m1 = compute_metrics(y, p, threshold=0.5)
    assert m1["tp"] == 2 and m1["tn"] == 2
    # With threshold=0.8 -> 0 TP, 2 TN, 2 FN
    m2 = compute_metrics(y, p, threshold=0.8)
    assert m2["tp"] == 0 and m2["fn"] == 2


# ---------------------------------------------------------------------------
# precision_at_recall
# ---------------------------------------------------------------------------


def test_precision_at_recall_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8])
    # At threshold 0.8 (>=target_recall=0.8), precision is 1.0 and recall is 1.0
    assert precision_at_recall(y, p, target_recall=0.8) == pytest.approx(1.0)


def test_precision_at_recall_returns_zero_when_unreachable() -> None:
    # 1 positive, always predicted as 0.5 -- recall can reach 1.0.
    # But if all probabilities are pushed to a corner, we test the unreachable
    # branch with an all-negative-class y_true (only one class -> return 0.0).
    y = np.array([0, 0, 0, 0])
    p = np.array([0.1, 0.2, 0.9, 0.8])
    assert precision_at_recall(y, p, target_recall=0.5) == 0.0


def test_precision_at_recall_raises_on_bad_target() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8])
    with pytest.raises(ValueError, match=r"target_recall"):
        precision_at_recall(y, p, target_recall=1.5)


def test_precision_at_recall_monotonic_sanity() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200)
    p = rng.random(size=200)
    p_high_recall = precision_at_recall(y, p, target_recall=0.90)
    p_low_recall = precision_at_recall(y, p, target_recall=0.50)
    # Lower target recall -> we can use a higher threshold -> at least as precise
    assert p_low_recall >= p_high_recall


# ---------------------------------------------------------------------------
# print_metrics + save_metrics_json
# ---------------------------------------------------------------------------


def test_print_metrics_no_error(capsys: pytest.CaptureFixture) -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_metrics(y, p)
    print_metrics("test-split", metrics)
    out = capsys.readouterr().out
    assert "test-split" in out
    assert "PR-AUC" in out
    assert "ROC-AUC" in out


def test_save_metrics_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "metrics.json"
    payload = {"model": "xgb", "val": {"pr_auc": 0.42}}
    out = save_metrics_json(payload, path)
    assert out == path
    assert path.is_file()
    loaded = json.loads(path.read_text())
    assert loaded == payload
