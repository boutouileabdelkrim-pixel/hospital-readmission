"""Unit tests for src.models.train_baseline (offline, tiny synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.data.clean import TARGET_BIN
from src.models.train_baseline import (
    LGBM_PARAMS,
    XGB_PARAMS,
    compute_scale_pos_weight,
    evaluate_model,
    split_xy,
    train_lightgbm,
    train_xgboost,
)


def _make_tiny(n: int = 300, n_features: int = 8, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic dataset with a learnable signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    # Signal: positive class favoured when first feature > 0
    logit = 2.0 * X[:, 0] + 0.5 * X[:, 1]
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < proba).astype(int)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df[TARGET_BIN] = y
    return df


# ---------------------------------------------------------------------------
# compute_scale_pos_weight
# ---------------------------------------------------------------------------


def test_compute_scale_pos_weight_basic() -> None:
    y = np.array([0, 0, 0, 1])
    assert compute_scale_pos_weight(y) == pytest.approx(3.0)


def test_compute_scale_pos_weight_raises_when_no_positive() -> None:
    y = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError, match=r"No positive"):
        compute_scale_pos_weight(y)


# ---------------------------------------------------------------------------
# split_xy
# ---------------------------------------------------------------------------


def test_split_xy_separates_target() -> None:
    df = _make_tiny(n=50)
    X, y = split_xy(df)
    assert TARGET_BIN not in X.columns
    assert X.shape == (50, 8)
    assert y.shape == (50,)


def test_split_xy_raises_on_missing_target() -> None:
    df = _make_tiny(n=20).drop(columns=[TARGET_BIN])
    with pytest.raises(KeyError, match=r"Missing target"):
        split_xy(df)


# ---------------------------------------------------------------------------
# Training smoke tests (very small data, very few rounds)
# ---------------------------------------------------------------------------


def _tiny_params(base: dict) -> dict:
    """Shrink a param dict for fast test runs."""
    p = dict(base)
    p["n_estimators"] = 30
    p["early_stopping_rounds"] = 5  # XGB only (LGB ignores unknown via callback)
    return p


def test_train_xgboost_runs_and_predicts() -> None:
    train = _make_tiny(n=300, seed=1)
    val = _make_tiny(n=100, seed=2)
    X_tr, y_tr = split_xy(train)
    X_va, y_va = split_xy(val)

    params = dict(XGB_PARAMS)
    params["n_estimators"] = 30
    params["early_stopping_rounds"] = 5

    model = train_xgboost(X_tr, y_tr, X_va, y_va, params=params)
    proba = model.predict_proba(X_va)[:, 1]
    assert proba.shape == (len(X_va),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_train_lightgbm_runs_and_predicts() -> None:
    train = _make_tiny(n=300, seed=1)
    val = _make_tiny(n=100, seed=2)
    X_tr, y_tr = split_xy(train)
    X_va, y_va = split_xy(val)

    params = dict(LGBM_PARAMS)
    params["n_estimators"] = 30

    model = train_lightgbm(X_tr, y_tr, X_va, y_va, params=params)
    proba = model.predict_proba(X_va)[:, 1]
    assert proba.shape == (len(X_va),)
    assert np.all((proba >= 0) & (proba <= 1))


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_full_metrics() -> None:
    train = _make_tiny(n=300, seed=1)
    val = _make_tiny(n=100, seed=2)
    X_tr, y_tr = split_xy(train)
    X_va, y_va = split_xy(val)

    params = dict(XGB_PARAMS)
    params["n_estimators"] = 30
    params["early_stopping_rounds"] = 5

    model = train_xgboost(X_tr, y_tr, X_va, y_va, params=params)
    metrics = evaluate_model(model, X_va, y_va, target_recall=0.80)

    for key in (
        "pr_auc",
        "roc_auc",
        "brier",
        "log_loss",
        "precision",
        "recall",
        "f1",
        "precision_at_recall_80",
    ):
        assert key in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["precision_at_recall_80"] <= 1.0
