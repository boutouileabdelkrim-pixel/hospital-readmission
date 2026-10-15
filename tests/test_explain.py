"""Unit tests for src.models.explain (offline, tiny synthetic XGBoost)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from src.data.clean import TARGET_BIN
from src.models.explain import (
    compute_shap_values,
    load_sample,
    save_bar_plot,
    save_summary_plot,
    save_top_features,
    top_features,
)


def _make_tiny_df(n: int = 100, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    logit = 1.5 * X[:, 0] - 0.5 * X[:, 1]
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df[TARGET_BIN] = y
    return df


def _train_tiny_booster(df: pd.DataFrame) -> xgb.Booster:
    X = df.drop(columns=[TARGET_BIN])
    y = df[TARGET_BIN].to_numpy()
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "max_depth": 3, "nthread": 1}
    return xgb.train(params, dtrain, num_boost_round=5)


def test_load_sample_missing_file_raises(tmp_path: Path, monkeypatch) -> None:
    import src.models.explain as mod

    monkeypatch.setattr(mod, "TEST_PARQUET", tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError, match=r"Missing split"):
        load_sample("test")


def test_compute_shap_values_shape() -> None:
    df = _make_tiny_df(n=60)
    booster = _train_tiny_booster(df)
    X = df.drop(columns=[TARGET_BIN])
    sv = compute_shap_values(booster, X)
    assert sv.shape == (60, X.shape[1])
    assert np.isfinite(sv).all()


def test_top_features_sorted_descending() -> None:
    shap_values = np.array([[1.0, 0.5, 0.1], [1.2, 0.4, 0.2]])
    names = ["a", "b", "c"]
    feats = top_features(shap_values, names, k=3)
    assert [f["feature"] for f in feats] == ["a", "b", "c"]
    assert feats[0]["mean_abs_shap"] > feats[1]["mean_abs_shap"]


def test_top_features_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"feature_names length"):
        top_features(np.zeros((10, 3)), ["a", "b"], k=2)


def test_save_top_features_json(tmp_path: Path) -> None:
    feats = [{"feature": "x", "mean_abs_shap": 0.42}]
    out = save_top_features(feats, path=tmp_path / "top.json")
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["top_features"][0]["feature"] == "x"


def test_save_summary_and_bar_plots(tmp_path: Path) -> None:
    df = _make_tiny_df(n=40)
    booster = _train_tiny_booster(df)
    X = df.drop(columns=[TARGET_BIN])
    sv = compute_shap_values(booster, X)

    summary = save_summary_plot(sv, X, path=tmp_path / "sum.png")
    bar = save_bar_plot(sv, X, path=tmp_path / "bar.png")
    assert summary.is_file() and summary.stat().st_size > 1000
    assert bar.is_file() and bar.stat().st_size > 1000
