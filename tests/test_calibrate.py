"""Unit tests for src.models.calibrate (offline, tiny synthetic data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from src.data.clean import TARGET_BIN
from src.models.calibrate import (
    BoosterProbaWrapper,
    IsotonicCalibrator,
    compare_brier,
    fit_isotonic_calibrator,
    save_metrics,
    save_reliability_diagram,
)


def _make_df(n: int = 300, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    logit = 1.2 * X[:, 0]
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df[TARGET_BIN] = y
    return df


def _train_booster(df: pd.DataFrame) -> xgb.Booster:
    X = df.drop(columns=[TARGET_BIN])
    y = df[TARGET_BIN].to_numpy()
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "max_depth": 3, "nthread": 1}
    return xgb.train(params, dtrain, num_boost_round=10)


def test_booster_wrapper_predict_proba_shape() -> None:
    df = _make_df(n=100)
    booster = _train_booster(df)
    wrapper = BoosterProbaWrapper(booster)
    X = df.drop(columns=[TARGET_BIN])
    p = wrapper.predict_proba(X)
    assert p.shape == (100, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((p >= 0) & (p <= 1))


def test_fit_isotonic_calibrator_returns_isotonic_calibrator() -> None:
    df = _make_df(n=300)
    booster = _train_booster(df)
    X = df.drop(columns=[TARGET_BIN])
    y = df[TARGET_BIN].to_numpy()

    calibrator = fit_isotonic_calibrator(booster, X, y, n_splits=5, seed=42)

    assert isinstance(calibrator, IsotonicCalibrator)
    p = calibrator.predict_proba(X)
    assert p.shape == (300, 2)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_fit_isotonic_calibrator_raises_on_too_few_samples() -> None:
    df = _make_df(n=8)  # too few for 5-fold CV
    booster = _train_booster(df)
    X = df.drop(columns=[TARGET_BIN])
    y = df[TARGET_BIN].to_numpy()
    with pytest.raises(ValueError, match=r"Not enough val samples"):
        fit_isotonic_calibrator(booster, X, y, n_splits=5)


def test_compare_brier_keys() -> None:
    y = np.array([0, 1, 0, 1])
    pu = np.array([0.1, 0.9, 0.2, 0.8])
    pc = np.array([0.2, 0.8, 0.3, 0.7])
    r = compare_brier(y, pu, pc)
    assert set(r.keys()) == {"brier_uncalibrated", "brier_calibrated", "brier_delta"}
    assert r["brier_delta"] == pytest.approx(r["brier_uncalibrated"] - r["brier_calibrated"])


def test_save_reliability_diagram(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    pu = rng.random(200)
    pc = (pu + 0.05).clip(0, 1)
    out = save_reliability_diagram(y, pu, pc, path=tmp_path / "curve.png")
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_save_metrics_json(tmp_path: Path) -> None:
    payload = {"brier": {"brier_calibrated": 0.12}}
    out = save_metrics(payload, path=tmp_path / "m.json")
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded == payload
