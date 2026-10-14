"""Unit tests for src.models.tune_optuna (offline, synthetic data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from src.data.clean import TARGET_BIN
from src.models.tune_optuna import (
    _build_xgb_params,
    merge_metadata,
    objective,
    retrain_best,
    run_study,
    save_history_plot,
    save_tuned_xgb,
    suggest_params,
)

# Silence optuna logs in tests
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tiny(n: int = 400, n_features: int = 6, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic dataset with a learnable signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    logit = 1.5 * X[:, 0] + 0.5 * X[:, 1] - 1.0
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < proba).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df[TARGET_BIN] = y
    return df


@pytest.fixture(scope="module")
def tiny_splits() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df_tr = _make_tiny(n=400, seed=1)
    df_va = _make_tiny(n=150, seed=2)
    X_tr = df_tr.drop(columns=[TARGET_BIN])
    y_tr = df_tr[TARGET_BIN].to_numpy(dtype=int)
    X_va = df_va.drop(columns=[TARGET_BIN])
    y_va = df_va[TARGET_BIN].to_numpy(dtype=int)
    return X_tr, X_va, y_tr, y_va


# ---------------------------------------------------------------------------
# suggest_params
# ---------------------------------------------------------------------------


def test_suggest_params_keys() -> None:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = suggest_params(trial)
    expected = {
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "gamma",
    }
    assert set(params.keys()) == expected


def test_suggest_params_bounds() -> None:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    p = suggest_params(trial)
    assert 3 <= p["max_depth"] <= 9
    assert 0.01 <= p["learning_rate"] <= 0.2
    assert 0.6 <= p["subsample"] <= 1.0
    assert 0.6 <= p["colsample_bytree"] <= 1.0
    assert 1 <= p["min_child_weight"] <= 10
    assert 0.0 <= p["gamma"] <= 5.0


# ---------------------------------------------------------------------------
# _build_xgb_params
# ---------------------------------------------------------------------------


def test_build_xgb_params_merges_fixed_params() -> None:
    sampled = {"max_depth": 5, "learning_rate": 0.05}
    params = _build_xgb_params(sampled, scale_pos_weight=7.78)
    assert params["max_depth"] == 5
    assert params["scale_pos_weight"] == 7.78
    assert params["tree_method"] == "hist"
    assert params["eval_metric"] == "aucpr"
    assert params["random_state"] == 42


# ---------------------------------------------------------------------------
# objective (single trial)
# ---------------------------------------------------------------------------


def test_objective_returns_float_in_range(
    tiny_splits: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    X_tr, X_va, y_tr, y_va = tiny_splits
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    value = objective(
        trial,
        X_tr,
        y_tr,
        X_va,
        y_va,
        scale_pos_weight=3.0,
        n_estimators=20,
        early_stopping_rounds=5,
    )
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# run_study + retrain_best + save_tuned_xgb
# ---------------------------------------------------------------------------


def test_run_study_completes_two_trials(
    tiny_splits: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    X_tr, X_va, y_tr, y_va = tiny_splits
    study = run_study(X_tr, y_tr, X_va, y_va, scale_pos_weight=3.0, n_trials=2, seed=0)
    assert len(study.trials) == 2
    assert study.best_value >= 0.0


def test_retrain_best_and_save(
    tiny_splits: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    X_tr, X_va, y_tr, y_va = tiny_splits
    # Build a tiny study manually (single "best" trial)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(
            t, X_tr, y_tr, X_va, y_va, 3.0, n_estimators=15, early_stopping_rounds=3
        ),
        n_trials=1,
    )
    model = retrain_best(study, X_tr, y_tr, X_va, y_va, scale_pos_weight=3.0)
    proba = model.predict_proba(X_va)[:, 1]
    assert proba.shape == (len(X_va),)

    path = tmp_path / "xgb_tuned.json"
    save_tuned_xgb(model, path=path)
    assert path.is_file()
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# save_history_plot + merge_metadata
# ---------------------------------------------------------------------------


def test_save_history_plot_creates_png(
    tiny_splits: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    X_tr, X_va, y_tr, y_va = tiny_splits
    study = run_study(X_tr, y_tr, X_va, y_va, 3.0, n_trials=2, seed=0)
    out = save_history_plot(study, path=tmp_path / "hist.png")
    assert out.is_file()
    assert out.stat().st_size > 1000  # PNG non-trivial


def test_merge_metadata_creates_and_updates(
    tiny_splits: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    X_tr, X_va, y_tr, y_va = tiny_splits
    study = run_study(X_tr, y_tr, X_va, y_va, 3.0, n_trials=2, seed=0)
    meta_path = tmp_path / "metadata.json"

    # First call: file does not exist -> creates it
    merge_metadata(study, {"pr_auc": 0.25}, {"pr_auc": 0.26}, 6, 3.0, path=meta_path)
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text())
    assert "xgb_tuned" in meta["models"]
    assert "tuned_at" in meta

    # Second call: file exists -> updates in place
    merge_metadata(study, {"pr_auc": 0.30}, {"pr_auc": 0.31}, 6, 3.0, path=meta_path)
    meta2 = json.loads(meta_path.read_text())
    assert meta2["models"]["xgb_tuned"]["metrics"]["val"]["pr_auc"] == 0.30
