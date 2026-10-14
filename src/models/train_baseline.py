"""Train baseline XGBoost and LightGBM models on the Phase-4 features.

Both models use early stopping on the validation set with PR-AUC as the
selection metric. `scale_pos_weight` handles the ~11% positive rate.
No hyperparameter tuning here — that is Phase 6 (Optuna).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.clean import TARGET_BIN
from src.features.clinical import TEST_PARQUET, TRAIN_PARQUET, VAL_PARQUET
from src.models.evaluate import (
    compute_metrics,
    precision_at_recall,
    print_metrics,
    save_metrics_json,
)
from src.utils.paths import MODELS_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

XGB_MODEL_PATH = MODELS_DIR / "xgb_baseline.json"
LGBM_MODEL_PATH = MODELS_DIR / "lgbm_baseline.txt"
METADATA_PATH = MODELS_DIR / "metadata.json"
METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 1000

# Baseline hyperparameters (deliberately not tuned here).
XGB_PARAMS: dict = {
    "n_estimators": NUM_BOOST_ROUND,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LGBM_PARAMS: dict = {
    "n_estimators": NUM_BOOST_ROUND,
    "max_depth": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "metric": "average_precision",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test parquets. Errors if any is missing."""
    for p in (TRAIN_PARQUET, VAL_PARQUET, TEST_PARQUET):
        if not p.exists():
            raise FileNotFoundError(f"Missing split: {p}. Run `make features` first.")
    train = pd.read_parquet(TRAIN_PARQUET)
    val = pd.read_parquet(VAL_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)
    return train, val, test


def split_xy(df: pd.DataFrame, target: str = TARGET_BIN) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X, y) where X excludes the target column."""
    if target not in df.columns:
        raise KeyError(f"Missing target column: {target}")
    y = df[target].to_numpy(dtype=int)
    X = df.drop(columns=[target])
    return X, y


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Return (n_neg / n_pos), the standard XGBoost imbalance weight."""
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive samples in y")
    return n_neg / n_pos


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """Fit an XGBClassifier with early stopping on (X_val, y_val)."""
    params = dict(XGB_PARAMS if params is None else params)
    params.setdefault("scale_pos_weight", compute_scale_pos_weight(y_train))
    params.setdefault("random_state", RANDOM_STATE)
    params.setdefault("n_jobs", -1)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    logger.info(
        "XGBoost trained: best_iteration=%s, best_score=%.4f",
        model.best_iteration,
        model.best_score,
    )
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict | None = None,
) -> lgb.LGBMClassifier:
    """Fit an LGBMClassifier with early stopping on (X_val, y_val)."""
    params = dict(LGBM_PARAMS if params is None else params)
    params.setdefault("scale_pos_weight", compute_scale_pos_weight(y_train))
    params.setdefault("random_state", RANDOM_STATE)
    params.setdefault("n_jobs", -1)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    logger.info(
        "LightGBM trained: best_iteration=%s, best_score=%.4f",
        model.best_iteration_,
        model.best_score_["valid_0"]["average_precision"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation + persistence
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    target_recall: float = 0.80,
) -> dict:
    """Compute full metrics + precision@recall for a fitted model."""
    proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, proba)
    metrics[f"precision_at_recall_{int(target_recall * 100)}"] = precision_at_recall(
        y, proba, target_recall=target_recall
    )
    return metrics


def save_xgb(model: xgb.XGBClassifier, path: Path = XGB_MODEL_PATH) -> Path:
    """Save the underlying booster directly.

    We bypass `XGBClassifier.save_model()` because xgboost 2.1.x calls
    sklearn's `_estimator_type`, which was removed in sklearn >= 1.6.
    The booster's `save_model` produces the same JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(path))
    logger.info("Saved XGBoost model: %s", path)
    return path


def save_lgbm(model: lgb.LGBMClassifier, path: Path = LGBM_MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))
    logger.info("Saved LightGBM model: %s", path)
    return path


def save_metadata(
    xgb_model: xgb.XGBClassifier,
    lgbm_model: lgb.LGBMClassifier,
    xgb_metrics: dict,
    lgbm_metrics: dict,
    n_features: int,
    scale_pos_weight: float,
    path: Path = METADATA_PATH,
) -> Path:
    """Write a JSON metadata file describing what was trained."""
    payload = {
        "trained_at": datetime.now(UTC).isoformat(),
        "n_features": int(n_features),
        "scale_pos_weight": float(scale_pos_weight),
        "xgboost_version": xgb.__version__,
        "lightgbm_version": lgb.__version__,
        "models": {
            "xgb": {
                "path": str(XGB_MODEL_PATH.name),
                "params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
                "best_iteration": int(getattr(xgb_model, "best_iteration", -1)),
                "metrics": xgb_metrics,
            },
            "lgbm": {
                "path": str(LGBM_MODEL_PATH.name),
                "params": {k: v for k, v in LGBM_PARAMS.items() if k != "n_jobs"},
                "best_iteration": int(getattr(lgbm_model, "best_iteration_", -1)),
                "metrics": lgbm_metrics,
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Saved metadata: %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train, val, test = load_splits()
    logger.info("Loaded splits: train=%s val=%s test=%s", train.shape, val.shape, test.shape)

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)
    n_features = X_train.shape[1]
    spw = compute_scale_pos_weight(y_train)
    print(f"Features          : {n_features}")
    print(f"scale_pos_weight  : {spw:.4f}")
    print(f"Train positives   : {int(y_train.sum()):,} / {len(y_train):,}")
    print()

    # --- XGBoost ---
    print("Training XGBoost (early stopping on val PR-AUC)...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_val = evaluate_model(xgb_model, X_val, y_val)
    xgb_test = evaluate_model(xgb_model, X_test, y_test)
    print_metrics("XGBoost — validation", xgb_val)
    print_metrics("XGBoost — test", xgb_test)
    print()

    # --- LightGBM ---
    print("Training LightGBM (early stopping on val PR-AUC)...")
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
    lgbm_val = evaluate_model(lgbm_model, X_val, y_val)
    lgbm_test = evaluate_model(lgbm_model, X_test, y_test)
    print_metrics("LightGBM — validation", lgbm_val)
    print_metrics("LightGBM — test", lgbm_test)
    print()

    # --- Persist ---
    save_xgb(xgb_model)
    save_lgbm(lgbm_model)
    save_metadata(
        xgb_model=xgb_model,
        lgbm_model=lgbm_model,
        xgb_metrics={"val": xgb_val, "test": xgb_test},
        lgbm_metrics={"val": lgbm_val, "test": lgbm_test},
        n_features=n_features,
        scale_pos_weight=spw,
    )
    save_metrics_json(
        {
            "n_features": n_features,
            "scale_pos_weight": spw,
            "xgb": {"val": xgb_val, "test": xgb_test},
            "lgbm": {"val": lgbm_val, "test": lgbm_test},
        },
        METRICS_PATH,
    )

    print("✅ Baseline training complete.")


if __name__ == "__main__":
    main()
