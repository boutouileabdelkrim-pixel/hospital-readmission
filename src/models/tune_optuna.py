"""Optuna hyperparameter tuning for XGBoost on the Phase-4 features.

Runs a TPE study maximizing validation PR-AUC, then retrains the best
configuration. Saves:
- models/xgb_tuned.json (gitignored)
- models/metadata.json (merged, adds "xgb_tuned" entry)
- reports/optuna_tuning.json (committed)
- reports/optuna_history.png (committed — for portfolio)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.models.evaluate import (
    compute_metrics,
    precision_at_recall,
    print_metrics,
    save_metrics_json,
)
from src.models.train_baseline import (
    compute_scale_pos_weight,
    load_splits,
    split_xy,
)
from src.utils.paths import MODELS_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

TUNED_MODEL_PATH = MODELS_DIR / "xgb_tuned.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
TUNING_JSON_PATH = REPORTS_DIR / "optuna_tuning.json"
TUNING_PLOT_PATH = REPORTS_DIR / "optuna_history.png"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS_CEILING = 1500
N_TRIALS_DEFAULT = 20


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def suggest_params(trial: optuna.Trial) -> dict:
    """Sample an XGBoost hyperparameter configuration from an Optuna trial."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _build_xgb_params(
    sampled: dict,
    scale_pos_weight: float,
    n_estimators: int = N_ESTIMATORS_CEILING,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> dict:
    """Combine sampled params with fixed training params."""
    params = dict(sampled)
    params.update(
        {
            "n_estimators": n_estimators,
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "early_stopping_rounds": early_stopping_rounds,
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    )
    return params


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    scale_pos_weight: float,
    n_estimators: int = N_ESTIMATORS_CEILING,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> float:
    """Optuna objective: validation PR-AUC (maximize)."""
    sampled = suggest_params(trial)
    params = _build_xgb_params(
        sampled,
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    proba = model.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, proba))


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


def run_study(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    scale_pos_weight: float,
    n_trials: int,
    seed: int = RANDOM_STATE,
) -> optuna.Study:
    """Create and run an Optuna study maximizing val PR-AUC."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgb_readmission_tuning",
    )
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study


def retrain_best(
    study: optuna.Study,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    scale_pos_weight: float,
) -> xgb.XGBClassifier:
    """Retrain XGBoost with the best sampled params."""
    params = _build_xgb_params(study.best_params, scale_pos_weight=scale_pos_weight)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_tuned_xgb(model: xgb.XGBClassifier, path: Path = TUNED_MODEL_PATH) -> Path:
    """Save the booster directly (bypasses sklearn <-> xgboost 2.1 mismatch)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(path))
    logger.info("Saved tuned XGBoost: %s", path)
    return path


def save_history_plot(study: optuna.Study, path: Path = TUNING_PLOT_PATH) -> Path:
    """Save a PNG of trial PR-AUC values and best-so-far curve."""
    trials = [t for t in study.trials if t.value is not None]
    numbers = [t.number for t in trials]
    values = [float(t.value) for t in trials]
    best_so_far = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(numbers, values, alpha=0.5, label="Trial PR-AUC", color="#377eb8")
    ax.plot(numbers, best_so_far, color="#e41a1c", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title(f"Optuna — XGBoost tuning ({len(trials)} trials)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved history plot: %s", path)
    return path


def merge_metadata(
    study: optuna.Study,
    tuned_val: dict,
    tuned_test: dict,
    n_features: int,
    scale_pos_weight: float,
    path: Path = METADATA_PATH,
) -> Path:
    """Merge an "xgb_tuned" entry into models/metadata.json."""
    meta = json.loads(path.read_text()) if path.exists() else {"models": {}}
    meta.setdefault("models", {})
    meta["models"]["xgb_tuned"] = {
        "path": TUNED_MODEL_PATH.name,
        "best_params": study.best_params,
        "best_value_val_pr_auc": float(study.best_value),
        "n_trials": len(study.trials),
        "n_features": int(n_features),
        "scale_pos_weight": float(scale_pos_weight),
        "metrics": {"val": tuned_val, "test": tuned_test},
    }
    meta["tuned_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("Merged metadata: %s", path)
    return path


def _baseline_comparison() -> dict | None:
    """Load baseline PR-AUC values for comparison (None if missing)."""
    if not BASELINE_METRICS_PATH.exists():
        return None
    b = json.loads(BASELINE_METRICS_PATH.read_text())
    return {
        "xgb_baseline_val_pr_auc": float(b["xgb"]["val"]["pr_auc"]),
        "lgbm_baseline_val_pr_auc": float(b["lgbm"]["val"]["pr_auc"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Optuna tuning for XGBoost")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS_DEFAULT,
        help=f"Number of Optuna trials (default: {N_TRIALS_DEFAULT})",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    train, val, test = load_splits()
    logger.info("Loaded splits: train=%s val=%s test=%s", train.shape, val.shape, test.shape)

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)
    spw = compute_scale_pos_weight(y_train)

    print(f"n_trials         : {args.n_trials}")
    print(f"n_features       : {X_train.shape[1]}")
    print(f"scale_pos_weight : {spw:.4f}")
    print()

    study = run_study(X_train, y_train, X_val, y_val, spw, args.n_trials, seed=args.seed)

    print(f"Best trial        : {study.best_trial.number}")
    print(f"Best val PR-AUC   : {study.best_value:.4f}")
    print(f"Best params       : {study.best_params}")
    print()

    model = retrain_best(study, X_train, y_train, X_val, y_val, spw)
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    val_metrics = compute_metrics(y_val, val_proba)
    val_metrics["precision_at_recall_80"] = precision_at_recall(y_val, val_proba, 0.80)
    test_metrics = compute_metrics(y_test, test_proba)
    test_metrics["precision_at_recall_80"] = precision_at_recall(y_test, test_proba, 0.80)

    print_metrics("XGBoost tuned — validation", val_metrics)
    print_metrics("XGBoost tuned — test", test_metrics)
    print()

    save_tuned_xgb(model)
    save_history_plot(study)

    payload = {
        "n_trials": len(study.trials),
        "best_trial": int(study.best_trial.number),
        "best_value_val_pr_auc": float(study.best_value),
        "best_params": study.best_params,
        "n_features": int(X_train.shape[1]),
        "scale_pos_weight": float(spw),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_comparison": _baseline_comparison(),
        "tuned_at": datetime.now(UTC).isoformat(),
    }
    save_metrics_json(payload, TUNING_JSON_PATH)
    merge_metadata(study, val_metrics, test_metrics, X_train.shape[1], spw)

    print("✅ Optuna tuning complete.")


if __name__ == "__main__":
    main()
