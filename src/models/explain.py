"""SHAP explainability for the tuned XGBoost model.

Produces global feature importance (mean |SHAP|) and a beeswarm summary
plot. Loads the model from `models/xgb_tuned.json` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.data.clean import TARGET_BIN
from src.features.clinical import TEST_PARQUET, TRAIN_PARQUET
from src.models.tune_optuna import TUNED_MODEL_PATH
from src.utils.paths import REPORTS_DIR

logger = logging.getLogger(__name__)

SHAP_SUMMARY_PATH = REPORTS_DIR / "shap_summary.png"
SHAP_BAR_PATH = REPORTS_DIR / "shap_bar.png"
SHAP_TOP_PATH = REPORTS_DIR / "shap_top_features.json"

DEFAULT_SAMPLE = 5000
DEFAULT_TOP_K = 20


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_booster(path: Path = TUNED_MODEL_PATH) -> xgb.Booster:
    """Load the raw XGBoost Booster from the JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned model: {path}. Run `make tune-xgb` first.")
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def load_sample(
    split: str = "test",
    sample_size: int | None = DEFAULT_SAMPLE,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a (possibly subsampled) split for SHAP analysis.

    Args:
        split: "test" or "train".
        sample_size: if smaller than the split, subsample uniformly.
        seed: random state for the subsample.
    """
    path = TEST_PARQUET if split == "test" else TRAIN_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}. Run `make features` first.")
    df = pd.read_parquet(path)
    if TARGET_BIN not in df.columns:
        raise KeyError(f"Missing target column '{TARGET_BIN}' in {path}")

    y = df[TARGET_BIN].to_numpy(dtype=int)
    X = df.drop(columns=[TARGET_BIN])

    if sample_size is not None and sample_size < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    return X, y


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def compute_shap_values(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for the positive class (probability space)."""
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        # Binary classifier sometimes returns [neg_class, pos_class]
        values = values[1] if len(values) == 2 else values[0]
    return np.asarray(values)


def top_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Return the top-k features sorted by mean absolute SHAP value."""
    if shap_values.ndim != 2:
        raise ValueError(f"shap_values must be 2D, got {shap_values.shape}")
    if len(feature_names) != shap_values.shape[1]:
        raise ValueError("feature_names length does not match shap_values shape")
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:k]
    return [{"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order]


def save_summary_plot(
    shap_values: np.ndarray, X: pd.DataFrame, path: Path = SHAP_SUMMARY_PATH
) -> Path:
    """Save the SHAP beeswarm summary plot as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=DEFAULT_TOP_K)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot: %s", path)
    return path


def save_bar_plot(shap_values: np.ndarray, X: pd.DataFrame, path: Path = SHAP_BAR_PATH) -> Path:
    """Save the SHAP bar plot (global importance) as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=DEFAULT_TOP_K)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP bar plot: %s", path)
    return path


def save_top_features(features: list[dict], path: Path = SHAP_TOP_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"top_features": features}, indent=2, sort_keys=True))
    logger.info("Saved top features: %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SHAP analysis for tuned XGBoost")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    booster = load_booster()
    X, y = load_sample(args.split, args.sample_size, args.seed)
    print(f"Loaded {args.split} split: X={X.shape}, positives={int(y.sum())}")

    print("Computing SHAP values...")
    shap_values = compute_shap_values(booster, X)
    print(f"SHAP values shape: {shap_values.shape}")

    save_summary_plot(shap_values, X)
    save_bar_plot(shap_values, X)

    feats = top_features(shap_values, list(X.columns), k=args.top_k)
    save_top_features(feats)
    print()
    print(f"Top-{args.top_k} features by mean |SHAP|:")
    for i, f in enumerate(feats, 1):
        print(f"  {i:>3}. {f['feature']:<40s}  {f['mean_abs_shap']:.5f}")

    print()
    print("✅ SHAP analysis complete.")


if __name__ == "__main__":
    main()
