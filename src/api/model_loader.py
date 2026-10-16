"""Model + feature pipeline loader for the FastAPI layer.

Reuses the exact same feature engineering as Phase 4 (clinical.py) and
the trained tuned XGBoost from Phase 6.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.features.clinical import (
    TEST_PARQUET,
    add_derived_features,
    encode_age,
    encode_medications,
    group_diag_columns,
)
from src.models.tune_optuna import TUNED_MODEL_PATH

logger = logging.getLogger(__name__)

RISK_BANDS = {
    "low": (0.0, 0.10),
    "medium": (0.10, 0.30),
    "high": (0.30, 1.0),
}
DEFAULT_THRESHOLD = 0.15


class ModelBundle:
    """Holds the booster plus the fixed feature column list."""

    def __init__(self, booster: xgb.Booster, feature_names: list[str]) -> None:
        self.booster = booster
        self.feature_names = feature_names

    def predict_proba(self, X: pd.DataFrame) -> float:
        """Return the probability of positive class for a single-row DataFrame."""
        # Align to training columns: add missing (0), drop extras, keep order.
        X = X.reindex(columns=self.feature_names, fill_value=0)
        dmat = xgb.DMatrix(X)
        return float(self.booster.predict(dmat)[0])


@lru_cache(maxsize=1)
def load_model_bundle(model_path: Path = TUNED_MODEL_PATH) -> ModelBundle:
    """Load the tuned XGBoost and determine its feature column layout."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run `make tune-xgb` first, or copy "
            f"the artifact from the Kaggle output."
        )
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    if not TEST_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing feature parquet: {TEST_PARQUET}. Run `make features` first."
        )
    test_df = pd.read_parquet(TEST_PARQUET)
    feature_names = [c for c in test_df.columns if c != "readmitted_30d"]

    logger.info("Loaded model with %d features", len(feature_names))
    return ModelBundle(booster, feature_names)


def build_features(payload: dict) -> pd.DataFrame:
    """Transform a raw API payload into the model's feature matrix (1 row)."""
    df = pd.DataFrame([payload])
    # The payload uses the same columns as the raw clean parquet
    df = encode_age(df)
    df = group_diag_columns(df)
    df = encode_medications(df)
    df = add_derived_features(df)
    df = pd.get_dummies(df, dtype="int8")
    return df


def classify_risk_band(probability: float) -> str:
    """Return 'low', 'medium' or 'high' based on the probability."""
    for band, (lo, hi) in RISK_BANDS.items():
        if lo <= probability < hi:
            return band
    return "high"


__all__ = [
    "DEFAULT_THRESHOLD",
    "RISK_BANDS",
    "ModelBundle",
    "build_features",
    "classify_risk_band",
    "load_model_bundle",
]
