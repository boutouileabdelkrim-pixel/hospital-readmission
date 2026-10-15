"""Post-hoc probability calibration for the tuned XGBoost model.

Uses manual cross-fitted isotonic regression instead of
`CalibratedClassifierCV`, because sklearn 1.9.x's calibration module is
incompatible with custom estimators that do not inherit from
sklearn's internal `_CalibratedClassifier` base.

Pipeline:
1. Split val into K folds (stratified).
2. On each fold: fit IsotonicRegression on (raw_pred, y) of the other folds.
3. For the final calibrator: fit IsotonicRegression on ALL val data.
4. Apply final calibrator to test raw probabilities.
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
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src.data.clean import TARGET_BIN
from src.features.clinical import TEST_PARQUET, VAL_PARQUET
from src.models.explain import load_booster
from src.utils.paths import REPORTS_DIR

logger = logging.getLogger(__name__)

CALIB_PLOT_PATH = REPORTS_DIR / "calibration_curve.png"
CALIB_METRICS_PATH = REPORTS_DIR / "calibration_metrics.json"

DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Adapter: Booster -> predict_proba
# ---------------------------------------------------------------------------


class BoosterProbaWrapper:
    """Wrap an xgb.Booster to expose the sklearn `predict_proba` contract.

    This is a plain object (not a sklearn estimator) used only for
    internal calibration. Not meant to be passed to CalibratedClassifierCV.
    """

    def __init__(self, booster: xgb.Booster) -> None:
        self.booster = booster

    def predict_proba(self, X) -> np.ndarray:
        dmat = xgb.DMatrix(X)
        p1 = self.booster.predict(dmat)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Calibrated estimator (returned by fit_isotonic_calibrator)
# ---------------------------------------------------------------------------


class IsotonicCalibrator:
    """A fitted isotonic calibrator on top of a raw XGBoost Booster.

    Exposes the same `predict_proba` API as sklearn classifiers.
    """

    def __init__(self, booster: xgb.Booster, iso: IsotonicRegression) -> None:
        self.booster = booster
        self.iso = iso

    def predict_proba(self, X) -> np.ndarray:
        raw = BoosterProbaWrapper(self.booster).predict_proba(X)[:, 1]
        calibrated = self.iso.predict(raw)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_split(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    path = VAL_PARQUET if split == "val" else TEST_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}. Run `make features` first.")
    df = pd.read_parquet(path)
    if TARGET_BIN not in df.columns:
        raise KeyError(f"Missing target column '{TARGET_BIN}' in {path}")
    y = df[TARGET_BIN].to_numpy(dtype=int)
    X = df.drop(columns=[TARGET_BIN])
    return X, y


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def fit_isotonic_calibrator(
    booster: xgb.Booster,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
) -> IsotonicCalibrator:
    """Fit an isotonic calibrator via manual cross-fitting on val.

    Returns a fitted IsotonicCalibrator that can be used to transform
    raw booster probabilities into calibrated probabilities.
    """
    if len(y_val) < n_splits * 2:
        raise ValueError(f"Not enough val samples ({len(y_val)}) for {n_splits}-fold CV")
    raw = BoosterProbaWrapper(booster)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_predictions = np.zeros(len(y_val), dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_val, y_val)):
        X_tr = X_val.iloc[train_idx]
        X_te = X_val.iloc[test_idx]
        y_tr = y_val[train_idx]
        p_tr = raw.predict_proba(X_tr)[:, 1]
        p_te = raw.predict_proba(X_te)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr, y_tr)
        oof_predictions[test_idx] = iso.predict(p_te)
        logger.info("Fit fold %d/%d", fold_idx + 1, n_splits)

    # Final calibrator trained on all val data
    final_iso = IsotonicRegression(out_of_bounds="clip")
    final_iso.fit(raw.predict_proba(X_val)[:, 1], y_val)

    logger.info("Final isotonic calibrator fitted on %d val samples", len(y_val))
    return IsotonicCalibrator(booster, final_iso)


def compare_brier(
    y_true: np.ndarray,
    proba_uncalibrated: np.ndarray,
    proba_calibrated: np.ndarray,
) -> dict:
    """Return Brier scores for calibrated and uncalibrated probabilities."""
    b_un = brier_score_loss(y_true, proba_uncalibrated)
    b_cal = brier_score_loss(y_true, proba_calibrated)
    return {
        "brier_uncalibrated": float(b_un),
        "brier_calibrated": float(b_cal),
        "brier_delta": float(b_un - b_cal),
    }


def save_reliability_diagram(
    y_true: np.ndarray,
    proba_uncalibrated: np.ndarray,
    proba_calibrated: np.ndarray,
    path: Path = CALIB_PLOT_PATH,
    n_bins: int = 10,
) -> Path:
    """Save a reliability diagram comparing calibrated vs uncalibrated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    frac_pos_un, mean_pred_un = calibration_curve(
        y_true, proba_uncalibrated, n_bins=n_bins, strategy="quantile"
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_true, proba_calibrated, n_bins=n_bins, strategy="quantile"
    )

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred_un, frac_pos_un, "o-", label="XGBoost (uncalibrated)")
    ax.plot(mean_pred_cal, frac_pos_cal, "s-", label="XGBoost (isotonic)")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved calibration curve: %s", path)
    return path


def save_metrics(payload: dict, path: Path = CALIB_METRICS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Saved calibration metrics: %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Calibrate tuned XGBoost")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    args = parser.parse_args()

    booster = load_booster()
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"Val  : {X_val.shape}  positives={int(y_val.sum())}")
    print(f"Test : {X_test.shape} positives={int(y_test.sum())}")
    print()

    print(f"Fitting isotonic calibrator ({args.n_splits}-fold cross-fit on val)...")
    calibrator = fit_isotonic_calibrator(booster, X_val, y_val, n_splits=args.n_splits)

    raw = BoosterProbaWrapper(booster)
    proba_test_un = raw.predict_proba(X_test)[:, 1]
    proba_test_cal = calibrator.predict_proba(X_test)[:, 1]

    brier = compare_brier(y_test, proba_test_un, proba_test_cal)
    print()
    print(f"Brier (uncalibrated) : {brier['brier_uncalibrated']:.4f}")
    print(f"Brier (calibrated)   : {brier['brier_calibrated']:.4f}")
    print(f"Delta                : {brier['brier_delta']:+.4f}")
    print()

    save_reliability_diagram(y_test, proba_test_un, proba_test_cal, n_bins=args.bins)
    save_metrics(
        {
            "n_bins": args.bins,
            "n_splits": args.n_splits,
            "brier": brier,
            "n_test": len(y_test),
            "positives_test": int(y_test.sum()),
        }
    )
    print("✅ Calibration complete.")


if __name__ == "__main__":
    main()
