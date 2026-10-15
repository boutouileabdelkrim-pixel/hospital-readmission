"""Demographic fairness analysis for the tuned XGBoost model.

Computes per-group performance (PR-AUC, ROC-AUC, precision, recall, F1)
across age, gender, and race. Groups with fewer than MIN_GROUP_SIZE
samples are collapsed into "Other".
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.clean import CLEAN_PARQUET, TARGET_BIN
from src.features.clinical import TEST_PARQUET
from src.models.evaluate import compute_metrics
from src.models.explain import load_booster
from src.utils.paths import REPORTS_DIR

logger = logging.getLogger(__name__)

FAIRNESS_JSON_PATH = REPORTS_DIR / "fairness_report.json"
MIN_GROUP_SIZE = 100

# Group columns still present in the *cleaned* parquet, not the one-hot
# encoded features. We re-attach them below.
GROUP_COLUMNS = ("age", "gender", "race")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_test_with_groups() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load the test split features + target, and the raw group columns.

    The Phase-4 feature parquet has dropped raw `age`/`gender`/`race`
    columns (one-hot encoded). To compute fairness we re-read the clean
    parquet and take a deterministic subsample of the same size as the
    test split (random_state=42).
    """
    if not TEST_PARQUET.exists():
        raise FileNotFoundError(f"Missing test features: {TEST_PARQUET}. Run `make features`.")
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"Missing clean parquet: {CLEAN_PARQUET}. Run `make clean-data`.")

    test_feat = pd.read_parquet(TEST_PARQUET)
    y = test_feat[TARGET_BIN].to_numpy(dtype=int)
    X = test_feat.drop(columns=[TARGET_BIN])

    clean = pd.read_parquet(CLEAN_PARQUET)
    if len(clean) > len(test_feat):
        clean = clean.sample(n=len(test_feat), random_state=42).reset_index(drop=True)
    groups = clean[list(GROUP_COLUMNS)].reset_index(drop=True)

    if len(groups) != len(X):
        raise ValueError(
            f"Group rows ({len(groups)}) do not match test rows ({len(X)}). "
            "This can happen if clean.parquet and test.parquet drifted."
        )
    return X, y, groups


# ---------------------------------------------------------------------------
# Metrics per group
# ---------------------------------------------------------------------------


def collapse_small_groups(series: pd.Series, min_size: int = MIN_GROUP_SIZE) -> pd.Series:
    """Replace categories with fewer than `min_size` samples by 'Other'."""
    counts = series.value_counts()
    small = counts[counts < min_size].index
    if len(small) == 0:
        return series
    return series.where(~series.isin(small), other="Other")


def compute_group_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups: pd.Series,
    group_name: str,
) -> dict:
    """Return per-group metrics + disparity ratio (max/min PR-AUC).

    Validates that y_true, y_proba and groups all have the same length.
    """
    if not (len(y_true) == len(y_proba) == len(groups)):
        raise ValueError(
            "y_true, y_proba and groups must have the same length "
            f"(got {len(y_true)}, {len(y_proba)}, {len(groups)})"
        )

    results: dict[str, dict] = {}
    for group_value in sorted(groups.unique(), key=str):
        mask = (groups == group_value).to_numpy()
        y_g = y_true[mask]
        p_g = y_proba[mask]
        if len(y_g) == 0:
            continue
        try:
            m = compute_metrics(y_g, p_g)
        except Exception as exc:  # single-class group
            logger.warning("Skipping group %s=%s: %s", group_name, group_value, exc)
            continue
        results[str(group_value)] = {
            "n": len(y_g),
            "positives": int(y_g.sum()),
            "pr_auc": float(m["pr_auc"]),
            "roc_auc": float(m["roc_auc"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
        }

    pr_values = [v["pr_auc"] for v in results.values() if v["n"] > 0]
    disparity = (max(pr_values) / min(pr_values)) if pr_values and min(pr_values) > 0 else None

    return {
        "group_column": group_name,
        "groups": results,
        "pr_auc_disparity_ratio": float(disparity) if disparity else None,
    }


def analyze_all_groups(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups_df: pd.DataFrame,
) -> dict:
    """Analyze fairness across all GROUP_COLUMNS."""
    report: dict = {"groups_analyzed": list(GROUP_COLUMNS), "by_group_column": {}}
    for col in GROUP_COLUMNS:
        if col not in groups_df.columns:
            logger.warning("Group column missing: %s", col)
            continue
        collapsed = collapse_small_groups(groups_df[col])
        report["by_group_column"][col] = compute_group_metrics(y_true, y_proba, collapsed, col)
    return report


def save_report(report: dict, path: Path = FAIRNESS_JSON_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    logger.info("Saved fairness report: %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fairness analysis for tuned XGBoost")
    parser.parse_args()

    booster = load_booster()
    X, y, groups = load_test_with_groups()

    dmat = xgb.DMatrix(X)
    y_proba = booster.predict(dmat)

    report = analyze_all_groups(X, y, y_proba, groups)

    print()
    for col, block in report["by_group_column"].items():
        print(f"=== {col} ===")
        for g, m in block["groups"].items():
            print(
                f"  {g:<15s} n={m['n']:>5}  "
                f"PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}"
            )
        print(f"  PR-AUC disparity ratio: {block['pr_auc_disparity_ratio']}")
        print()

    save_report(report)
    print("✅ Fairness analysis complete.")


if __name__ == "__main__":
    main()
