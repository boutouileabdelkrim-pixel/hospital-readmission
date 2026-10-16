"""Data drift monitoring with Evidently.

Compares the training feature distribution (reference) against the test
distribution (current) and writes an HTML report + a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.data.clean import TARGET_BIN
from src.features.clinical import TEST_PARQUET, TRAIN_PARQUET
from src.utils.paths import REPORTS_DIR

logger = logging.getLogger(__name__)

DRIFT_HTML_PATH = REPORTS_DIR / "drift_report.html"
DRIFT_JSON_PATH = REPORTS_DIR / "drift_summary.json"

DEFAULT_SAMPLE = 5000
DEFAULT_SEED = 42


def load_pair(
    sample_size: int = DEFAULT_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference (train) and current (test) feature DataFrames."""
    for p in (TRAIN_PARQUET, TEST_PARQUET):
        if not p.exists():
            raise FileNotFoundError(f"Missing split: {p}. Run `make features` first.")
    ref = pd.read_parquet(TRAIN_PARQUET).drop(columns=[TARGET_BIN])
    cur = pd.read_parquet(TEST_PARQUET).drop(columns=[TARGET_BIN])
    if sample_size and sample_size < len(ref):
        ref = ref.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    if sample_size and sample_size < len(cur):
        cur = cur.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    # Evidently wants numeric columns only
    ref = ref.select_dtypes(include="number")
    cur = cur.select_dtypes(include="number")
    return ref, cur


def build_report(ref: pd.DataFrame, cur: pd.DataFrame) -> Report:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    return report


def summarize(report: Report) -> dict:
    """Extract a small JSON-friendly summary from the Evidently report."""
    payload = report.as_dict()
    drift_metric = next(
        (m for m in payload.get("metrics", []) if m.get("metric") == "DataDriftTable"),
        None,
    )
    if drift_metric is None:
        return {"drifted": None, "share_drifted": None, "per_column": {}}

    result = drift_metric.get("result", {})
    n_drifted = int(result.get("number_of_drifted_columns", 0))
    n_total = int(result.get("number_of_columns", 0))
    share = (n_drifted / n_total) if n_total else 0.0
    per_col = {
        col: bool(v.get("drift_detected", False))
        for col, v in result.get("drift_by_columns", {}).items()
    }
    return {
        "drifted": n_drifted > 0,
        "n_drifted_columns": n_drifted,
        "n_total_columns": n_total,
        "share_drifted": share,
        "per_column": per_col,
    }


def save_html(report: Report, path: Path = DRIFT_HTML_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(path))
    logger.info("Saved drift HTML: %s", path)
    return path


def save_json(summary: dict, path: Path = DRIFT_JSON_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info("Saved drift JSON: %s", path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evidently drift report")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    ref, cur = load_pair(args.sample_size, args.seed)
    print(f"Reference: {ref.shape}  Current: {cur.shape}")

    report = build_report(ref, cur)
    summary = summarize(report)

    print()
    print(
        f"Drifted columns    : {summary.get('n_drifted_columns')}/{summary.get('n_total_columns')}"
    )
    print(f"Share drifted      : {summary.get('share_drifted'):.3f}")
    drifted_cols = [c for c, d in summary.get("per_column", {}).items() if d]
    if drifted_cols:
        print(
            f"Columns drifted    : {', '.join(drifted_cols[:10])}{'...' if len(drifted_cols) > 10 else ''}"
        )
    print()

    save_html(report)
    save_json(summary)
    print("✅ Drift report complete.")


if __name__ == "__main__":
    main()
