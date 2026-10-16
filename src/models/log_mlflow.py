"""Log the 3 trained models (xgb_baseline, lgbm_baseline, xgb_tuned) to MLflow.

Reads:
- models/xgb_baseline.json, models/lgbm_baseline.txt, models/xgb_tuned.json
- models/metadata.json (params + metrics per model)
- reports/baseline_metrics.json, reports/optuna_tuning.json (metrics)

Writes:
- mlruns.db (SQLite backend, gitignored)
- mlartifacts/ (gitignored)
- reports/mlflow_runs.json (committed)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow

from src.tracking.mlflow_utils import (
    configure_tracking,
    export_runs_to_json,
    log_metrics_from_dict,
    log_params_from_dict,
    start_run,
)
from src.utils.paths import MODELS_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

XGB_BASELINE = MODELS_DIR / "xgb_baseline.json"
LGBM_BASELINE = MODELS_DIR / "lgbm_baseline.txt"
XGB_TUNED = MODELS_DIR / "xgb_tuned.json"
METADATA = MODELS_DIR / "metadata.json"
BASELINE_METRICS = REPORTS_DIR / "baseline_metrics.json"
OPTUNA_TUNING = REPORTS_DIR / "optuna_tuning.json"
RUNS_JSON = REPORTS_DIR / "mlflow_runs.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def log_baseline_model(name: str, model_path: Path, metrics: dict) -> None:
    """Log one of the two baseline models (XGB or LGBM) to MLflow."""
    with start_run(run_name=name, tags={"stage": "baseline", "model": name}):
        # Params come from metadata.json
        meta = _load_json(METADATA)
        params = meta["models"][name].get("params", {})
        log_params_from_dict(params)

        # Metrics: flatten the val + test sub-dicts
        val = metrics.get("val", {})
        test = metrics.get("test", {})
        flat = {f"val_{k}": v for k, v in val.items()}
        flat.update({f"test_{k}": v for k, v in test.items()})
        log_metrics_from_dict(flat)

        # Artifacts
        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="model")

        logger.info("Logged %s", name)


def log_tuned_model() -> None:
    """Log the Optuna-tuned XGBoost with its best params and metrics."""
    tuning = _load_json(OPTUNA_TUNING)
    with start_run(run_name="xgb_tuned", tags={"stage": "tuned", "model": "xgb_tuned"}):
        log_params_from_dict(tuning.get("best_params", {}))

        val = tuning.get("val_metrics", {})
        test = tuning.get("test_metrics", {})
        flat = {f"val_{k}": v for k, v in val.items()}
        flat.update({f"test_{k}": v for k, v in test.items()})
        flat["best_value_val_pr_auc"] = tuning.get("best_value_val_pr_auc", 0.0)
        flat["n_trials"] = tuning.get("n_trials", 0)
        log_metrics_from_dict(flat)

        if XGB_TUNED.exists():
            mlflow.log_artifact(str(XGB_TUNED), artifact_path="model")

        # Also log the Optuna history plot for the portfolio
        plot = REPORTS_DIR / "optuna_history.png"
        if plot.exists():
            mlflow.log_artifact(str(plot), artifact_path="plots")

        logger.info("Logged xgb_tuned")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    configure_tracking()

    baseline = _load_json(BASELINE_METRICS)

    log_baseline_model("xgb", XGB_BASELINE, baseline["xgb"])
    log_baseline_model("lgbm", LGBM_BASELINE, baseline["lgbm"])
    log_tuned_model()

    export_runs_to_json(RUNS_JSON)
    print(f"✅ MLflow logging complete. Runs exported to {RUNS_JSON}")


if __name__ == "__main__":
    main()
