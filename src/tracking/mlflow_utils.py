"""Thin wrapper around MLflow for tracking the hospital readmission models.

Keeps the MLflow surface small so tests run offline (they use a tmp SQLite
backend and never touch the network).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from src.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = PROJECT_ROOT / "mlruns.db"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"
EXPERIMENT_NAME = "hospital-readmission"


def configure_tracking(
    db_path: Path = DEFAULT_DB_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> str:
    """Configure MLflow to use a local SQLite backend + local artifact store.

    Returns the tracking URI (useful for tests and debugging).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    # Force local artifact store (avoid the default ./mlruns/ dir)
    os.environ["MLFLOW_ARTIFACT_ROOT"] = str(artifact_dir)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    return tracking_uri


def get_or_create_experiment(name: str = EXPERIMENT_NAME) -> str:
    """Return the experiment_id, creating the experiment if needed."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    experiment_id = mlflow.create_experiment(name)
    logger.info("Created MLflow experiment '%s' (id=%s)", name, experiment_id)
    return experiment_id


def log_metrics_from_dict(metrics: dict[str, float]) -> None:
    """Log a flat dict of metric name -> float, skipping non-numeric values."""
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            clean[k] = float(v)
    mlflow.log_metrics(clean)


def log_params_from_dict(params: dict[str, Any]) -> None:
    """Log a flat dict of param name -> value (values are str()-ified)."""
    clean: dict[str, str] = {k: str(v) for k, v in params.items() if v is not None}
    mlflow.log_params(clean)


@contextmanager
def start_run(
    run_name: str,
    experiment_name: str = EXPERIMENT_NAME,
    tags: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Context manager: ensures experiment exists, starts a run, yields it."""
    get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        yield run


def export_runs_to_json(out_path: Path) -> Path:
    """Export the list of runs in the default experiment to a JSON file."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run a training first.")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    payload = json.loads(runs.to_json(orient="records"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Exported %d runs to %s", len(payload), out_path)
    return out_path


__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "DEFAULT_DB_PATH",
    "EXPERIMENT_NAME",
    "configure_tracking",
    "export_runs_to_json",
    "get_or_create_experiment",
    "log_metrics_from_dict",
    "log_params_from_dict",
    "start_run",
]
