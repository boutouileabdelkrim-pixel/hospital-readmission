"""Unit tests for src.tracking.mlflow_utils (offline, tmp SQLite backend)."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from src.tracking.mlflow_utils import (
    configure_tracking,
    export_runs_to_json,
    get_or_create_experiment,
    log_metrics_from_dict,
    log_params_from_dict,
    start_run,
)


@pytest.fixture
def tmp_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point MLflow at a tmp SQLite backend + tmp artifact dir."""
    db = tmp_path / "test_mlruns.db"
    artifacts = tmp_path / "test_artifacts"
    configure_tracking(db_path=db, artifact_dir=artifacts)
    # Use a unique experiment name per test
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.EXPERIMENT_NAME",
        f"test-{tmp_path.name}",
    )
    return tmp_path


def test_configure_tracking_creates_sqlite(tmp_tracking: Path) -> None:
    uri = mlflow.get_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert "test_mlruns.db" in uri


def test_get_or_create_experiment_is_idempotent(tmp_tracking: Path) -> None:
    name = f"exp-{tmp_tracking.name}"
    id1 = get_or_create_experiment(name)
    id2 = get_or_create_experiment(name)
    assert id1 == id2


def test_start_run_logs_params_and_metrics(tmp_tracking: Path) -> None:
    name = f"run-{tmp_tracking.name}"
    with start_run(run_name="my-run", experiment_name=name):
        log_params_from_dict({"max_depth": 5, "lr": 0.05, "none_val": None})
        log_metrics_from_dict({"pr_auc": 0.25, "roc_auc": 0.66, "bad": "skip-me"})

    exp = mlflow.get_experiment_by_name(name)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["params.max_depth"] == "5"
    assert row["params.lr"] == "0.05"
    assert float(row["metrics.pr_auc"]) == pytest.approx(0.25)


def test_export_runs_to_json(tmp_tracking: Path) -> None:
    name = f"export-{tmp_tracking.name}"
    # Monkey-patch EXPERIMENT_NAME on the module used by export
    import src.tracking.mlflow_utils as mu

    original = mu.EXPERIMENT_NAME
    mu.EXPERIMENT_NAME = name
    try:
        with start_run(run_name="r1", experiment_name=name):
            log_metrics_from_dict({"pr_auc": 0.3})
        out = tmp_tracking / "runs.json"
        export_runs_to_json(out)
        assert out.is_file()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
    finally:
        mu.EXPERIMENT_NAME = original
