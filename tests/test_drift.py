"""Unit tests for src.monitoring.drift (offline, tiny synthetic data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.monitoring.drift import (
    build_report,
    load_pair,
    save_html,
    save_json,
    summarize,
)


def _make_df(n: int = 200, drift: bool = False, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shift = 3.0 if drift else 0.0
    return pd.DataFrame(
        {
            "f0": rng.normal(shift, 1.0, size=n),
            "f1": rng.normal(0.0, 1.0, size=n),
            "f2": rng.integers(0, 5, size=n),
        }
    )


def test_load_pair_raises_on_missing(tmp_path: Path, monkeypatch) -> None:
    import src.monitoring.drift as mod

    monkeypatch.setattr(mod, "TRAIN_PARQUET", tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError, match=r"Missing split"):
        load_pair(sample_size=100)


def test_build_report_runs_on_synthetic() -> None:
    ref = _make_df(n=200, drift=False, seed=1)
    cur = _make_df(n=200, drift=False, seed=2)
    report = build_report(ref, cur)
    assert report is not None


def test_summarize_returns_expected_keys() -> None:
    ref = _make_df(n=200, drift=False, seed=1)
    cur = _make_df(n=200, drift=False, seed=2)
    summary = summarize(build_report(ref, cur))
    assert "n_drifted_columns" in summary
    assert "share_drifted" in summary
    assert isinstance(summary["per_column"], dict)


def test_detects_drift_when_shifted() -> None:
    ref = _make_df(n=500, drift=False, seed=1)
    cur = _make_df(n=500, drift=True, seed=2)
    summary = summarize(build_report(ref, cur))
    assert summary["n_drifted_columns"] >= 1
    assert summary["per_column"]["f0"] is True


def test_save_html_and_json(tmp_path: Path) -> None:
    ref = _make_df(n=100, drift=False, seed=1)
    cur = _make_df(n=100, drift=False, seed=2)
    report = build_report(ref, cur)
    summary = summarize(report)

    html = save_html(report, path=tmp_path / "r.html")
    js = save_json(summary, path=tmp_path / "r.json")
    assert html.is_file() and html.stat().st_size > 1000
    assert js.is_file()
    data = json.loads(js.read_text())
    assert "n_drifted_columns" in data
