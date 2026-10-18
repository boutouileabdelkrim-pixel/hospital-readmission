"""Unit tests for src.models.cost_analysis (offline, synthetic data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.models.cost_analysis import (
    baseline_cost,
    confusion_at_threshold,
    cost_at_default_05,
    optimal_threshold,
    save_cost_curve,
    save_json,
    savings_vs_baseline,
    total_cost,
)

# --- confusion --------------------------------------------------------------


def test_confusion_perfect_classifier() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8])
    tn, fp, fn, tp = confusion_at_threshold(y, p, 0.5)
    assert (tn, fp, fn, tp) == (2, 0, 0, 2)


def test_confusion_at_default_threshold_can_miss_positives() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.4, 0.45, 0.55, 0.6])
    _tn, _fp, fn, tp = confusion_at_threshold(y, p, 0.7)
    assert tp == 0
    assert fn == 2


# --- costs ------------------------------------------------------------------


def test_baseline_cost_equals_positives_times_fn_cost() -> None:
    y = np.array([0, 1, 0, 1, 1])
    assert baseline_cost(y, fn_cost=1000.0) == pytest.approx(3000.0)


def test_total_cost_at_low_threshold_triggers_many_fps() -> None:
    y = np.array([0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.6])
    # t=0.5 -> tn=0, fp=3, fn=0, tp=1 -> cost = 0*FN + 3*FP + 1*FP = 4*FP
    cost = total_cost(y, p, 0.5, fn_cost=1000.0, fp_cost=100.0)
    assert cost == pytest.approx(400.0)


def test_savings_positive_at_optimal_threshold() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.15).astype(int)
    # A simple informative signal: positive class gets higher scores
    p = np.where(y == 1, rng.uniform(0.4, 0.9, size=500), rng.uniform(0.05, 0.5, size=500))
    result = optimal_threshold(y, p, fn_cost=15_000.0, fp_cost=500.0)
    assert result["savings_usd"] > 0
    assert 0.0 < result["optimal_threshold"] < 1.0


# --- analysis ---------------------------------------------------------------


def test_optimal_threshold_beats_default_05_on_synthetic() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(1000) < 0.11).astype(int)
    p = np.clip(0.11 + 0.4 * (y - 0.11) + rng.normal(0, 0.15, size=1000), 0, 1)
    opt = optimal_threshold(y, p, fn_cost=15_000.0, fp_cost=500.0)
    default = cost_at_default_05(y, p, fn_cost=15_000.0, fp_cost=500.0)
    assert opt["savings_usd"] >= default["savings_usd"]


def test_save_cost_curve_and_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    y = (rng.random(200) < 0.15).astype(int)
    p = rng.random(200)
    png = save_cost_curve(y, p, path=tmp_path / "curve.png")
    assert png.is_file() and png.stat().st_size > 1000

    payload = {"optimal_threshold": 0.12, "savings_usd": 12345.0}
    js = save_json(payload, path=tmp_path / "cost.json")
    assert js.is_file()
    assert json.loads(js.read_text()) == payload


def test_savings_vs_baseline_zero_at_high_threshold() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.05, 0.05, 0.05])
    # All below threshold -> no TP, no FP -> savings = 0 (all FNs)
    assert savings_vs_baseline(y, p, 0.99, fn_cost=1000.0, fp_cost=100.0) == pytest.approx(0.0)
