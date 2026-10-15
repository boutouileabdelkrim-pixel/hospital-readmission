"""Unit tests for src.models.fairness (offline, synthetic groups)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.models.fairness import (
    analyze_all_groups,
    collapse_small_groups,
    compute_group_metrics,
    save_report,
)


def test_collapse_small_groups_merges_small_into_other() -> None:
    s = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 3)
    out = collapse_small_groups(s, min_size=6)
    assert set(out.unique()) == {"A", "Other"}
    assert (out == "Other").sum() == 8  # B + C


def test_collapse_small_groups_no_change() -> None:
    s = pd.Series(["A"] * 10 + ["B"] * 10)
    out = collapse_small_groups(s, min_size=5)
    assert set(out.unique()) == {"A", "B"}


def test_compute_group_metrics_basic() -> None:
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, size=n)
    p = rng.random(n)
    groups = pd.Series(rng.choice(["A", "B"], size=n))
    r = compute_group_metrics(y, p, groups, "grp")
    assert r["group_column"] == "grp"
    assert "A" in r["groups"] and "B" in r["groups"]
    assert r["groups"]["A"]["n"] + r["groups"]["B"]["n"] == n
    assert r["pr_auc_disparity_ratio"] is not None


def test_compute_group_metrics_raises_on_length_mismatch() -> None:
    y = np.zeros(5)
    p = np.zeros(4)
    with pytest.raises(ValueError, match=r"same length"):
        compute_group_metrics(y, p, pd.Series(["A"] * 5), "grp")


def test_analyze_all_groups_produces_all_columns() -> None:
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, size=n)
    p = rng.random(n)
    X = pd.DataFrame({"f0": rng.normal(size=n)})
    groups_df = pd.DataFrame(
        {
            "age": rng.choice(["[40-50)", "[50-60)", "[60-70)"], size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "race": rng.choice(["Caucasian", "AfricanAmerican"], size=n),
        }
    )
    report = analyze_all_groups(X, y, p, groups_df)
    for col in ("age", "gender", "race"):
        assert col in report["by_group_column"]
        assert "groups" in report["by_group_column"][col]


def test_save_report_writes_json(tmp_path: Path) -> None:
    report = {"groups_analyzed": ["age"], "by_group_column": {}}
    out = save_report(report, path=tmp_path / "r.json")
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded == report
