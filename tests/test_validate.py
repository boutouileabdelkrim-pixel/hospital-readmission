"""Unit tests for src.data.validate (offline, synthetic fixtures)."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data.validate import (
    EXPECTED_COLUMNS,
    SchemaError,
    summarize,
    validate_schema,
)


def _make_df(n: int = 100) -> pd.DataFrame:
    """Build a synthetic DataFrame matching the expected schema."""
    data: dict[str, list] = {col: ["x"] * n for col in EXPECTED_COLUMNS}
    df = pd.DataFrame(data)
    df["encounter_id"] = list(range(n))
    df["patient_nbr"] = list(range(n))
    df["readmitted"] = ["NO"] * n
    return df


def test_validate_schema_ok() -> None:
    df = _make_df(n=100)
    validate_schema(df, min_rows=10, max_rows=1000)


def test_validate_schema_missing_columns() -> None:
    df = _make_df(n=100).drop(columns=["age"])
    with pytest.raises(SchemaError, match="Missing columns"):
        validate_schema(df, min_rows=10, max_rows=1000)


def test_validate_schema_extra_columns() -> None:
    df = _make_df(n=100)
    df["new_feature"] = 0
    with pytest.raises(SchemaError, match="Unexpected extra columns"):
        validate_schema(df, min_rows=10, max_rows=1000)


def test_validate_schema_bad_row_count() -> None:
    df = _make_df(n=5)
    with pytest.raises(SchemaError, match="Row count"):
        validate_schema(df, min_rows=10, max_rows=1000)


def test_validate_schema_duplicate_ids() -> None:
    df = _make_df(n=100)
    df.loc[1, "encounter_id"] = df.loc[0, "encounter_id"]
    with pytest.raises(SchemaError, match="Duplicate encounter_id"):
        validate_schema(df, min_rows=10, max_rows=1000)


def test_validate_schema_bad_target_values() -> None:
    df = _make_df(n=100)
    df.loc[0, "readmitted"] = "MAYBE"
    with pytest.raises(SchemaError, match="Unexpected target values"):
        validate_schema(df, min_rows=10, max_rows=1000)


def test_summarize_prints_without_error(capsys: pytest.CaptureFixture) -> None:
    df = _make_df(n=50)
    summarize(df)
    captured = capsys.readouterr().out
    assert "Rows" in captured
    assert "Target distribution" in captured
    assert "NO" in captured
