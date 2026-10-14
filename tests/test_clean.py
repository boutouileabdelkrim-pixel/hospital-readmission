"""Unit tests for src.data.clean (offline, synthetic fixtures)."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data.clean import (
    DECEASED_DISCHARGE_IDS,
    TARGET_BIN,
    binarize_target,
    clean_dataframe,
    drop_unused_columns,
    filter_deceased,
    impute_categorical_unknown,
)


def _make_raw(n: int = 20) -> pd.DataFrame:
    """Minimal synthetic raw DataFrame matching clean.py expectations."""
    df = pd.DataFrame(
        {
            "encounter_id": range(n),
            "patient_nbr": range(1000, 1000 + n),
            "race": ["Caucasian"] * n,
            "payer_code": [None] * n,
            "medical_specialty": ["Cardiology"] * n,
            "discharge_disposition_id": [1] * n,
            "diag_1": ["250.83"] * n,
            "diag_2": [None] * n,
            "diag_3": [None] * n,
            "weight": [None] * n,
            "examide": ["No"] * n,
            "citoglipton": ["No"] * n,
            "readmitted": ["NO"] * (n - 4) + ["<30"] * 2 + [">30"] * 2,
        }
    )
    return df


def test_filter_deceased_removes_listed_ids() -> None:
    df = _make_raw(n=10)
    df.loc[0, "discharge_disposition_id"] = 11
    df.loc[1, "discharge_disposition_id"] = 13
    out = filter_deceased(df)
    assert len(out) == 8
    remaining_ids = set(out["discharge_disposition_id"].unique())
    assert remaining_ids.isdisjoint(DECEASED_DISCHARGE_IDS)


def test_filter_deceased_keeps_others() -> None:
    df = _make_raw(n=10)
    out = filter_deceased(df)
    assert len(out) == 10


def test_drop_unused_columns_removes_expected() -> None:
    df = _make_raw(n=10)
    out = drop_unused_columns(df)
    for col in ("weight", "examide", "citoglipton", "encounter_id", "patient_nbr"):
        assert col not in out.columns


def test_drop_unused_columns_raises_on_missing() -> None:
    df = _make_raw(n=10).drop(columns=["weight"])
    with pytest.raises(KeyError, match=r"weight"):
        drop_unused_columns(df)


def test_impute_categorical_unknown() -> None:
    df = _make_raw(n=10)
    out = impute_categorical_unknown(df)
    assert out["payer_code"].isna().sum() == 0
    assert (out["payer_code"] == "Unknown").all()
    assert (out["diag_2"] == "Unknown").all()
    assert (out["diag_3"] == "Unknown").all()


def test_binarize_target_creates_binary() -> None:
    df = _make_raw(n=10)
    out = binarize_target(df)
    assert TARGET_BIN in out.columns
    assert "readmitted" not in out.columns
    assert set(out[TARGET_BIN].unique()).issubset({0, 1})
    assert int(out[TARGET_BIN].sum()) == 2


def test_clean_dataframe_end_to_end() -> None:
    df = _make_raw(n=20)
    df.loc[0, "discharge_disposition_id"] = 11
    out = clean_dataframe(df)

    assert len(out) == 19
    assert TARGET_BIN in out.columns
    for col in ("weight", "examide", "citoglipton", "encounter_id", "patient_nbr"):
        assert col not in out.columns
    assert out[TARGET_BIN].isna().sum() == 0


def test_clean_dataframe_resets_index() -> None:
    df = _make_raw(n=20)
    df.loc[0, "discharge_disposition_id"] = 11
    out = clean_dataframe(df)
    assert out.index.tolist() == list(range(len(out)))
