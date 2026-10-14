"""Unit tests for src.features.clinical (offline, synthetic fixtures)."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data.clean import TARGET_BIN
from src.features.clinical import (
    AGE_ORDER,
    MED_COLUMNS,
    MED_MAP,
    add_derived_features,
    encode_age,
    encode_medications,
    group_diag_columns,
    group_icd9,
    make_splits,
    one_hot_encode,
)


def _make_engineered(n: int = 100, pos_frac: float = 0.15) -> pd.DataFrame:
    """Synthetic cleaned DataFrame with all columns clinical.py expects.

    Guarantees exactly `n` rows total, with ~`pos_frac` positive examples.
    """
    n_pos = max(1, round(n * pos_frac))
    n_pos = min(n_pos, n - 1)  # keep at least 1 negative
    n_neg = n - n_pos
    target = [0] * n_neg + [1] * n_pos

    df = pd.DataFrame(
        {
            "age": [AGE_ORDER[i % len(AGE_ORDER)] for i in range(n)],
            "race": ["Caucasian"] * n,
            "gender": ["Female"] * n,
            "diag_1": ["250.83"] * n,
            "diag_2": ["428.0"] * n,
            "diag_3": ["Unknown"] * n,
            "number_outpatient": [0] * n,
            "number_emergency": [1] * n,
            "number_inpatient": [2] * n,
            TARGET_BIN: target,
        }
    )
    for col in MED_COLUMNS:
        df[col] = ["No"] * n
    return df


# --- group_icd9 ------------------------------------------------------------


def test_group_icd9_diabetes() -> None:
    assert group_icd9("250.83") == "Diabetes"
    assert group_icd9("250") == "Diabetes"


def test_group_icd9_circulatory() -> None:
    assert group_icd9("428.0") == "Circulatory"
    assert group_icd9("410") == "Circulatory"


def test_group_icd9_respiratory() -> None:
    assert group_icd9("486") == "Respiratory"


def test_group_icd9_v_and_e_codes() -> None:
    assert group_icd9("V57") == "Other"
    assert group_icd9("E849") == "Other"


def test_group_icd9_unknown() -> None:
    assert group_icd9("Unknown") == "Unknown"
    assert group_icd9(None) == "Unknown"
    assert group_icd9("") == "Unknown"


# --- encode_age ------------------------------------------------------------


def test_encode_age_maps_all_buckets() -> None:
    df = pd.DataFrame({"age": AGE_ORDER})
    out = encode_age(df)
    assert "age_ord" in out.columns
    assert "age" not in out.columns
    assert out["age_ord"].tolist() == list(range(len(AGE_ORDER)))


def test_encode_age_raises_on_unknown() -> None:
    df = pd.DataFrame({"age": ["[0-10)", "999+"]})
    with pytest.raises(ValueError, match=r"Unknown age"):
        encode_age(df)


# --- group_diag_columns ----------------------------------------------------


def test_group_diag_columns_replaces_diag_with_group() -> None:
    df = pd.DataFrame({"diag_1": ["250.83"], "diag_2": ["428.0"], "diag_3": ["Unknown"]})
    out = group_diag_columns(df)
    for col in ("diag_1", "diag_2", "diag_3"):
        assert col not in out.columns
        assert f"{col}_group" in out.columns
    assert out["diag_1_group"].iloc[0] == "Diabetes"
    assert out["diag_2_group"].iloc[0] == "Circulatory"
    assert out["diag_3_group"].iloc[0] == "Unknown"


# --- encode_medications ----------------------------------------------------


def test_encode_medications_maps_all_values() -> None:
    # 3 rows so we can set one value per row across all 21 med columns.
    df = pd.DataFrame({col: ["No"] * 3 for col in MED_COLUMNS})
    df.loc[0, "metformin"] = "Steady"
    df.loc[1, "metformin"] = "Down"
    df.loc[2, "metformin"] = "Up"

    out = encode_medications(df)

    for col in MED_COLUMNS:
        assert out[col].dtype == "int8"
    assert out["metformin"].iloc[0] == MED_MAP["Steady"]
    assert out["metformin"].iloc[1] == MED_MAP["Down"]
    assert out["metformin"].iloc[2] == MED_MAP["Up"]


def test_encode_medications_raises_on_unknown() -> None:
    df = pd.DataFrame({col: ["No"] * 2 for col in MED_COLUMNS})
    df.loc[0, "metformin"] = "Sometimes"
    with pytest.raises(ValueError, match=r"metformin"):
        encode_medications(df)


# --- add_derived_features --------------------------------------------------


def test_add_derived_service_utilization() -> None:
    df = _make_engineered(n=5)
    df = encode_medications(df)
    out = add_derived_features(df)
    # outpatient=0, emergency=1, inpatient=2 -> 3 for every row
    assert (out["service_utilization"] == 3).all()


def test_add_derived_n_med_changes() -> None:
    df = _make_engineered(n=4)
    df = encode_medications(df)
    # Row 0: no change
    # Row 1: 1 change
    # Row 2: 2 changes
    # Row 3: 3 changes
    df.loc[1, "metformin"] = MED_MAP["Down"]
    df.loc[2, "metformin"] = MED_MAP["Up"]
    df.loc[2, "insulin"] = MED_MAP["Down"]
    df.loc[3, "metformin"] = MED_MAP["Up"]
    df.loc[3, "insulin"] = MED_MAP["Down"]
    df.loc[3, "glipizide"] = MED_MAP["Up"]

    out = add_derived_features(df)

    assert out["n_med_changes"].tolist() == [0, 1, 2, 3]


# --- make_splits -----------------------------------------------------------


def test_make_splits_sizes_and_share() -> None:
    df = _make_engineered(n=1000)
    train, val, test = make_splits(df, TARGET_BIN)
    total = len(train) + len(val) + len(test)
    assert total == 1000
    assert 0.68 <= len(train) / total <= 0.72
    assert 0.13 <= len(val) / total <= 0.17
    assert 0.13 <= len(test) / total <= 0.17


def test_make_splits_stratification() -> None:
    df = _make_engineered(n=2000, pos_frac=0.15)
    train, val, test = make_splits(df, TARGET_BIN)
    for split in (train, val, test):
        rate = split[TARGET_BIN].mean()
        assert 0.10 <= rate <= 0.20, f"rate={rate}"


# --- one_hot_encode --------------------------------------------------------


def test_one_hot_encode_aligns_columns() -> None:
    df = _make_engineered(n=100)
    train, val, test = make_splits(df, TARGET_BIN)
    train_e, val_e, test_e = one_hot_encode(train, val, test)
    assert list(train_e.columns) == list(val_e.columns)
    assert list(train_e.columns) == list(test_e.columns)
    # No object columns left
    assert train_e.select_dtypes(include="object").empty
