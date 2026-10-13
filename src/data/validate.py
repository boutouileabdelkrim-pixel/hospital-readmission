"""Validate the raw Diabetes 130-US hospitals dataset schema.

Ensures columns, row count, and target labels match the upstream format
before any downstream cleaning or feature engineering is applied.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.paths import RAW_DIR

logger = logging.getLogger(__name__)

DIABETIC_CSV = RAW_DIR / "diabetic_data.csv"

# 50 columns, in the original order delivered by UCI.
EXPECTED_COLUMNS: list[str] = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "readmitted",
]

TARGET_COLUMN = "readmitted"
VALID_TARGETS = {"NO", ">30", "<30"}

# Row count guard rails (real dataset has ~101,766 rows).
MIN_ROWS = 95_000
MAX_ROWS = 110_000


class SchemaError(ValueError):
    """Raised when the raw dataset does not match the expected schema."""


def load_raw(path: Path = DIABETIC_CSV) -> pd.DataFrame:
    """Load the raw CSV, treating '?' and empty strings as NaN."""
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")
    return pd.read_csv(path, na_values=["?", ""], keep_default_na=True)


def validate_schema(
    df: pd.DataFrame,
    *,
    min_rows: int = MIN_ROWS,
    max_rows: int = MAX_ROWS,
) -> None:
    """Validate that df matches the expected schema.

    Raises SchemaError on any mismatch.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing columns: {missing}")

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        raise SchemaError(f"Unexpected extra columns: {extra}")

    if not (min_rows <= len(df) <= max_rows):
        raise SchemaError(f"Row count {len(df)} outside expected range [{min_rows}, {max_rows}]")

    if df["encounter_id"].duplicated().any():
        dup = int(df["encounter_id"].duplicated().sum())
        raise SchemaError(f"Duplicate encounter_id values found ({dup} duplicates)")

    if df[TARGET_COLUMN].isna().any():
        raise SchemaError(f"Target column '{TARGET_COLUMN}' contains NaN")

    observed = set(df[TARGET_COLUMN].unique())
    unexpected = observed - VALID_TARGETS
    if unexpected:
        raise SchemaError(f"Unexpected target values: {sorted(unexpected)}")


def summarize(df: pd.DataFrame) -> None:
    """Print a compact summary for human eyes."""
    print(f"Rows           : {len(df):,}")
    print(f"Columns        : {df.shape[1]}")
    print(f"Unique patients: {df['patient_nbr'].nunique():,}")
    print()
    print("Target distribution:")
    counts = df[TARGET_COLUMN].value_counts(dropna=False)
    for label, n in counts.items():
        pct = 100 * n / len(df)
        print(f"  {label!s:>5s}  {n:>8,d}  ({pct:5.2f}%)")
    print()
    print("Top-5 columns with missing values:")
    miss = df.isna().sum().sort_values(ascending=False).head(5)
    for col, n in miss.items():
        pct = 100 * n / len(df)
        print(f"  {col:25s}  {n:>8,d}  ({pct:5.2f}%)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_raw()
    validate_schema(df)
    print(f"✅ Schema OK — {df.shape[0]:,} rows x {df.shape[1]} cols")
    print()
    summarize(df)


if __name__ == "__main__":
    main()
