"""Clean the raw Diabetes 130-US hospitals dataset.

Transforms `data/raw/diabetic_data.csv` (101,766 encounters, 50 cols, with
'?' as missing and numeric codes for some categoricals) into
`data/processed/clean.parquet` ready for feature engineering.

Key decisions (see reports/phase3_summary.md):
- Binarize target: `readmitted_30d = (readmitted == "<30")`.
- Drop `weight` (96.9% missing), `examide`/`citoglipton` (constant),
  `encounter_id`/`patient_nbr` (identifiers, not features).
- Impute categorical NaN -> "Unknown" (race, payer_code, medical_specialty,
  diag_1/2/3). Keep "None" for max_glu_serum / A1Cresult (means "test not done").
- Drop encounters where the patient died (discharge_disposition_id in
  {11, 13, 14, 19, 20, 21}) — such patients cannot be readmitted.
- Explicit dtypes -> avoids pandas DtypeWarning and non-determinism.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.paths import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

RAW_CSV = RAW_DIR / "diabetic_data.csv"
CLEAN_PARQUET = PROCESSED_DIR / "clean.parquet"

TARGET_RAW = "readmitted"
TARGET_BIN = "readmitted_30d"

# Numeric columns we want as int64 (IDs and small counts).
INT_COLUMNS: list[str] = [
    "encounter_id",
    "patient_nbr",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Columns to drop entirely.
DROP_COLUMNS: list[str] = [
    "weight",  # 96.9% missing
    "examide",  # constant "No"
    "citoglipton",  # constant "No"
    "encounter_id",  # identifier
    "patient_nbr",  # identifier (also leakage risk across rows)
]

# Categorical NaN -> "Unknown".
IMPUTE_UNKNOWN: list[str] = [
    "race",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
]

# discharge_disposition_id codes meaning the patient died (or hospice).
# Source: IDS_mapping.csv + published literature on this dataset.
DECEASED_DISCHARGE_IDS: set[int] = {11, 13, 14, 19, 20, 21}


def load_raw_typed(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV with explicit dtypes (fixes DtypeWarning).

    - `?` and empty strings -> NaN.
    - Numeric ID/count columns -> int64.
    - All other columns -> str.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "?"])

    for col in INT_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Expected column missing from raw CSV: {col}")
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")

    logger.info("Loaded raw CSV: %s rows x %s cols", *df.shape)
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add `readmitted_30d` (0/1) and drop the raw 3-class target."""
    if TARGET_RAW not in df.columns:
        raise KeyError(f"Missing target column: {TARGET_RAW}")
    df = df.copy()
    df[TARGET_BIN] = (df[TARGET_RAW] == "<30").astype("int8")
    df = df.drop(columns=[TARGET_RAW])
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier + useless columns. Errors if a listed column is absent."""
    missing = [c for c in DROP_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns to drop: {missing}")
    return df.drop(columns=DROP_COLUMNS)


def impute_categorical_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN in categorical columns with the literal string 'Unknown'."""
    df = df.copy()
    for col in IMPUTE_UNKNOWN:
        if col not in df.columns:
            raise KeyError(f"Missing categorical column: {col}")
        df[col] = df[col].fillna("Unknown")
    return df


def filter_deceased(df: pd.DataFrame) -> pd.DataFrame:
    """Remove encounters where the patient died (cannot be readmitted)."""
    if "discharge_disposition_id" not in df.columns:
        raise KeyError("Missing discharge_disposition_id column")
    mask = df["discharge_disposition_id"].isin(DECEASED_DISCHARGE_IDS)
    n_dropped = int(mask.sum())
    logger.info("Filtering deceased/hospice encounters: %s rows dropped", n_dropped)
    return df.loc[~mask].reset_index(drop=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline.

    Order matters:
    1. filter deceased
    2. drop unused columns
    3. impute categorical NaN
    4. binarize target
    """
    df = filter_deceased(df)
    df = drop_unused_columns(df)
    df = impute_categorical_unknown(df)
    df = binarize_target(df)
    return df


def summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the cleaned DataFrame."""
    print(f"Rows           : {len(df):,}")
    print(f"Columns        : {df.shape[1]}")
    print()
    print(f"Target `{TARGET_BIN}` distribution:")
    counts = df[TARGET_BIN].value_counts().sort_index()
    for label, n in counts.items():
        pct = 100 * n / len(df)
        print(f"  {label}  {n:>8,d}  ({pct:5.2f}%)")
    print()
    print("Remaining NaN per column (only cols with NaN):")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        print("  (none)")
    else:
        for col, n in miss.items():
            pct = 100 * n / len(df)
            print(f"  {col:25s}  {n:>8,d}  ({pct:5.2f}%)")


def save_parquet(df: pd.DataFrame, path: Path = CLEAN_PARQUET) -> Path:
    """Write the cleaned DataFrame as snappy-compressed parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MB)", path, size_mb)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw = load_raw_typed()
    clean = clean_dataframe(raw)

    print(f"Before cleaning: {raw.shape[0]:,} rows x {raw.shape[1]} cols")
    print(f"After cleaning : {clean.shape[0]:,} rows x {clean.shape[1]} cols")
    print()

    summary(clean)
    print()

    out = save_parquet(clean)
    print(f"✅ Saved: {out}")


if __name__ == "__main__":
    main()
