"""Clinical feature engineering for the hospital readmission project.

Transforms `data/processed/clean.parquet` into model-ready train/val/test
splits saved as parquet under `data/features/`.

Key transformations (all deterministic):
- Ordinal encode `age` (10-year buckets -> 0..9).
- Group ICD-9 codes in diag_1/2/3 into 17 clinical chapters + 'Other'.
- Ordinal encode 21 medication columns (No=0, Steady=1, Down=2, Up=3).
- Derived: service_utilization = outpatient + emergency + inpatient.
- Derived: n_med_changes = count of meds with Up or Down.
- One-hot encode categorical columns (fit on train only to avoid leakage).
- Stratified 70/15/15 split performed BEFORE one-hot encoding.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.clean import CLEAN_PARQUET, TARGET_BIN
from src.utils.paths import FEATURES_DIR

logger = logging.getLogger(__name__)

TRAIN_PARQUET = FEATURES_DIR / "train.parquet"
VAL_PARQUET = FEATURES_DIR / "val.parquet"
TEST_PARQUET = FEATURES_DIR / "test.parquet"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGE_ORDER: list[str] = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]

MED_COLUMNS: list[str] = [
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
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Natural dose ordering: No -> Steady -> Down -> Up
MED_MAP: dict[str, int] = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}

# ICD-9 chapter grouping. Order matters: Diabetes (250) sits inside the
# Endocrine range (240-279) and must be checked first.
ICD9_GROUPS: list[tuple[str, int, int]] = [
    ("Diabetes", 250, 250),
    ("Neoplasms", 140, 239),
    ("Endocrine", 240, 279),
    ("Blood", 280, 289),
    ("Mental", 290, 319),
    ("Nervous", 320, 389),
    ("Circulatory", 390, 459),
    ("Respiratory", 460, 519),
    ("Digestive", 520, 579),
    ("Genitourinary", 580, 629),
    ("Pregnancy", 630, 679),
    ("Skin", 680, 709),
    ("Musculoskeletal", 710, 739),
    ("Congenital", 740, 759),
    ("Perinatal", 760, 779),
    ("Symptoms", 780, 799),
    ("Injury", 800, 999),
]

# ---------------------------------------------------------------------------
# Feature transformations
# ---------------------------------------------------------------------------


def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    """Map age buckets to ordinal 0..9. Errors if an unknown value appears."""
    if "age" not in df.columns:
        raise KeyError("Missing 'age' column")
    df = df.copy()
    mapping = {bucket: i for i, bucket in enumerate(AGE_ORDER)}
    df["age_ord"] = df["age"].map(mapping)
    unknown = df["age_ord"].isna()
    if unknown.any():
        bad = df.loc[unknown, "age"].unique().tolist()
        raise ValueError(f"Unknown age values: {bad}")
    df["age_ord"] = df["age_ord"].astype("int8")
    df = df.drop(columns=["age"])
    return df


def group_icd9(code: object) -> str:
    """Map a raw ICD-9 code to its clinical chapter.

    Returns 'Unknown' for missing values (including the string 'Unknown'),
    'Other' for V/E codes and codes outside all chapters.
    """
    if code is None:
        return "Unknown"
    s = str(code).strip()
    if s == "" or s.lower() == "unknown" or s == "nan":
        return "Unknown"
    if s[0] in ("V", "E"):
        return "Other"
    try:
        numeric_part = s.split(".")[0]
        num = int(numeric_part)
    except (ValueError, IndexError):
        return "Other"
    for name, lo, hi in ICD9_GROUPS:
        if lo <= num <= hi:
            return name
    return "Other"


def group_diag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace diag_1/2/3 with grouped columns diag_X_group."""
    df = df.copy()
    for col in ("diag_1", "diag_2", "diag_3"):
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
        df[f"{col}_group"] = df[col].map(group_icd9)
        df = df.drop(columns=[col])
    return df


def encode_medications(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode the 21 medication columns via MED_MAP."""
    df = df.copy()
    for col in MED_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Missing medication column: {col}")
        encoded = df[col].map(MED_MAP)
        if encoded.isna().any():
            bad = df.loc[encoded.isna(), col].unique().tolist()
            raise ValueError(f"Unknown values in {col}: {bad}")
        df[col] = encoded.astype("int8")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add service_utilization and n_med_changes.

    Requires `encode_medications` to have run first (meds must be int8).
    """
    df = df.copy()

    required_util = ["number_outpatient", "number_emergency", "number_inpatient"]
    for col in required_util:
        if col not in df.columns:
            raise KeyError(f"Missing column for service_utilization: {col}")

    df["service_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    ).astype("int16")

    # n_med_changes: how many meds were adjusted (Down=2 or Up=3)
    med_matrix = df[MED_COLUMNS].to_numpy()
    df["n_med_changes"] = (med_matrix >= 2).sum(axis=1).astype("int8")

    return df


# ---------------------------------------------------------------------------
# Splitting + encoding
# ---------------------------------------------------------------------------


def make_splits(
    df: pd.DataFrame,
    target: str = TARGET_BIN,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split. Returns (train, val, test)."""
    if target not in df.columns:
        raise KeyError(f"Missing target column: {target}")

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target]
    )
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=seed,
        stratify=train_val[target],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def one_hot_encode(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-hot encode all object columns; fit columns on train only."""
    ohe_cols = train.select_dtypes(include="object").columns.tolist()

    train_enc = pd.get_dummies(train, columns=ohe_cols, prefix=ohe_cols, dtype="int8")
    val_enc = pd.get_dummies(val, columns=ohe_cols, prefix=ohe_cols, dtype="int8")
    test_enc = pd.get_dummies(test, columns=ohe_cols, prefix=ohe_cols, dtype="int8")

    # Align val/test to train's columns (missing categories -> 0)
    cols = train_enc.columns
    val_enc = val_enc.reindex(columns=cols, fill_value=0)
    test_enc = test_enc.reindex(columns=cols, fill_value=0)

    return train_enc, val_enc, test_enc


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Write the three splits as snappy-compressed parquet."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PARQUET, index=False, compression="snappy")
    val.to_parquet(VAL_PARQUET, index=False, compression="snappy")
    test.to_parquet(TEST_PARQUET, index=False, compression="snappy")
    logger.info("Wrote %s", TRAIN_PARQUET)
    logger.info("Wrote %s", VAL_PARQUET)
    logger.info("Wrote %s", TEST_PARQUET)


def print_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    total = len(train) + len(val) + len(test)
    print()
    print(f"{'Split':<7}{'Rows':>8}{'Share':>10}{'Positive':>12}")
    for name, split in (("Train", train), ("Val", val), ("Test", test)):
        share = 100 * len(split) / total
        pos = split[TARGET_BIN].mean()
        print(f"{name:<7}{len(split):>8,}{share:>9.2f}%{pos:>12.4f}")
    print()
    print(f"Total features (excl. target): {train.shape[1] - 1}")
    print(f"Total rows                   : {total:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {CLEAN_PARQUET}. Run `make clean-data` first."
        )

    df = pd.read_parquet(CLEAN_PARQUET)
    logger.info("Loaded clean parquet: %s rows x %s cols", *df.shape)

    df = encode_age(df)
    df = group_diag_columns(df)
    df = encode_medications(df)
    df = add_derived_features(df)

    train, val, test = make_splits(df, TARGET_BIN)
    train, val, test = one_hot_encode(train, val, test)

    save_splits(train, val, test)
    print_summary(train, val, test)
    print()
    print("✅ Feature engineering complete.")


if __name__ == "__main__":
    main()
