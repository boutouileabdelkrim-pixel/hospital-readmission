# Phase 3 — EDA + Cleaning Summary

## Input / Output

| | Rows | Cols | Notes |
|---|---|---|---|
| Raw (`data/raw/diabetic_data.csv`) | 101,766 | 50 | ~19 MB, some `?` values |
| Clean (`data/processed/clean.parquet`) | see step 9 output | 47 | snappy compressed |

## Cleaning decisions

| Decision | Rationale |
|---|---|
| Binarize target: `readmitted_30d = (readmitted == "<30")` | Project scope = 30-day readmission |
| Drop `weight` | 96.86 % missing -> no signal |
| Drop `examide`, `citoglipton` | Constant value `No` in every row |
| Drop `encounter_id`, `patient_nbr` | Identifiers, leakage risk across rows |
| Keep `max_glu_serum`, `A1Cresult` as-is (value `"None"` = test not performed) | Clinically meaningful, not a missing value |
| Impute `race`, `payer_code`, `medical_specialty`, `diag_1/2/3` NaN -> `"Unknown"` | Avoid dropping 40 %+ rows; "unknown" is itself informative |
| Drop deceased/hospice discharges (`discharge_disposition_id ∈ {11, 13, 14, 19, 20, 21}`) | Patient cannot be readmitted after death |
| Explicit dtypes (`dtype=str` then numeric cast) | Fix `DtypeWarning`, guarantee determinism |

## Cleaned target distribution

See step 9 output (expected ~11 % positive class, `readmitted_30d = 1`).

-> **Strong class imbalance** preserved. Modeling will use PR-AUC as the
primary metric and `scale_pos_weight` for XGBoost / LightGBM.

## Residual NaN

After cleaning, remaining NaN may appear in `gender` (a handful of
`Unknown/Invalid` rows). To be addressed in Phase 4 if non-negligible.

## Next phase — Feature Engineering (Phase 4)

- Ordinal-encode `age` buckets
- Group ICD-9 `diag_1/2/3` into top-N categories + "Other"
- Create `service_utilization = number_outpatient + number_emergency + number_inpatient`
- Encode medication columns (No/Steady/Up/Down)
- One-hot / target-encode low-cardinality categoricals
- Stratified train/val/test split (70/15/15) BEFORE any fitting encoder
- Save `data/features/{train,val,test}.parquet`
