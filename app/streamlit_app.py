"""Streamlit dashboard for the Hospital Readmission Early Warning model.

Self-contained: no import from the repo's `src/` package (Streamlit Cloud
only deploys the `app/` folder). Data + model live in `app/data/`.

Run locally: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths (self-contained)
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODEL_PATH = DATA_DIR / "xgb_tuned.json"
TEST_PATH = DATA_DIR / "test.parquet"
FAIRNESS_PATH = DATA_DIR / "fairness_report.json"

TARGET_COLUMN = "readmitted_30d"

st.set_page_config(
    page_title="Hospital Readmission Early Warning",
    page_icon="🏥",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Feature engineering (inlined, mirrors src/features/clinical.py)
# ---------------------------------------------------------------------------

AGE_ORDER = [
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

MED_COLUMNS = [
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

MED_MAP = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}

ICD9_GROUPS = [
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


def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {bucket: i for i, bucket in enumerate(AGE_ORDER)}
    df = df.copy()
    df["age_ord"] = df["age"].map(mapping).astype("int8")
    return df.drop(columns=["age"])


def group_icd9(code: object) -> str:
    if code is None:
        return "Unknown"
    s = str(code).strip()
    if s == "" or s.lower() == "unknown" or s == "nan":
        return "Unknown"
    if s[0] in ("V", "E"):
        return "Other"
    try:
        num = int(s.split(".")[0])
    except (ValueError, IndexError):
        return "Other"
    for name, lo, hi in ICD9_GROUPS:
        if lo <= num <= hi:
            return name
    return "Other"


def group_diag_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("diag_1", "diag_2", "diag_3"):
        df[f"{col}_group"] = df[col].map(group_icd9)
        df = df.drop(columns=[col])
    return df


def encode_medications(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MED_COLUMNS:
        df[col] = df[col].map(MED_MAP).astype("int8")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["service_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    ).astype("int16")
    df["n_med_changes"] = (df[MED_COLUMNS].to_numpy() >= 2).sum(axis=1).astype("int8")
    return df


def build_features(payload: dict, feature_names: list[str]) -> pd.DataFrame:
    """Raw payload -> model-ready 1-row DataFrame aligned to training columns."""
    df = pd.DataFrame([payload])
    df = encode_age(df)
    df = group_diag_columns(df)
    df = encode_medications(df)
    df = add_derived_features(df)
    df = pd.get_dummies(df, dtype="int8")
    return df.reindex(columns=feature_names, fill_value=0)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_booster() -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster


@st.cache_data
def load_test() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_parquet(TEST_PATH)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y, list(X.columns)


@st.cache_data
def load_fairness() -> dict:
    if not FAIRNESS_PATH.exists():
        return {}
    return json.loads(FAIRNESS_PATH.read_text())


# ---------------------------------------------------------------------------
# SHAP helper
# ---------------------------------------------------------------------------


def shap_waterfall(booster: xgb.Booster, X_row: pd.DataFrame) -> plt.Figure:
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X_row)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    exp = shap.Explanation(
        values=shap_vals[0],
        base_values=float(np.array(explainer.expected_value).ravel()[0]),
        data=X_row.iloc[0].to_numpy(),
        feature_names=list(X_row.columns),
    )
    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

booster = load_booster()
X_test, y_test, feature_names = load_test()
fairness = load_fairness()

st.title("🏥 Hospital Readmission Early Warning")
st.markdown(
    "Predict **30-day readmission risk** from clinical features. "
    "Trained on the UCI Diabetes 130-US hospitals dataset (101,766 encounters → 198 features). "
    "PR-AUC **0.232** · ROC-AUC **0.666** · Brier (calibrated) **0.096**."
)

tab_patient, tab_cost, tab_card = st.tabs(
    ["🎯 Single patient", "💰 Cost analysis", "📊 Model card"]
)


# --- Tab 1 -----------------------------------------------------------------

with tab_patient:
    st.header("Score a single patient")
    st.caption("Adjust fields below — the prediction and SHAP explanation update live.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.selectbox("Age bucket", AGE_ORDER, index=7)
        gender = st.selectbox("Gender", ["Female", "Male"])
        race = st.selectbox(
            "Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Other", "Unknown"]
        )
        time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
    with c2:
        n_inpatient = st.slider("Prior inpatient visits", 0, 20, 1)
        n_emergency = st.slider("Prior emergency visits", 0, 20, 0)
        n_outpatient = st.slider("Prior outpatient visits", 0, 20, 0)
        n_medications = st.slider("Number of medications", 0, 80, 15)
    with c3:
        num_lab = st.slider("Lab procedures", 0, 130, 40)
        num_diag = st.slider("Number of diagnoses", 0, 16, 7)
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        change = st.selectbox("Medication change", ["No", "Ch"])

    payload = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": time_in_hospital,
        "payer_code": "MC",
        "medical_specialty": "Cardiology",
        "num_lab_procedures": num_lab,
        "num_procedures": 0,
        "num_medications": n_medications,
        "number_outpatient": n_outpatient,
        "number_emergency": n_emergency,
        "number_inpatient": n_inpatient,
        "diag_1": "250.83",
        "diag_2": "428.0",
        "diag_3": "Unknown",
        "number_diagnoses": num_diag,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "metformin": "Steady",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": "No",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "insulin": insulin,
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": change,
        "diabetesMed": "Yes",
    }

    X_row = build_features(payload, feature_names)
    proba = float(booster.predict(xgb.DMatrix(X_row))[0])

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("30-day readmission probability", f"{proba:.1%}")
    band = "🔴 High" if proba >= 0.30 else ("🟡 Medium" if proba >= 0.10 else "🟢 Low")
    m2.metric("Risk band", band)
    decision = "⚠️ Flag for follow-up" if proba >= 0.15 else "✅ Routine"
    m3.metric("Decision (threshold 0.15)", decision)

    st.subheader("Why this prediction? (SHAP)")
    st.caption("Bars to the right push risk **up**; bars to the left push risk **down**.")
    fig = shap_waterfall(booster, X_row)
    st.pyplot(fig)
    plt.close(fig)


# --- Tab 2 -----------------------------------------------------------------

with tab_cost:
    st.header("Cost-sensitive threshold optimization")
    st.markdown(
        "In clinical practice, a **false negative** (missed readmission) costs far "
        "more than a **false positive** (unnecessary follow-up). Adjust the "
        "assumptions to see how the optimal decision threshold shifts."
    )

    c1, c2 = st.columns(2)
    with c1:
        fn_cost = st.number_input(
            "False negative cost (USD)",
            min_value=1_000,
            max_value=100_000,
            value=15_000,
            step=1_000,
        )
    with c2:
        fp_cost = st.number_input(
            "False positive cost (USD)",
            min_value=50,
            max_value=10_000,
            value=500,
            step=50,
        )

    min_precision = st.slider(
        "Minimum precision floor (operational constraint)",
        min_value=0.05,
        max_value=0.50,
        value=0.15,
        step=0.05,
    )

    # Cost evaluation on the loaded test sample
    sample_size = min(5000, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_s = X_test.iloc[idx].reset_index(drop=True)
    y_s = y_test[idx]

    y_proba = booster.predict(xgb.DMatrix(X_s))

    # Compute costs and precisions across thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = np.zeros_like(thresholds)
    precisions = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        y_pred = (y_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_s == 1)).sum())
        fp = int(((y_pred == 1) & (y_s == 0)).sum())
        fn = int(((y_pred == 0) & (y_s == 1)).sum())
        costs[i] = fn * fn_cost + (fp + tp) * fp_cost
        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    base_cost = int(y_s.sum()) * fn_cost
    feasible = precisions >= min_precision
    if feasible.any():
        feasible_costs = np.where(feasible, costs, np.inf)
        best_idx = int(np.argmin(feasible_costs))
    else:
        best_idx = int(np.argmin(costs))

    best_t = float(thresholds[best_idx])
    savings = base_cost - float(costs[best_idx])
    savings_per_1000 = savings / len(y_s) * 1000

    m1, m2, m3 = st.columns(3)
    m1.metric("Optimal threshold", f"{best_t:.3f}")
    m2.metric("Savings vs baseline", f"${savings:,.0f}")
    m3.metric("Savings per 1000 patients", f"${savings_per_1000:,.0f}")

    st.info(
        f"At threshold **{best_t:.3f}** (precision {precisions[best_idx]:.2f}), the model "
        f"saves **{100 * savings / base_cost:.1f}%** of the total baseline cost "
        f"(**${base_cost:,.0f}** over {len(y_s):,} patients)."
    )

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(thresholds, costs / 1e6, color="#377eb8", label="Total cost (model)")
    ax.axhline(base_cost / 1e6, color="black", linestyle="--", label="Baseline (no model)")
    ax.axvline(best_t, color="#e41a1c", linestyle=":", label=f"Optimal t = {best_t:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="Default t = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total cost (millions USD)")
    ax.set_title("Cost vs threshold (live)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# --- Tab 3 -----------------------------------------------------------------

with tab_card:
    st.header("Model card")
    st.markdown(
        """
| Attribute | Value |
|-----------|-------|
| Model | XGBoost tuned (Optuna TPE, 20 trials) |
| Features | 198 |
| Dataset | UCI Diabetes 130-US hospitals (ID 296, CC BY 4.0) |
| Encounters | 101,766 raw → 99,343 clean |
| Class imbalance | 11.4 % positive (30-day readmission) |
| Val PR-AUC | 0.2321 |
| Test PR-AUC | 0.2295 |
| Test ROC-AUC | 0.6733 |
| Brier (calibrated) | 0.0961 (vs 0.2120 uncalibrated) |
| Optimal threshold (val, precision ≥ 0.15) | 0.41 |
"""
    )

    if fairness:
        st.subheader("Fairness — PR-AUC disparity per group")
        rows = []
        for col, block in fairness.get("by_group_column", {}).items():
            rows.append(
                {
                    "Group column": col,
                    "Disparity ratio (max/min PR-AUC)": round(
                        block.get("pr_auc_disparity_ratio", 0) or 0, 2
                    ),
                    "Groups": len(block.get("groups", {})),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Known limitations")
    st.markdown(
        "- No deep learning (tree-based only, justified by small dataset + interpretability)\n"
        "- No NLP on clinical notes (dataset limitation)\n"
        "- Fairness on age is moderate (disparity 2.77) due to small <18 and >90 cohorts\n"
        "- Not deployed to production hospital infrastructure (demo only)\n"
    )

st.sidebar.markdown(
    "### Links\n"
    "- [GitHub repository](https://github.com/boutouileabdelkrim-pixel/hospital-readmission)\n"
    "- [API docs (local)](http://localhost:8000/docs)\n"
)
st.sidebar.caption(
    f"Model: `xgb_tuned.json` — {len(feature_names)} features, "
    f"test sample: {len(X_test):,} patients."
)
