# Hospital Readmission Early Warning

[![CI](https://github.com/boutouileabdelkrim-pixel/hospital-readmission/actions/workflows/ci.yml/badge.svg)](https://github.com/boutouileabdelkrim-pixel/hospital-readmission/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hospital-readmission999.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-113%20passed-brightgreen.svg)](#testing)

> 🎯 **[Live demo →](https://hospital-readmission999.streamlit.app)** — interactive Streamlit dashboard with SHAP explanations and cost-sensitive threshold optimization.

Predict 30-day hospital readmission risk from clinical data with **calibrated probabilities**, **SHAP interpretability**, **demographic fairness analysis**, and **cost-sensitive decision optimization**. Production-ready pipeline: MLflow, FastAPI, Docker, GitHub Actions CI, Streamlit.

---

## 💰 Business Impact

**Not all errors are equal in healthcare.** In collaboration with clinical literature, we use asymmetric cost assumptions:

| Error type | Cost (USD) | Source |
|------------|------------|--------|
| **False negative** (missed readmission) | **$15,000** | CMS penalty + readmission cost |
| **False positive** (unnecessary follow-up) | **$500** | Post-discharge call + consult |

The **optimal decision threshold is not 0.5** but **0.41**, subject to an operational precision floor of 15 %:

| Threshold | Patients flagged | Total cost | Savings vs baseline |
|-----------|------------------|------------|---------------------|
| Baseline (no model) | — | $25.46M | $0 |
| **Model @ 0.50 (default)** | 28 % | $13.60M | **$11.86M (46.6 %)** |
| **Model @ 0.41 (optimal, precision ≥ 15 %)** | 47 % | $10.49M | **$14.97M (58.8 %)** |

**+$3.1M additional savings** compared to the naive 0.5 threshold — while staying deployable (flags < 50 % of patients).

> ⚠️ **Why a precision floor?** Without it, the unconstrained cost minimum flags **88 % of patients**, which is clinically useless. This is the kind of pragmatic constraint a real hospital would impose — and it's why the "ML optimal" ≠ "business optimal".

Full analysis: [`reports/cost_analysis.json`](reports/cost_analysis.json) · Curve: [`reports/cost_curve.png`](reports/cost_curve.png)

---

## 🎯 Problem

Hospital readmissions within 30 days are costly and often preventable. This project builds an early-warning system that flags high-risk patients at discharge so care teams can prioritize follow-up.

- **Task**: binary classification (readmitted within 30 days: yes/no)
- **Dataset**: [Diabetes 130-US hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI ID 296, CC BY 4.0), 101,766 encounters → 99,343 after cleaning, 198 engineered features
- **Class imbalance**: 11.4 % positive rate
- **Primary metric**: PR-AUC (ROC-AUC as secondary)
- **Clinical priority**: high recall at controlled precision
- **Fairness**: per-group performance across age, gender, ethnicity

---

## 📊 Results

| Model | Split | PR-AUC | ROC-AUC | Brier |
|-------|-------|--------|---------|-------|
| XGBoost baseline | val | 0.2263 | 0.6663 | 0.2163 |
| LightGBM baseline | val | 0.2291 | 0.6658 | 0.2144 |
| **XGBoost tuned (Optuna)** | **val** | **0.2321** | 0.6655 | 0.2132 |
| XGBoost tuned | test | 0.2295 | 0.6733 | 0.2121 |

Random baseline PR-AUC = 0.114 → **~2× lift**.

### Calibration

Isotonic regression, 5-fold cross-fit on validation:

| | Brier (test) |
|--|--|
| Uncalibrated | 0.2120 |
| **Calibrated** | **0.0961** |

**−55 % Brier error** → probabilities usable for clinical decision-making.

![Reliability diagram](reports/calibration_curve.png)

### Interpretability (SHAP)

Top-5 features by mean |SHAP|:

| # | Feature | Mean \|SHAP\| |
|---|---------|-------------|
| 1 | `number_inpatient` | 0.2784 |
| 2 | `discharge_disposition_id` | 0.2015 |
| 3 | `age_ord` | 0.0599 |
| 4 | `payer_code_Unknown` | 0.0571 |
| 5 | **`service_utilization` (engineered)** | 0.0520 |

![SHAP summary](reports/shap_summary.png)

### Fairness

| Group column | PR-AUC disparity ratio |
|--------------|------------------------|
| gender | **1.09** (near-equitable) |
| race | 1.71 (moderate) |
| age | 2.77 (driven by <18 and >90 cohorts, n<300) |

Full report: [`reports/fairness_report.json`](reports/fairness_report.json)

---

## 🏗️ Architecture

```
hospital-readmission/
├── src/
│   ├── data/           # download, validate, clean
│   ├── features/       # clinical feature engineering (ICD-9 grouping, meds encoding)
│   ├── models/         # train_baseline, tune_optuna, log_mlflow, explain, calibrate,
│   │                   # fairness, cost_analysis
│   ├── api/            # FastAPI (schemas, model_loader, main)
│   ├── monitoring/     # Evidently drift
│   ├── tracking/       # MLflow helpers
│   └── utils/          # paths
├── app/                # Streamlit dashboard (self-contained)
├── tests/              # 113 pytest tests
├── docker/             # Dockerfile (multi-stage, uv-based, non-root)
├── reports/            # metrics, plots, calibration, fairness, cost, drift
├── notebooks/          # EDA
├── .github/workflows/  # CI (2 jobs: quality + api-smoke)
├── pyproject.toml
└── Makefile
```

---

## 🚀 Quickstart

```bash
git clone git@github.com:boutouileabdelkrim-pixel/hospital-readmission.git
cd hospital-readmission
uv sync --all-groups

# Full pipeline
make data
make clean-data
make features
make train-baseline
make tune-xgb
make cost-analysis
make drift

# Dashboard
make streamlit    # or: uv run streamlit run app/streamlit_app.py

# API
uv run uvicorn src.api.main:app --port 8000

# Quality
make test    # 113 tests
make lint
```

---

## 🛠️ Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, numpy, pyarrow |
| ML | scikit-learn, XGBoost 2.x, LightGBM 4.x, Optuna |
| Interpretability | SHAP, fairlearn, evidently |
| MLOps | MLflow 2.22, FastAPI, uvicorn, Docker |
| Dashboard | Streamlit |
| Quality | pytest, ruff, pre-commit, GitHub Actions |

---

## 📈 MLOps pipeline

1. **Data**: `make data` → UCI / Kaggle fallback → `data/raw/`
2. **Clean**: `make clean-data` → `data/processed/clean.parquet`
3. **Features**: `make features` → stratified 70/15/15 splits
4. **Baseline**: `make train-baseline` → XGBoost + LightGBM
5. **Tuning**: `make tune-xgb` → Optuna TPE, 20 trials
6. **Cost analysis**: `make cost-analysis` → optimal threshold + savings
7. **Tracking**: `make mlflow` → `mlruns.db` (SQLite)
8. **Interpretability + Fairness**: `python -m src.models.explain|calibrate|fairness`
9. **Drift**: `make drift` → `reports/drift_report.html`
10. **Serve**: `uvicorn src.api.main:app`
11. **Dashboard**: `make streamlit`
12. **Containerize**: `make docker-build`

---

## 🧪 Testing

```bash
make test   # 113 tests, ~10 s, no network, no large files
make lint   # ruff check + format
```

Tests use fixtures and mocks. No real dataset required for `make test`.

CI runs on every push: 2 jobs (`Lint + Tests` + `API Smoke Test`).

---

## ⚠️ Limitations

- **No deep learning** (tree-based only, justified by dataset size + interpretability)
- **No NLP** on clinical notes (dataset limitation)
- **Not deployed to production hospital infrastructure** — demo only
- **Fairness on age is moderate** (disparity 2.77) due to small <18 and >90 cohorts — documented, not corrected
- **MIMIC-IV not used** (restricted access, reproducibility requirement)
- **Streamlit Cloud free tier** throttles CPU after sustained use — for production, deploy to a paid tier or Docker/K8s

---

## 📜 License

MIT

## 🙏 Acknowledgements

- UCI ML Repository — Diabetes 130-US hospitals dataset
- SHAP, Optuna, MLflow, Evidently, FastAPI, Streamlit teams
