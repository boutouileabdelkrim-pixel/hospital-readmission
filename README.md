# Hospital Readmission Early Warning

[![CI](https://github.com/boutouileabdelkrim-pixel/hospital-readmission/actions/workflows/ci.yml/badge.svg)](https://github.com/boutouileabdelkrim-pixel/hospital-readmission/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predict 30-day hospital readmission risk from clinical data, with calibrated
probabilities, SHAP-based interpretability, and demographic fairness analysis.

> 🚧 **Work in progress** — Phase 1: skeleton + CI.

---

## 🎯 Problem

Hospital readmissions within 30 days are costly and often preventable. This
project builds a production-ready early-warning system that flags high-risk
patients at discharge, so care teams can prioritize follow-up.

- **Task** : binary classification (readmitted within 30 days: yes / no)
- **Challenge** : strong class imbalance (~11 % positive rate)
- **Primary metric** : PR-AUC (with ROC-AUC as secondary)
- **Clinical priority** : high recall at controlled precision
- **Explainability** : SHAP values for clinician-facing explanations
- **Fairness** : per-group performance across age, gender, ethnicity
- **Calibration** : calibrated probabilities for clinical decision-making

## 📦 Dataset

- **Diabetes 130-US hospitals (1999–2008)** — UCI ML Repository, CC BY 4.0
- ~100 k encounters, 50 features, small enough for local and Kaggle workflows

## 🛠️ Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, numpy, pyarrow |
| ML | scikit-learn, XGBoost, LightGBM, Optuna |
| Interpretability | SHAP, fairlearn |
| MLOps | MLflow, FastAPI, uvicorn, Docker |
| Quality | pytest, ruff, pre-commit, GitHub Actions |

## 🏗️ Project structure

```text
hospital-readmission/
├── src/
│   ├── data/           # download, clean, validate
│   ├── features/       # clinical feature engineering
│   ├── models/         # train, evaluate, calibration, fairness
│   ├── api/            # FastAPI serving
│   ├── monitoring/     # drift detection
│   ├── tracking/       # MLflow helpers
│   └── utils/          # paths, logging, helpers
├── tests/              # pytest suite
├── notebooks/          # EDA, fairness analysis
├── docker/             # Dockerfile + compose
├── docs/               # documentation
├── data/               # raw / processed / features (gitignored)
├── models/             # trained artifacts (gitignored)
├── reports/            # metrics + figures
├── .github/workflows/  # CI
├── pyproject.toml
└── README.md
```

## 🚀 Quickstart

```bash
git clone git@github.com:boutouileabdelkrim-pixel/hospital-readmission.git
cd hospital-readmission
uv sync --all-groups
uv run pytest -v
uv run ruff check src tests
```

## 📜 License

MIT — see LICENSE file.
