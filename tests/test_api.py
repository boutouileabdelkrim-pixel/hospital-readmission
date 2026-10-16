"""Unit tests for the FastAPI layer (offline, uses TestClient)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def _valid_payload() -> dict:
    return {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "payer_code": "MC",
        "medical_specialty": "Cardiology",
        "num_lab_procedures": 40,
        "num_procedures": 0,
        "num_medications": 12,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "diag_1": "250.83",
        "diag_2": "428.0",
        "diag_3": "Unknown",
        "number_diagnoses": 7,
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
        "insulin": "Steady",
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": "Ch",
        "diabetesMed": "Yes",
    }


# --- root ------------------------------------------------------------------


def test_root_returns_metadata() -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "endpoints" in body
    assert "/predict" in body["endpoints"]


# --- health ----------------------------------------------------------------


def test_health_when_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the model loader to raise
    with patch("src.api.main.load_model_bundle", side_effect=FileNotFoundError("no model")):
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["model_loaded"] is False


def test_health_when_model_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.feature_names = ["f0", "f1", "f2"]
    with patch("src.api.main.load_model_bundle", return_value=fake):
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_features"] == 3


# --- predict ---------------------------------------------------------------


def test_predict_returns_expected_schema() -> None:
    fake = MagicMock()
    fake.predict_proba.return_value = 0.42
    with patch("src.api.main.load_model_bundle", return_value=fake):
        r = client.post("/predict", json=_valid_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["threshold"] == 0.15
    assert body["will_be_readmitted_30d"] is True
    assert body["risk_band"] in {"low", "medium", "high"}


def test_predict_rejects_invalid_payload() -> None:
    payload = _valid_payload()
    payload["time_in_hospital"] = 99  # > 14
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_returns_503_when_model_missing() -> None:
    with patch("src.api.main.load_model_bundle", side_effect=FileNotFoundError("no model")):
        r = client.post("/predict", json=_valid_payload())
    assert r.status_code == 503


# --- risk band -------------------------------------------------------------


def test_classify_risk_band() -> None:
    from src.api.model_loader import classify_risk_band

    assert classify_risk_band(0.05) == "low"
    assert classify_risk_band(0.15) == "medium"
    assert classify_risk_band(0.50) == "high"
    assert classify_risk_band(0.99) == "high"
