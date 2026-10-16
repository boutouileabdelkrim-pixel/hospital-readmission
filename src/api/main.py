"""FastAPI app exposing the hospital readmission early-warning model."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from src.api.model_loader import (
    DEFAULT_THRESHOLD,
    build_features,
    classify_risk_band,
    load_model_bundle,
)
from src.api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.models.tune_optuna import TUNED_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hospital Readmission Early Warning API",
    description=(
        "Predicts 30-day hospital readmission risk from clinical features. "
        "Trained on Diabetes 130-US hospitals (UCI)."
    ),
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Returns whether the model is loadable."""
    try:
        bundle = load_model_bundle()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_path=str(TUNED_MODEL_PATH.name),
            n_features=len(bundle.feature_names),
        )
    except FileNotFoundError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_path=str(TUNED_MODEL_PATH.name),
            n_features=0,
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    """Score a single patient. Returns probability + risk band."""
    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        payload = features.model_dump(by_alias=True)
        X = build_features(payload)
        proba = bundle.predict_proba(X)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        probability=round(proba, 6),
        threshold=DEFAULT_THRESHOLD,
        will_be_readmitted_30d=proba >= DEFAULT_THRESHOLD,
        risk_band=classify_risk_band(proba),
    )


@app.get("/")
def root() -> dict:
    """Root endpoint. Points to the docs."""
    return {
        "name": "hospital-readmission-api",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": ["/health", "/predict"],
    }
