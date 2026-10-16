"""Pydantic schemas for the FastAPI serving layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """Raw patient features as accepted by the API (pre-encoding)."""

    race: str = Field(..., description="Patient race")
    gender: str = Field(..., description="Patient gender")
    age: str = Field(..., description="Age bucket (e.g. '[60-70)')")
    admission_type_id: int = Field(..., ge=1, le=8)
    discharge_disposition_id: int = Field(..., ge=1, le=30)
    admission_source_id: int = Field(..., ge=1, le=25)
    time_in_hospital: int = Field(..., ge=1, le=14)
    payer_code: str
    medical_specialty: str
    num_lab_procedures: int = Field(..., ge=0)
    num_procedures: int = Field(..., ge=0)
    num_medications: int = Field(..., ge=0)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    diag_1: str
    diag_2: str
    diag_3: str
    number_diagnoses: int = Field(..., ge=0)
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    insulin: str
    glyburide_metformin: str = Field(..., alias="glyburide-metformin")
    glipizide_metformin: str = Field(..., alias="glipizide-metformin")
    glimepiride_pioglitazone: str = Field(..., alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: str = Field(..., alias="metformin-rosiglitazone")
    metformin_pioglitazone: str = Field(..., alias="metformin-pioglitazone")
    change: str
    diabetes_med: str = Field(..., alias="diabetesMed")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Prediction returned by the API."""

    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated readmission risk")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Decision threshold used")
    will_be_readmitted_30d: bool = Field(..., description="Hard label at the threshold")
    risk_band: str = Field(..., description="One of: low, medium, high")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    n_features: int
