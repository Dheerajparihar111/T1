# ============================================================
# schemas.py
# Pydantic Schemas — Define data shapes for API input/output
# Think of these as "contracts" for what data looks like
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ------------------------------------------------------------
# HealthData Schema
# Represents ONE row from the Supabase health_data table
# Optional fields = handles missing/null values gracefully
# ------------------------------------------------------------
class HealthData(BaseModel):
    """
    Maps to the 'health_data' table in Supabase.
    All fields are Optional so missing data doesn't crash the app.
    """
    id: Optional[int] = None
    name: Optional[str] = None
    age: Optional[float] = None
    glucose: Optional[float] = None
    bmi: Optional[float] = None
    hba1c: Optional[float] = None
    blood_pressure: Optional[float] = None
    cholesterol: Optional[float] = None
    insulin: Optional[float] = None
    created_at: Optional[str] = None  # Supabase returns timestamps as strings


# ------------------------------------------------------------
# PredictionResult Schema
# The ML model's prediction output
# ------------------------------------------------------------
class PredictionResult(BaseModel):
    """
    Holds the diabetes risk prediction from the ML model.
    """
    risk_level: str = Field(
        ...,
        description="One of: 'Low Risk', 'Medium Risk', 'High Risk'",
        examples=["High Risk"]
    )
    probability: float = Field(
        ...,
        description="Probability score between 0.0 and 1.0",
        examples=[0.87]
    )


# ------------------------------------------------------------
# PredictionResponse Schema
# The FULL JSON response returned by /predict endpoint
# ------------------------------------------------------------
class PredictionResponse(BaseModel):
    """
    Complete API response with user data, prediction, risk factors,
    and personalized health recommendations.
    """
    user_data: HealthData                    # Raw health metrics from Supabase
    prediction: PredictionResult             # ML model output
    risk_factors: List[str]                  # List of concerning health values
    recommendations: List[str]               # Personalized prevention tips

    class Config:
        # Allow extra fields from Supabase (future-proofing)
        extra = "allow"


# ------------------------------------------------------------
# ErrorResponse Schema
# Standardized error format for all API errors
# ------------------------------------------------------------
class ErrorResponse(BaseModel):
    """
    Returned when something goes wrong (DB error, model error, etc.)
    """
    error: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Technical error details")


# ------------------------------------------------------------
# HealthCheckResponse Schema
# Used by the /health endpoint to verify the API is running
# ------------------------------------------------------------
class HealthCheckResponse(BaseModel):
    """
    Simple health check response.
    """
    status: str = "ok"
    message: str = "Diabetes Prediction API is running"
    version: str = "1.0.0"
