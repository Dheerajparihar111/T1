# ============================================================
# main.py
# FastAPI Application Entry Point
# All API routes, startup logic, and error handling
# ============================================================

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Local module imports
from schemas import (
    PredictionResponse,
    PredictionResult,
    HealthData,
    HealthCheckResponse,
)
from supabase_client import fetch_latest_health_data, fetch_health_data_by_name
from predict import load_model_and_scaler, predict_diabetes_risk
from recommendations import identify_risk_factors, generate_recommendations

# ============================================================
# Logging Configuration
# Shows timestamped logs in terminal for debugging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# Global Model State
# Stores model & scaler after loading at startup
# Avoids reloading from disk on every API request (much faster)
# ============================================================
app_state = {
    "model": None,
    "scaler": None,
    "model_loaded": False,
}


# ============================================================
# Lifespan — Startup & Shutdown Events
# Loads the ML model ONCE when the server starts
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Called automatically by FastAPI on startup and shutdown.
    We load the ML model here so it's ready before any requests arrive.
    """
    # ---- STARTUP ----
    logger.info("🚀 Starting Diabetes Prediction API...")
    logger.info("🧠 Loading ML model and scaler...")

    try:
        model, scaler = load_model_and_scaler()
        app_state["model"] = model
        app_state["scaler"] = scaler
        app_state["model_loaded"] = True
        logger.info("✅ ML model and scaler loaded successfully!")
    except FileNotFoundError as e:
        # Don't crash the whole app — just mark model as not loaded
        logger.error(f"❌ Model files missing: {e}")
        logger.warning("⚠️  API will start but /predict endpoint won't work without model files")
        app_state["model_loaded"] = False
    except Exception as e:
        logger.error(f"❌ Unexpected error loading model: {e}")
        app_state["model_loaded"] = False

    logger.info("🌐 API is ready to accept requests!")

    yield  # App runs here — everything after yield is shutdown code

    # ---- SHUTDOWN ----
    logger.info("👋 Shutting down Diabetes Prediction API...")


# ============================================================
# FastAPI App Initialization
# ============================================================
app = FastAPI(
    title="🏥 AI Diabetes Prediction API",
    description="""
    ## AI-Powered Diabetes Risk Prediction System
    
    This API connects to Supabase to fetch user health data,
    runs it through a trained ML model, and returns:
    
    - **Risk Level**: Low / Medium / High
    - **Probability Score**: 0.0 to 1.0
    - **Risk Factors**: What's causing concern
    - **Recommendations**: Personalized prevention advice
    
    ### Quick Start
    1. POST health data to Supabase via your frontend
    2. Call `GET /predict` to get the diabetes risk prediction
    3. (Optional) Pass `?name=John` to predict for a specific user
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",           # Swagger UI at /docs
    redoc_url="/redoc",         # ReDoc UI at /redoc
)


# ============================================================
# CORS Middleware
# Allows your React frontend to call this API
# In production, replace "*" with your actual frontend domain
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Replace with ["https://yourfrontend.com"] in production
    allow_credentials=True,
    allow_methods=["*"],           # Allow GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],           # Allow all headers
)


# ============================================================
# ROUTE 1: Health Check
# Quick endpoint to verify the API is alive
# ============================================================
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Check if API is running",
)
async def health_check():
    """
    Simple health check endpoint.
    Returns status 200 if the API is running correctly.
    """
    return HealthCheckResponse(
        status="ok",
        message="Diabetes Prediction API is running",
        version="1.0.0"
    )


# ============================================================
# ROUTE 2: Model Status
# Check if the ML model was loaded successfully
# ============================================================
@app.get(
    "/model-status",
    tags=["System"],
    summary="Check if ML model is loaded",
)
async def model_status():
    """
    Returns whether the ML model and scaler are loaded and ready.
    """
    return {
        "model_loaded": app_state["model_loaded"],
        "model_type": type(app_state["model"]).__name__ if app_state["model"] else None,
        "status": "ready" if app_state["model_loaded"] else "model files missing",
    }


# ============================================================
# ROUTE 3: Main Prediction Endpoint
# Fetches health data from Supabase → runs ML → returns prediction
# ============================================================
@app.get(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict diabetes risk for a user",
    responses={
        200: {"description": "Successful prediction with risk level and recommendations"},
        404: {"description": "No health data found in database"},
        503: {"description": "ML model not loaded"},
        500: {"description": "Internal server error"},
    }
)
async def predict_diabetes(
    name: Optional[str] = Query(
        default=None,
        description="Optional: Filter by user name. If not provided, fetches the latest record.",
        examples=["John Doe"]
    )
):
    """
    ## Main Prediction Endpoint
    
    **Flow:**
    1. Fetches health data from Supabase (`health_data` table)
    2. Extracts 5 ML features: glucose, bmi, hba1c, age, blood_pressure
    3. Scales features and runs the diabetes ML model
    4. Identifies risk factors from health values
    5. Generates personalized prevention recommendations
    6. Returns complete JSON response
    
    **Query Parameters:**
    - `name` (optional): Get prediction for a specific user by name
    
    **Example:** `GET /predict?name=John`
    """

    # ---- Guard: Check model is loaded ----
    if not app_state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ML model not available",
                "detail": "Model files (diabetes_model.pkl, scaler.pkl) are missing. "
                          "Please place them in the 'models/' directory."
            }
        )

    # ---- STEP 1: Fetch Health Data from Supabase ----
    try:
        if name:
            # Fetch specific user by name
            logger.info(f"📡 Fetching health data for user: '{name}'")
            raw_data = await fetch_health_data_by_name(name)
        else:
            # Fetch the most recently added record
            logger.info("📡 Fetching latest health record from Supabase...")
            raw_data = await fetch_latest_health_data()

    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Database connection failed",
                "detail": str(e)
            }
        )

    # ---- Guard: No data found ----
    if not raw_data:
        user_msg = f"'{name}'" if name else "any user"
        raise HTTPException(
            status_code=404,
            detail={
                "error": "No health data found",
                "detail": f"No records found for {user_msg} in the health_data table. "
                          f"Please submit health data via the frontend form first."
            }
        )

    logger.info(f"✅ Health data fetched for: {raw_data.get('name', 'Unknown')}")

    # ---- STEP 2: Run ML Prediction ----
    try:
        risk_level, probability = predict_diabetes_risk(
            health_data=raw_data,
            model=app_state["model"],
            scaler=app_state["scaler"],
        )
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "detail": f"ML model error: {str(e)}"
            }
        )

    # ---- STEP 3: Identify Risk Factors ----
    risk_factors = identify_risk_factors(raw_data)

    # ---- STEP 4: Generate Recommendations ----
    recommendations = generate_recommendations(raw_data, risk_level)

    # ---- STEP 5: Build & Return Response ----
    response = PredictionResponse(
        user_data=HealthData(**{
            k: v for k, v in raw_data.items()
            if k in HealthData.model_fields  # Only include known schema fields
        }),
        prediction=PredictionResult(
            risk_level=risk_level,
            probability=probability,
        ),
        risk_factors=risk_factors,
        recommendations=recommendations,
    )

    logger.info(
        f"🎉 Prediction complete | User: {raw_data.get('name')} | "
        f"Risk: {risk_level} | Probability: {probability}"
    )

    return response


# ============================================================
# ROUTE 4: Predict for Specific User ID
# Alternative endpoint to fetch by database record ID
# ============================================================
@app.get(
    "/predict/{record_id}",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict diabetes risk by record ID",
)
async def predict_by_id(record_id: int):
    """
    Gets prediction for a specific health record by its database ID.
    
    **Path Parameters:**
    - `record_id`: The integer ID from the health_data table
    
    **Example:** `GET /predict/42`
    """
    if not app_state["model_loaded"]:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    # Import here to avoid circular imports
    from supabase_client import supabase
    import os

    table_name = os.getenv("SUPABASE_TABLE_NAME", "health_data")

    try:
        response = (
            supabase
            .table(table_name)
            .select("*")
            .eq("id", record_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"No health record found with ID: {record_id}"
            )

        raw_data = response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Run prediction pipeline
    risk_level, probability = predict_diabetes_risk(
        raw_data, app_state["model"], app_state["scaler"]
    )
    risk_factors = identify_risk_factors(raw_data)
    recommendations = generate_recommendations(raw_data, risk_level)

    return PredictionResponse(
        user_data=HealthData(**{k: v for k, v in raw_data.items() if k in HealthData.model_fields}),
        prediction=PredictionResult(risk_level=risk_level, probability=probability),
        risk_factors=risk_factors,
        recommendations=recommendations,
    )


# ============================================================
# Global Exception Handler
# Catches any unhandled exceptions and returns clean JSON error
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"💥 Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        }
    )


# ============================================================
# Root Endpoint — API Welcome Message
# ============================================================
@app.get("/", tags=["System"])
async def root():
    """Welcome message with available endpoints."""
    return {
        "message": "🏥 Welcome to the AI Diabetes Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health_check": "GET /health",
            "model_status": "GET /model-status",
            "predict_latest": "GET /predict",
            "predict_by_name": "GET /predict?name=John",
            "predict_by_id": "GET /predict/{id}",
            "docs": "GET /docs",
        },
        "tech_stack": {
            "framework": "FastAPI",
            "database": "Supabase (PostgreSQL)",
            "ml_library": "Scikit-learn",
            "language": "Python 3.11+",
        }
    }


# ============================================================
# Run the App Directly (for development)
# In production, use: uvicorn main:app --host 0.0.0.0 --port 8000
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # Auto-reload on code changes (dev only)
        log_level="info",
    )
