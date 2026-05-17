# ============================================================
# predict.py
# Machine Learning Prediction Module
# Loads the trained model and makes diabetes risk predictions
# ============================================================

import os
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================
# STEP 5A: Define Feature Order
# This MUST match the order features were in during model training
# If you trained with [glucose, bmi, hba1c, age, blood_pressure]
# then we must prepare them in the SAME order here
# ============================================================

# The 5 features the ML model was trained on (ORDER MATTERS!)
FEATURE_NAMES = [
    "glucose",          # Fasting blood glucose (mg/dL)
    "bmi",              # Body Mass Index (kg/m²)
    "hba1c",            # HbA1c percentage (%)
    "age",              # Age in years
    "blood_pressure",   # Systolic blood pressure (mmHg)
]

# Default "safe" values used when a field is missing
# These are approximate population averages — prevents crashes on null data
DEFAULT_VALUES = {
    "glucose": 100.0,         # Normal fasting glucose
    "bmi": 25.0,              # Normal BMI threshold
    "hba1c": 5.5,             # Normal HbA1c
    "age": 40.0,              # Middle adult age
    "blood_pressure": 120.0,  # Normal systolic BP
}


# ============================================================
# STEP 5B: Load Model and Scaler
# Load the pre-trained .pkl files once at startup
# ============================================================

def load_model_and_scaler():
    """
    Loads the trained diabetes ML model and feature scaler from disk.
    
    Returns:
        Tuple of (model, scaler) — both loaded sklearn objects
        
    Raises:
        FileNotFoundError: If .pkl files don't exist at specified paths
        Exception: If files are corrupt or incompatible
    """
    # Read file paths from environment variables
    model_path = os.getenv("MODEL_PATH", "models/diabetes_model.pkl")
    scaler_path = os.getenv("SCALER_PATH", "models/scaler.pkl")

    # Check model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Model file not found at: '{model_path}'\n"
            f"   Make sure diabetes_model.pkl is in the 'models/' folder"
        )

    # Check scaler file exists
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"❌ Scaler file not found at: '{scaler_path}'\n"
            f"   Make sure scaler.pkl is in the 'models/' folder"
        )

    try:
        # Load the trained RandomForest/LogisticRegression/etc. model
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
            logger.info(f"✅ Model loaded from: {model_path}")

        # Load the StandardScaler/MinMaxScaler used during training
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
            logger.info(f"✅ Scaler loaded from: {scaler_path}")

        return model, scaler

    except pickle.UnpicklingError as e:
        raise Exception(f"❌ Could not unpickle model/scaler files: {str(e)}")
    except Exception as e:
        raise Exception(f"❌ Error loading model files: {str(e)}")


# ============================================================
# STEP 4: Prepare Feature Array
# Extract & clean the 5 ML input features from health data
# ============================================================

def prepare_features(health_data: Dict[str, Any]) -> np.ndarray:
    """
    Extracts the 5 ML features from the health data dictionary.
    Handles missing values by substituting safe defaults.
    
    Args:
        health_data: Dictionary with health measurements from Supabase
        
    Returns:
        numpy array of shape (1, 5) ready for model.predict()
    """
    feature_values = []

    for feature_name in FEATURE_NAMES:
        raw_value = health_data.get(feature_name)

        if raw_value is None:
            # Missing value — use a safe default so we don't crash
            default = DEFAULT_VALUES[feature_name]
            logger.warning(
                f"⚠️ '{feature_name}' is missing — using default value: {default}"
            )
            feature_values.append(default)
        else:
            # Convert to float (handles int, string numbers, etc.)
            try:
                feature_values.append(float(raw_value))
            except (ValueError, TypeError):
                default = DEFAULT_VALUES[feature_name]
                logger.warning(
                    f"⚠️ '{feature_name}' has invalid value '{raw_value}' "
                    f"— using default: {default}"
                )
                feature_values.append(default)

    # Create a 2D numpy array: shape (1, 5) — 1 sample, 5 features
    feature_array = np.array([feature_values])
    logger.info(f"📐 Feature array prepared: {dict(zip(FEATURE_NAMES, feature_values))}")
    return feature_array


# ============================================================
# STEP 6: Make Prediction
# Scale features and run the ML model
# ============================================================

def determine_risk_level(probability: float) -> str:
    """
    Converts a probability score into a human-readable risk level.
    
    Risk Logic:
        < 0.30 → Low Risk   (green zone)
        < 0.70 → Medium Risk (yellow zone)
        >= 0.70 → High Risk  (red zone)
    
    Args:
        probability: Float between 0.0 and 1.0
        
    Returns:
        One of: "Low Risk", "Medium Risk", "High Risk"
    """
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.70:
        return "Medium Risk"
    else:
        return "High Risk"


def predict_diabetes_risk(
    health_data: Dict[str, Any],
    model,
    scaler
) -> Tuple[str, float]:
    """
    Core prediction function:
    1. Extracts features from health data
    2. Scales features using the trained scaler
    3. Runs the ML model to get probability
    4. Converts probability to risk level
    
    Args:
        health_data: Dictionary with health metrics
        model: Loaded sklearn model (diabetes_model.pkl)
        scaler: Loaded sklearn scaler (scaler.pkl)
        
    Returns:
        Tuple of (risk_level: str, probability: float)
    """
    # Step A: Prepare feature array from health data
    features = prepare_features(health_data)
    logger.info(f"📊 Raw features: {features}")

    # Step B: Scale the features using the same scaler used during training
    # This is critical — raw values must be normalized the same way as training data
    scaled_features = scaler.transform(features)
    logger.info(f"📏 Scaled features: {scaled_features}")

    # Step C: Get probability from the model
    # predict_proba returns [[prob_class_0, prob_class_1]]
    # class 1 = diabetic, so we want index [0][1]
    probabilities = model.predict_proba(scaled_features)
    diabetes_probability = float(probabilities[0][1])
    logger.info(f"🎯 Raw probability: {diabetes_probability:.4f}")

    # Step D: Round to 2 decimal places for clean display
    diabetes_probability = round(diabetes_probability, 4)

    # Step E: Determine risk level based on probability thresholds
    risk_level = determine_risk_level(diabetes_probability)
    logger.info(f"🚦 Risk Level: {risk_level} (probability: {diabetes_probability})")

    return risk_level, diabetes_probability
