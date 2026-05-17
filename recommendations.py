# ============================================================
# recommendations.py
# Intelligent Health Recommendations Engine
# Generates personalized prevention tips based on health values
# ============================================================

from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================
# CLINICAL THRESHOLDS
# These are standard medical reference values
# Used to determine if a value is "concerning"
# ============================================================

THRESHOLDS = {
    # Fasting plasma glucose levels (mg/dL)
    "glucose": {
        "normal": 100,        # Below 100 = Normal
        "prediabetes": 125,   # 100-125 = Prediabetes
        "diabetes": 126,      # 126+ = Diabetic range
    },
    # Body Mass Index (kg/m²)
    "bmi": {
        "normal": 24.9,       # Below 25 = Normal weight
        "overweight": 29.9,   # 25-29.9 = Overweight
        "obese": 30,          # 30+ = Obese
    },
    # HbA1c — 3-month average blood sugar (%)
    "hba1c": {
        "normal": 5.6,        # Below 5.7 = Normal
        "prediabetes": 6.4,   # 5.7-6.4 = Prediabetes
        "diabetes": 6.5,      # 6.5+ = Diabetic range
    },
    # Systolic blood pressure (mmHg)
    "blood_pressure": {
        "normal": 120,        # Below 120 = Normal
        "elevated": 129,      # 120-129 = Elevated
        "high": 130,          # 130+ = High (Hypertension Stage 1)
        "very_high": 140,     # 140+ = Hypertension Stage 2
    },
    # Total cholesterol (mg/dL)
    "cholesterol": {
        "desirable": 200,     # Below 200 = Desirable
        "borderline": 239,    # 200-239 = Borderline high
        "high": 240,          # 240+ = High
    },
    # Serum insulin (mU/L)
    "insulin": {
        "normal_max": 25,     # Normal fasting insulin
        "high": 25,           # Above 25 = Elevated
    },
    # Age risk threshold
    "age": {
        "risk": 45,           # 45+ increases diabetes risk
    }
}


# ============================================================
# STEP 7A: Identify Risk Factors
# Check which health values are outside normal ranges
# ============================================================

def identify_risk_factors(health_data: Dict[str, Any]) -> List[str]:
    """
    Analyzes each health metric and identifies concerning values.
    
    Args:
        health_data: Dictionary with the user's health measurements
        
    Returns:
        List of strings describing each risk factor found
    """
    risk_factors = []

    # ------- Glucose Check -------
    glucose = health_data.get("glucose")
    if glucose is not None:
        if glucose >= THRESHOLDS["glucose"]["diabetes"]:
            risk_factors.append(
                f"⚠️ Diabetic-range fasting glucose: {glucose} mg/dL (Normal: <100 mg/dL)"
            )
        elif glucose >= THRESHOLDS["glucose"]["normal"]:
            risk_factors.append(
                f"⚠️ Elevated fasting glucose (prediabetes range): {glucose} mg/dL"
            )

    # ------- BMI Check -------
    bmi = health_data.get("bmi")
    if bmi is not None:
        if bmi >= THRESHOLDS["bmi"]["obese"]:
            risk_factors.append(
                f"⚠️ Obesity detected: BMI {bmi:.1f} (Healthy range: 18.5–24.9)"
            )
        elif bmi >= THRESHOLDS["bmi"]["normal"]:
            risk_factors.append(
                f"⚠️ Overweight: BMI {bmi:.1f} (Healthy range: 18.5–24.9)"
            )

    # ------- HbA1c Check -------
    hba1c = health_data.get("hba1c")
    if hba1c is not None:
        if hba1c >= THRESHOLDS["hba1c"]["diabetes"]:
            risk_factors.append(
                f"⚠️ HbA1c in diabetic range: {hba1c}% (Normal: <5.7%)"
            )
        elif hba1c >= THRESHOLDS["hba1c"]["normal"]:
            risk_factors.append(
                f"⚠️ HbA1c in prediabetes range: {hba1c}% (Normal: <5.7%)"
            )

    # ------- Blood Pressure Check -------
    bp = health_data.get("blood_pressure")
    if bp is not None:
        if bp >= THRESHOLDS["blood_pressure"]["very_high"]:
            risk_factors.append(
                f"⚠️ Stage 2 Hypertension: {bp} mmHg (Optimal: <120 mmHg)"
            )
        elif bp >= THRESHOLDS["blood_pressure"]["high"]:
            risk_factors.append(
                f"⚠️ Stage 1 Hypertension: {bp} mmHg — increases cardiovascular risk"
            )
        elif bp >= THRESHOLDS["blood_pressure"]["normal"]:
            risk_factors.append(
                f"⚠️ Elevated blood pressure: {bp} mmHg"
            )

    # ------- Cholesterol Check -------
    cholesterol = health_data.get("cholesterol")
    if cholesterol is not None:
        if cholesterol >= THRESHOLDS["cholesterol"]["high"]:
            risk_factors.append(
                f"⚠️ High cholesterol: {cholesterol} mg/dL (Desirable: <200 mg/dL)"
            )
        elif cholesterol >= THRESHOLDS["cholesterol"]["desirable"]:
            risk_factors.append(
                f"⚠️ Borderline high cholesterol: {cholesterol} mg/dL"
            )

    # ------- Insulin Check -------
    insulin = health_data.get("insulin")
    if insulin is not None and insulin > THRESHOLDS["insulin"]["high"]:
        risk_factors.append(
            f"⚠️ Elevated insulin: {insulin} mU/L — may indicate insulin resistance"
        )

    # ------- Age Check -------
    age = health_data.get("age")
    if age is not None and age >= THRESHOLDS["age"]["risk"]:
        risk_factors.append(
            f"ℹ️ Age {int(age)}: Risk of Type 2 diabetes increases after age 45"
        )

    # If all values look normal
    if not risk_factors:
        risk_factors.append("✅ All measured health indicators are within normal ranges")

    logger.info(f"📊 Identified {len(risk_factors)} risk factor(s)")
    return risk_factors


# ============================================================
# STEP 7B: Generate Recommendations
# Create personalized prevention advice based on health values
# ============================================================

def generate_recommendations(
    health_data: Dict[str, Any],
    risk_level: str
) -> List[str]:
    """
    Generates intelligent, personalized prevention recommendations
    based on the user's specific health values.
    
    Args:
        health_data: Dictionary with health measurements
        risk_level: "Low Risk", "Medium Risk", or "High Risk"
        
    Returns:
        List of personalized recommendation strings
    """
    recommendations = []

    # ------- Glucose-based Recommendations -------
    glucose = health_data.get("glucose")
    if glucose is not None:
        if glucose >= THRESHOLDS["glucose"]["diabetes"]:
            recommendations.extend([
                "🍎 Immediately eliminate sugary drinks, white bread, and processed foods from your diet",
                "🩺 Consult an endocrinologist for diabetes management and medication review",
                "📊 Monitor your blood glucose daily using a home glucometer",
                "🥗 Follow a low-glycemic index (GI) diet — choose whole grains, legumes, and vegetables",
            ])
        elif glucose >= THRESHOLDS["glucose"]["normal"]:
            recommendations.extend([
                "🥤 Reduce sugary beverages (sodas, juices) — replace with water or unsweetened tea",
                "🍞 Switch to complex carbohydrates: whole grain bread, brown rice, oats",
                "⏰ Practice intermittent fasting (consult your doctor first) to improve insulin sensitivity",
            ])

    # ------- BMI-based Recommendations -------
    bmi = health_data.get("bmi")
    if bmi is not None:
        if bmi >= THRESHOLDS["bmi"]["obese"]:
            recommendations.extend([
                "🏃 Aim for at least 150 minutes of moderate exercise per week (30 min/day, 5 days)",
                "🥦 Follow a calorie-deficit diet: increase vegetables, reduce ultra-processed foods",
                "👨‍⚕️ Consider consulting a dietitian for a structured weight loss plan",
                "💪 Include strength training 2-3 times per week to improve insulin sensitivity",
            ])
        elif bmi >= THRESHOLDS["bmi"]["normal"]:
            recommendations.extend([
                "🚶 Add daily walks — even 20-30 minutes reduces diabetes risk significantly",
                "🥗 Replace processed snacks with whole foods: nuts, fruits, yogurt",
                "⚖️ Target losing 5-7% of body weight to dramatically reduce diabetes risk",
            ])

    # ------- HbA1c-based Recommendations -------
    hba1c = health_data.get("hba1c")
    if hba1c is not None:
        if hba1c >= THRESHOLDS["hba1c"]["diabetes"]:
            recommendations.extend([
                "🩺 Seek immediate medical evaluation — HbA1c in diabetic range requires treatment",
                "💊 Discuss Metformin or other diabetes medications with your doctor",
                "📅 Get HbA1c retested every 3 months to track progress",
            ])
        elif hba1c >= THRESHOLDS["hba1c"]["normal"]:
            recommendations.extend([
                "📅 Get HbA1c retested in 3-6 months to monitor your trend",
                "🍽️ Practice portion control — use the plate method (½ vegetables, ¼ protein, ¼ grains)",
            ])

    # ------- Blood Pressure-based Recommendations -------
    bp = health_data.get("blood_pressure")
    if bp is not None:
        if bp >= THRESHOLDS["blood_pressure"]["high"]:
            recommendations.extend([
                "🧂 Reduce sodium intake below 2,300 mg/day — avoid processed and fast foods",
                "🍌 Increase potassium-rich foods: bananas, sweet potatoes, spinach, beans",
                "🧘 Practice stress management: meditation, deep breathing, or yoga daily",
                "🚭 Quit smoking if applicable — it raises blood pressure and worsens diabetes risk",
                "💊 Discuss blood pressure medication with your doctor if lifestyle changes are insufficient",
            ])
        elif bp >= THRESHOLDS["blood_pressure"]["normal"]:
            recommendations.extend([
                "🧂 Moderate your salt intake — cook at home more often to control sodium",
                "🏊 Add aerobic exercise to lower blood pressure naturally",
            ])

    # ------- Cholesterol-based Recommendations -------
    cholesterol = health_data.get("cholesterol")
    if cholesterol is not None and cholesterol >= THRESHOLDS["cholesterol"]["desirable"]:
        recommendations.extend([
            "🐟 Eat fatty fish (salmon, mackerel, sardines) twice a week for omega-3 fatty acids",
            "🫒 Replace saturated fats with healthy fats: olive oil, avocados, nuts",
            "🌾 Increase soluble fiber intake: oatmeal, apples, beans help lower LDL cholesterol",
        ])

    # ------- General Recommendations (always included) -------
    general = [
        "💧 Drink 8-10 glasses of water daily to support metabolism and kidney function",
        "😴 Aim for 7-9 hours of quality sleep — poor sleep increases insulin resistance",
        "🩺 Schedule a comprehensive health check-up with your doctor every 6 months",
    ]

    # ------- Risk-Level Specific Additions -------
    if risk_level == "High Risk":
        recommendations.insert(0, "🚨 HIGH PRIORITY: Please consult a healthcare provider within the next 2 weeks")
        recommendations.append("📱 Consider joining a Diabetes Prevention Program (DPP) — clinically proven to reduce risk by 58%")
    elif risk_level == "Medium Risk":
        recommendations.insert(0, "⚡ You have modifiable risk factors — lifestyle changes can significantly reduce your risk")

    # Add general recommendations at the end
    recommendations.extend(general)

    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)

    logger.info(f"💡 Generated {len(unique_recommendations)} personalized recommendation(s)")
    return unique_recommendations
