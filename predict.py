def predict_diabetes(age, bmi, glucose):

    risk_score = 0

    if glucose > 140:
        risk_score += 40

    if bmi > 30:
        risk_score += 30

    if age > 45:
        risk_score += 20

    if risk_score >= 60:
        risk = "High Risk"
        advice = "Consult doctor immediately. Reduce sugar intake and start exercise."

    elif risk_score >= 30:
        risk = "Medium Risk"
        advice = "Improve diet, sleep, and physical activity."

    else:
        risk = "Low Risk"
        advice = "Maintain healthy lifestyle."

    return {
        "risk_score": risk_score,
        "risk_level": risk,
        "advice": advice
    }