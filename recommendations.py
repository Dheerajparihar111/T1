def generate_recommendations(data, risk_level):

    recommendations = []

    glucose = data["glucose"]
    bmi = data["bmi"]

    if glucose > 140:
        recommendations.append(
            "Reduce sugary drinks and sweets."
        )

    if bmi > 25:
        recommendations.append(
            "Exercise at least 30 minutes daily."
        )

    recommendations.extend([
        "Drink more water.",
        "Sleep 7-8 hours daily.",
        "Eat more vegetables.",
        "Avoid junk food."
    ])

    return recommendations