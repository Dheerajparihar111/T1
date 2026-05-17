import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("diabetes.csv")

# Features
X = df[[
    "Glucose",
    "BMI",
    "Age",
    "BloodPressure"
]]

# Target
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier()

model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "models/diabetes_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model trained successfully")