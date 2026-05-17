# Models Directory

Place your trained model files here:

- `diabetes_model.pkl` — The trained scikit-learn classifier
- `scaler.pkl` — The fitted StandardScaler/MinMaxScaler

## Important Notes

1. The model was trained with these 5 features IN THIS ORDER:
   - glucose
   - bmi
   - hba1c
   - age
   - blood_pressure

2. The scaler MUST be the same one used during training — it normalizes
   the input values using the same parameters (mean/std) as training.

3. Do NOT commit these files to Git if they contain sensitive training data.
   Store them in cloud storage (AWS S3, Google Cloud, etc.) for production.

## Expected Model Interface

Your model must support:
```python
model.predict_proba(X)  # Returns [[prob_class_0, prob_class_1], ...]
```

Compatible model types: LogisticRegression, RandomForestClassifier,
GradientBoostingClassifier, SVC (with probability=True), etc.
