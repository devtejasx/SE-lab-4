from fastapi import FastAPI
import joblib
import pandas as pd
import shap
import numpy as np

app = FastAPI()

# Load models
rf = joblib.load("rf.pkl")
xgb = joblib.load("xgb.pkl")
columns = joblib.load("columns.pkl")

explainer = shap.TreeExplainer(rf)


@app.get("/")
def home():
    return {"message": "API Running"}


@app.get("/predict")
def predict(age: int, dose: int, exposure: int):

    data = {
        "mutation_type": "intron22",
        "exon": 22,
        "severity": "severe",
        "age_first_treatment": age,
        "dose_intensity": dose,
        "exposure_days": exposure
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in columns:
        if col not in df:
            df[col] = 0

    df = df[columns]

    # Prediction
    p1 = rf.predict_proba(df)[0][1]
    p2 = xgb.predict_proba(df)[0][1]
    risk = (p1 + p2) / 2

    # SHAP
    shap_values = explainer.shap_values(df)
    shap_vals = np.array(shap_values).flatten()

    # Feature importance
    feature_importance = {}
    for i in range(len(df.columns)):
        feature_importance[df.columns[i]] = float(shap_vals[i])

    # Top feature
    top_index = np.argmax(np.abs(shap_vals))
    top_feature = df.columns[top_index]

    return {
        "risk_score": float(risk),
        "reason": str(top_feature),
        "importance": feature_importance
    }