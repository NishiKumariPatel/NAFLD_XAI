from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap

# ==============================
# LOAD MODEL, SCALER, FEATURES
# ==============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")  # features used during training

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

app = FastAPI(title="DibLiv NAFLD Risk API with Robust XAI")

# ==============================
# INPUT DATA MODEL
# ==============================
class PatientInput(BaseModel):
    age: float
    gender: int        # not used in model
    bmi: float
    waist: float
    ast: float
    alt: float
    glucose: float
    hba1c: float
    triglycerides: float
    hdl: float
    ldl: float
    insulin: float     # not used in model
    homa: float        # not used in model

# ==============================
# PREDICTION ENDPOINT
# ==============================
@app.post("/predict")
def predict(data: PatientInput):
    # Map JSON keys to model features
    input_df = pd.DataFrame([[
        data.age,
        data.bmi,
        data.waist,
        data.ast,
        data.alt,
        data.triglycerides,
        data.hdl,
        data.ldl,
        200,    # total_cholesterol placeholder
        data.glucose,
        data.hba1c,
        120,    # systolic placeholder
        80      # diastolic placeholder
    ]], columns=feature_columns)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    risk_level = "High Risk" if prediction == 1 else "Low Risk"

    # ==============================
    # SHAP EXPLANATION (Robust)
    # ==============================
    shap_values = explainer.shap_values(input_scaled)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        class1_shap = shap_values[1]  # contributions for "High Risk"
    else:
        class1_shap = shap_values

    # Flatten any shape safely
    class1_shap_flat = class1_shap.flatten().tolist()

    # Map feature contributions
    feature_contributions = {
        col: round(float(val), 3)
        for col, val in zip(feature_columns, class1_shap_flat)
    }

    # ==============================
    # RETURN RESULTS
    # ==============================
    return {
        "fibrosis_prediction": int(prediction),
        "probability": round(float(probability), 3),
        "risk_level": risk_level,
        "feature_contributions": feature_contributions,
        "note": "This is a screening tool. Consult a medical professional."
    }





