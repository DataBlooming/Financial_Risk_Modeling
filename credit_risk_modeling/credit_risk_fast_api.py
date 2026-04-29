# Predict using API inference endpoint

import numpy as np
import joblib
import csv
import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

# Deploy endpoint
# ===========================
# Load solar model and scaler
# ===========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

MODEL_VERSION = "CREDIT_RISK_v1_2026_04"
LOG_FILE = "prediction_log.csv"

# ===========================
# Initialize log
# ===========================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "features",
            "prediction",
            "probability",
            "model_version"
        ])
        
# ===========================
# Request format
# ===========================

class InputData(BaseModel):
    features: list[float]


# ===========================
# Initialize FastAPI
# ===========================
app = FastAPI(title="Credit Risk API")


# ===========================
# Health Check
# ===========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ===========================
# Predict Endpoint
# ===========================
@app.post("/predict")
def predict(data: InputData):

    x = np.array(data.features).reshape(1, -1)
    x_scaled = scaler.transform(x)

    prob = model.predict_proba(x_scaled)[0][1]
    pred = int(prob > 0.5)

    # log
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            data.features,
            float(pred),
            float(prob),
            MODEL_VERSION
        ])

    return {
        "prediction": pred,
        "probability": float(prob),
        "model_version": MODEL_VERSION
    }


# =======================================
# Run Real-time Credit Risk Scoring API
# =======================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)