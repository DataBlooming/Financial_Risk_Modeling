import json
import time
import logging
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

# ------------------
# Logging setup
# ------------------
logging.basicConfig(
    filename="prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s")

# ------------------
# Load artifact（model + threshold + version）
# ------------------
artifact = joblib.load("artifact.pkl")

model = artifact["model"]
threshold = artifact["threshold"]
model_version = artifact.get("version", "v1")

# ------------------
# Feature engineering
# ------------------
def engineer_features(df):
    df = df.copy()

    # -----------------------------
    # Balance features
    # -----------------------------
    df["Difference_Orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["Difference_Dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["error_balance_orig"] = (df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"])
    df["error_balance_dest"] = (df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"])

    df["is_balance_error_orig"] = (df["error_balance_orig"].abs() > 1).astype(int)
    df["is_balance_error_dest"] = (df["error_balance_dest"].abs() > 1).astype(int)

    # -----------------------------
    # Type encoding
    # -----------------------------
    df['type_encoded'] = df['type_encoded'] if 'type_encoded' in df else df['type_encoded']

    return df


# ------------------
# Feature schema
# ------------------
FEATURE_COLUMNS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
       'newbalanceDest', 'Difference_Orig', 'Difference_Dest', 'error_balance_orig', 
       'error_balance_dest', 'is_balance_error_orig',
       'is_balance_error_dest', 'type_encoded']

# ------------------
# Business decision rule
# ------------------
def decision(score, high=0.95, low=0.7):
    if score > high:
        return "BLOCK"
    elif score > low:
        return "REVIEW"
    else:
        return "ALLOW"

# ------------------
# FastAPI app
# ------------------
app = FastAPI()

# ------------------
# Input schema
# ------------------
class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_encoded: int

# ------------------
# Prediction endpoint
# ------------------
@app.post("/predict")
def predict(data: Transaction):

    # 1. input → dataframe
    df = pd.DataFrame([data.dict()])
    
    # 1. feature engineering
    df = engineer_features(df)
    df = df[FEATURE_COLUMNS]

    # 2. model inference
    score = model.predict_proba(df)[0, 1]

    # 3. threshold decision
    is_fraud = int(score >= threshold)

    # 4. business decision
    action = decision(score)

    # 5. logging
    log_record = {
        "timestamp": time.time(),
        "model_version": model_version,
        "input": data.dict(),
        "score": float(score),
        "threshold": float(threshold),
        "is_fraud": is_fraud,
        "action": action
    }

    logging.info(json.dumps(log_record))

    # 6. response
    return {
        "score": float(score),
        "is_fraud": is_fraud,
        "action": action,
        "model_version": model_version
    }

# ------------------
# Run server (Optional)
# ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)