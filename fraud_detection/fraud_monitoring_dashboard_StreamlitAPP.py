import streamlit as st
import joblib
import pandas as pd

# -------------------------
# Load model
# -------------------------
model = joblib.load("model.pkl")
threshold = joblib.load("threshold.pkl")

st.title("Fraud Detection System")

# -------------------------
# Feature engineering 
# -------------------------
def engineer_features(df):
    df = df.copy()

    df["Difference_Orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["Difference_Dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["error_balance_orig"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    )
    df["error_balance_dest"] = (
        df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    )

    df["is_balance_error_orig"] = (df["error_balance_orig"].abs() > 1).astype(int)
    df["is_balance_error_dest"] = (df["error_balance_dest"].abs() > 1).astype(int)

    return df


# -------------------------
# Input
# -------------------------
amount = st.number_input("Amount", 0.0)
oldbalanceOrg = st.number_input("Old Balance Org", 0.0)
newbalanceOrig = st.number_input("New Balance Orig", 0.0)
oldbalanceDest = st.number_input("Old Balance Dest", 0.0)
newbalanceDest = st.number_input("New Balance Dest", 0.0)
type_encoded = st.selectbox("Type", [0, 1, 2, 3, 4])


# -------------------------
# Predict
# -------------------------
if st.button("Predict"):

    # raw input
    data = pd.DataFrame([{
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "type_encoded": type_encoded
    }])

    # feature engineering
    data = engineer_features(data)

    # ensure correct order
    FEATURE_COLUMNS = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "Difference_Orig",
        "Difference_Dest",
        "error_balance_orig",
        "error_balance_dest",
        "is_balance_error_orig",
        "is_balance_error_dest",
        "type_encoded"
    ]

    data = data[FEATURE_COLUMNS]

    # prediction
    score = model.predict_proba(data)[0, 1]

    # decision
    if score >= threshold:
        label = "FRAUD"
    else:
        label = "NORMAL"

    # -------------------------
    # Output
    # -------------------------
    st.subheader("Result")

    st.write(f"Score: {score:.4f}")
    st.write(f"Threshold: {threshold:.4f}")

    if label == "FRAUD":
        st.error("FRAUD")
    else:
        st.success("NORMAL")