import streamlit as st
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt

# -------------------------
# 1. Load model / threshold
# -------------------------
model = joblib.load("model.pkl")
threshold = joblib.load("threshold.pkl")

st.title("Fraud Detection Monitoring Dashboard")


# -------------------------
# 2. Load logs
# -------------------------
def load_logs(path="prediction.log"):
    data = []

    try:
        with open(path, "r") as f:
            for line in f:
                try:
                    log = line.strip().split(" - ", 1)
                    if len(log) == 2:
                        record = json.loads(log[1])
                        data.append(record)
                except:
                    pass
    except FileNotFoundError:
        return pd.DataFrame()

    return pd.DataFrame(data)


df = load_logs()

if df.empty:
    st.warning("No log data found yet.")
    st.stop()


# -------------------------
# 3. Action mapping 
# -------------------------
def get_action(score):
    if score > 0.95:
        return "BLOCK"
    elif score > threshold:
        return "REVIEW"
    else:
        return "ALLOW"


df["action"] = df["score"].apply(get_action)


# -------------------------
# 4. Overview
# -------------------------
st.subheader("Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", len(df))
col2.metric("Avg Score", round(df["score"].mean(), 4))
col3.metric("Fraud Rate", round((df["action"] == "BLOCK").mean(), 4))


# -------------------------
# 5. Action distribution
# -------------------------
st.subheader("Action Distribution")
st.bar_chart(df["action"].value_counts())

# -------------------------
# 6. Score distribution
# -------------------------
st.subheader("Score Distribution")
st.histogram = df["score"].hist()
st.write("Score Histogram")
st.pyplot()

# -------------------------
# 7. Score trend
# -------------------------
st.subheader("Score Trend")
st.line_chart(df["score"])


# -------------------------
# 8. Filter view
# -------------------------
st.subheader("Filtered Logs")

option = st.selectbox(
    "Filter by action",
    ["ALL", "ALLOW", "REVIEW", "BLOCK"]
)

if option != "ALL":
    filtered = df[df["action"] == option]
else:
    filtered = df

st.dataframe(filtered.tail(50))


# -------------------------
# 9. Explain decision 
# -------------------------
st.subheader("Decision Explanation (Sample)")

if len(df) > 0:
    sample = df.iloc[-1]

    st.write(f"Score: {sample['score']:.4f}")
    st.write(f"Threshold (REVIEW): {threshold}")
    st.write(f"Final Action: {sample['action']}")

    if sample["action"] == "BLOCK":
        st.error("High risk transaction detected → BLOCK")
    elif sample["action"] == "REVIEW":
        st.warning("Medium risk → requires manual review")
    else:
        st.success("Low risk → allow transaction")


# -------------------------
# 10. Raw logs
# -------------------------
st.subheader("Recent Predictions")
st.dataframe(df.tail(20))


# -------------------------
# 11. Model info
# -------------------------
st.subheader("Model Info")
st.write(f"Model: {type(model).__name__}")
st.write(f"Threshold: {threshold}")