# Fraud Detection ML Pipeline (End-to-End System)

##  Project Overview

This project builds an end-to-end machine learning system for detecting fraudulent online payment transactions.  
It leverages transaction-level financial data to identify abnormal behavior and support real-time risk decision-making.

The system includes the full ML lifecycle: data processing, feature engineering, model training, evaluation, threshold optimization, deployment, and monitoring.

---

##  Objectives

- Detect fraudulent transactions with high accuracy
- Build a robust end-to-end ML pipeline
- Enable real-time inference via API
- Provide system-level monitoring and observability

---

##  Project Structure
fraud_detection_project/
│
├── fraud_detection_ml_pipeline.ipynb   # End-to-end ML pipeline (training + evaluation)
│
├── api.py                              # FastAPI service for real-time inference
│
├── fraud_monitoring_dashboard.py       # Streamlit monitoring dashboard
│
├── model.pkl                           # Trained Random Forest model
│
├── threshold.pkl                       # Optimal decision threshold
│
├── prediction.log                      # Prediction logs for monitoring
│
└── README.md                           # Project documentation

---

##  Workflow Overview

1. Data Understanding & Preparation  
2. Feature Engineering  
3. Model Training (Random Forest, XGBoost)  
4. Model Evaluation (ROC-AUC, PR-AUC, Confusion Matrix)  
5. Post-model Optimization (Threshold Tuning)  
6. Model Interpretability (Feature Importance)  
7. End-to-End Pipeline Execution  
8. Experiment Tracking (MLflow Logging)  
9. Real-time Inference System  
10. Model Monitoring & Logging  
11. FastAPI Deployment  
12. Streamlit Dashboard (Monitoring & Observability)

---

##  Model Performance

Two tree-based models were evaluated:

### Random Forest (Best Model)
- ROC-AUC: 0.9996  
- PR-AUC: 0.9975  

### XGBoost
- ROC-AUC: 0.9977  
- PR-AUC: 0.9817  

Random Forest was selected as the final model due to superior ROC-AUC performance.

---

##  Threshold Optimization

- Optimal threshold: **0.73**
- Converts predicted probabilities into binary fraud decisions
- Enables risk-based action mapping

---

##  Feature Importance

Key drivers of fraud detection:

- Balance inconsistency in origin account  
- Transaction-induced balance difference  
- Balance error indicators  

These features capture abnormal financial behavior patterns commonly associated with fraud.

---

##  System Output

The system produces:

- Fraud probability score  
- Decision output:
  - ALLOW
  - REVIEW
  - BLOCK

---

##  System Architecture

- **Model Serving:** FastAPI  
- **Monitoring Dashboard:** Streamlit  
- **Logging:** Prediction log tracking  
- **Experiment Tracking:** MLflow  

---

##  Monitoring & Observability

- Real-time prediction logging  
- Score distribution tracking  
- Action distribution monitoring  
- System-level observability dashboard

---

##  How to Run

### 1. Start API
```bash
uvicorn api:app --reload

### 2. Start Dashboard
streamlit run fraud_monitoring_dashboard.py
