# Credit Risk Modeling (Machine Learning Project)

This project focuses on building a machine learning system for **credit risk prediction**, aiming to estimate the probability of a customer experiencing 90-day past due delinquency or worse.

The project demonstrates an end-to-end ML workflow including data preprocessing, feature engineering, model training, evaluation, and API-based deployment.

---

## Objective

The goal is to predict credit default risk using historical financial and behavioral features, enabling better risk assessment for lending decisions.

---

## Machine Learning Pipeline

The project follows a structured ML workflow:

1. Data Loading (credit dataset)
2. Data Cleaning & Missing Value Handling
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
   - Missing value indicators
   - Outlier treatment (capping at quantiles)
5. Train/Test Split (stratified sampling)
6. Feature Scaling (StandardScaler)
7. Model Training
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Voting Classifier (ensemble)
8. Model Evaluation
   - ROC-AUC Score
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
9. Model Selection based on performance
10. Model Persistence (`model.pkl`, `scaler.pkl`)
11. API Deployment (FastAPI inference service)

---

## Models Used

- Logistic Regression (baseline interpretable model)
- Random Forest (non-linear relationships)
- XGBoost (high-performance gradient boosting model)
- Voting Classifier (ensemble approach)

---

## Evaluation Metrics

Due to class imbalance, the following metrics are emphasized:

- ROC-AUC (primary metric)
- Precision
- Recall
- F1-score
- Confusion Matrix analysis

Special attention is given to **recall**, as missing a default case is more costly in credit risk scenarios.

---

## API Inference Service

A FastAPI-based service is implemented for real-time credit risk prediction.

### Features:
- REST API endpoint for inference
- Input validation using Pydantic
- Scaled feature transformation
- Probability-based prediction output
- Prediction logging with timestamp and model versioning

## Project Structure
credit_risk_modeling/
├── credit_risk_modeling_ml_pipeline.ipynb
├── credit_risk_fast_api.py
├── model.pkl
├── scaler.pkl
├── README.md

## 🛠 Tech Stack
- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- FastAPI
- Joblib
- Seaborn / Matplotlib
- Jupyter Notebook
