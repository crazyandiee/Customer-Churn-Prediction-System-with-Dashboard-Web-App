# 📘 Customer Churn Prediction System
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-orange)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-lightgrey)
![App](https://img.shields.io/badge/Framework-Streamlit-brightgreen?logo=streamlit)
![Deploy](https://img.shields.io/badge/Deployment-Render%20%7C%20HuggingFace-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 🧩 Project Overview

This project aims to predict customer churn for a subscription-based telecom business using machine learning techniques.
The goal is to identify customers likely to discontinue service, analyze the factors driving churn, and provide actionable insights to reduce attrition.

---

## 🧠 Business Problem

Customer churn directly impacts a company’s recurring revenue and long-term profitability.
By building a predictive model, businesses can:

* Identify high-risk customers early
* Target them with retention campaigns
* Estimate potential cost savings from churn prevention

---

## 🧰 Tech Stack

| Layer                      | Tools                                   |
| -------------------------- | --------------------------------------- |
| **Data Handling**          | Python, Pandas, NumPy                   |
| **Modeling**               | Scikit-learn, XGBoost, RandomForest     |
| **Explainability**         | SHAP (feature importance)               |
| **Visualization**          | Matplotlib, Seaborn                     |
| **Versioning**             | Git + GitHub                            |
| **Deployment (Next Step)** | Streamlit + Render / HuggingFace Spaces |

---

## 📂 Project Structure
```
churn-prediction/
│
├── data/
│   ├── raw/                                      # Original source data
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Raw dataset
│   │
│   ├── processed/                                # Cleaned datasets
│   └── telco_cleaned.csv                         # Final cleaned dataset
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb                    # Data preparation & feature engineering
│   ├── 02_model_training.ipynb                   # Model training & evaluation
│   └── 03_model_interpretation.ipynb             # SHAP explainability
│
├── models/
│   └── churn_best_model.pkl                      # Trained best model (saved)
│
├── streamlit_app.py                              # Deployed web app
├── requirements.txt                              # Required packages
└── README.md
```

---

## 📊 Dataset Information

### Dataset: Telco Customer Churn (Kaggle)
Source: Telco Customer Churn Dataset (Kaggle)
Rows: 7,043 customers
Target Variable: Churn (Yes = 1, No = 0)

Key columns:
* tenure → months with the company
* Contract → type of subscription
* PaymentMethod → billing mode
* MonthlyCharges, TotalCharges → spending behavior
* TechSupport, OnlineSecurity, StreamingTV → service features

---

## 🚀 Model Training & Evaluation

Models tested:
* Logistic Regression (baseline)
* Random Forest
* XGBoost (best performer)

Metrics used:
* Accuracy, Precision, Recall, F1-Score, ROC-AUC
* 🏁 Best Model: XGBoost — ROC-AUC ≈ 0.85

---

## 🔍 Explainability (SHAP)

Used SHAP for:
* Global feature importance
* Local explanations (why each customer is likely to churn)

### Top churn drivers:
* Contract type → Month-to-month increases churn
* Short tenure → Strong churn indicator
* Electronic check payments → Higher churn risk
* Lack of tech support → Higher churn probability
* Multiple services → Lower churn risk

---
## 🌐 Streamlit Web App

Interactive web app for real-time churn prediction.

### Features:
* CSV upload support
* Instant churn probability scoring
* Risk segmentation: Low / Medium / High / Very High
* Downloadable results
* Visual churn distribution chart

Local Run:
pip install -r requirements.txt
streamlit run streamlit_app.py

---

## 📸 App Preview


---

## 💾 Requirements
* streamlit
* pandas
* numpy
* scikit-learn
* xgboost
* joblib



