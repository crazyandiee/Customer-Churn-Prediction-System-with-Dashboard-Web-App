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
└── README.md
```

---

## 📊 Dataset Information

### Dataset: Telco Customer Churn (Kaggle)
Records: 7,043 customers
Features: 21 (demographics, services, contracts, payments)
Target: Churn (Yes = 1, No = 0)

### Key Columns:
* tenure → months with company
* Contract → month-to-month, one-year, two-year
* PaymentMethod → electronic check, mailed check, etc.
* MonthlyCharges, TotalCharges → billing amounts
* InternetService, TechSupport, StreamingTV, etc.
