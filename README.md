# 📘 Customer Churn Prediction System

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
