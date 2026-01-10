# 💳 Credit Risk Delinquency Prediction System

**ML-Driven Financial Risk Analytics**

## 📌 Overview

This project implements an end-to-end credit risk delinquency prediction system designed to identify high-risk customers before they default. By leveraging machine learning, this solution enables financial institutions to move from reactive collections to proactive risk management.

The project covers the complete ML lifecycle: from **exploratory data analysis (EDA)** and **feature engineering** to **ensemble modeling** and **real-time API deployment** via Flask.

---

## 📉 Business Problem

Financial institutions face significant losses due to customer delinquency. Traditional rule-based systems often struggle with non-linear behavioral patterns, leading to missed risks or unnecessary credit denials.

**Objective:** Develop a predictive engine to determine the likelihood of a customer becoming delinquent based on financial, behavioral, and historical payment data.

---

## 📊 Dataset & Features

The system processes structured customer-level data with a focus on:

* **Financial Attributes:** Annual income, credit score, credit utilization ratios, and outstanding balances.
* **Payment Behavior:** History of on-time vs. late payments and frequency of missed payments.
* **Demographics:** Employment status, card types, and categorical indicators.

> **Note:** The dataset exhibits inherent **class imbalance**, which was addressed during the modeling phase to ensure the system accurately identifies the minority "at-risk" class.

---

## 🛠️ Methodology

### 1. Data Engineering & EDA

* Handled missing values and outliers to ensure data integrity.
* Performed feature encoding (One-Hot/Label) and scaling.
* Analyzed delinquency drivers through correlation matrices and distribution plots.

### 2. Modeling & Ensemble Learning

We evaluated a range of algorithms to balance interpretability with predictive power:

* **Logistic Regression:** Established a baseline for regulatory transparency.
* **Decision Trees:** Captured non-linear relationships.
* **XGBoost:** Utilized gradient boosting for high-performance classification.
* **Neural Networks:** Explored deep learning patterns in customer behavior.
* **Ensemble Strategy:** Combined models to improve robustness and reduce variance.

### 3. Evaluation Metrics

Standard accuracy is misleading in credit risk. We prioritized:

* **ROC-AUC:** Primary metric for ranking risk.
* **Precision-Recall / F1-Score:** To balance the cost of false positives vs. false negatives.

---

## 🚀 Deployment

The final model is serialized and served via a **Flask-based REST API**, allowing for real-time inference.

---

## 💻 Tech Stack

* **Language:** Python
* **Data Science:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost, TensorFlow/Keras
* **Deployment:** Flask, Gunicorn
* **Serialization:** Pickle / Joblib

---

## 📁 Project Structure

```text
├── data/
│   ├── raw/                 # Original data files
│   └── processed/           # Cleaned and engineered features
├── notebooks/
│   ├── EDA.ipynb            # Data exploration and visualization
│   └── Modeling.ipynb       # Model training and evaluation
├── models/
│   ├── xgboost_model.pkl    # Serialized model files
│   └── ensemble_model.pkl
├── app/
│   ├── app.py               # Flask API entry point
│   └── preprocessing.py     # Production inference pipeline
├── requirements.txt         # Project dependencies
└── README.md

```

---

## ⚙️ How to Run

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/credit-risk-delinquency-prediction.git
cd credit-risk-delinquency-prediction

```


2. **Install Dependencies**
```bash
pip install -r requirements.txt

```


3. **Start the API**
```bash
python app/app.py

```


*The API will be live at `http://127.0.0.1:5000`.*

---
