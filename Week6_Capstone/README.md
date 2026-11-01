# Customer Churn Prediction and Analysis Using Python

**Week 6 Capstone Project — Data Science Using Python**

## Overview
This project analyzes customer churn using the IBM Telco Customer Churn dataset and builds machine-learning classifiers to identify customers who may leave a telecommunications service.

## Objectives
- Inspect and clean customer data.
- Explore churn patterns across tenure, contract type, payment method, internet service, and monthly charges.
- Prepare features for machine learning.
- Train and compare classification models.
- Evaluate models using precision, recall, F1-score, confusion matrix, and ROC-AUC.
- Save a reusable model and preprocessing artifacts.

## Dataset
- Records: 7,043
- Columns: 21
- Target: `Churn`
- Identifier excluded from model inputs: `customerID`

Dataset source:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Key Findings
- Overall churn rate: **26.54%**.
- Average tenure: **17.98 months** for churned customers versus **37.57 months** for non-churned customers.
- Churn rate by contract: month-to-month **42.71%**, one-year **11.27%**, two-year **2.83%**.
- Electronic-check customers had an observed churn rate of **45.29%**.
- Fiber-optic customers had an observed churn rate of **41.89%**.

These are descriptive associations and do not establish causation.

## Models Evaluated
- Dummy Classifier
- Logistic Regression
- Random Forest
- Gradient Boosting

## Final Model: Logistic Regression
At the default probability threshold of 0.50:

| Metric | Score |
|---|---:|
| Accuracy | 74.02% |
| Precision | 50.69% |
| Recall | 78.61% |
| F1-score | 61.64% |
| ROC-AUC | 84.14% |

Logistic Regression was selected because it achieved the highest churn-class recall and F1-score among the evaluated models. Gradient Boosting had a marginally higher ROC-AUC (0.8422 versus 0.8414).

## Project Structure
```text
Week6_Capstone/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── Week6_Customer_Churn_Analysis.ipynb
├── models/
├── screenshots/
├── reports/
├── presentation/
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup and Run
1. Clone or download this repository.
2. Open a terminal in the project directory.
3. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Launch JupyterLab:
   ```bash
   jupyter lab
   ```
6. Open `notebooks/Week6_Customer_Churn_Analysis.ipynb` and run the cells in order.

## Methodology Notes
- Eleven blank `TotalCharges` values were converted to numeric missing values and filled with `0.0`; all 11 corresponded to customers with zero tenure.
- `Churn` was encoded as 0 = No and 1 = Yes.
- Categorical variables were one-hot encoded, producing 30 model features.
- Data was split using a stratified 80/20 train-test split (`random_state=42`).
- The scaler was fitted only on training data to avoid data leakage.
- The final model, scaler, and feature-column order were saved as joblib artifacts.

## Limitations
- This is a cross-sectional educational classification project, not a temporal forecasting model.
- Individual predictions can be incorrect; model outputs are risk estimates.
- Further validation, probability calibration, cost-based threshold selection, fairness checks, and monitoring would be required before operational deployment.

## Report
See `reports/Customer_Churn_Prediction_Final_Report.docx` for the project methodology, numerical findings, charts, model evaluation, limitations, and conclusion.

## Author
Add your name, institution, and contact details before publishing.
