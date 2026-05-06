# Customer Churn Prediction – End-to-End Machine Learning Project

## Overview
This project builds a complete machine learning pipeline to predict customer churn, focusing on **recall optimization**, which is critical in churn prevention scenarios.

The goal is not only to predict churn, but to **enable actionable business decisions** that preserve revenue.

---

## Problem Statement
Customer churn directly impacts revenue. Missing a churner (false negative) is significantly more costly than incorrectly flagging a non-churner.

Therefore, the model prioritizes:
- **Recall (macro)** over accuracy
- Business-oriented evaluation

---

## Methodology

### 1. Data Processing
- Handling categorical features with multiple encoding strategies:
  - TargetEncoder
  - CatBoostEncoder
  - Frequency / Count Encoding
- Numerical transformations:
  - PowerTransformer
  - Scaling & normalization
- Feature selection:
  - Recursive Feature Elimination (RFE)

---

### 2. Model Training
Multiple estimators evaluated:
- Logistic Regression (baseline)
- Random Forest
- Extra Trees
- XGBoost
- LightGBM (winner)

---

### 3. Optimization Strategy

#### Optuna Hyperparameter Optimization
- Objective aligned to **recall_macro**
- Early stopping applied to reduce training time

#### Threshold Tuning
- Decision threshold optimized using `predict_proba`
- Business-driven adjustment to improve recall

---

### 4. Model Evaluation

| Model Type | Recall (Test) |
|-----------|--------------|
| Baseline Logistic Regression | ~0.71 |
| Threshold Tuned Baseline | ~0.74 |
| Optimized Model (LightGBM) | **~0.756** |

---

## Key Insights

- The model shows **strong generalization**, with minimal gap between train and test recall.
- No significant overfitting observed.
- Threshold tuning alone delivers competitive performance at lower computational cost.
- Feature importance highlights key churn drivers:
  - Contract type
  - Tenure
  - Internet service
  - Total charges

---

## Business Impact

The model enables:
- Early identification of high-risk customers
- Targeted retention campaigns
- Revenue preservation strategies

> The real value of the model is not predicting churn —  
> but enabling timely action that prevents it.

---

## Tech Stack
- Python
- Scikit-learn
- LightGBM
- XGBoost
- Optuna
- Pandas / NumPy

---

## Next Steps
- Cost-sensitive optimization
- Uplift modeling for campaign effectiveness
- Real-time scoring pipeline

---

## Development Notes

Parts of this project were developed with the assistance of AI-based coding and writing tools for code refinement, structuring, documentation, and analytical brainstorming.

All analytical decisions, business interpretations, modeling choices, validation procedures, and final project integration were conducted and reviewed by the author.

Author : Enio Rubens  
Data Science & Analytics
