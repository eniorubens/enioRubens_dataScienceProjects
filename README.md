# Data Science Projects

**Author:** [Enio Rubens](https://github.com/eniorubens)

A portfolio of end-to-end data science projects covering the full analytics spectrum — from descriptive statistics and inferential testing to predictive modeling and prescriptive optimization. Projects include reproducible notebooks, modular source code and automated test suites; the language status of each edition is documented in its own README.

---

## Projects

### 1. [ds_toolkit](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/871f29b92677490e01defe715740d30081c4f83f/ds_toolkit) — Reusable ML Library
![Status](https://img.shields.io/badge/status-active-brightgreen)

A modular Python toolkit providing reusable components shared across all projects.

**Key modules:**
- `opt_binary_clf_pipe` — Optuna-based pipeline optimizer for binary classification (encoding → feature selection → model → threshold)
- `multilang` — Static translation dictionary for bilingual (EN/PT) notebook output

**Technologies:** Optuna · scikit-learn · imbalanced-learn · category_encoders · MLflow

---

### 2. [Customer Churn Prediction](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/ef4fc19f71fde9561818eeaca80024985ea8c5a4/customer-churn-prediction) — End-to-End ML Pipeline
![Status](https://img.shields.io/badge/status-completed-blue)

Predicts customer churn (26.5% base rate) with **76.4% Recall Macro**, translating into an estimated **$3M annual revenue preserved** at **4.7x ROI**. Reframes ML as a business optimization problem.

**Methods:** Imbalanced classification · BalancedRandomForestClassifier · Threshold optimization · PhiK correlation · EDA · 90-day phased deployment roadmap

**Technologies:** scikit-learn · imbalanced-learn · Optuna · MLflow · FastAPI · pytest · category_encoders · PhiK

---

### 3. [Customer Lifetime Value](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/ef4fc19f71fde9561818eeaca80024985ea8c5a4/Customer%20Lifetime%20Value) — Probabilistic BTYD Modeling
![Status](https://img.shields.io/badge/status-completed-blue)

Estimates expected future revenue per customer over a **180-day horizon** using probabilistic Buy-Till-You-Die models with temporal holdout validation.

**Methods:** RFM feature engineering · BG/NBD model (purchase frequency & P(alive)) · Gamma-Gamma model (monetary value) · MAP & MCMC fitting · Customer segmentation on CLTV × P(alive) matrix

**Technologies:** pymc-marketing · Pandas · NumPy · scikit-learn · Flask API · pytest

---

### 4. [Customer Segmentation & Next Best Action](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/ef4fc19f71fde9561818eeaca80024985ea8c5a4/Customer_Segmentation_and_Next_Best_Action) — Prescriptive Analytics
![Status](https://img.shields.io/badge/status-completed-blue)

End-to-end pipeline from transactional data to actionable marketing decisions: **RFM → K-Means segmentation → Repurchase probability → Budget-constrained prescriptive optimization**, achieving **2.4x improvement in campaign ROI** vs. uniform campaigns.

**Methods:** K-Means clustering (Elbow + Silhouette, k=6) · Calibrated probabilistic classifier · Incentive simulation · Expected profit calculation · Budget optimizer

**Technologies:** scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · pytest · GitHub Actions CI

---

### 5. [Marketing Campaign Optimization and Retention Analytics](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/ef4fc19f71fde9561818eeaca80024985ea8c5a4/Marketing_Campaign_Optimization_and_Retention_Analytics_EN) — Full Analytics Pipeline
![Status](https://img.shields.io/badge/status-completed-blue)

A four-layer analytics pipeline — **descriptive → inferential → predictive → prescriptive** — for campaign performance analysis and user retention, with automated user classification into 6 action categories and multi-sheet Excel export.

**Methods:** Channel performance analysis · Z-test A/B testing · Supervised retention modeling · False positive / bias diagnosis · Financial ROI simulation · Automated recommendation system

**Technologies:** Python · Pandas · NumPy · scikit-learn · Statsmodels · Matplotlib · OpenPyXL

---

### 6. [Cross-Sell Association Rules](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/67340b649f62a1ace72e7ef38dc9bca064a09998/Cross_Sell_Association_Rules) — Market Basket Analysis
![Status](https://img.shields.io/badge/status-completed-blue)

Identifies genuine cross-selling opportunities in retail transactional data using Association Rules. The core insight: **high item frequency ≠ meaningful association** — whole milk and vegetables, the two most popular items, show lift of 0.77, proving popularity alone misleads. Rules are filtered by lift > 1 and Zhang's metric > 0.2 to surface statistically significant combinations.

**Methods:** Apriori algorithm · Frequent itemset mining · Multi-metric evaluation (Support, Confidence, Lift, Conviction, Zhang's metric) · One-hot encoding · Transaction aggregation

**Technologies:** Python · Pandas · NumPy · MLxtend · Matplotlib · Seaborn

---

### 7. [Seoul Bike Sharing Demand](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Bike-Sharing-Demand) — Temporal Demand Forecasting
![Status](https://img.shields.io/badge/status-completed-blue)

Forecasts hourly bicycle demand under normal operating conditions using nine years of mobility and meteorological data. A sealed temporal holdout confirmed CatBoost with **MAE 1,118.1** and **R² 0.839**; adaptive conformal calibration subsequently raised 90% interval coverage from **81.0% to 90.1%** while reducing mean interval width by **25.6%**.

**Methods:** Expanding meteorological-year CV · Dynamic estimator-specific preprocessing · Optuna · MLflow · CatBoost · Residual diagnostics · SHAP · Adaptive conformal inference · Operational replay

**Technologies:** Python · Pandas · scikit-learn · CatBoost · XGBoost · LightGBM · Optuna · MLflow · SHAP · pytest · Jupyter

---

## Skills Summary

| Area | Methods & Techniques |
|------|----------------------|
| **Supervised Learning** | Random Forest, CatBoost, Gradient Boosting, Logistic Regression, Threshold Optimization |
| **Probabilistic Modeling** | BG/NBD, Gamma-Gamma, MCMC, MAP, Adaptive Conformal Inference |
| **Unsupervised Learning** | K-Means Clustering, RFM Segmentation |
| **Prescriptive Analytics** | Budget Optimization, Expected Profit Simulation, Next Best Action |
| **Inferential Statistics** | Z-test, A/B Testing, Hypothesis Testing |
| **MLOps** | Optuna, MLflow, FastAPI, pytest, GitHub Actions CI |
| **Data Engineering** | Feature Pipelines, Encoding Strategies, Imbalanced Data Handling |
| **Market Basket Analysis** | Apriori, Association Rules, Lift, Conviction, Zhang's Metric |

## Core Technologies

Python · Pandas · NumPy · scikit-learn · imbalanced-learn · Optuna · MLflow · pymc-marketing · FastAPI · Flask · pytest · Matplotlib · Seaborn · Statsmodels · category_encoders · MLxtend · Jupyter

---

*Portuguese is the canonical language of newly developed notebooks; English editions are published as separate portfolio deliverables when completed.*
