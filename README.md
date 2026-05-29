# Data Science Projects

**Author:** [Enio Rubens](https://github.com/eniorubens)

A portfolio of end-to-end data science projects covering the full analytics spectrum — from descriptive statistics and inferential testing to predictive modeling and prescriptive optimization. All projects are production-ready, documented in English and Portuguese, and include test suites.

---

## Projects

### 1. [ds_toolkit](https://github.com/eniorubens/ds_toolkit) — Reusable ML Library
![Status](https://img.shields.io/badge/status-active-brightgreen)

A modular Python toolkit providing reusable components shared across all projects.

**Key modules:**
- `opt_binary_clf_pipe` — Optuna-based pipeline optimizer for binary classification (encoding → feature selection → model → threshold)
- `multilang` — Static translation dictionary for bilingual (EN/PT) notebook output

**Technologies:** Optuna · scikit-learn · imbalanced-learn · category_encoders · MLflow

---

### 2. [Customer Churn Prediction v3](https://github.com/eniorubens/Customer-Churn-Prediction-v3) — End-to-End ML Pipeline
![Status](https://img.shields.io/badge/status-completed-blue)

Predicts customer churn (26.5% base rate) with **76.4% Recall Macro**, translating into an estimated **$3M annual revenue preserved** at **4.7x ROI**. Reframes ML as a business optimization problem.

**Methods:** Imbalanced classification · BalancedRandomForestClassifier · Threshold optimization · PhiK correlation · EDA · 90-day phased deployment roadmap

**Technologies:** scikit-learn · imbalanced-learn · Optuna · MLflow · FastAPI · pytest · category_encoders · PhiK

---

### 3. [Customer Lifetime Value](https://github.com/eniorubens/Customer-Lifetime-Value) — Probabilistic BTYD Modeling
![Status](https://img.shields.io/badge/status-completed-blue)

Estimates expected future revenue per customer over a **180-day horizon** using probabilistic Buy-Till-You-Die models with temporal holdout validation.

**Methods:** RFM feature engineering · BG/NBD model (purchase frequency & P(alive)) · Gamma-Gamma model (monetary value) · MAP & MCMC fitting · Customer segmentation on CLTV × P(alive) matrix

**Technologies:** pymc-marketing · Pandas · NumPy · scikit-learn · Flask API · pytest

---

### 4. [Customer Segmentation & Next Best Action v2](https://github.com/eniorubens/Customer_Segmentation_and_Next_Best_Action_v2) — Prescriptive Analytics
![Status](https://img.shields.io/badge/status-completed-blue)

End-to-end pipeline from transactional data to actionable marketing decisions: **RFM → K-Means segmentation → Repurchase probability → Budget-constrained prescriptive optimization**, achieving **2.4x improvement in campaign ROI** vs. uniform campaigns.

**Methods:** K-Means clustering (Elbow + Silhouette, k=6) · Calibrated probabilistic classifier · Incentive simulation · Expected profit calculation · Budget optimizer

**Technologies:** scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · pytest · GitHub Actions CI

---

### 5. [Marketing Campaign Optimization and Retention Analytics](https://github.com/eniorubens/Marketing-Campaign-Optimization-and-Retention-Analytics) — Full Analytics Pipeline
![Status](https://img.shields.io/badge/status-completed-blue)

A four-layer analytics pipeline — **descriptive → inferential → predictive → prescriptive** — for campaign performance analysis and user retention, with automated user classification into 6 action categories and multi-sheet Excel export.

**Methods:** Channel performance analysis · Z-test A/B testing · Supervised retention modeling · False positive / bias diagnosis · Financial ROI simulation · Automated recommendation system

**Technologies:** Python · Pandas · NumPy · scikit-learn · Statsmodels · Matplotlib · OpenPyXL

---

### 6. [Obsidian AI Wiki](https://github.com/eniorubens/Obsidian) — Personal Knowledge Base
![Status](https://img.shields.io/badge/status-active-brightgreen)

A structured, LLM-maintained personal wiki for accumulating AI knowledge from papers, articles, and tutorials. Organized as a growing graph of interlinked pages (models, techniques, papers, people, datasets, analyses) with three operational modes: **Ingest → Query → Lint**.

**Technologies:** Markdown · Obsidian vault format · CLAUDE.md schema

---

## Skills Summary

| Area | Methods & Techniques |
|------|----------------------|
| **Supervised Learning** | Random Forest, Gradient Boosting, Logistic Regression, Threshold Optimization |
| **Probabilistic Modeling** | BG/NBD, Gamma-Gamma, MCMC, MAP |
| **Unsupervised Learning** | K-Means Clustering, RFM Segmentation |
| **Prescriptive Analytics** | Budget Optimization, Expected Profit Simulation, Next Best Action |
| **Inferential Statistics** | Z-test, A/B Testing, Hypothesis Testing |
| **MLOps** | Optuna, MLflow, FastAPI, pytest, GitHub Actions CI |
| **Data Engineering** | Feature Pipelines, Encoding Strategies, Imbalanced Data Handling |

## Core Technologies

Python · Pandas · NumPy · scikit-learn · imbalanced-learn · Optuna · MLflow · pymc-marketing · FastAPI · Flask · pytest · Matplotlib · Seaborn · Statsmodels · category_encoders · Jupyter

---

*All projects include bilingual notebooks (English / Portuguese) and production-ready APIs or exports.*
