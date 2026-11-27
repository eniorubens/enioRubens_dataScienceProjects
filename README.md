# Data Science Projects

[![Tests](https://github.com/eniorubens/enioRubens_dataScienceProjects/actions/workflows/tests.yml/badge.svg)](https://github.com/eniorubens/enioRubens_dataScienceProjects/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Author: [Enio Rubens](https://github.com/eniorubens)

Language: English | [Português (Brasil)](README.pt-BR.md)

A portfolio of end-to-end data science projects that range from descriptive statistics and inferential testing to predictive modeling and prescriptive optimization. Projects include reproducible notebooks, modular source code and automated test suites; the language status of each edition is documented in its own README.

---

## Portfolio approach

The analytical projects combine business framing, modular implementation, evidence-based validation and an executive communication layer. Each report makes the supported claims, limitations and reproduction path explicit; deployment readiness is stated per project rather than assumed.

## Quick access

| Project | Domain | Executive report | Code and tests |
| :--- | :--- | :--- | :--- |
| Customer Churn Prediction | Retention / classification | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/) | [Repository](./customer-churn-prediction) |
| Customer Lifetime Value | Probabilistic marketing | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/) | [Repository](./Customer%20Lifetime%20Value) |
| Customer Segmentation & Next Best Action | Segmentation / prescription | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/) | [Repository](./Customer_Segmentation_and_Next_Best_Action) |
| Marketing Campaign Optimization | Full analytics pipeline | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) | [Repository](./Marketing_Campaign_Optimization_and_Retention_Analytics) |
| Cross-Sell Association Rules | Market basket analysis | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/) | [Repository](./Cross_Sell_Association_Rules) |
| Seoul Bike Sharing Demand | Time series / conformal inference | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Bike-Sharing-Demand/) | [Repository](./Bike-Sharing-Demand) |
| Marketing Campaign Uplift Modeling | Causal targeting | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/) | [Repository](./Marketing_Promotion_Campaign_Uplift_Modelling) |
| Consumer Complaint Intelligence | Imbalanced NLP | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Consumer_Complaint_Intelligence/) | [Repository](./Consumer_Complaint_Intelligence) |
| NASA C-MAPSS | Predictive maintenance | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/) | [Repository](./CMAPSS_Predictive_Maintenance) |
| Model Risk & Responsible ML | Model-risk governance | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/ModelRisk_ResponsibleML/) | [Repository](./ModelRisk_ResponsibleML) |
| E-commerce Delivery Network & SLA Risk | Delivery risk / temporal ML | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/E-commerce_Delivery_Network_SLA_Risk/) | [Repository](./E-commerce_Delivery_Network_SLA_Risk) |
| Delivery Routing & Time-Window Optimization | Logistics / prescriptive optimization | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Delivery_Routing_Time_Window_Optimization/en.html) | [Repository](./Delivery_Routing_Time_Window_Optimization) |
| Claim Frequency per Policy | Insurance / count regression | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Claim_Frequency_per_Policy/en.html) | [Repository](./Claim_Frequency_per_Policy) |
| Retail Pricing & Promotion (Multilevel) | Retail / multilevel models | [View report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Dunnhumby_Breakfast-at-the-Frat/en.html) | [Repository](./Dunnhumby_Breakfast-at-the-Frat) |
| ds_toolkit | Reusable ML library | [Library README](./ds_toolkit) | [Code](./ds_toolkit) |

---

## Projects

### 1. [ds_toolkit](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/ds_toolkit)
![Status](https://img.shields.io/badge/status-active-brightgreen)

A modular Python toolkit providing reusable components shared across all projects. It contains two modules: `opt_binary_clf_pipe`, an Optuna-based pipeline optimizer for binary classification that covers encoding, feature selection, model and threshold, and `multilang`, a static translation dictionary for bilingual (EN/PT) notebook output.

Technologies: Optuna · scikit-learn · imbalanced-learn · category_encoders · MLflow

---

### 2. [Customer Churn Prediction](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/customer-churn-prediction)
![Status](https://img.shields.io/badge/status-completed-blue)

Predicts customer churn (26.6% base rate) with 76.4% Recall Macro and 83.5% ROC AUC on the current hold-out artifact. Revenue preservation and ROI remain illustrative, not causally validated.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/README.pt-BR.md)

Methods: Imbalanced classification · BalancedRandomForestClassifier · Threshold optimization · PhiK correlation · EDA · 90-day phased deployment roadmap

Technologies: scikit-learn · imbalanced-learn · Optuna · MLflow · FastAPI · pytest · category_encoders · PhiK

---

### 3. [Customer Lifetime Value](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Customer%20Lifetime%20Value)
![Status](https://img.shields.io/badge/status-completed-blue)

Estimates expected future revenue per customer over a 180-day horizon using probabilistic Buy-Till-You-Die models with temporal holdout validation.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer%20Lifetime%20Value/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer%20Lifetime%20Value/README.pt-BR.md)

Methods: RFM feature engineering · BG/NBD model (purchase frequency & P(alive)) · Gamma-Gamma model (monetary value) · MAP & MCMC fitting · Customer segmentation on CLTV × P(alive) matrix

Technologies: pymc-marketing · Pandas · NumPy · scikit-learn · Flask API · pytest

---

### 4. [Customer Segmentation & Next Best Action](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Customer_Segmentation_and_Next_Best_Action)
![Status](https://img.shields.io/badge/status-completed-blue)

End-to-end pipeline from transactional data to actionable marketing decisions: RFM, K-Means segmentation, repurchase probability and budget-constrained prescriptive optimization. The project reports expected-profit simulations under explicit assumptions; causal uplift and superiority over a uniform campaign are not established.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer_Segmentation_and_Next_Best_Action/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer_Segmentation_and_Next_Best_Action/README.pt-BR.md)

Methods: K-Means clustering (Elbow + Silhouette, k=6) · Calibrated probabilistic classifier · Incentive simulation · Expected profit calculation · Budget optimizer

Technologies: scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · pytest

---

### 5. [Marketing Campaign Optimization and Retention Analytics](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Marketing_Campaign_Optimization_and_Retention_Analytics)
![Status](https://img.shields.io/badge/status-completed-blue)

A four-layer analytics pipeline (descriptive, inferential, predictive and prescriptive) for campaign performance analysis and user retention, with automated user classification into 6 action categories and multi-sheet Excel export.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/README.pt-BR.md)

Methods: Channel performance analysis · Z-test A/B testing · Supervised retention modeling · False positive / bias diagnosis · Financial ROI simulation · Automated recommendation system

Technologies: Python · Pandas · NumPy · scikit-learn · Statsmodels · Matplotlib · OpenPyXL

---

### 6. [Cross-Sell Association Rules](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Cross_Sell_Association_Rules)
![Status](https://img.shields.io/badge/status-completed-blue)

Identifies cross-selling opportunities in retail transactional data using association rules. High item frequency does not imply meaningful association: whole milk and vegetables, the two most popular items, show a lift of 0.77. Rules are filtered by lift > 1 and Zhang's metric > 0.2 and are not causal evidence.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Cross_Sell_Association_Rules/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Cross_Sell_Association_Rules/README.pt-BR.md)

Methods: Apriori algorithm · Frequent itemset mining · Multi-metric evaluation (support, confidence, lift, conviction, Zhang's metric) · One-hot encoding · Transaction aggregation

Technologies: Python · Pandas · NumPy · MLxtend · Matplotlib · Seaborn

---

### 7. [Seoul Bike Sharing Demand](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Bike-Sharing-Demand) (featured)
![Status](https://img.shields.io/badge/status-completed-blue)

Forecasts hourly bicycle demand under normal operating conditions using nine years of mobility and meteorological data. A sealed temporal holdout confirmed CatBoost with MAE 1,118.1 and R² 0.839. On the development folds, adaptive conformal calibration raised 90% interval coverage from 81.0% to 90.1% while reducing mean interval width by 25.6%.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Bike-Sharing-Demand/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Bike-Sharing-Demand/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Bike-Sharing-Demand/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Bike-Sharing-Demand/README.pt-BR.md)

Methods: Expanding meteorological-year CV · Dynamic estimator-specific preprocessing · Optuna · MLflow · CatBoost · Residual diagnostics · SHAP · Adaptive conformal inference · Operational replay

Technologies: Python · Pandas · scikit-learn · CatBoost · XGBoost · LightGBM · Optuna · MLflow · SHAP · pytest · Jupyter

---

### 8. [Marketing Campaign Uplift Modeling](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Marketing_Promotion_Campaign_Uplift_Modelling) (featured)
![Status](https://img.shields.io/badge/status-completed-blue)

Estimates who should receive an e-mail because the treatment changes their outcome, using the randomized Hillstrom experiment. The bilingual workflow compares response targeting, meta-learners, causal forests, and uplift trees, then evaluates budget-constrained policies and ROI scenarios. In the pre-registered sealed test, the primary uplift candidate did not demonstrate superiority over the baseline, so the project recommends a prospective randomized pilot rather than deployment.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Promotion_Campaign_Uplift_Modelling/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Promotion_Campaign_Uplift_Modelling/README.pt-BR.md)

Methods: Randomization audit · S/T/X/R-learners · Causal forest · Uplift trees · Qini/AUUC · Sealed holdout · Quantile heterogeneity profiles · Policy learning · IPW evaluation · ROI sensitivity

Technologies: Python · Pandas · scikit-learn · EconML · CausalML · Matplotlib · pytest · Jupyter

---

### 9. [Consumer Complaint Intelligence](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Consumer_Complaint_Intelligence) (featured)
![Status](https://img.shields.io/badge/status-completed-blue)

Builds a reproducible `en-US` complaint-routing system over a 3.8-million-row CFPB snapshot. The project progresses from bounded Polars/DuckDB audits and temporal deduplication to TF-IDF classifiers, hard-negative mining, a hierarchical detector, frozen inference bundles, and two pre-registered sealed evaluations. Neither evaluation was retrospectively promoted: S8 passed 2 of 3 gates and V2.1-C passed 3 of 4, so both remain `NOT_CONFIRMED` and `deploy=false`. A third cycle (V3) compared frozen MiniLM sentence embeddings with a TF-IDF control on rolling-origin folds restricted to development partitions; no MiniLM candidate met the critical-precision floor in every fold, and the frozen development package uses the TF-IDF detector. The V3 confirmation protocol is approved, but its execution awaits the complete 2026 source, so V3 has no confirmatory result.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Consumer_Complaint_Intelligence/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Consumer_Complaint_Intelligence/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Consumer_Complaint_Intelligence/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Consumer_Complaint_Intelligence/README.pt-BR.md)

Methods: Imbalanced multiclass NLP · Word/character TF-IDF · LinearSVC · SGD logistic regression · Temporal splitting · Group-aware deduplication · Hard-negative mining · Hierarchical classification · Sentence embeddings (MiniLM) · Threshold calibration · Bootstrap intervals · Sealed confirmatory evaluation

Technologies: Python · Polars · DuckDB · scikit-learn · imbalanced-learn · sentence-transformers · joblib · Kaggle · Jupyter · unittest

---

### 10. [NASA C-MAPSS: predictive maintenance](CMAPSS_Predictive_Maintenance/README.md) (featured)

Estimates remaining useful life on simulated FD001 engine trajectories using causal sensor histories. An Extra Trees pipeline selected 93 of 205 attributes and achieved MAE 16.23 cycles and RMSE 23.80 cycles in the official 100-engine benchmark. Model selection followed that comparison; the result is not an additional independent post-selection evaluation. Alerts, hypothetical costs and empirical uncertainty bands remain exploratory, not industrial validation.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/CMAPSS_Predictive_Maintenance/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/CMAPSS_Predictive_Maintenance/README.pt-BR.md)

Methods: Engine-grouped validation · Causal rolling features · Optuna · Feature selection · Residual diagnostics · Permutation importance · Illustrative maintenance alerts and costs

Technologies: Python · Pandas · scikit-learn · Extra Trees · XGBoost · HistGradientBoosting · Optuna · Plotly · Jupyter · GitHub Pages

---

### 11. [Model Risk & Responsible ML](ModelRisk_ResponsibleML/README.md)

Audits calibration, fairness, uncertainty and final-test isolation on the UCI Adult dataset. The bilingual portfolio documents a frozen HistGradientBoosting evaluation, threshold 0.31 and 18 automated contract tests. The result is explicitly exploratory and `production_eligible=false`.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/ModelRisk_ResponsibleML/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/ModelRisk_ResponsibleML/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/ModelRisk_ResponsibleML/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/ModelRisk_ResponsibleML/README.pt-BR.md)

Methods: Group-aware partitions · Platt calibration · Threshold audit · Wilson interval for group proportions · Fairness diagnostics · Contract tests

Technologies: Python · Pandas · scikit-learn · Matplotlib · Jupyter · unittest · GitHub Pages

---

### 12. [E-commerce Delivery Network & SLA Risk](E-commerce_Delivery_Network_SLA_Risk/README.md)

Studies whether an Olist order can be classified as a delivery-SLA breach at purchase time under temporal drift, using temporal validation, seller-to-destination network features, Haversine distance and calibrated models. The official final evaluation is inconclusive, the post-hoc analysis is exploratory, and `production_eligible=false` remains explicit.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/E-commerce_Delivery_Network_SLA_Risk/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/E-commerce_Delivery_Network_SLA_Risk/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/README.pt-BR.md)

Methods: Expanding-window validation · Graph-theoretic seller-to-state network features as-of purchase · Haversine distance · CatBoost · Logistic baseline · Optuna · Calibration · Contract tests · Bilingual notebooks

Technologies: Python · Pandas · scikit-learn · CatBoost · XGBoost · LightGBM · Optuna · Plotly · Folium · pytest · GitHub Actions

---

### 13. [Delivery Routing & Time-Window Optimization](Delivery_Routing_Time_Window_Optimization/README.md)

Compares OR-Tools with two constructive heuristics on the Solomon benchmark for vehicle routing with capacity and time windows. In the primary v1 evaluation, OR-Tools matched the SINTEF reference fleet on 24 of 42 instances, with a 3.40% median distance gap among those matches. Fixed-plan delay simulations and reoptimization show how route plans respond to operational changes. The v2 re-evaluation is post-hoc and the project is a benchmark study with `production_eligible = false`.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Delivery_Routing_Time_Window_Optimization/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Delivery_Routing_Time_Window_Optimization/en.html) · README: [English](Delivery_Routing_Time_Window_Optimization/README.md) · [Português](Delivery_Routing_Time_Window_Optimization/README.pt-BR.md)

Methods: Vehicle routing with time windows · Lexicographic evaluation · Independent feasibility checks · Priority penalties · Warm-start reoptimization

Technologies: Python · OR-Tools · NumPy · pandas · pytest · Jupyter · GitHub Pages

---

### 14. [Claim Frequency per Policy](Claim_Frequency_per_Policy/README.en-US.md)

Estimates motor insurance claims per 100 policy-years on the French `freMTPL2freq` benchmark (678,013 records). A global rate, Poisson and NB2 with an exposure offset were compared on validation; the Poisson was frozen and evaluated once on the reserved test set, behind an integrity gate and a single-run lock. The test supports a gain in mean Poisson deviance (0.31902 vs 0.33079; paired difference −0.01177, 95% CI −0.01317 to −0.01018), while calibration by coverage duration shows strong underestimation for very short exposures. `production_eligible = false`.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Claim_Frequency_per_Policy/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Claim_Frequency_per_Policy/en.html) · README: [English](Claim_Frequency_per_Policy/README.en-US.md) · [Português](Claim_Frequency_per_Policy/README.pt-BR.md)

Methods: Poisson and NB2 GLMs with a log(Exposure) offset (own IRLS) · Deterministic hash partition · Poisson deviance and predictive log-likelihood · Paired bootstrap by policy and by profile · Segment calibration · Zero diagnostics · Verifiable freezing and single final evaluation

Technologies: Python · NumPy · pandas · SciPy · statsmodels · Matplotlib · pytest · Jupyter · conda

---

### 15. [Retail Pricing & Promotion with Multilevel Models](Dunnhumby_Breakfast-at-the-Frat/README.md)

Estimates how discount depth, feature, display and temporary price reductions are associated with unit sales in the dunnhumby *Breakfast at the Frat* panel (524,926 weekly sales records from 77 stores and 55 products over 156 weeks), with crossed random intercepts for store, product and store × product. Under a pre-registered protocol, the intercept-only multilevel model was kept over random discount slopes on two internal temporal folds and scored once on the last 52 weeks: per-pair MAE of log units 0.454, against 0.917 for pooled OLS and close to a GPBoost challenger (0.452). Planned contrasts with Holm show that the discount slope differs across categories (−1.08 to −2.42), and the scenarios read a display at 20% off as about 1.57× the modelled sales of a 20% TPR, as conditional associations.

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Dunnhumby_Breakfast-at-the-Frat/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Dunnhumby_Breakfast-at-the-Frat/en.html) · README: [English](Dunnhumby_Breakfast-at-the-Frat/README.md) · [Português](Dunnhumby_Breakfast-at-the-Frat/README.pt-BR.md)

Methods: Crossed linear mixed models (lme4) · ICC and boundary-corrected likelihood-ratio tests · Within/between (Mundlak) price decomposition · Two-way cluster-robust inference · Pre-registered temporal validation with hash freezing and a single final scoring · Parametric-bootstrap test of random slopes · Holm-adjusted planned contrasts · Scenario contrasts with joint covariance and observed support

Technologies: Python · R · lme4 · rpy2 · pandas · NumPy · SciPy · statsmodels · GPBoost · Optuna · Matplotlib · pytest · Jupyter · conda

---

## Skills summary

| Area | Methods and techniques |
|------|----------------------|
| Supervised learning | random forest, CatBoost, gradient boosting, logistic regression, threshold optimization |
| Natural language processing | TF-IDF, sparse linear models, hard-negative mining, hierarchical classification, imbalanced text classification |
| Probabilistic modeling | BG/NBD, Gamma-Gamma, MCMC, MAP |
| Time series and forecasting | expanding-window CV, temporal holdout, demand forecasting, residual diagnostics |
| Uncertainty quantification | adaptive conformal inference, conditional coverage, Winkler score, temporal bootstrap |
| Model explainability | SHAP, permutation importance, ablation analysis |
| Unsupervised learning | K-Means clustering, RFM segmentation |
| Prescriptive analytics | budget optimization, expected profit simulation, next best action |
| Inferential statistics | Z-test, A/B testing, hypothesis testing |
| Count data modeling | Poisson and negative binomial (NB2) GLMs, exposure offset, dispersion and zero diagnostics, segment calibration |
| Multilevel models | Crossed random effects, partial pooling (shrinkage), ICC, boundary likelihood-ratio tests, random slopes, parametric bootstrap |
| Causal inference | CATE estimation, meta-learners, causal forests, uplift evaluation, policy learning |
| MLOps | Optuna, MLflow, FastAPI, pytest, GitHub Actions CI |
| Data engineering | Polars, DuckDB, feature pipelines, encoding strategies, imbalanced data handling |
| Market basket analysis | Apriori, association rules, lift, conviction, Zhang's metric |

## Core technologies

Python · Pandas · Polars · DuckDB · NumPy · scikit-learn · imbalanced-learn · CatBoost · XGBoost · LightGBM · EconML · CausalML · Optuna · MLflow · SHAP · feature-engine · pymc-marketing · FastAPI · Flask · pytest · Matplotlib · Seaborn · Statsmodels · R/lme4 · rpy2 · GPBoost · category_encoders · MLxtend · Jupyter

---

*Portuguese is the canonical language of newly developed notebooks; English editions are published as separate portfolio deliverables when completed.*

## License

Code is released under the [MIT License](LICENSE). Third-party datasets (CFPB, Kaggle, UCI, NASA, Hillstrom/MineThatData, Solomon/SINTEF, CVRPLIB, OpenML/freMTPL2freq, dunnhumby Source Files) remain under their original terms; see each project's README for sources.
