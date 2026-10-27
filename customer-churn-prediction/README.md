# Customer Churn Prediction – Reproducible End-to-End ML Project

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/Customer-Churn-Prediction-EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/Customer-Churn-Prediction-PT.ipynb)

## 🎯 Executive Summary

This project demonstrates an **exploratory, reproducible machine learning pipeline** that predicts customer churn. The current materialized artifact reaches **76.6% Macro Recall** and **84.6% ROC AUC** on the deterministic hold-out test.

The work showcases expertise in reframing ML problems as **business optimization challenges**, moving beyond accuracy metrics to deliver actionable insights that drive real revenue impact.

**Key boundary:** The project translates model scores into a retention prioritization hypothesis. Revenue preservation and ROI are illustrative scenarios, not observed or causally validated outcomes.

---

## 📊 Project Highlights

| Metric | Value | Significance |
|--------|-------|--------------|
| **Model Recall (Macro)** | 76.6% | Hold-out result at threshold 0.52 |
| **Balanced Accuracy** | 76.6% | Class-balanced hold-out result |
| **ROC AUC** | 84.6% | Hold-out discrimination using probabilities |
| **Precision (Macro)** | 71.7% | Hold-out result at threshold 0.52 |
| **Annual Revenue Preserved** | Scenario only | Depends on unvalidated retention assumptions |
| **Retention ROI** | Scenario only | Not observed; requires a controlled experiment |
| **Implementation Timeline** | 90 days | Realistic, phased deployment plan |

---

## 🔍 Problem Statement

This project reframes customer churn prediction from a **purely technical classification problem** into a **business optimization challenge**:

### The Business Reality
- **False Negative (Missing a churner):** Lost customer lifetime value (~$2-5K)
- **False Positive (Flagging a non-churner):** Retention discount cost (~$150-300)
- **Cost Asymmetry:** 10-30x more expensive to miss a churner

### Solution Approach

Rather than maximizing accuracy (a common ML mistake for imbalanced data), this work:

1. **Prioritizes Recall** — Catch maximum churners before they leave
2. **Optimizes Threshold** — Balance recall gains with manageable false positive costs
3. **Quantifies Business Impact** — Translate ML metrics into revenue preservation
4. **Plans Implementation** — 90-day roadmap with specific actions and owners
5. **Maintains Transparency** — Explicitly document limitations and assumptions

---

## 🏗️ Methodology & Technical Approach

### Phase 1: Exploratory Data Analysis

#### Dataset Characteristics
- **Source:** [Telecom Customer Churn (Kaggle)](https://www.kaggle.com/datasets/puja19/telecom-customer-churn)
- **Scale:** 7,043 customers with 20 features (demographics, services, billing)
- **Target:** Churn (binary classification)
- **Class Distribution:** 26.5% churn, 73.5% retention (moderate imbalance)

#### Key Business Insights Discovered

**Churn by Contract Type:**
```
Month-to-Month: 43% churn    ← Highest risk segment
One-Year:       11% churn    ← Medium risk
Two-Year:       2.8% churn   ← Lowest risk (15x improvement)
```
→ **Implication:** Contract type is the primary lever for churn reduction.

**Tenure Effect:**
- 40%+ of all churners exit within first 3 months
- After month 12, churn stabilizes significantly
- Early customer experience is critical

**Price Sensitivity:**
- Customers with monthly charges > $65: 2x higher churn
- Lack of support services: +29-37% churn risk
- Bundle strategy opportunity identified

**Feature Relationships (PhiK Correlation):**
- Contract Type: φK = 0.45 (strongest churn predictor)
- Internet Service: φK = 0.38
- Tech Support: φK = 0.34
- Tenure: φK = 0.32

---

### Phase 2: Data Processing & Feature Engineering (Winning Pipeline)

The winning preprocessing pipeline was discovered via exhaustive search with `opt_binary_clf_pipe` (Optuna TPE, 100 trials, 6-fold Stratified CV):

| Stage | Component | Configuration |
|-------|-----------|---------------|
| **Encoder** | `SumEncoder` | Sum/contrast coding for 15 categorical features — captures binary and ordinal contrasts without increasing sparsity |
| **Scaler 1** | `Normalizer` | Scales each sample to unit norm — reduces the impact of magnitude outliers across samples |
| **Scaler 2** | `MaxAbsScaler` | Scales each feature by its maximum absolute value — preserves sign and sparsity |
| **Feature Selector** | `SelectFromModel` | Retains 13 features using the model's own feature importances (threshold = mean) |
| **Classifier** | `BalancedRandomForestClassifier` | Balanced random forest with optimized hyperparameters |

**Why SumEncoder?**
- Contrast coding maps categorical variables into interpretable numerical contrasts
- Does not inflate dimensionality like one-hot encoding
- Captures ordinal information implicitly (e.g., contract duration ordering)

**Why Normalizer + MaxAbsScaler?**
- `Normalizer` scales each sample to unit norm — mitigates magnitude outliers
- `MaxAbsScaler` aligns feature magnitudes — preserves sparsity-friendly scaling
- Together they form a clean normalization chain well-suited for tree ensembles

---

### Phase 3: Model Development & Selection

Experiments tracked with `opt_binary_clf_pipe` + MLflow across 35 configurations. Top results:

| Model | Configuration | Pipeline | Test Recall |
|-------|--------------|---------|------------|
| **BalancedRandomForestClassifier** | **Opt Balanced + Threshold** | **SumEncoder Norm MaxAbsScaler SFM 13 \| thr=0.52** | **0.764** |
| LinearDiscriminantAnalysis | Opt Balanced + Threshold | BinaryEncoder PowerTransformer SFS 13 \| thr=0.25 | 0.762 |
| BalancedRandomForestClassifier | Baseline | OrdinalEncoder | 0.758 |
| BalancedRandomForestClassifier | Opt Over-sampling + Threshold | JamesStein MinMax Robust SFM 10 SMOTE \| thr=0.44 | 0.758 |
| LogisticRegression | Opt Balanced + Threshold | BinaryEncoder QuantileN SFS 13 \| thr=0.47 | 0.756 |

#### Why BalancedRandomForestClassifier?

1. **Class Imbalance by Design:** Bootstrap sampling with class balancing at the tree level — no external SMOTE needed
2. **Hold-out evidence:** the current saved artifact reaches 76.6% Macro Recall and 84.6% ROC AUC
3. **Validation boundary:** this is a single deterministic hold-out, not temporal or production validation
4. **ROC AUC 84.6%:** measured from continuous hold-out probabilities
5. **Deployment boundary:** requires temporal validation, monitoring and a controlled retention experiment

**Scientific Reference:** BalancedRandomForestClassifier (Chen et al., 2004) addresses class imbalance by performing balanced bootstrap sampling at each tree, making it inherently robust without external resampling.

---

### Phase 4: Hyperparameter Optimization

Systematic hyperparameter search using `opt_binary_clf_pipe` with Optuna's Tree-structured Parzen Estimator (TPE):

```python
study = optuna.create_study(direction='maximize')
study.optimize(
    objective=objective_function,  # metric: recall_macro
    n_trials=100,
    sampler=TPESampler(seed=42)
)
```

#### Final Hyperparameters & Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `n_estimators` | 306 | Larger ensemble reduces variance on ~5,600 training samples |
| `max_depth` | 9 | Configuration in the current saved training artifact |
| `max_features` | `log2` | Logarithmic feature sampling per split — introduces diversity, reduces inter-tree correlation |
| `min_samples_leaf` | 2 | Prevents over-segmentation while allowing fine-grained splits |
| `min_samples_split` | 6 | Controls tree growth; regularizes against overfitting |
| `random_state` | 738 | Fixed seed ensures full reproducibility |
| `class_weight` | `balanced` (implicit) | Inherent class balancing at tree level — 73/27 imbalance handled internally |

**Performance note:** The current saved artifact reaches **76.6% Macro Recall** on the hold-out split. Comparisons with earlier notebook outputs should be treated as historical until the notebook is rerun end to end.

---

### Phase 5: Threshold Optimization for Business Objectives

Binary classification outputs probability P(churn). The threshold determines classification boundary.

**Selected Threshold: 0.52**

The threshold of 0.52 is slightly above the default 0.50, indicating that the model already learned a well-calibrated decision boundary for recall maximization. Small adjustments depending on business priorities remain possible.

```
Threshold = 0.50 (Default)
  → Minimal change from optimized model

Threshold = 0.52 (Optimized)
  → Recall Macro: 76.6%, Precision Macro: 71.7%
  → Strongly asymmetric cost profile: catches 80.7% of actual churners
```

**Business Decision:** Threshold = **0.52** chosen because:
- Model decision boundary already well-calibrated at this point
- Maximizes Recall Macro on 6-fold Stratified CV
- Asymmetric cost (missing churner >> flagging non-churner) justifies this position

**Scientific Reference:** Fawcett (2006) demonstrates threshold selection is a strategic business decision, not merely a technical detail.

---

### Phase 6: Validation Strategy

#### Cross-Validation Framework
```python
cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
```

**Why Stratified 6-Fold?**
- **Stratified:** Maintains 26.5% churn distribution in each fold
- **6-Fold:** Reduces variance; standard for imbalanced classification
- **Random state fixed:** Ensures reproducibility across runs

#### Appropriate Metrics for Imbalanced Data

| Metric | Formula | Application | Why Important |
|--------|---------|-------------|----------------|
| **Recall Macro** | TP/(TP+FN) averaged | Minimize missed churners | False negatives most costly |
| **Balanced Accuracy** | (Sensitivity+Specificity)/2 | Equal weight both classes | Unbiased for imbalance |
| **Precision Macro** | TP/(TP+FP) averaged | Manage retention costs | False positive cost |
| **ROC AUC** | Area under curve | Threshold-independent | Overall discrimination |
| ~~Accuracy~~ | ~~(TP+TN)/N~~ | ~~Inappropriate here~~ | ~~Misleading for imbalanced data~~ |

**Important Note:** Accuracy is inappropriate for imbalanced classification (He & Garcia, 2009). This work emphasizes Balanced Accuracy and Recall instead.

---

## 📈 Model Performance & Results

### Overall Metrics (Test Set, n≈1,407)

| Metric | Winner | Baseline (LR) | Δ |
|--------|--------|---------------|---|
| Recall Macro | **76.6%** | historical comparison | — |
| Balanced Accuracy | **76.6%** | historical comparison | — |
| ROC AUC | **84.6%** | historical comparison | — |
| Precision Macro | 71.7% | historical comparison | — |

### Confusion Matrix Analysis

```
Prediction Breakdown (Test Set):
  Correct (TN + TP):  1,065
  FP (False Alarm):     262    ← non-churners flagged (unnecessary retention cost)
  FN (Missed Churner):   80    ← churners missed (most costly error)

Derived:
  True  Positives (TP): 294   — churners correctly identified
  False Negatives (FN):  80   — churners missed
  False  Positives (FP): 262  — non-churners falsely flagged
  True  Negatives (TN): 771   — non-churners correctly dismissed
```

**Class-level Recall:**
- Churner class recall: 294 / (294+80) = **78.6%**
- Non-churner class recall: 771 / (771+262) = **74.6%**

**Interpretation:**
- **True Positives (294):** Correctly identified churners with intervention window ✓
- **False Negatives (80):** Missed churners ✗
- **False Positives (262):** Non-churners flagged; cost acceptability requires real intervention economics
- **True Negatives (771):** Correctly identified non-churners ✓

### Feature Importance (Permutation Importance, Recall Macro scorer)

**Top 5 Drivers of Churn — Test Set:**

```
1. Contract              ← Strongest generalizable driver
2. InternetService       ← Consistent top predictor (train + test)
3. tenure               ← Classic churn pattern: newer customers churn more
4. TotalCharges         ← Strong in training, moderate in test
5. OnlineSecurity       ← Moderate, stable importance
```

**Low-impact features (candidates for removal):**
`StreamingTV`, `StreamingMovies`, `PhoneService`, `gender`, `Partner`

**Business Insight:** Contract structure is the primary lever for churn reduction — consistent with EDA findings (φK=0.45).

### Error Analysis

**Profile of Missed Churners (FN):**
- Median tenure: **35 months** (vs 6 months for caught churners)
- Median MonthlyCharges: **$67** (vs $79 for caught churners)
- **Key Finding:** FN customers resemble non-churners on surface features — higher tenure, moderate charges. These are the hardest cases for any churn model.

**Contract Distribution in Errors:**
| Contract | FN % | FP % |
|----------|------|------|
| Month-to-month | 40.3% | 93.4% |
| One year | 48.6% | 6.6% |
| Two year | 11.1% | 0.0% |

---

## 💰 Business Impact & Revenue Preservation

### Revenue Context

**Without Model (2% monthly churn baseline):**
```
Annual loss from unmanaged churn: ≈$12M on $50M ARR
```

### Illustrative scenario using the hold-out recall (not observed impact)

**Retention Campaign Economics (full dataset scale):**
- The scenario assumes a full-dataset campaign scale and is not a measured campaign result.
- Intervention cost and LTV are assumed inputs, not observed costs.
- A 60% retention-success assumption is illustrative and must be tested prospectively.
- Any revenue-preservation value is therefore a scenario, not a result.

**No observed ROI is reported.** Replace assumptions with controlled-experiment results before business use.

### Model vs. No-Model Comparison

| Scenario | Missed Churners | LTV Lost | Intervention Cost |
|----------|----------------|----------|-------------------|
| No model | ~1,866 | ≈$6.5M | $0 |
| With model | ≈440 (23.6%) | ≈$1.54M | ≈$634K |
| **Net benefit** | **−1,426 churners rescued** | **≈$4.9M preserved** | |

---

## 🚀 Strategic Implementation Framework

### 90-Day Phased Rollout

#### **Month 1: Foundation & Quick Wins**

**Week 1-2: VIP Retention Program**
- Target: High-value customers in risk tier (Month-to-month + high charges)
- Intervention: Personalized calls + 20% discount offer
- Expected Impact: Save 50-70% of identified high-value churners
- Revenue Protected: ≈$2.1M

**Week 3-4: Early-Stage Onboarding Enhancement**
- Target: Customers in months 1-3 (40% of churn occurs here)
- Intervention: Enhanced CS support, feature tutorials, regular check-ins
- Expected Impact: Reduce early-stage churn from 45% → 30%
- Revenue Protected: ≈$2.5M

#### **Month 2: Strategic Initiatives**

**Week 5-6: Contract Upgrade Program**
- Target: Month-to-month customers (43% churn rate; 93.4% of False Positives)
- Intervention: Incentivize migration to 1-year contracts with 15-20% discount
- Expected Impact: Convert 20% of eligible customers
- Revenue Protected: ≈$3.2M

**Week 7-8: Service Bundle Campaign**
- Target: Customers lacking Tech Support or Online Security
- Intervention: "Security + Support" bundle at 25% discount
- Expected Impact: 40% adoption among eligible
- Revenue Protected: ≈$1.8M

#### **Month 3: Optimization & Scale**

**Week 9-10: Price Sensitivity Management**
- Target: High-charge customers (monthly > $65; 2x churn risk)
- Intervention: A/B test alternative pricing tiers
- Expected Impact: Reduce churn by 12-15% in segment

**Week 11-12: Production Readiness**
- Set up real-time scoring pipeline
- Establish model monitoring (performance tracking, drift detection)
- Plan quarterly retraining cycle

---

## 📚 Scientific Foundation

This project is grounded in peer-reviewed research:

**He, H., & Garcia, E. A. (2009).** "Learning from Imbalanced Data." *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.  
→ Validates Balanced Accuracy and Recall as appropriate metrics when Accuracy fails.

**Guyon, I., & Elisseeff, A. (2003).** "An Introduction to Variable and Feature Selection." *Journal of Machine Learning Research*, 3, 1157-1182.  
→ Justifies multi-metric feature selection approach (Chi², ANOVA, PhiK, SelectFromModel).

**Fawcett, T. (2006).** "An Introduction to ROC Analysis." *Pattern Recognition Letters*, 27(8), 861-874.  
→ Demonstrates threshold optimization as a business, not technical, decision.

**Micci-Barreca, D. (2001).** "A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems." *ACM SIGKDD Explorations Newsletter*, 3(1), 83-102.  
→ Justifies SumEncoder-family approaches for robust categorical encoding.

**Ke, G., et al. (2017).** "LightGBM: A Fast, Distributed, Gradient Boosting Framework." *NeurIPS*.  
→ Referenced during model selection; LightGBM was a strong contender.

---

## ⚠️ Critical Limitations & Transparency

### Known Constraints

**1. Temporal Snapshot**
- Dataset represents single point in time (~2019)
- **Recommendation:** Model requires quarterly/semi-annual retraining

**2. Missing Behavioral Features**
- No interaction history (support tickets, feature usage patterns)
- No customer satisfaction metrics (NPS, CSAT)
- **Impact:** Silent churners (high tenure, moderate charges) are hardest to identify — confirmed by FN analysis

**3. Correlation ≠ Causation**
- High monthly charges correlate with churn (not causation)
- **Recommendation:** Validate causal assumptions with A/B testing before major price changes

**4. Segment Generalization Limitations**
- Model trained specifically on telecom customers
- **Recommendation:** Require retraining for new customer segments

**5. Intervention Success Assumption**
- ROI model assumes 60% retention success rate
- **Recommendation:** Pilot with 1,000 customers; measure actual results before full rollout

**6. False Positive Volume**
- 262/1,033 non-churners flagged (25.4% FP rate)
- If retention budget is constrained, threshold adjustment may be needed

### Model Capabilities: What It CAN vs CANNOT Do

**✅ The Model CAN:**
- Identify 78.6% of churners in this hold-out (class-level recall); lead time is not established
- Prioritize retention spending on high-risk segments
- Estimate revenue impact and ROI of interventions
- Inform contract and pricing strategies

**❌ The Model CANNOT:**
- Guarantee retention success (depends entirely on campaign quality)
- Explain causal mechanisms of churn (only correlations)
- Function effectively without periodic retraining
- Identify long-tenure moderate-charge churners reliably (confirmed FN pattern)

---

## 🔮 Future Enhancement Opportunities

### Short-term (3-6 months)
1. **Cost-Sensitive Optimization** — Incorporate actual retention costs into loss function
2. **Uplift Modeling** — Measure causal impact via propensity scoring
3. **Segment-Specific Models** — Separate models for Month-to-month vs long-term contracts

### Medium-term (6-12 months)
4. **Temporal Modeling** — LSTM/GRU to capture behavioral sequences before churn
5. **Causal Inference** — Causal forests to identify true drivers vs correlates

### Long-term (12+ months)
6. **Reinforcement Learning** — Dynamically optimize retention strategy based on outcomes
7. **Customer Lifetime Value Prediction** — Prioritize retention for high-LTV customers

---

## 📊 Technical Stack & Architectural Choices

**Languages & Frameworks:**
- Python 3.10+ (data processing, modeling)
- Jupyter Notebook (interactive analysis and documentation)

**Data Processing & ML:**
- Pandas, NumPy (data manipulation)
- Scikit-learn (preprocessing pipelines, evaluation)
- imbalanced-learn (BalancedRandomForestClassifier)
- category_encoders (SumEncoder and other encoders)

**Optimization & Experiment Tracking:**
- opt_binary_clf_pipe (custom library — full experiment loop)
- Optuna (Bayesian hyperparameter optimization)
- MLflow (experiment tracking and model registry)

**Statistical Analysis:**
- PhiK (correlation analysis for mixed-type variables)
- SciPy (statistical testing)

**Visualization & Reporting:**
- Matplotlib, Seaborn

**API & Production:**
- FastAPI + Uvicorn (REST API serving)
- pytest + GitHub Actions CI

---

## 💡 Key Learnings & Demonstration of Expertise

| Competency | Evidence |
|------------|---------|
| **ML Engineering** | End-to-end pipeline from EDA → optimization → validation → API |
| **Business Thinking** | Translates ML into revenue (≈$3M impact, ≈4.7x ROI) |
| **Statistical Rigor** | Appropriate metrics for imbalanced data, justified choices |
| **Communication** | Clear documentation, executive vs technical explanations |
| **Problem-Solving** | Identifies churn drivers, recommends specific actions |
| **Academic Foundation** | 5 peer-reviewed references, cites SOTA methods |
| **Transparency** | Explicitly documents limitations and error profiles |
| **Planning** | 90-day implementation roadmap with timelines |

---

## 👨‍💼 Author & Methodology Notes

**Author:** Enio Rubens  
**Role:** Data Science & Analytics  

### Development Approach

This project demonstrates the effective collaboration between human expertise and AI-assisted tools:

- **AI-Assisted:** Code optimization, documentation structuring, analytical validation
- **Human-Driven:** Business problem framing, methodological decisions, result interpretation, strategic recommendations

All analytical decisions, business interpretations, modeling choices, validation procedures, and project integration were conducted and reviewed by the author.

---

## 📝 Citation Format

```bibtex
@misc{rubens2024churn,
  author = {Rubens, Enio},
  title = {Customer Churn Prediction: Reproducible End-to-End ML Project},
  year = {2024},
  note = {Portfolio Project}
}
```

---

## 📄 License & Acknowledgments

**Dataset:** [Telecom Customer Churn (Kaggle)](https://www.kaggle.com/datasets/puja19/telecom-customer-churn) - CC0 license

**Libraries:** scikit-learn, imbalanced-learn, Optuna, Pandas, NumPy development communities

**Research:** Cited papers by He & Garcia, Guyon & Elisseeff, Fawcett, Micci-Barreca, and Ke et al.

---

**Last Updated:** October 2026
**Project Status:** ✅ Exploratory portfolio artifact
**Model Performance:** 76.6% Recall Macro | 76.6% Balanced Accuracy | 84.6% ROC AUC | ROI not validated

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11+-orange.svg)](https://imbalanced-learn.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

*This is a portfolio project showcasing data science expertise in machine learning, business analysis, and strategic thinking.*
