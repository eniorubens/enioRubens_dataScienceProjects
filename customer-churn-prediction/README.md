# Customer churn prediction: reproducible end-to-end ML project

[English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/Customer-Churn-Prediction-EN.ipynb) | [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/Customer-Churn-Prediction-PT.ipynb)

## Scope and evidence

Can support: exploratory churn-risk prioritization and reproducible inference for a controlled retention pilot.

Does not support: causal retention claims, observed revenue preservation, or production use without temporal validation, monitoring and a controlled experiment.

Engineering surface: reusable `churn_project/` modules, a persisted training artifact, FastAPI endpoints in `serve_model.py`, and automated tests.

Governance layer: the [executive report](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/) states which hold-out evidence supports and why revenue/ROI remain illustrative.

## Executive summary

The project implements an exploratory, reproducible machine learning pipeline that predicts customer churn. The current materialized artifact reaches 76.4% Macro Recall (equivalent to balanced accuracy) and 83.5% ROC AUC on the deterministic hold-out test.

The classification problem is reframed as a business decision: accuracy metrics are replaced by Macro Recall, and the model scores are translated into a retention prioritization hypothesis.

Key boundary: revenue preservation and ROI are illustrative, assumption-based scenarios, not observed or causally validated outcomes.

---

## Project highlights

| Metric | Value | Significance |
|--------|-------|--------------|
| Macro Recall (equivalent to balanced accuracy) | 76.4% | Hold-out result at threshold 0.52 |
| ROC AUC | 83.5% | Hold-out discrimination using continuous scores |
| Precision (Macro) | 71.2% | Hold-out result at threshold 0.52 |
| Annual revenue preserved | Scenario only | Depends on unvalidated retention assumptions |
| Retention ROI | Scenario only | Not observed; requires a controlled experiment |

---

## Problem statement

The project reframes customer churn prediction from a purely technical classification problem into a business decision with asymmetric costs.

### The business reality

A false negative (missing a churner) means lost customer lifetime value, assumed at about $2-5K. A false positive (flagging a non-churner) means the cost of a retention discount, assumed at about $150-300. The dollar figures are illustrative assumptions, not values from the dataset or a source. Under these assumptions, missing a churner costs 10 to 30 times more than flagging a non-churner.

### Solution approach

Rather than maximizing accuracy, which misleads on imbalanced data, the work prioritizes Macro Recall to catch the maximum number of churners before they leave. The threshold was chosen to maximize this metric, not through a cost calculation. The translation of metrics into revenue preservation is done only as an illustrative scenario, and limitations and assumptions are documented explicitly.

---

## Methodology and technical approach

### Phase 1: exploratory data analysis

The dataset is [Telecom Customer Churn (Kaggle)](https://www.kaggle.com/datasets/puja19/telecom-customer-churn), with 7,043 customers and 20 features (demographics, services, billing). The target is Churn (binary classification), with 26.5% churn on the 7,043 raw rows (26.6% after dropping 11 rows with blank `TotalCharges`; 7,032 rows used).

#### Key business findings

Churn by contract type shows the largest difference in risk:

```
Month-to-Month: 43% churn    ← Highest risk segment
One-Year:       11% churn    ← Medium risk
Two-Year:       2.8% churn   ← Lowest risk (15x improvement)
```

Contract type is therefore the main lever associated with churn reduction.

On tenure, 40%+ of all churners exit within the first 3 months, and after month 12 churn stabilizes significantly; the early customer experience is critical. On price, customers with monthly charges above $65 show 2x higher churn, and the lack of support services is associated with a 29-37% higher churn risk, which suggests exploring a bundle strategy.

By PhiK correlation, contract type is the strongest churn predictor (φK = 0.45), followed by Internet Service (0.38), Tech Support (0.34) and Tenure (0.32).

---

### Phase 2: data processing and feature engineering (winning pipeline)

The winning preprocessing pipeline was selected by the search with `opt_binary_clf_pipe` (Optuna TPE, 100 trials, 6-fold stratified CV):

| Stage | Component | Configuration |
|-------|-----------|---------------|
| Encoder | `SumEncoder` | Sum (deviation) coding for 15 categorical features: each level is contrasted with the grand mean, without encoding order among levels |
| Scaler 1 | `Normalizer` | Scales each sample (row) to unit norm; it does not address per-feature outliers. Selected by the search, with no clear rationale for tree models and no ablation |
| Scaler 2 | `MaxAbsScaler` | Scales each feature by its maximum absolute value; preserves sign and sparsity |
| Feature selector | `SelectFromModel` | Retains 13 features using the model's own feature importances (threshold = mean) |
| Classifier | `BalancedRandomForestClassifier` | Balanced random forest with optimized hyperparameters |

Sum (or deviation) coding maps each categorical variable into contrast columns in which each level is compared with the grand mean. It does not inflate dimensionality as much as one-hot encoding, but it does not represent the order of the levels, such as contract duration.

`Normalizer` rescales each row to unit norm (row-wise, not per-feature). Tree ensembles are largely insensitive to this operation, and row normalization has no clear rationale before a tree model. It appears in the pipeline because the Optuna search selected it, and its contribution was not isolated by an ablation. `MaxAbsScaler` aligns feature magnitudes while preserving a sparsity-friendly scaling.

---

### Phase 3: model development and selection

Experiments were tracked with `opt_binary_clf_pipe` and MLflow across 35 configurations. Top results:

| Model | Configuration | Pipeline | Test Recall |
|-------|--------------|---------|------------|
| BalancedRandomForestClassifier | Opt Balanced + Threshold | SumEncoder Norm MaxAbsScaler SFM 13 \| thr=0.52 | 0.764 |
| LinearDiscriminantAnalysis | Opt Balanced + Threshold | BinaryEncoder PowerTransformer SFS 13 \| thr=0.25 | 0.762 |
| BalancedRandomForestClassifier | Baseline | OrdinalEncoder | 0.758 |
| BalancedRandomForestClassifier | Opt Over-sampling + Threshold | JamesStein MinMax Robust SFM 10 SMOTE \| thr=0.44 | 0.758 |
| LogisticRegression | Opt Balanced + Threshold | BinaryEncoder QuantileN SFS 13 \| thr=0.47 | 0.756 |

> Selection protocol and its limit. Inside the search, every trial is scored by 6-fold CV on the training split only, the whole pipeline (encoder, scalers, selector, sampler, model) is refit per fold, and each threshold is chosen on out-of-fold training predictions. The test split is used only to report. The one exception is the final ranking above: the champion was picked by `Test Recall`. The top three optimized candidates are statistically tied (test 0.764 / 0.762 / 0.758; out-of-fold 0.768 / 0.769 / 0.768 for BRF / LDA / BRF-SMOTE), so the choice of BRF is a tie-break, not a demonstrated superiority. `Baseline + Threshold tuning` rows for LogisticRegression (thr=0.22 / 0.42) were tuned on the test split and should be ignored in comparisons. The `Test`/`Train` columns also differ in meaning by row type: in `Baseline` rows they are cross-validation means on the training split (validation fold and in-fold, e.g. RF `Train` 0.998), in `Opt …` rows `Train` is out-of-fold and `Test` is the held-out test, and in `Baseline + Threshold tuning` rows `Test` comes from a threshold sweep on the test itself. Only `Opt …` rows are comparable with each other: the BalancedRandomForest `Baseline` figure (0.758) is a cross-validation mean to compare with the champion's out-of-fold 0.768, not with its test 0.764.

#### Why BalancedRandomForestClassifier?

BalancedRandomForestClassifier addresses class imbalance by design: it uses bootstrap sampling with class balancing at the tree level, with no external SMOTE needed (Chen et al., 2004). The notebook champion (artifact rebuilt by `train_model.py` on the same stratified 80/20 split, seed 42) reaches 76.4% Macro Recall (95% bootstrap CI 74.0–78.7%) and 83.5% ROC AUC (95% CI 81.4–85.6%) on 1,407 test customers (374 churners). ROC AUC is measured from continuous hold-out scores. The 6-fold out-of-fold recall on the training split is 76.8% (AUC 84.2%), so there is no sign of optimistic bias in the reported figure.

The validation is a single deterministic hold-out, not temporal or production validation. Production use requires temporal validation, monitoring and a controlled retention experiment.

---

### Phase 4: hyperparameter optimization

The hyperparameter search used `opt_binary_clf_pipe` with Optuna's Tree-structured Parzen Estimator (TPE):

```python
study = optuna.create_study(direction='maximize')
study.optimize(
    objective=objective_function,  # metric: recall_macro
    n_trials=100,
    sampler=TPESampler(seed=42)
)
```

#### Final hyperparameters and justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `n_estimators` | 306 | Larger ensemble reduces variance on ~5,600 training samples |
| `max_depth` | 9 | Configuration in the current saved training artifact |
| `max_features` | `log2` | Logarithmic feature sampling per split; introduces diversity, reduces inter-tree correlation |
| `min_samples_leaf` | 2 | Prevents over-segmentation while allowing fine-grained splits |
| `min_samples_split` | 6 | Controls tree growth; regularizes against overfitting |
| `random_state` | 738 | Fixed seed ensures full reproducibility |
| `class_weight` | `balanced` (implicit) | Inherent class balancing at tree level; 73/27 imbalance handled internally |

Performance note: the current saved artifact reaches 76.4% Macro Recall on the hold-out split. Comparisons with earlier notebook outputs should be treated as historical until the notebook is rerun end to end.

---

### Phase 5: threshold selection

The classifier outputs a continuous risk score, and the threshold determines the classification boundary. This score is not a calibrated probability: Balanced Random Forest outputs are not calibrated, and no calibration analysis was run.

The selected threshold was 0.52, slightly above the default 0.50. It was chosen to maximize Macro Recall on the out-of-fold predictions of the 6-fold stratified CV on the training split, not through a cost calculation. A cost-based threshold would require real intervention and customer-loss costs, which the dataset does not provide.

```
Threshold = 0.50 (Default)
  → Minimal change from optimized model

Threshold = 0.52 (chosen by Macro Recall)
  → Recall Macro: 76.4%, Precision Macro: 71.2%
  → Catches 80.5% of actual churners (301 of 374)
```

The choice of threshold depends on the context of use and is not merely a technical detail (Fawcett, 2006).

---

### Phase 6: validation strategy

```python
cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
```

The cross-validation is stratified to keep the 26.5% churn rate in each fold, uses 6 folds to reduce variance in imbalanced classification, and fixes the random state to ensure reproducibility across runs.

#### Appropriate metrics for imbalanced data

| Metric | Formula | Application | Why important |
|--------|---------|-------------|----------------|
| Macro Recall (equivalent to balanced accuracy) | TP/(TP+FN) averaged per class = (Sensitivity+Specificity)/2 | Minimize missed churners | False negatives most costly; not biased by imbalance |
| Precision Macro | TP/(TP+FP) averaged | Manage retention costs | False positive cost |
| ROC AUC | Area under curve | Threshold-independent | Overall discrimination |
| Accuracy | (TP+TN)/N | Inappropriate here | Misleading for imbalanced data |

With a single positive class, Macro Recall and balanced accuracy are the same quantity. Accuracy is inappropriate for imbalanced classification (He & Garcia, 2009), so this work emphasizes Macro Recall.

---

## Model performance and results

### Overall metrics (test set, n≈1,407)

The table below compares the champion with a reference logistic regression (one-hot encoding, standardized numeric features, `class_weight="balanced"`, untuned 0.50 threshold), trained on the same 80/20 split with seed 42 and evaluated on the same 1,407 test customers (374 churners).

| Metric | Winner (BRF) | Baseline (LR) | Difference (BRF − LR) |
|--------|--------|---------------|---|
| Macro Recall (equivalent to balanced accuracy) | 76.4% | 74.8% | +1.6 p.p. |
| Precision Macro | 71.2% | 69.8% | +1.4 p.p. |
| ROC AUC | 83.5% | 83.5% | 0.0 p.p. (95% bootstrap CI −1.0 to +1.0) |
| Gini (2 × AUC − 1) | 0.671 | 0.670 | +0.001 |
| KS | 0.535 | 0.509 | +0.026 |
| PR AUC (average precision) | 0.635 | 0.618 | +0.017 |
| Brier score (lower is better) | 0.172 | 0.174 | −0.001 |
| Precision in the top-score decile | 73.0% | 69.5% | +3.5 p.p. |
| Lift in the top-score decile | 2.75 | 2.61 | +0.14 |
| Churners captured in the top-score decile | 27.5% | 26.2% | +1.3 p.p. |

The logistic regression ties the champion on discrimination: the ROC AUC difference is zero, with a bootstrap confidence interval (2,000 resamples of the test set) between −1.0 and +1.0 percentage points. The BRF advantages in KS, PR AUC and top-decile lift are small and were not formally tested, and the Macro Recall difference mixes the model with the threshold, which was tuned only for the BRF. The mean test score of both models (0.42) sits well above the churn base rate (0.266), which confirms that neither produces calibrated probabilities; class balancing pushes the scores upward. Where interpretability matters, the logistic regression delivers the same risk ordering with a simpler model. The comparison is reproduced by the `evaluate_baseline.py` script, run from this folder with the dataset at `dataset/telco.csv` (`python evaluate_baseline.py`).

### Confusion matrix analysis

```
Prediction breakdown (test set):
  Correct (TN + TP):  1,048
  FP (false alarm):     286    ← non-churners flagged (unnecessary retention cost)
  FN (missed churner):   73    ← churners missed (most costly error)

Derived:
  True  Positives (TP): 301   churners correctly identified
  False Negatives (FN):  73   churners missed
  False  Positives (FP): 286  non-churners falsely flagged
  True  Negatives (TN): 747   non-churners correctly dismissed
```

Churner-class recall is 301 / (301+73) = 80.5%, and non-churner-class recall is 747 / (747+286) = 72.3%. The 301 true positives are correctly identified churners, and the 73 false negatives are missed churners. The 286 false positives are flagged non-churners, and whether this cost is acceptable requires real intervention economics. The 747 true negatives are correctly identified non-churners.

### Feature importance (permutation importance, Macro Recall scorer)

The top five drivers of churn on the test set:

```
1. Contract              ← Strongest generalizable driver
2. InternetService       ← Consistent top predictor (train + test)
3. tenure               ← Classic churn pattern: newer customers churn more
4. TotalCharges         ← Strong in training, moderate in test
5. OnlineSecurity       ← Moderate, stable importance
```

The low-impact features, candidates for removal, are `StreamingTV`, `StreamingMovies`, `PhoneService`, `gender` and `Partner`. Contract structure is the main lever associated with churn reduction, consistent with the exploratory findings (φK=0.45).

### Error analysis

The missed churners (FN) have a median tenure of 35 months (vs 6 months for caught churners) and a median MonthlyCharges of $67 (vs $79). These customers resemble non-churners on surface features, with higher tenure and moderate charges, and are the hardest cases for any churn model.

Contract distribution in errors:

| Contract | FN % | FP % |
|----------|------|------|
| Month-to-month | 40.3% | 93.4% |
| One year | 48.6% | 6.6% |
| Two year | 11.1% | 0.0% |

---

## Business impact and revenue preservation

### Revenue context

Without a model (2% monthly churn baseline):

```
Annual loss from unmanaged churn: ≈$12M on $50M ARR
```

### Illustrative scenario using the hold-out recall (not observed impact)

The scenario below is illustrative, assumption-based and non-causal. It assumes a full-dataset campaign scale and is not a measured campaign result. Intervention cost and LTV are assumed inputs, not observed costs, and the 60% retention-success assumption must be tested prospectively. Any revenue-preservation value is therefore a scenario, not a result.

Basis: at the hold-out threshold (0.52) the model captures 80.5% of churners (301 of 374) and flags 27.7% of non-churners (286 of 1,033). Scale: ≈7,043 customers (≈1,866 churners). Assumed inputs: US$250 per intervention, US$3,500 LTV per churner, 60% retention success.

No observed ROI is reported. The assumptions must be replaced with controlled-experiment results before business use.

### Model vs. no-model comparison

| Scenario | Churners lost | LTV lost | Intervention cost |
|----------|----------------|----------|-------------------|
| No model | ~1,866 | ≈$6.5M | $0 |
| With model | ≈965 (≈364 missed + ≈601 flagged but not retained) | ≈$3.4M | ≈$734K (≈2,935 flagged) |
| Illustrative difference | ≈901 churners retained | ≈$3.2M preserved | ≈4.3x (preserved LTV / intervention cost) |

---

## References

He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data." *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284. Cited for the choice of metrics suited to imbalanced data, where accuracy fails.

Guyon, I., & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." *Journal of Machine Learning Research*, 3, 1157-1182. Cited as the basis of the multi-metric feature selection approach (Chi², ANOVA, PhiK, SelectFromModel).

Fawcett, T. (2006). "An Introduction to ROC Analysis." *Pattern Recognition Letters*, 27(8), 861-874. Cited for threshold selection as a decision that depends on the context of use.

Ke, G., et al. (2017). "LightGBM: A Fast, Distributed, Gradient Boosting Framework." *NeurIPS*. Referenced during model selection; LightGBM was a strong contender.

---

## Limitations and transparency

### Known constraints

Temporal snapshot: the dataset represents a single point in time (~2019), and the model would require quarterly or semi-annual retraining.

Missing behavioral features: there is no interaction history (support tickets, feature usage patterns) and no customer satisfaction metrics (NPS, CSAT). Silent churners (high tenure, moderate charges) are the hardest to identify, as the FN analysis confirms.

Correlation is not causation: high monthly charges correlate with churn without being shown to cause it. Causal assumptions should be validated with A/B testing before major price changes.

Segment generalization: the model was trained specifically on telecom customers, and new customer segments require retraining.

Intervention success assumption: the ROI scenario assumes 60% retention success. A pilot with 1,000 customers would measure actual results before a full rollout.

False positive volume: 286 of 1,033 non-churners were flagged (27.7% FP rate). If the retention budget is constrained, threshold adjustment may be needed.

### What the model can and cannot do

On this hold-out, the model identifies 80.5% of churners (class-level recall); lead time is not established. It allows retention spending to be prioritized on high-risk segments, informs contract and pricing strategies, and supports illustrative revenue and ROI scenarios with explicit assumptions.

The model does not guarantee retention success, which depends on campaign quality. It also does not explain causal mechanisms of churn (only correlations), does not work effectively without periodic retraining, and does not reliably identify long-tenure, moderate-charge churners, a confirmed FN pattern.

---

## Next steps

The realistic next steps need no data beyond what is described here. Probability calibration would allow the scores to be read as probabilities. A cost-based threshold would require real retention and customer-loss costs. Segment-specific models (month-to-month and long-term contracts) can be compared with the single model. If timestamped data were available, out-of-time validation would replace the random hold-out. Uplift modeling depends on data from a retention experiment.

---

## Technical stack

The project uses Python 3.10+ and Jupyter Notebook (interactive analysis and documentation). Data processing and ML use Pandas and NumPy, Scikit-learn (preprocessing pipelines, evaluation), imbalanced-learn (BalancedRandomForestClassifier) and category_encoders (SumEncoder and other encoders). Optimization and experiment tracking use opt_binary_clf_pipe (custom library for the full experiment loop), Optuna (Bayesian hyperparameter optimization) and MLflow (experiment tracking and model registry). Statistical analysis uses PhiK (correlation for mixed-type variables) and SciPy (statistical testing), and visualization uses Matplotlib and Seaborn. The API uses FastAPI and Uvicorn, and testing uses pytest with GitHub Actions CI.

---

## Author and methodology notes

Author: Enio Rubens

Role: Data Science & Analytics

AI-assisted tools were used for code optimization, documentation structuring and analytical validation. The business problem framing, methodological decisions, result interpretation and recommendations were driven by the author. All analytical decisions, business interpretations, modeling choices, validation procedures and project integration were conducted and reviewed by the author.

---

## Citation format

```bibtex
@misc{rubens2026churn,
  author = {Rubens, Enio},
  title = {Customer Churn Prediction: Reproducible End-to-End ML Project},
  year = {2026},
  note = {Portfolio Project}
}
```

---

## License and acknowledgments

Dataset: [Telecom Customer Churn (Kaggle)](https://www.kaggle.com/datasets/puja19/telecom-customer-churn), CC0 license.

Libraries: scikit-learn, imbalanced-learn, Optuna, Pandas and NumPy development communities.

Research: cited papers by He & Garcia, Guyon & Elisseeff, Fawcett and Ke et al.

---

Last updated: October 2026

Project status: exploratory portfolio artifact

Model performance: 76.4% Macro Recall (equivalent to balanced accuracy) | 83.5% ROC AUC | ROI not validated

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11+-orange.svg)](https://imbalanced-learn.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

Data science portfolio project on customer churn prediction.
