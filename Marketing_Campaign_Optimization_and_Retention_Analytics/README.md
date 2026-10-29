# Marketing Campaign Optimization and Retention Analytics

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/notebooks/Marketing_Campaign_Optimization_and_Retention_Analytics_EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/notebooks/Marketing_Campaign_Optimization_and_Retention_Analytics_PT.ipynb)

**Read:** [Executive report · Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) · [English report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/en.html)

Complete Marketing Analytics pipeline with descriptive, inferential, predictive, and prescriptive analysis applied to campaign optimization, user retention, and operational recommendation generation.

---

## 🎯 Scope and evidence

**Can support:** descriptive, inferential, predictive and prescriptive exploration of campaign and retention decisions.

**Does not support:** a production scoring service, causal campaign impact, or automatic decisions without further engineering and validation.

**Engineering surface:** notebook-led analysis plus a small `src/marketing_campaign` package (data preparation, RFM segmentation, funnel, A/B test and power, customer value and ROI, retention model, recommendations) covered by 205 pytest tests that reproduce the figures of the executive report. Both notebooks call this package for every calculation (cleaning, segmentation, funnel, A/B test, value and ROI, retention model, error analysis, recommendations, exports and charts). The two notebooks run exactly the same code and differ only in the `LANG` line; every text, including the insight paragraphs that quote the computed numbers, lives in the message catalog, and a test fails if the code of the two notebooks ever diverges. The refactor also fixed one defect: users flagged with the wrong language (`is_correct_lang` = "No") are now routed to "Fix Language", a branch the original notebook code never reached. There is no serving API.

**Governance layer:** [Executive report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) connects analytical findings to explicit business limitations.

## Objective

This project implements an end-to-end analytics solution for digital marketing campaigns, integrating:

- exploratory analysis;
- conversion funnel analysis;
- A/B testing;
- predictive retention modeling;
- bias diagnosis in model errors;
- ROI simulation;
- expected value calculation per user;
- automated operational recommendation generation.

The project was conceptually inspired by the paper:

> *Optimising Marketing Strategies by Customer Segments and Lifetime Values, with A/B Testing*  
> Guha, Echagarruga & Tian (2021)

but expanded into a modern data science pipeline applied to the marketing analytics context.

---

## Main Features

### Descriptive Analysis
- Overall conversion rate
- Retention rate
- Channel performance
- Personalization impact
- Conversion and retention by language

### A/B Testing
- Z-test for difference between proportions
- Statistical interpretation of uplift
- Inferential validation of variants

### Machine Learning
- Supervised pipeline for retention prediction
- Automated preprocessing
- Categorical variable encoding
- Evaluation using:
  - ROC AUC
  - Recall
  - Accuracy
  - Precision

### Error Diagnosis
- False positive analysis
- Over-representation of groups in model errors
- Comparison between dataset composition and error composition
- Diagnosis of potential operational biases

### Financial Simulation
- Estimated ROI by channel
- Expected profit
- Expected value per user
- Operational prioritization based on financial impact

### Prescriptive Recommendation
Automatic classification of users into actions such as:

- Prioritize and Scale
- Maintain and Optimize
- Detailed Analysis
- Fix Language
- Review Pricing
- Low Priority

### Export
- Automated multi-sheet Excel export
- Consolidation of key analytical results

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- OpenPyXL

---

## Analytical Structure

The pipeline was structured into four analytical layers:

### 1. Descriptive
Understanding campaign and user behavior.

### 2. Inferential
Statistical validation of observed differences through A/B testing.

### 3. Predictive
Modeling user retention probability.

### 4. Prescriptive
Generation of operational recommendations based on expected value and ROI.

---

## Code and Tests

```text
Marketing_Campaign_Optimization_and_Retention_Analytics/
├── notebooks/                      # Analysis notebooks (EN and PT-BR)
├── dataset/marketing_new.csv       # Input dataset
├── src/marketing_campaign/         # Reusable, documented analysis code
│   ├── config.py                   # Business constants and segment actions
│   ├── data.py                     # Loading, cleaning, stable-user base
│   ├── segmentation.py             # RFM proxies and K-Means segments
│   ├── funnel.py                   # Funnel and channel performance
│   ├── experiment.py               # Z-test, sample size, Monte Carlo power
│   ├── economics.py                # Customer value and simulated ROI
│   ├── retention_model.py          # Logistic regression for retention
│   ├── recommendations.py          # Expected value, actions, exports
│   ├── diagnostics.py              # Language impact, model errors, false-positive bias
│   ├── catalog.py                  # Texts of both languages (data only)
│   ├── i18n.py                     # Applies the catalog: localize, t, say
│   ├── reports.py                  # Experiment tables and pre-test messages
│   ├── insights.py                 # Insight and executive-summary texts
│   ├── plots.py                    # Chart helpers (EN and PT texts)
│   ├── narrative.py                # Helpers for the narrative cells
│   └── pipeline.py                 # Runs the whole analysis
├── tests/                          # pytest suite (205 tests)
└── pyproject.toml                  # Package and test configuration
```

The suite protects the numbers of the executive report: 10,037 raw records, 9,757 cleaned exposures, 6,849 stable users (460 variant switchers excluded), 703 conversions (10.26%), 463 retained (65.86%), z = 0.0779 with p = 0.937870, 3,916 users per group and 80.0% simulated power, the incremental ROI by channel, ROC-AUC 0.956 with 2 false positives, and the 6,849-row recommendation table.

```bash
cd Marketing_Campaign_Optimization_and_Retention_Analytics
pip install -e ".[dev]"
python -m pytest -q
```

The recommendation step writes `output/crm_next_best_action.csv` (one row per user) and the Excel workbook locally. Both files hold user-level rows, so they are kept in the project folder and are not versioned in the GitHub repository. Of the roughly 580 users with a positive expected profit (the exact count moves by a few users across library versions), 22 have the wrong language and receive "Fix Language -> then Loyalty Campaign"; the rest receive "Loyalty Campaign".

The notebooks import `marketing_campaign` (install it with `pip install -e .` from the project folder; without installing, they fall back to `./src`). The English notebook uses `LANG = "en"` and the Portuguese one `LANG = "pt"`, which only changes the labels of the tables and texts, not the calculations. Running them rewrites the local CRM exports in `output/`, which are not versioned in the repository.

---

## Key Results

- No statistically significant difference between the A/B variants (z = 0.0779, p = 0.94), with both groups below the planned sample size
- Better performance in campaigns using the correct language
- Identification of segments over-represented in false positives
- ROI simulation showing strong variation across channels
- Automated prioritization of users with higher expected value

---

## Academic Reference

GUHA, P.; ECHAGARRUGA, C.; TIAN, E. Q.  
*Optimising Marketing Strategies by Customer Segments and Lifetime Values, with A/B Testing*.  
Applied Marketing Analytics, v. 7, n. 2, p. 144–153, 2021.

---

## Author

Enio Rubens  
Data Scientist | Marketing Analytics | Machine Learning | Predictive Modeling
