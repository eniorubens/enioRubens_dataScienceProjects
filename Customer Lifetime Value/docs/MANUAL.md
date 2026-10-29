# Customer Lifetime Value Prediction — Technical Manual

> **Technical manual.** Full reference for the API, modules, validation methodology and
> segmentation framework. The project overview, results and limitations are in the
> [README](../README.md).

A production-ready CLTV modelling project using probabilistic BTYD models.  
**BG/NBD + Gamma-Gamma** via [pymc-marketing](https://www.pymc-marketing.io) — the active successor to the archived `lifetimes` library.

---

## Overview

**Customer Lifetime Value (CLTV)** is the total revenue a business can expect from a single customer account throughout their relationship. Unlike historical spend, forward-looking CLTV predicts *future* value — enabling marketing teams to allocate budget by expected return rather than past performance.

This project estimates CLTV using two probabilistic models fitted on transactional RFM data:

**BG/NBD (BetaGeoModel):** Estimates how many purchases each customer will make in the next N days and the probability they are still active. The model assumes customers purchase at an individual Poisson rate while "alive" and permanently churn at a Beta-distributed probability. This produces a closed-form expression for expected future purchases.

**Gamma-Gamma (GammaGammaModel):** Estimates the expected monetary value per future transaction. Given that a customer makes purchases, what is the average order value? The model treats each customer's average as Gamma-distributed, with the distribution shape itself varying across the customer population.

**CLTV = expected purchases × expected spend per transaction**, integrated over the prediction horizon (default: 180 days).

The difference between *historical CLTV* (total past revenue) and *forward-looking CLTV* (predicted future revenue) is fundamental: this project predicts the future.

---

## Project Structure

```
Customer Lifetime Value/
├── src/
│   ├── __init__.py          # clean public API
│   ├── preprocessing.py     # OnlineRetailPreprocessor — load, clean, RFM, split
│   ├── cltv_model.py        # CLTVModel — BG/NBD + Gamma-Gamma wrapper
│   ├── segmentation.py      # CustomerSegmenter — quartile + P(alive) matrix
│   ├── evaluation.py        # CLTVEvaluator — temporal holdout validation
│   └── utils.py             # logging, chart formatting helpers
├── tests/
│   ├── conftest.py          # shared fixtures (synthetic data, no real dataset)
│   ├── test_preprocessing.py
│   ├── test_cltv_model.py
│   ├── test_segmentation.py
│   ├── test_evaluation.py
│   └── test_integration.py  # end-to-end pipeline + Flask API
├── notebooks/
│   ├── Customer_Lifetime_Value_EN.ipynb  # English walkthrough
│   └── Customer_Lifetime_Value_PT.ipynb  # Portuguese walkthrough
├── models/
│   └── .gitkeep             # bgm_model.nc, gg_model.nc, metadata.json after training
├── dataset/
│   ├── online_retail.csv
│   └── variable_explanation.csv
├── train_model.py           # CLI — train and save models
├── serve_model.py           # Flask API — score customers
├── environment.yml          # conda-forge environment
├── pyproject.toml
└── README.md
```

---

## Quick Start

### Environment Setup

```bash
conda env create -f environment.yml
conda activate cltv_env
```

### Train Model

```bash
# Default: MAP fitting, 180-day horizon, dataset/online_retail.csv
python train_model.py

# Custom options
python train_model.py --data dataset/online_retail.csv --output models/ --t 365

# Full Bayesian (MCMC) — slower, ~5-10 minutes
python train_model.py --fit-method mcmc
```

### Run API

```bash
python serve_model.py
# API available at http://localhost:5000
```

### Run Notebooks

```bash
jupyter lab notebooks/
```

### Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Dataset

**Source:** UCI Machine Learning Repository — Online Retail Dataset  
**Citation:** Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208.

| Attribute | Value |
|---|---|
| Rows (raw) | 541,909 |
| Customers (raw) | 4,372 |
| Period | Dec 2010 – Dec 2011 |
| Country | United Kingdom |

### Cleaning Pipeline

| Step | Rows removed |
|---|---|
| Parse InvoiceDate | 0 |
| Remove null CustomerID | ~135,080 |
| Remove cancellations (InvoiceNo starts with 'C') | ~8,905 |
| Remove Quantity ≤ 0 or UnitPrice ≤ 0 | ~40 |
| Cap TotalPrice at 99.5th percentile | ~1,923 |
| **Total remaining** | **~395,961** |

After building the RFM matrix: ~4,309 customers. After filtering to modeling base (frequency > 0): **~2,762 customers** used for CLTV prediction.

---

## Methodology

### RFM Framework

Each customer is represented by four features computed over the observation period:

| Feature | Description |
|---|---|
| `frequency` | Number of repeat purchase periods after the first transaction |
| `recency` | Days between first and most recent purchase |
| `T` | Customer age in days (first purchase → observation period end) |
| `monetary_value` | Average revenue per repeat transaction (GBP) |

### BG/NBD Model

The BG/NBD model describes two competing processes for each customer:

1. **Purchase process:** While alive, purchases arrive at a Poisson rate λ, which is Gamma-distributed across customers (parameters `r`, `α`).
2. **Dropout process:** After each transaction, the customer churns permanently with probability `p`, which is Beta-distributed across customers (parameters `a`, `b`).

The model produces:
- **Expected future purchases** in `t` days: `expected_num_purchases(data, future_t=t)`
- **Probability of still being alive**: `expected_probability_alive(data)`

Parameters: `r`, `alpha` (purchase rate), `a`, `b` (dropout probability)

### Gamma-Gamma Model

Given that a customer makes repeat purchases, the Gamma-Gamma model estimates the expected average order value. It assumes:
- Each customer's transaction value is drawn from a Gamma distribution with shape parameter `p`.
- The scale of that Gamma distribution is itself Gamma-distributed across customers (parameters `q`, `v`).

Output: **Expected spend per transaction** — `expected_customer_spend(data)`

Parameters: `p`, `q`, `v`

**Independence assumption:** The Gamma-Gamma model requires that purchase frequency and average monetary value are *independent* across customers. This is tested empirically using Pearson correlation before fitting (|r| < 0.3 threshold).

### CLTV Combination

```
CLTV(t) = expected_purchases(t) × expected_spend_per_transaction
```

Computed via `GammaGammaModel.expected_customer_lifetime_value()` which integrates both models over the prediction horizon with optional discounting.

### MAP vs MCMC

| Method | Speed | Output | Use when |
|---|---|---|---|
| MAP (default) | Fast (seconds) | Point estimate | Exploratory analysis, production scoring |
| MCMC | Slow (5-10 min) | Full posterior | Uncertainty quantification, research |

Use `--fit-method mcmc` in `train_model.py` or `fit_method="mcmc"` in `CLTVModel` for MCMC.

---

## Model Validation

### Methodology: Temporal Calibration / Holdout Split

The observation period is split into:
- **Calibration (75%):** Model learns purchase patterns
- **Holdout (25%):** Model predicts, actual purchases are compared

k-fold cross-validation is **not appropriate** for BTYD models because it shuffles time — a customer's future purchases would appear in the training fold. This violates the temporal structure the models are designed to exploit.

### Why Not Cross-Validation?

In k-fold CV, each fold can contain both early and late transactions from the same customer. The calibration/holdout framework preserves causality: the model only sees information available at the calibration cutoff when predicting holdout behaviour.

### Validation Metrics

After training, run `python train_model.py` to see validation results:

```
── Holdout Validation Results ──────────────────────────
  rmse_frequency:      [fill after training]
  mae_frequency:       [fill after training]
  pearson_r:           [fill after training]
  assumption_holds:    [fill after training]
  n_customers_holdout: [fill after training]
```

---

## Segmentation Framework

Customers are segmented on two dimensions after CLTV prediction:

1. **CLTV quartile** — `Low Value`, `Mid Value`, `High Value`, `Top Value`
2. **P(alive) group** — above/below the median probability-alive threshold

### Decision Matrix

|  | Low P(alive) | High P(alive) |
|---|---|---|
| **Low/Mid Value** | Low Priority | Low-Cost Nurture |
| **High/Top Value** | Reactivation Priority | Retain and Grow |

### Action Descriptions

| Action | Description |
|---|---|
| **Retain and Grow** | High-value customers still purchasing. Invest in loyalty, cross-sell, and VIP treatment. |
| **Reactivation Priority** | High-value customers going quiet. Win-back campaigns — highest ROI opportunity. |
| **Low-Cost Nurture** | Active but lower-value customers. Automated email flows, low-cost touchpoints. |
| **Low Priority** | Low value + low activity. Minimal investment; monitor for re-entry into higher tiers. |

### Segment Statistics

After training, run the notebook to populate:

| Action | Customers | Expected Revenue | Avg CLTV |
|---|---|---|---|
| Retain and Grow | [fill] | [fill] | [fill] |
| Reactivation Priority | [fill] | [fill] | [fill] |
| Low-Cost Nurture | [fill] | [fill] | [fill] |
| Low Priority | [fill] | [fill] | [fill] |

---

## Business Insights

After training, quantified insights (fill these from notebook output):

- **Top 10% of customers** account for approximately **[X]%** of total expected 180-day revenue.
- **Reactivation Priority** segment: **[Y]** customers with **GBP [Z]** combined expected value — the highest ROI win-back opportunity.
- **Retain and Grow** segment: **[Y]** customers with **GBP [Z]** combined expected value — protect this group.
- Budget recommendation: concentrate 70%+ of acquisition and retention spend on High/Top Value segments.

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check with model status |
| GET | `/model/info` | Model metadata and parameters |
| POST | `/predict/single` | Score a single customer |
| POST | `/predict/batch` | Score a list of customers |

### GET `/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "fit_date": "2024-01-15T10:30:00",
  "fit_method": "map"
}
```

### GET `/model/info`

```json
{
  "fit_method": "map",
  "fit_date": "2024-01-15T10:30:00",
  "n_customers": 2762,
  "bgm_params": {"r": 0.24, "alpha": 4.41, "a": 0.79, "b": 2.42},
  "gg_params": {"p": 6.25, "q": 3.74, "v": 15.44},
  "validation": {"rmse_frequency": 1.23, "mae_frequency": 0.87}
}
```

### POST `/predict/single`

Request:
```json
{
  "customer_id": "17850",
  "frequency": 4,
  "recency": 120.0,
  "T": 300.0,
  "monetary_value": 25.50,
  "t": 180
}
```

Response:
```json
{
  "customer_id": "17850",
  "predicted_purchases": 2.34,
  "probability_alive": 0.87,
  "expected_spend": 28.10,
  "cltv": 65.77,
  "cltv_segment": "High Value",
  "marketing_action": "Retain and Grow"
}
```

### POST `/predict/batch`

Request: JSON array of the same schema as `/predict/single`.  
Response: JSON array of the same schema as the single response.

### Validation Rules

| Field | Rule |
|---|---|
| `frequency` | integer ≥ 0 |
| `recency` | float ≥ 0 |
| `T` | float ≥ `recency` |
| `monetary_value` | float > 0 (for repeat customers) |
| `t` | integer 1–730 |

### Error Responses

| Status | Example message |
|---|---|
| 400 | `"T must be >= recency"` |
| 400 | `"Missing fields: ['frequency']"` |
| 503 | `"Model not loaded. Run train_model.py first."` |

---

## Module Reference

### `OnlineRetailPreprocessor`

```python
from src import OnlineRetailPreprocessor

prep = OnlineRetailPreprocessor()

# Full pipeline (recommended)
rfm, modeling_base, cleaning_summary = prep.prepare("dataset/online_retail.csv")

# Step-by-step
df_raw    = prep.load("dataset/online_retail.csv")
df_clean, summary = prep.clean(df_raw)
rfm       = prep.build_rfm(df_clean)
base      = prep.get_modeling_base(rfm)

# Temporal split for validation
cal, holdout = prep.calibration_holdout_split(df_clean)
```

### `CLTVModel`

```python
from src import CLTVModel

model = CLTVModel(fit_method="map")   # or "mcmc"
model.fit(modeling_base)              # returns self (method chaining)

cltv_df = model.predict_cltv(modeling_base, t=180)
probs   = model.predict_probability_alive(modeling_base)
preds   = model.predict_purchases(modeling_base, t=90)
spend   = model.predict_expected_spend(modeling_base)   # frequency > 0 only

model.save("models/")
model = CLTVModel.load("models/")
```

### `CustomerSegmenter`

```python
from src import CustomerSegmenter

seg = CustomerSegmenter(alive_threshold=0.8)   # or None to use median
segmented_df, threshold = seg.segment(cltv_df, cltv_col="cltv_180d")
matrix  = seg.build_decision_matrix()
summary = seg.segment_summary(segmented_df)
```

### `CLTVEvaluator`

```python
from src import CLTVEvaluator

ev = CLTVEvaluator()
pearson  = ev.pearson_independence_check(rfm)
results  = ev.evaluate(model, calibration_holdout_df)
ax       = ev.calibration_plot(model, calibration_holdout_df)
```

### Adapting to Other Datasets

To use `OnlineRetailPreprocessor` with a different transactional dataset:

1. **Override `load()`** to read your data format and rename columns to the canonical set: `InvoiceNo`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`.
2. **Override `clean()`** if your date format, cancellation encoding, or outlier strategy differs.
3. `build_rfm()`, `calibration_holdout_split()`, and `get_modeling_base()` use only the cleaned/canonical columns and need no changes.

---

## Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage HTML report
pytest tests/ --cov=src --cov-report=html
```

Coverage target: **>80%** in `src/`.

Tests use **synthetic data only** — no dependency on the real dataset. Session-scoped fixtures avoid repeated model fitting.

---

## Limitations & Future Work

**Current limitations:**
- The dataset contains **revenue, not margin**. CLTV figures are revenue-based and will be higher than margin-based estimates.
- The BG/NBD model assumes **permanent churn** — once a customer stops buying, they never return. This simplification underestimates reactivation potential.
- The Gamma-Gamma model assumes **independence between frequency and monetary value**. This is tested empirically (Pearson |r| < 0.3) but should be verified for new datasets.
- No customer-level covariates (demographics, channel, product category) are used.

**Migration note:** This project migrated from the `lifetimes` library (GitHub-archived, 2023) to `pymc-marketing`, the officially maintained successor with active development and Bayesian tooling via PyMC.

**Potential extensions:**
- **Covariates:** Pareto/NBD with covariates using pymc-marketing's covariate support
- **Discount rate:** Configure a positive discount rate for NPV-based CLTV
- **Cohort analysis:** Fit separate models per acquisition cohort
- **Full Bayesian (MCMC):** Use `fit_method='mcmc'` for uncertainty intervals on CLTV estimates
- **Margin data:** Multiply by gross margin rate for true profit-based CLTV

---

## References

1. Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). "Counting your customers" the easy way: An alternative to the Pareto/NBD model. *Marketing Science*, 24(2), 275–284.

2. Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary value. Retrieved from http://www.brucehardie.com/notes/025/

3. Schmittlein, D. C., Morrison, D. G., & Colombo, R. (1987). Counting your customers: Who are they and what will they do next? *Management Science*, 33(1), 1–24.

4. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208.

5. PyMC-Marketing documentation: https://www.pymc-marketing.io

6. Cameron Davidson-Pilon, `lifetimes` library (predecessor): https://github.com/CamDavidsonPilon/lifetimes
