# Customer Lifetime Value Prediction

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/9f494bfca025a50df36dc980a01c647e2907709f/Customer%20Lifetime%20Value/notebooks/Customer_Lifetime_Value_EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/9f494bfca025a50df36dc980a01c647e2907709f/Customer%20Lifetime%20Value/notebooks/Customer_Lifetime_Value_PT.ipynb)

> **Forward-looking revenue estimation at the individual customer level —
> not historical averages, but probabilistic forecasts of future value.**

---

## Overview

This project builds a production-ready CLTV (Customer Lifetime Value) pipeline
using probabilistic BTYD models on transactional e-commerce data. It estimates,
for each customer, how often they will purchase, how much they will spend per
transaction, and how likely they are to still be active — then combines these
into a 180-day revenue forecast and a prioritised marketing action per customer.

The output is not a segment label. It is a defensible, individual-level
allocation signal for marketing, sales, and finance teams.

---

## Business Problem

Not all customers have the same future value. Uniform budget allocation wastes
spend on low-return customers and underinvests in high-value ones. The model
answers three questions that drive allocation decisions:

- **Marketing:** Where do I concentrate campaign spend to maximise expected return?
- **Sales:** Which accounts should my team actively manage rather than leaving to self-serve?
- **Finance:** What is the defensible revenue projection for the next two quarters,
  disaggregated by retention risk?

---

## Results (Online Retail UCI, 180-day horizon)

| Metric | Value |
|---|---|
| Total expected revenue (180d) | GBP 99,751,217 |
| Top 10% customers — revenue share | 40.1% |
| Top 20% customers — revenue share | ~60% |
| Customers in modelling base | 2,762 |

**Segment breakdown:**

| Segment | Customers | Expected Revenue | Share |
|---|---|---|---|
| Retain and Grow | 1,000 | GBP 66,937,937 | 67.1% |
| Reactivation Priority | 381 | GBP 12,853,917 | 12.9% |
| Low Priority | 1,000 | GBP 13,358,174 | 13.4% |
| Low-Cost Nurture | 381 | GBP 6,601,190 | 6.6% |

Retain and Grow + Reactivation Priority = **80% of expected revenue from 56% of customers.**

---

## Methodology

### 1 — Data Preparation

The [Online Retail UCI](https://archive.ics.uci.edu/dataset/352/online+retail) dataset
contains 541,909 raw invoice line items from a UK-based e-commerce retailer
(Dec 2010 – Dec 2011). The cleaning pipeline removes four categories of
problematic records:

| Step | Action | Rows removed |
|---|---|---|
| Null CustomerID | Anonymous sessions — no individual attribution | ~135,080 |
| Cancellations | InvoiceNo starting with 'C' | ~8,905 |
| Invalid qty / price | Non-positive Quantity or UnitPrice | ~40 |
| Outlier cap | TotalPrice capped at 99.5th percentile | ~1,923 |

Result: ~4,309 customers in the RFM matrix; 2,762 in the modelling base.

### 2 — RFM Matrix

Each customer's purchase history is compressed into four numbers that map
directly to the generative assumptions of the BTYD models:

- **frequency** — repeat purchases after the first transaction
- **recency** — days between first and most recent purchase
- **T** — customer age from first purchase to observation window end
- **monetary_value** — average revenue per repeat transaction

### 3 — Model Validation

Validation uses a **temporal calibration / holdout split** (75% / 25%),
not k-fold cross-validation. k-fold shuffles time — future purchases
leak into the training fold, inflating every validation metric and
producing a model that fails immediately in real deployment.

### 4 — BG/NBD Model *(Fader, Hardie & Lee, 2005)*

Estimates, for each customer:
- Expected number of future transactions over a given time window
- Probability of still being active (`P(alive)`)

The model combines a Poisson purchase process (with Gamma-distributed
heterogeneity across customers) and a Beta-Geometric dropout mechanism
(permanent churn with Beta-distributed probability across customers).

**Fitted parameters (MAP, Online Retail):**

| Parameter | Value | Interpretation |
|---|---|---|
| `r` | 1.5829 | Purchase rate heterogeneity — pronounced right tail |
| `α` | 85.0371 | Rate scale — mean purchase rate ≈ 0.0186/day (~1 per 54 days) |
| `a` | 0.0294 | Beta shape α — churn probability |
| `b` | 0.9707 | Beta shape β — mean dropout ≈ 2.9% per transaction |

### 5 — Gamma-Gamma Model *(Fader & Hardie, 2013)*

Estimates the expected monetary value per future transaction using
Bayesian shrinkage: individual spend estimates are pulled toward the
population mean proportionally to how little data exists for that customer.
A customer with 2 transactions is corrected substantially; one with 50 is
barely adjusted.

**Prerequisite:** purchase frequency and monetary value must be independent
across customers. Tested empirically via Pearson correlation — `|r| = 0.134 < 0.3` ✓

**Fitted parameters (MAP, Online Retail):**

| Parameter | Value | Interpretation |
|---|---|---|
| `p` | 1.9806 | Shape of spend distribution |
| `q` | 4.1655 | Population-level scale shape |
| `v` | 621.07 | Scale — implied mean spend ≈ p×v/q ≈ £295 |

### 6 — CLTV & Segmentation

```
CLTV(t) = E[transactions in t days | BG/NBD] × E[spend per transaction | Gamma-Gamma]
```

Customers are segmented on two orthogonal dimensions:

- **CLTV quartile** (Low / Mid / High / Top Value) — how much is this customer worth?
- **P(alive) threshold** (median of modelled population) — is this customer still reachable?

| CLTV | P(alive) | Segment | Recommended Action |
|---|---|---|---|
| High | High | Retain and Grow | Loyalty incentives, account management |
| High | Low | Reactivation Priority | Win-back campaigns — window is narrowing |
| Low | High | Low-Cost Nurture | Automated flows, frequency development |
| Low | Low | Low Priority | Passive retention only |

---

## Project Architecture

```
Customer Lifetime Value/
├── src/
│   ├── preprocessing.py   # OnlineRetailPreprocessor
│   ├── cltv_model.py      # CLTVModel — BG/NBD + Gamma-Gamma wrapper
│   ├── segmentation.py    # CustomerSegmenter + decision matrix
│   ├── evaluation.py      # CLTVEvaluator — holdout validation + Pearson test
│   └── utils.py           # logging, chart formatting helpers
├── tests/                 # 43 tests — pytest, synthetic data fixtures
├── notebooks/
│   ├── Customer_Lifetime_Value_EN.ipynb
│   └── Customer_Lifetime_Value_PT.ipynb
├── train_model.py         # CLI — --data, --output, --fit-method, --t
├── serve_model.py         # Flask API — /health /model/info /predict/*
└── models/                # bgm_model.nc + gg_model.nc + metadata.json
```

### Test Coverage

```
src/__init__.py       100%
src/evaluation.py      97%
src/preprocessing.py   95%
src/segmentation.py    95%
src/cltv_model.py      89%
src/utils.py           70%
─────────────────────────
TOTAL                  91%   (43 passed, 0 warnings)
```

### Model Persistence

Models are serialised via `idata.to_netcdf()` / `az.from_netcdf()` (ArviZ).
**joblib and pickle are not used** — PyMC objects are not safe to pickle.

---

## Flask API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model status |
| `/model/info` | GET | MAP parameters and fit metadata |
| `/predict/single` | POST | Score one customer |
| `/predict/batch` | POST | Score a list of customers |

**`/predict/single` response:**
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

---

## Tech Stack

| Layer | Tools |
|---|---|
| Modelling | `pymc-marketing` (BetaGeoModel, GammaGammaModel) |
| Inference | MAP (default, seconds) / MCMC (full posterior, optional) |
| Data | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Serialisation | `arviz` (NetCDF) |
| API | `Flask` |
| Testing | `pytest`, `pytest-cov` |
| CI | GitHub Actions |
| Notebooks | Bilingual EN + PT via [`multilang`](../../ds_toolkit/multilang) `LangMap` |

> **Note:** this project migrated from the `lifetimes` library (archived 2023)
> to `pymc-marketing`, the actively maintained successor.

---

## Known Limitations

1. **Revenue, not margin** — CLTV figures are revenue-based; profitability depends
   on product mix and fulfilment costs not available in this dataset.
2. **Permanent churn assumption** — BG/NBD does not model reactivation.
   The Pareto/NBD model *(Schmittlein, Morrison & Colombo, 1987)* relaxes this.
3. **No covariates** — no demographics, acquisition channel, or product category.
4. **Independence assumption** — Gamma-Gamma requires frequency ⊥ monetary_value.
   Always verify with a Pearson test before applying to a new dataset.

---

## References

1. Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). "Counting your customers" the easy
   way. *Marketing Science*, 24(2), 275–284.
2. Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary value.
   [brucehardie.com/notes/025](http://www.brucehardie.com/notes/025/)
3. Schmittlein, D. C., Morrison, D. G., & Colombo, R. (1987). Counting your customers:
   Who are they and what will they do next? *Management Science*, 33(1), 1–24.
4. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry.
   *Journal of Database Marketing*, 19(3), 197–208.
5. PyMC-Marketing (2024). [Customer Lifetime Value module](https://www.pymc-marketing.io/en/stable/notebooks/clv/clv_quickstart.html).

---

## Development Notes

Parts of this project were developed with the assistance of AI-based coding and
writing tools for code refinement, structuring, documentation, and analytical
brainstorming. All analytical decisions, business interpretations, modelling
choices, validation procedures, and final project integration were conducted
and reviewed by the author.

**Author:** Enio Rubens — Data Science & Analytics
