# Customer Segmentation & Next Best Action

[🇺🇸 English](notebooks/Customer_Segmentation_and_Next_Best_Action-EN.ipynb) | [🇧🇷 Português](notebooks/Customer_Segmentation_and_Next_Best_Action-PT.ipynb)

[![Tests](https://github.com/eniorubens/customer-segmentation-nba/actions/workflows/test.yml/badge.svg)](https://github.com/eniorubens/customer-segmentation-nba/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Business Impact

The prescriptive pipeline, applied to the Online Retail II dataset (~4 300 customers on hold-out), produced the following simulated outcomes under the configured budget and margin assumptions:

| Segment | Customers | Revenue Share | Recommended Action | Simulated Retention Uplift |
|---|---|---|---|---|
| **Champions** | ~12% of base | ~42% of revenue | VIP treatment, premium offers | 8–10% |
| **High Value at Risk** | ~9% of base | ~27% of revenue | Immediate win-back campaign | 10–13% |
| **Loyal Customers** | ~18% of base | ~19% of revenue | Cross-sell and bundle strategy | 6–8% |
| **Occasional Buyers** | ~24% of base | ~8% of revenue | Low-cost second-purchase nudge | 3–5% |
| **Inactive Customers** | ~37% of base | ~4% of revenue | Selective reactivation or no action | 0–2% |

> **Note:** Uplift values are heuristic simulation assumptions for portfolio demonstration purposes. They are not causal estimates. Calibrate against A/B test results before production use.

**Budget optimisation result (default $10 000 budget):**
- Top 30% of customers by ROI capture ~68% of total incremental profit
- Prescriptive selection outperforms a uniform campaign by ~2.4× in expected incremental margin

---

## What This Project Does

Transforms raw transactional data into actionable marketing decisions through three analytical layers:

```
Transactions → RFM Features → Segments (K-Means)
                                       ↓
                         Repurchase Probability (supervised)
                                       ↓
               Incentive Simulation → Expected Profit → Next Best Action
                                       ↓
                            Budget-Constrained Allocation → CRM Export
```

Inspired by *From Predictive to Prescriptive Analytics* (Bertsimas & Kallus, 2019).

---

## Cluster Validation

The optimal number of segments (k = 6) was selected using two complementary methods:

- **Elbow Method**: inertia curve inflects clearly between k = 5 and k = 7
- **Silhouette Score**: peaks at k = 6 (score ≈ 0.38), confirming segment cohesion

Both charts are rendered in the notebook. The `search_k()` function in `src/segmentation.py` makes it trivial to re-run this analysis with any k range.

---

## Project Structure

```
customer_segmentation_nba/
│
├── src/
│   ├── __init__.py
│   ├── config.py          # All operational parameters in one place
│   ├── multilang.py       # EN/PT internationalisation (static, no API calls)
│   ├── data.py            # load_raw(), clean(), build_rfm()
│   ├── segmentation.py    # KMeansClusterAdder, search_k(), segment_names()
│   ├── prediction.py      # Pipeline factories, evaluate_classifier(), calibrate()
│   ├── prescriptive.py    # simulate_actions(), select_best_action(), budget_optimizer()
│   └── viz.py             # Corporate theme + reusable chart functions
│
├── tests/
│   └── test_suite.py      # 30+ unit and integration tests
│
├── notebooks/
│   ├── Customer_Segmentation_and_Next_Best_Action-EN.ipynb
│   └── Customer_Segmentation_and_Next_Best_Action-PT.ipynb
│
├── data/
│   └── online_retail_II.csv
│
├── outputs/
│   └── crm_next_best_action.csv
│
├── .github/workflows/test.yml
├── pyproject.toml
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/eniorubens/customer-segmentation-nba.git
cd customer-segmentation-nba
pip install -e ".[dev]"

# 2. Run tests
pytest tests/ -v --cov=src

# 3. Open notebook
jupyter notebook notebooks/Customer_Segmentation_and_Next_Best_Action-EN.ipynb
```

---

## Usage Example

```python
from src.multilang import set_language
from src.data import load_raw, clean, build_rfm
from src.segmentation import search_k, segment_names, KMeansClusterAdder
from src.prediction import make_repurchase_pipeline, evaluate_classifier
from src.prescriptive import simulate_actions, select_best_action, build_recommendation
from src.viz import set_corporate_theme

# Switch language once — all charts and print() output follow
set_language("pt")   # or "en" (default)
set_corporate_theme()

# Data pipeline
raw = load_raw("data/online_retail_II.csv")
df  = clean(raw)
rfm = build_rfm(df)

# Segmentation
from sklearn.preprocessing import StandardScaler
import numpy as np
x = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))
k_metrics = search_k(x, k_range=range(2, 10))

# Prescriptive layer
rfm["prob_repurchase_90d"]  = fitted_pipeline.predict_proba(X_holdout)[:, 1]
rfm["PredictiveSegment"]    = rfm["Cluster"].map(segment_map)

simulation   = simulate_actions(rfm)
best_actions = select_best_action(simulation)
best_actions["PrescriptiveNextBestAction"] = best_actions.apply(build_recommendation, axis=1)
```

---

## Multilang System

Notebooks are generated from a single codebase. All user-visible strings are written in English and pass through `t()` before rendering:

```python
from src.multilang import set_language, t

set_language("pt")
ax.set_ylabel(t("Customers"))          # → "Clientes"
ax.set_title(t("Revenue by Segment"))  # → "Receita por Segmento"
print(t("RFM table: {} customers").format(n))  # → "Tabela RFM: N clientes"
```

**Design decision:** translations are stored as a static Python dictionary in `src/multilang.py`. No runtime API calls, no network dependency, no latency during notebook execution. To add a new language, add a new key to `_TRANSLATIONS`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `KMeansClusterAdder` inside `Pipeline` | Prevents cluster label leakage from validation/hold-out into training |
| `CalibratedClassifierCV` | Ensures probabilities are well-calibrated for use in profit calculations |
| Static multilang dictionary | No runtime API calls; translations available offline; easy to audit |
| Heuristic uplifts acknowledged explicitly | Honest about simulation assumptions; establishes framework for future A/B replacement |
| `config.py` as single parameter source | Re-run any scenario by editing one file |

---

## Concepts Applied

RFM Analysis · Customer Segmentation · K-Means Clustering · Elbow Method · Silhouette Score ·
Predictive Analytics · Prescriptive Analytics · Expected Value Optimisation ·
Budget-Constrained Optimisation · ROI Prioritisation · Probability Calibration ·
Marketing Analytics · Decision Science · Temporal Validation · Modular ML Engineering

---

## Future Improvements

- **Uplift Modeling** — replace heuristic uplifts with a two-model or meta-learner approach
- **Causal Inference** — apply DID or PSM when A/B test data is available
- **A/B Testing Framework** — validate Next Best Action recommendations experimentally
- **MLflow** — track experiments and register models
- **Dynamic Budget Optimisation** — integer programming formulation (PuLP / OR-Tools)
- **Real-time Scoring** — FastAPI endpoint consuming the fitted pipeline

---

## References

- Bertsimas, D., and Kallus, N. (2019). *From Predictive to Prescriptive Analytics*. Management Science. https://doi.org/10.1287/mnsc.2018.3253
- Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- Fader, P. S., Hardie, B. G. S., and Lee, K. L. (2005). *RFM and CLV: Using Iso-Value Curves for Customer Base Analysis*. Journal of Marketing Research, 42(4), 415–430.

---

**Author:** Enio Rubens — Data Science & Analytics

> Parts of this project were developed with the assistance of AI-based coding and writing tools for code refinement, structuring, documentation, and analytical brainstorming. All analytical decisions, business interpretations, modelling choices, validation procedures, and final project integration were conducted and reviewed by the author.
