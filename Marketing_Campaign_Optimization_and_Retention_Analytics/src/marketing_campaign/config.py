"""Business constants and defaults shared by the analysis modules.

The values are the ones used in the notebooks. Costs and revenues are
configurable assumptions, not observed financial data.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "marketing_new.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

COST_PER_IMPRESSION: dict[str, float] = {
    "House Ads": 1.0,
    "Email": 0.8,
    "Social Media": 1.5,
    "Paid Search": 2.5,
    "Organic Search": 0.5,
    "Referral": 1.0,
    "Display": 2.0,
}
DEFAULT_COST = 1.2
REVENUE_PER_CONVERSION = 45.0
ESTIMATED_CUSTOMER_VALUE = 120.0
RANDOM_STATE = 42
SELECTED_K = 3
ALPHA = 0.05
POWER = 0.80
MINIMUM_LIFT_PP = 2.0
ORGANIC_CONVERSION_RATE = 0.20
MC_N_SIMULATIONS = 10_000
RETENTION_LTV_SHARE = 0.15

SEGMENT_LABELS = ["High Value", "Medium Value", "Low Value"]

SEGMENT_ACTIONS: dict[str, dict[str, object]] = {
    "High Value": {
        "action": "Loyalty Campaign",
        "rationale": (
            "High-value customers - maintain engagement and reduce churn."
        ),
        "max_incentive": 15.0,
    },
    "Medium Value": {
        "action": "Promotional Offer",
        "rationale": (
            "Customers with potential - increase conversion and "
            "average ticket."
        ),
        "max_incentive": 8.0,
    },
    "Low Value": {
        "action": "Win-Back Campaign or No Action",
        "rationale": (
            "Low ROI - evaluate whether reactivation cost is justified."
        ),
        "max_incentive": 3.0,
    },
}
NO_ACTION = "No Action - Negative Incremental Profit"
