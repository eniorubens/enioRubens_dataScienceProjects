"""
config.py
---------
Single source of truth for all operational parameters.
Edit this file to re-run the pipeline with a different scenario.
"""

from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

_DATA_CANDIDATES = [
    DATA_DIR / "online_retail.csv",
    DATA_DIR / "online_retail_II.csv",
]
for _candidate in _DATA_CANDIDATES:
    if _candidate.exists():
        DATA_PATH = _candidate
        break
else:
    # Keep a deterministic default when dataset is not present yet.
    DATA_PATH = _DATA_CANDIDATES[0]

EXPORT_PATH = OUTPUT_DIR / "crm_next_best_action.csv"

# ── Segmentation ───────────────────────────────────────────────────────────
N_CLUSTERS: int = 6          # number of RFM behavioural segments

# ── Predictive layer ───────────────────────────────────────────────────────
PREDICTION_HORIZON_DAYS: int = 90    # target window for future repurchase
HOLDOUT_FRACTION: float      = 0.15  # independent hold-out fraction

# ── Prescriptive / financial layer ────────────────────────────────────────
BUDGET_TOTAL:    float = 10_000.00   # total available incentive budget ($)
MARGIN_RATE:     float = 0.35        # estimated gross margin over revenue
UNIFORM_UPLIFT:  float = 0.07        # uplift for uniform-campaign scenario (Scenario B)
UNIFORM_COST:    float = 5.00        # cost of uniform incentive ($)

# ── Sensitivity analysis ───────────────────────────────────────────────────
NOISE_LEVELS: list[float] = [-0.15, -0.10, -0.05, 0.00, +0.05, +0.10, +0.15]
BUDGET_CURVE_POINTS: int  = 50       # granularity of budget-curve sweep

# ── Incentive action catalogue ─────────────────────────────────────────────
# Keys are action codes; values carry cost and *heuristic* baseline uplift.
# NOTE: uplifts are simulation assumptions, NOT causal estimates.
# Replace with empirical A/B or uplift-model results when available.
INCENTIVE_CATALOGUE: dict[str, dict] = {
    "No Action":        {"cost": 0.00, "base_uplift": 0.00},
    "Low Incentive":    {"cost": 2.00, "base_uplift": 0.03},
    "Medium Incentive": {"cost": 5.00, "base_uplift": 0.07},
    "High Incentive":   {"cost": 10.00, "base_uplift": 0.12},
}

# ── Per-segment uplift multipliers ────────────────────────────────────────
# Segment names must match those produced by segmentation.segment_names().
SEGMENT_UPLIFT_MULTIPLIER: dict[str, float] = {
    "Champions":          1.0,
    "Loyal Customers":    1.1,
    "High Value at Risk": 1.3,
    "Occasional Buyers":  0.8,
    "Low Value":          0.6,
    "Inactive Customers": 0.5,
}
