"""
prescriptive.py
---------------
Prescriptive analytics layer: incentive simulation, expected-profit
optimisation and Next Best Action recommendation.

All user-visible text passes through ``t()`` so recommendations are
rendered in the active language without code duplication.

Design note on uplifts
----------------------
All uplift values are *heuristic simulation assumptions*, NOT causal
estimates. They are suitable for portfolio demonstration and sensitivity
analysis. Replace with empirical A/B test results or an uplift model
(e.g. two-model approach, meta-learner) before production use.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import (
    BUDGET_TOTAL,
    BUDGET_CURVE_POINTS,
    INCENTIVE_CATALOGUE,
    MARGIN_RATE,
    NOISE_LEVELS,
    SEGMENT_UPLIFT_MULTIPLIER,
)
from src.multilang import t


# ── Incentive simulation ───────────────────────────────────────────────────

def simulate_actions(
    rfm: pd.DataFrame,
    prob_col: str = "prob_repurchase_90d",
    segment_col: str = "PredictiveSegment",
    revenue_col: str = "Monetary",
    catalogue: dict | None = None,
    multipliers: dict | None = None,
) -> pd.DataFrame:
    """Simulate all actions for every customer and compute expected profit.

    For each (customer, action) pair:

        expected_revenue = repurchase_prob × avg_revenue
        expected_profit  = expected_revenue × MARGIN_RATE − incentive_cost

    Uplift is applied to the baseline repurchase probability:

        adjusted_prob = min(1.0, base_prob + base_uplift × segment_multiplier)

    Parameters
    ----------
    rfm : pd.DataFrame
        Must include *prob_col*, *segment_col*, *revenue_col*.
    prob_col : str
        Column with baseline repurchase probability (0–1).
    segment_col : str
        Column with segment name (English).
    revenue_col : str
        Column with customer revenue used to estimate future spend.
    catalogue : dict, optional
        Override :data:`src.config.INCENTIVE_CATALOGUE`.
    multipliers : dict, optional
        Override :data:`src.config.SEGMENT_UPLIFT_MULTIPLIER`.

    Returns
    -------
    pd.DataFrame
        One row per (customer, action) with columns:
        ``Customer ID``, ``Action``, ``incentive_cost``,
        ``AdjustedProb``, ``ExpectedRevenue``,
        ``ExpectedProfit``, ``ExpectedProfit_NoAction``,
        ``IncrementalProfit``.
    """
    cat = catalogue or INCENTIVE_CATALOGUE
    mult = multipliers or SEGMENT_UPLIFT_MULTIPLIER

    rows = []
    for _, customer in rfm.iterrows():
        base_prob  = float(customer[prob_col])
        segment    = str(customer[segment_col])
        avg_rev    = float(customer[revenue_col])
        multiplier = mult.get(segment, 1.0)

        no_action_profit = base_prob * avg_rev * MARGIN_RATE

        for action, spec in cat.items():
            cost         = float(spec["cost"])
            base_uplift  = float(spec["base_uplift"])
            adj_prob     = min(1.0, base_prob + base_uplift * multiplier)
            exp_revenue  = adj_prob * avg_rev
            exp_profit   = exp_revenue * MARGIN_RATE - cost

            rows.append({
                "Customer ID":            customer["Customer ID"],
                "Action":                 action,
                "incentive_cost":         cost,
                "AdjustedProb":           adj_prob,
                "ExpectedRevenue":        exp_revenue,
                "ExpectedProfit":         exp_profit,
                "ExpectedProfit_NoAction": no_action_profit,
                "IncrementalProfit":      exp_profit - no_action_profit,
            })

    return pd.DataFrame(rows)


def select_best_action(simulation: pd.DataFrame) -> pd.DataFrame:
    """For each customer, select the action with the highest ``ExpectedProfit``.

    Parameters
    ----------
    simulation : pd.DataFrame
        Output of :func:`simulate_actions`.

    Returns
    -------
    pd.DataFrame
        One row per customer with the best action and its metrics.
    """
    idx = simulation.groupby("Customer ID")["ExpectedProfit"].idxmax()
    best = simulation.loc[idx].rename(columns={"Action": "BestAction"}).copy()
    best["RecommendedAction"] = best["BestAction"].map({
        "No Action":        "No Action",
        "Low Incentive":    "Low Incentive — $2 voucher",
        "Medium Incentive": "Medium Incentive — $5 voucher",
        "High Incentive":   "High Incentive — $10 voucher",
    })
    return best.reset_index(drop=True)


# ── Recommendation text ────────────────────────────────────────────────────

def build_recommendation(row: pd.Series) -> str:
    """Translate the optimal action into an executive recommendation.

    Uses ``t()`` so the output language matches the active locale.

    Parameters
    ----------
    row : pd.Series
        Must contain ``BestAction``, ``PredictiveSegment``,
        ``IncrementalProfit``, ``RecommendedAction``.

    Returns
    -------
    str
        Business recommendation sentence.
    """
    action_label = row["RecommendedAction"]
    segment      = row.get("PredictiveSegment", "")
    incremental  = row["IncrementalProfit"]

    if row["BestAction"] == "No Action" or incremental <= 0:
        return f"{action_label}: {t('preserve margin; incentive does not increase expected profit.')}"

    if segment in {"Champions", "High Value at Risk"} and row["BestAction"] == "High Incentive":
        return t("Maximum retention priority with aggressive campaign.")

    if segment in {"Champions", "High Value at Risk"}:
        return t("Prioritize relationship with {}.").format(action_label.lower())

    if segment == "Low Value" and row["BestAction"] == "High Incentive":
        return t("Review cost before aggressive campaign for low-value customer.")

    if segment == "Inactive Customers":
        return t("Test controlled reactivation with {}.").format(action_label.lower())

    return t("Execute {} based on positive expected profit.").format(action_label.lower())


# ── Budget optimisation ────────────────────────────────────────────────────

def allocate_budget(
    prescriptive_df: pd.DataFrame,
    budget: float = BUDGET_TOTAL,
) -> pd.DataFrame:
    """Greedy ROI-ranked budget allocation.

    Customers are ranked by ``IncrementalProfit`` descending.
    Incentives are assigned until the budget is exhausted.

    Parameters
    ----------
    prescriptive_df : pd.DataFrame
        Output of :func:`select_best_action` merged with RFM.
    budget : float
        Total available spend.

    Returns
    -------
    pd.DataFrame
        Input with an added boolean ``BudgetAllocated`` column.
    """
    df = prescriptive_df.copy()
    df = df.sort_values("IncrementalProfit", ascending=False).reset_index(drop=True)

    remaining = budget
    allocated = []
    for _, row in df.iterrows():
        cost = float(row["incentive_cost"])
        if cost > 0 and cost <= remaining:
            remaining -= cost
            allocated.append(True)
        else:
            allocated.append(False)

    df["BudgetAllocated"] = allocated
    return df


def build_budget_curve(
    prescriptive_df: pd.DataFrame,
    max_budget: float | None = None,
    n_points: int = BUDGET_CURVE_POINTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep budget levels and compute incremental profit and ROI at each point.

    Parameters
    ----------
    prescriptive_df : pd.DataFrame
        Output of :func:`select_best_action`.
    max_budget : float, optional
        Upper bound of sweep.  Defaults to total cost of all positive actions.
    n_points : int
        Number of budget levels to evaluate.

    Returns
    -------
    curve_df : pd.DataFrame
        Columns: ``Budget``, ``IncrementalProfit``, ``CustomersServed``,
        ``ActualCost``, ``ROI``.
    breakpoints : pd.DataFrame
        Curve values at 25 / 50 / 75 / 100 % of ``max_budget``.
    """
    candidates = (
        prescriptive_df
        .loc[prescriptive_df["incentive_cost"] > 0]
        .sort_values("IncrementalProfit", ascending=False)
        .reset_index(drop=True)
    )
    max_possible = float(candidates["incentive_cost"].sum()) if max_budget is None else max_budget

    curve_rows = []
    for b in np.linspace(0, max_possible, n_points):
        remaining = b
        profit = cost = clients = 0.0
        for _, row in candidates.iterrows():
            if row["incentive_cost"] <= remaining:
                remaining -= row["incentive_cost"]
                profit  += row["IncrementalProfit"]
                cost    += row["incentive_cost"]
                clients += 1
        curve_rows.append({
            "Budget":            b,
            "IncrementalProfit": profit,
            "CustomersServed":   int(clients),
            "ActualCost":        cost,
            "ROI":               profit / cost if cost > 0 else 0.0,
        })

    curve_df = pd.DataFrame(curve_rows)

    breakpoints_rows = []
    for pct in [0.25, 0.50, 0.75, 1.00]:
        target = max_possible * pct
        idx    = (curve_df["Budget"] - target).abs().idxmin()
        row    = curve_df.loc[idx]
        breakpoints_rows.append({
            "BudgetPercentile":  f"{int(pct * 100)}%",
            "Budget":            row["Budget"],
            "IncrementalProfit": row["IncrementalProfit"],
            "CustomersServed":   row["CustomersServed"],
            "ROI":               row["ROI"],
        })

    return curve_df, pd.DataFrame(breakpoints_rows)


# ── Sensitivity analysis ───────────────────────────────────────────────────

def sensitivity_analysis(
    prescriptive_df: pd.DataFrame,
    noise_levels: list[float] = NOISE_LEVELS,
) -> pd.DataFrame:
    """Simulate total incremental profit under probability perturbations.

    Parameters
    ----------
    prescriptive_df : pd.DataFrame
        Must contain ``prob_repurchase_90d``, ``Monetary``, ``incentive_cost``.
    noise_levels : list[float]
        Probability shifts to simulate (e.g. ``-0.10`` = −10 pp).

    Returns
    -------
    pd.DataFrame
        Columns: ``NoiseDelta``, ``TotalIncrementalProfit``.
    """
    rows = []
    for delta in noise_levels:
        df = prescriptive_df.copy()
        df["adj_prob"] = (df["prob_repurchase_90d"] + delta).clip(0, 1)
        df["adj_profit"] = (
            df["adj_prob"] * df["Monetary"] * MARGIN_RATE - df["incentive_cost"]
        )
        rows.append({
            "NoiseDelta":            delta,
            "TotalIncrementalProfit": df["adj_profit"].sum(),
        })
    return pd.DataFrame(rows)


# ── CRM export ─────────────────────────────────────────────────────────────

def build_crm_export(rfm: pd.DataFrame, prescriptive_df: pd.DataFrame) -> pd.DataFrame:
    """Join RFM features with prescriptive decisions for CRM delivery.

    Parameters
    ----------
    rfm : pd.DataFrame
    prescriptive_df : pd.DataFrame
        Output of :func:`select_best_action`.

    Returns
    -------
    pd.DataFrame
        CRM-ready table with ``priority_rank`` column.
    """
    prescription_cols = [
        "Customer ID", "BestAction", "RecommendedAction",
        "ExpectedProfit", "ExpectedProfit_NoAction",
        "IncrementalProfit", "PrescriptiveNextBestAction",
    ]
    present = [c for c in prescription_cols if c in prescriptive_df.columns]

    export_cols = {
        "Customer ID":               "customer_id",
        "SegmentName":               "segment_name",
        "Recency":                   "recency_days",
        "Frequency":                 "frequency",
        "Monetary":                  "monetary_total",
        "AverageTicket":             "average_ticket",
        "prob_repurchase_90d":       "prob_repurchase_90d",
        "BestAction":                "best_action_code",
        "RecommendedAction":         "recommended_campaign",
        "ExpectedProfit":            "expected_profit",
        "ExpectedProfit_NoAction":   "expected_profit_no_action",
        "IncrementalProfit":         "incremental_profit",
        "PrescriptiveNextBestAction": "crm_action_description",
    }

    drop_cols = [c for c in present if c != "Customer ID"]
    crm_base = (
        rfm.drop(columns=drop_cols, errors="ignore")
        .merge(prescriptive_df[present], on="Customer ID", how="inner")
    )

    available = [c for c in export_cols if c in crm_base.columns]
    crm_out = (
        crm_base[available]
        .rename(columns={k: export_cols[k] for k in available})
        .sort_values(
            ["segment_name", "incremental_profit"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )
    crm_out["priority_rank"] = (
        crm_out["incremental_profit"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return crm_out
