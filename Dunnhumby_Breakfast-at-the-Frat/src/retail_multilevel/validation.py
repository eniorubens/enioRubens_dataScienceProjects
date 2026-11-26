"""Data-quality contract of the analytical panel.

The checks are pure functions over a panel frame, so they are unit-tested on synthetic
data in CI and run on the real (licensed, local-only) panel by ``scripts/validate_data.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEY = ["WEEK_END_DATE", "STORE_NUM", "UPC"]
MODEL_COLS = [
    "LOG_UNITS",
    "LOG_PRICE",
    "LOG_BASE_PRICE",
    "DISCOUNT",
    "FEATURE",
    "DISPLAY",
    "TPR_ONLY",
    "STORE_NAME",
    "CATEGORY",
    "MANUFACTURER",
]
SEGMENTS = {"MAINSTREAM", "VALUE", "UPSCALE"}

# Expected values for the real dunnhumby panel (see docs/REGISTRO-DE-DECISOES.md, D6-D10; D8 revised to history-only imputation).
EXPECTED_REAL = {
    "rows": 524_926,
    "stores": 77,
    "upcs": 55,
    "weeks": 156,
    "base_imputed": 185,
    "base_ffill": 44,
    "base_fallback": 141,
    "outlier_upv": 67,
    "ambiguous_stores": {4503, 17627},
}


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one data-quality check."""

    name: str
    passed: bool
    detail: str = ""


def check_structure(panel: pd.DataFrame) -> list[CheckResult]:
    """Check key uniqueness and that model variables are present and finite."""
    dup = int(panel.duplicated(KEY).sum())
    n_missing = int(panel[MODEL_COLS].isna().sum().sum())
    logs = panel[["LOG_UNITS", "LOG_PRICE", "LOG_BASE_PRICE"]].to_numpy()
    n_inf = int((~np.isfinite(logs)).sum())
    return [
        CheckResult("unique_key", dup == 0, f"{dup} duplicated (week, store, upc)"),
        CheckResult("model_vars_not_missing", n_missing == 0, f"{n_missing} missing values"),
        CheckResult("logs_finite", n_inf == 0, f"{n_inf} non-finite log values"),
    ]


def check_consistency(panel: pd.DataFrame, spend_tol: float = 0.05) -> list[CheckResult]:
    """Check SPEND = UNITS x PRICE, promotion-flag exclusivity and price-below-base vs promotion."""
    rel = (panel["SPEND"] - panel["UNITS"] * panel["PRICE"]).abs() / panel["SPEND"]
    n_spend = int((rel >= spend_tol).sum())
    tpr_mixed = int(((panel["TPR_ONLY"] == 1) & ((panel["FEATURE"] == 1) | (panel["DISPLAY"] == 1))).sum())
    below = (panel["PRICE"] < 0.95 * panel["BASE_PRICE"]) & ~panel["BASE_IMPUTED"]
    below_no_promo = int((below & ~panel["ANY_PROMO"]).sum())
    fb = panel["BASE_FALLBACK"]
    src_ok = (
        panel["BASE_SOURCE"].isin({"observed", "ffill", "fallback_price"}).all()
        and bool((panel["BASE_IMPUTED"] == (panel["BASE_SOURCE"] != "observed")).all())
        and bool((fb == (panel["BASE_SOURCE"] == "fallback_price")).all())
        and bool(np.allclose(panel.loc[fb, "BASE_PRICE"], panel.loc[fb, "PRICE"]))
    )
    seg_ok = panel.loc[~panel["SEG_AMBIGUOUS"], "SEGMENT"].isin(SEGMENTS).all()
    seg_amb_empty = panel.loc[panel["SEG_AMBIGUOUS"], "SEGMENT"].isna().all()
    return [
        CheckResult("spend_equals_units_x_price", n_spend == 0, f"{n_spend} rows off by >= {spend_tol:.0%}"),
        CheckResult("tpr_exclusive_of_feature_display", tpr_mixed == 0, f"{tpr_mixed} rows"),
        CheckResult("price_below_base_implies_promo", below_no_promo == 0, f"{below_no_promo} rows"),
        CheckResult("base_source_consistent", src_ok, "fallback rows have BASE_PRICE = PRICE (discount 0)"),
        CheckResult("segments_valid", bool(seg_ok and seg_amb_empty), "known segment or empty if ambiguous"),
        CheckResult("in_model_excludes_upv_outliers", bool((panel["IN_MODEL"] == ~panel["OUTLIER_UPV"]).all())),
    ]


def check_expected_counts(panel: pd.DataFrame, expected: dict) -> list[CheckResult]:
    """Compare panel counts with the expected values recorded in the decision log."""
    observed = {
        "rows": len(panel),
        "stores": panel["STORE_NUM"].nunique(),
        "upcs": panel["UPC"].nunique(),
        "weeks": panel["WEEK"].nunique(),
        "base_imputed": int(panel["BASE_IMPUTED"].sum()),
        "base_ffill": int(panel["BASE_SOURCE"].eq("ffill").sum()),
        "base_fallback": int(panel["BASE_FALLBACK"].sum()),
        "outlier_upv": int(panel["OUTLIER_UPV"].sum()),
        "ambiguous_stores": {int(s) for s in panel.loc[panel["SEG_AMBIGUOUS"], "STORE_NUM"].unique()},
    }
    return [
        CheckResult(f"count_{k}", observed[k] == v, f"observed {observed[k]}, expected {v}")
        for k, v in expected.items()
    ]


def validate_panel(panel: pd.DataFrame, expected: dict | None = None) -> list[CheckResult]:
    """Run every check; ``expected`` adds the count checks (use :data:`EXPECTED_REAL` on real data)."""
    results = check_structure(panel) + check_consistency(panel)
    if expected is not None:
        results += check_expected_counts(panel, expected)
    return results
