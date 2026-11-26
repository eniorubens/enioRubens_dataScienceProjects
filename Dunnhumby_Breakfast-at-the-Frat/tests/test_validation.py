"""Data-quality checks: pass on a clean synthetic panel, fail on targeted corruptions."""

import numpy as np

from retail_multilevel.validation import validate_panel


def _failed(results):
    return {r.name for r in results if not r.passed}


def test_clean_synthetic_panel_passes_all_checks(panel):
    # the synthetic "price far above base" row is legitimate; only structural/consistency checks run
    assert _failed(validate_panel(panel)) == set()


def test_duplicated_key_is_detected(panel):
    import pandas as pd

    broken = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    assert "unique_key" in _failed(validate_panel(broken))


def test_non_finite_log_is_detected(panel):
    broken = panel.copy()
    broken.loc[0, "LOG_UNITS"] = -np.inf
    assert "logs_finite" in _failed(validate_panel(broken))


def test_price_below_base_without_promo_is_detected(panel):
    broken = panel.copy()
    i = broken.index[(broken["ANY_PROMO"] == False) & (~broken["BASE_IMPUTED"])][0]  # noqa: E712
    broken.loc[i, "PRICE"] = 0.5 * broken.loc[i, "BASE_PRICE"]
    broken.loc[i, "SPEND"] = broken.loc[i, "UNITS"] * broken.loc[i, "PRICE"]
    assert _failed(validate_panel(broken)) == {"price_below_base_implies_promo"}


def test_expected_counts(panel):
    ok = {"rows": len(panel), "ambiguous_stores": {2}}
    assert _failed(validate_panel(panel, ok)) == set()
    assert _failed(validate_panel(panel, {"rows": len(panel) + 1})) == {"count_rows"}
