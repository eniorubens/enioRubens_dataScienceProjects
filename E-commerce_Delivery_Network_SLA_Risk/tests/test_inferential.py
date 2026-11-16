import numpy as np
import pandas as pd

from ecom_delivery_sla.inferential import (
    coefficient_table,
    fit_logit_inferential,
    fit_ols_inferential,
    ols_diagnostics,
    standardized_ols,
    vif_table,
)


def _inferential_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 40
    promised = np.linspace(5, 25, n)
    state = np.where(np.arange(n) % 2, "RJ", "SP")
    price = rng.uniform(20, 200, n)
    delay = 0.4 * promised + (state == "RJ") * 1.5 + rng.normal(0, 0.2, n)
    return pd.DataFrame(
        {
            "delivery_delay_days": delay,
            "sla_breach": rng.binomial(1, 0.35, n).astype(bool),
            "promised_lead_days": promised,
            "total_price": price,
            "customer_state": state,
            "primary_seller_id": [f"s{i % 5}" for i in range(n)],
        }
    )


def test_inferential_outputs_include_cluster_hc3_and_diagnostics():
    frame = _inferential_frame()
    results = fit_ols_inferential(frame)
    table = coefficient_table(results["hc3"])
    diagnostics = ols_diagnostics(results["hc3"])
    standardized = standardized_ols(frame)
    vif = vif_table(results["hc3"])

    assert set(results) == {"hc3", "clustered"}
    assert {"term", "ci_lower", "ci_upper"}.issubset(table.columns)
    assert "breusch_pagan" in diagnostics
    assert np.isfinite(standardized.params).all()
    assert not vif.empty


def test_logit_inferential_returns_odds_ratio_table():
    frame = _inferential_frame()
    result, table = fit_logit_inferential(frame)

    assert result is not None
    assert {"term", "odds_ratio", "ci_lower", "ci_upper"}.issubset(table.columns)
