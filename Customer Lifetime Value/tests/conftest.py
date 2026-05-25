"""
conftest.py — Shared pytest fixtures (synthetic data only, no real dataset dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Raw transactions ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """
    50 synthetic transactions across 10 customers covering edge cases:
    multiple purchases, cancellations, NaN CustomerID, negative prices.
    """
    rng = np.random.default_rng(42)
    n = 50
    customer_ids: list = [f"{10000 + i}" for i in range(10)] * 5
    customer_ids[0] = float("nan")   # NaN CustomerID
    customer_ids[1] = float("nan")

    invoice_nos = [f"INV{str(i).zfill(4)}" for i in range(n)]
    invoice_nos[5] = "C00005"   # cancellation
    invoice_nos[6] = "C00006"   # cancellation

    quantities = rng.integers(1, 20, size=n).astype(float)
    quantities[3] = -2.0   # negative quantity
    quantities[4] = 0.0    # zero quantity

    unit_prices = rng.uniform(1.0, 50.0, size=n)
    unit_prices[7] = -5.0  # negative price
    unit_prices[8] = 0.0   # zero price

    # Dates spanning ~12 months
    base_date = pd.Timestamp("2010-12-01")
    dates = [base_date + pd.Timedelta(days=int(d)) for d in rng.integers(0, 370, size=n)]
    date_strs = [d.strftime("%m/%d/%y %H:%M") for d in dates]

    return pd.DataFrame(
        {
            "InvoiceNo": invoice_nos,
            "Quantity": quantities,
            "InvoiceDate": date_strs,
            "UnitPrice": unit_prices,
            "CustomerID": customer_ids,
        }
    )


# ── RFM matrix ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_rfm() -> pd.DataFrame:
    """
    Synthetic RFM matrix with 20 customers.
    Includes customers with frequency=0 (single purchase) and frequency>0.
    """
    rng = np.random.default_rng(7)
    n = 20
    ids = [f"C{str(i).zfill(4)}" for i in range(n)]
    frequency = [0] * 5 + list(rng.integers(1, 15, size=n - 5))
    recency = [0.0] * 5 + list(rng.uniform(10, 300, size=n - 5))
    T = [rng.integers(30, 370)] + list(rng.integers(50, 400, size=n - 1))
    monetary_value = [0.0] * 5 + list(rng.uniform(20, 500, size=n - 5))

    # BG/NBD requires recency <= T for all customers
    T_floats = [float(t) for t in T]
    recency = [min(r, t - 1.0) if r > 0 else r for r, t in zip(recency, T_floats)]

    return pd.DataFrame(
        {
            "customer_id": ids,
            "frequency": frequency,
            "recency": recency,
            "T": T_floats,
            "monetary_value": monetary_value,
        }
    )


@pytest.fixture
def sample_modeling_base(sample_rfm: pd.DataFrame) -> pd.DataFrame:
    """sample_rfm filtered to frequency > 0 and monetary_value > 0."""
    return sample_rfm.loc[
        (sample_rfm["frequency"] > 0) & (sample_rfm["monetary_value"] > 0)
    ].copy().reset_index(drop=True)


@pytest.fixture
def sample_calibration_holdout() -> pd.DataFrame:
    """
    Synthetic calibration/holdout DataFrame with 15 customers.
    """
    rng = np.random.default_rng(13)
    n = 15
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "frequency_cal": rng.integers(1, 10, size=n).astype(float),
            "recency_cal": rng.uniform(5, 200, size=n),
            "T_cal": rng.uniform(200, 280, size=n),
            "monetary_value_cal": rng.uniform(30, 400, size=n),
            "frequency_holdout": rng.integers(0, 5, size=n).astype(float),
            "T_holdout": rng.uniform(60, 100, size=n),
        }
    )


# ── Fitted model ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fitted_model(sample_modeling_base_session):
    """
    CLTVModel trained with small synthetic data using MAP (fast).
    Session-scoped to avoid repeated expensive fitting.
    """
    from src.cltv_model import CLTVModel

    model = CLTVModel(fit_method="map")
    model.fit(sample_modeling_base_session)
    return model


@pytest.fixture(scope="session")
def sample_modeling_base_session() -> pd.DataFrame:
    """Session-scoped version of sample_modeling_base for use by fitted_model."""
    rng = np.random.default_rng(7)
    n = 20
    ids = [f"C{str(i).zfill(4)}" for i in range(n)]
    frequency = list(rng.integers(1, 15, size=n))
    recency_raw = list(rng.uniform(10, 300, size=n))
    T = list(rng.integers(50, 400, size=n))
    monetary_value = list(rng.uniform(20, 500, size=n))

    # BG/NBD requires recency <= T for all customers
    recency = [min(r, float(t) - 1.0) for r, t in zip(recency_raw, T)]

    return pd.DataFrame(
        {
            "customer_id": ids,
            "frequency": frequency,
            "recency": recency,
            "T": [float(t) for t in T],
            "monetary_value": monetary_value,
        }
    )
