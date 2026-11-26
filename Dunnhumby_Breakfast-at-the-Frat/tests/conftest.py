"""Synthetic fixtures shaped like the dunnhumby sheets. No test reads local data."""

import numpy as np
import pandas as pd
import pytest

TX_COLS = [
    "STORE_NUM",
    "UPC",
    "WEEK_END_DATE",
    "UNITS",
    "VISITS",
    "HHS",
    "SPEND",
    "PRICE",
    "BASE_PRICE",
    "FEATURE",
    "DISPLAY",
    "TPR_ONLY",
]
WEEKS = pd.to_datetime(["2009-01-14", "2009-01-21", "2009-01-28"])


@pytest.fixture
def stores_raw():
    base = dict(
        STORE_NAME="X",
        ADDRESS_CITY_NAME="C",
        ADDRESS_STATE_PROV_CODE="TX",
        MSA_CODE=1,
        PARKING_SPACE_QTY=np.nan,
        SALES_AREA_SIZE_NUM=100,
        AVG_WEEKLY_BASKETS=10.0,
    )
    return pd.DataFrame(
        [
            {**base, "STORE_ID": 1, "SEG_VALUE_NAME": "VALUE"},
            {**base, "STORE_ID": 2, "SEG_VALUE_NAME": "MAINSTREAM"},
            {**base, "STORE_ID": 2, "SEG_VALUE_NAME": "UPSCALE"},
        ]
    )


@pytest.fixture
def products_raw():
    return pd.DataFrame(
        {
            "UPC": [10, 20],
            "DESCRIPTION": ["a", "b"],
            "MANUFACTURER": ["PRIVATE LABEL", "KING"],
            "CATEGORY": ["C", "C"],
            "SUB_CATEGORY": ["S", "S"],
            "PRODUCT_SIZE": ["1", "1"],
        }
    )


@pytest.fixture
def tx_raw():
    w = WEEKS
    rows = [
        # store, upc, week, units, visits, hhs, spend, price, base, F, D, T
        (1, 10, w[0], 4, 4, 4, 8.0, 2.0, 2.0, 0, 0, 0),
        (1, 10, w[1], 6, 6, 5, 9.0, 1.5, np.nan, 0, 0, 1),  # base imputed from week 0
        (1, 10, w[2], 1, 1, 1, 0.0, np.nan, 2.0, 0, 0, 0),  # invalid price -> dropped
        (2, 10, w[0], 60, 2, 2, 120.0, 2.0, 2.0, 0, 0, 0),  # units/visit = 30 -> outlier
        (2, 20, w[0], 3, 3, 1, 9.0, 3.0, np.nan, 0, 0, 0),  # visits/hh = 3 (not > 3)
        (2, 20, w[1], 3, 3, 1, 10.5, 3.5, np.nan, 0, 0, 0),  # no base in pair -> price
        (1, 20, w[0], 2, 2, 2, 4.0, 2.0, 0.0001, 1, 0, 0),  # price far above base
    ]
    return pd.DataFrame(rows, columns=TX_COLS)


@pytest.fixture
def panel(stores_raw, products_raw, tx_raw):
    from retail_multilevel.cleaning import build_panel

    return build_panel(stores_raw, products_raw, tx_raw)
