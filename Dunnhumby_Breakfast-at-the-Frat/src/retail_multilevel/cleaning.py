"""Cleaning rules and the analytical store x product x week panel.

Every rule is documented in docs/REGISTRO-DE-DECISOES.md (D6-D10; D8 revised). Rows are dropped only
when they are unusable (no valid price); every other anomaly becomes a boolean flag so
that its effect can be checked in sensitivity analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNITS_PER_VISIT_MAX = 5.0  # dunnhumby user guide: above this the row is suspect
VISITS_PER_HH_MAX = 3.0  # dunnhumby user guide
PRICE_ABOVE_BASE_TOL = 1.05  # price more than 5% above base price
N_WEEKS_TOTAL = 156

PROMO_COLS = ["FEATURE", "DISPLAY", "TPR_ONLY"]


def clean_stores(stores: pd.DataFrame) -> pd.DataFrame:
    """One row per store.

    Stores listed with more than one segment (4503 and 17627 in the source) keep their
    other attributes, get ``SEG_AMBIGUOUS = True`` and a missing ``SEGMENT``: the data
    do not tell which label is right, so no segment is invented (D6).
    """
    df = stores.rename(columns={"STORE_ID": "STORE_NUM", "SEG_VALUE_NAME": "SEGMENT"})
    n_seg = df.groupby("STORE_NUM")["SEGMENT"].transform("nunique")
    df = df.assign(SEG_AMBIGUOUS=n_seg > 1)
    df.loc[df["SEG_AMBIGUOUS"], "SEGMENT"] = np.nan
    df = df.drop_duplicates("STORE_NUM").reset_index(drop=True)
    return df.rename(
        columns={
            "ADDRESS_CITY_NAME": "CITY",
            "ADDRESS_STATE_PROV_CODE": "STATE",
            "SALES_AREA_SIZE_NUM": "SALES_AREA",
        }
    )


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    """Add ``PRIVATE_LABEL``."""
    return products.assign(PRIVATE_LABEL=products["MANUFACTURER"].eq("PRIVATE LABEL"))


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Apply the row-level rules to the transaction table.

    - D7: drop rows with missing or non-positive PRICE (all have SPEND = 0).
    - D8 (revised after the external audit): a missing BASE_PRICE is filled **only from earlier
      weeks** of the same store x UPC (last observed value carried forward), so every row is
      cleaned with information available at its own date. When the pair has no earlier base
      price, ``BASE_PRICE = PRICE`` (fallback), which forces ``DISCOUNT = LOG_DISC = 0`` in that
      row even if a promotion is running. ``BASE_SOURCE`` is ``"observed"``, ``"ffill"`` or
      ``"fallback_price"``; ``BASE_FALLBACK`` marks the last one; ``BASE_IMPUTED`` is true for
      both imputed kinds.
    - D9: flag ``OUTLIER_UPV`` (units/visit > 5) and ``OUTLIER_VPH`` (visits/HH > 3).
    - D10: flag ``PRICE_ABOVE_BASE`` (price > 1.05 x base).
    """
    df = tx.loc[tx["PRICE"].notna() & (tx["PRICE"] > 0)].copy()
    df = df.sort_values(["STORE_NUM", "UPC", "WEEK_END_DATE"], ignore_index=True)

    observed = df["BASE_PRICE"].notna()
    carried = df.groupby(["STORE_NUM", "UPC"])["BASE_PRICE"].ffill()  # earlier weeks only
    source = np.where(observed, "observed", np.where(carried.notna(), "ffill", "fallback_price"))
    df["BASE_SOURCE"] = pd.Series(source, index=df.index, dtype="object")
    df["BASE_FALLBACK"] = df["BASE_SOURCE"].eq("fallback_price")
    df["BASE_IMPUTED"] = ~observed
    df["BASE_PRICE"] = carried.fillna(df["PRICE"])

    df["OUTLIER_UPV"] = df["UNITS"] / df["VISITS"] > UNITS_PER_VISIT_MAX
    df["OUTLIER_VPH"] = df["VISITS"] / df["HHS"] > VISITS_PER_HH_MAX
    df["PRICE_ABOVE_BASE"] = df["PRICE"] > PRICE_ABOVE_BASE_TOL * df["BASE_PRICE"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model variables: logs, discount depth, time index and calendar fields."""
    weeks = np.sort(df["WEEK_END_DATE"].unique())
    week_idx = {w: i for i, w in enumerate(weeks)}
    out = df.assign(
        LOG_UNITS=np.log(df["UNITS"]),
        LOG_PRICE=np.log(df["PRICE"]),
        LOG_BASE_PRICE=np.log(df["BASE_PRICE"]),
        DISCOUNT=1.0 - df["PRICE"] / df["BASE_PRICE"],
        ANY_PROMO=df[PROMO_COLS].sum(axis=1) > 0,
        WEEK=df["WEEK_END_DATE"].map(week_idx).astype("int16"),
        WEEK_OF_YEAR=df["WEEK_END_DATE"].dt.isocalendar().week.astype("int16"),
        YEAR=df["WEEK_END_DATE"].dt.year.astype("int16"),
    )
    out["PAIR_ID"] = out["STORE_NUM"].astype(str) + "_" + out["UPC"].astype(str)
    out["PAIR_N_WEEKS"] = out.groupby("PAIR_ID")["WEEK"].transform("nunique").astype("int16")
    out["FULL_PANEL"] = out["PAIR_N_WEEKS"].eq(N_WEEKS_TOTAL)
    return out


def build_panel(stores: pd.DataFrame, products: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Join cleaned transactions with store and product attributes.

    ``IN_MODEL`` marks the default modelling sample: rows that are not units/visit
    outliers (D9). The rule is a flag, so sensitivity analyses can include them back.
    """
    st = clean_stores(stores)[
        ["STORE_NUM", "STORE_NAME", "CITY", "STATE", "MSA_CODE", "SEGMENT", "SEG_AMBIGUOUS",
         "SALES_AREA", "AVG_WEEKLY_BASKETS"]
    ]
    pr = clean_products(products)
    panel = add_features(clean_transactions(tx))
    panel = panel.merge(st, on="STORE_NUM", how="left", validate="many_to_one")
    panel = panel.merge(pr, on="UPC", how="left", validate="many_to_one")
    panel["IN_MODEL"] = ~panel["OUTLIER_UPV"]
    return panel


def cleaning_report(tx_raw: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Return the counts behind each rule, for the README and the notebooks."""
    return pd.Series(
        {
            "rows_raw": len(tx_raw),
            "dropped_invalid_price": len(tx_raw) - len(panel),
            "rows_panel": len(panel),
            "base_price_imputed": int(panel["BASE_IMPUTED"].sum()),
            "base_price_ffill": int(panel["BASE_SOURCE"].eq("ffill").sum()),
            "base_price_fallback": int(panel["BASE_FALLBACK"].sum()),
            "outlier_units_per_visit": int(panel["OUTLIER_UPV"].sum()),
            "outlier_visits_per_hh": int(panel["OUTLIER_VPH"].sum()),
            "price_above_base": int(panel["PRICE_ABOVE_BASE"].sum()),
            "rows_ambiguous_segment": int(panel["SEG_AMBIGUOUS"].sum()),
            "rows_in_model": int(panel["IN_MODEL"].sum()),
            "pairs": panel["PAIR_ID"].nunique(),
            "pairs_full_panel": panel.loc[panel["FULL_PANEL"], "PAIR_ID"].nunique(),
        }
    )
