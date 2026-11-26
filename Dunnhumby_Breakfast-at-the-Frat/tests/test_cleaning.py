"""Cleaning rules on the synthetic fixtures from conftest.py."""

import numpy as np
import pandas as pd

from retail_multilevel.cleaning import add_features, clean_stores, clean_transactions


def test_ambiguous_store_is_deduplicated_flagged_and_unsegmented(stores_raw):
    st = clean_stores(stores_raw).set_index("STORE_NUM")
    assert st.index.is_unique and len(st) == 2
    assert st.loc[2, "SEG_AMBIGUOUS"] and pd.isna(st.loc[2, "SEGMENT"])
    assert not st.loc[1, "SEG_AMBIGUOUS"] and st.loc[1, "SEGMENT"] == "VALUE"


def test_invalid_price_rows_are_dropped(tx_raw):
    out = clean_transactions(tx_raw)
    assert len(out) == len(tx_raw) - 1
    assert (out["PRICE"] > 0).all() and out["PRICE"].notna().all()


def test_base_price_imputed_within_pair_and_flagged(tx_raw):
    out = clean_transactions(tx_raw).set_index(["STORE_NUM", "UPC", "WEEK_END_DATE"])
    row = out.loc[(1, 10, pd.Timestamp("2009-01-21"))]
    assert row["BASE_PRICE"] == 2.0 and row["BASE_IMPUTED"]
    assert out.loc[(2, 20, pd.Timestamp("2009-01-14")), "BASE_PRICE"] == 3.0  # fallback to price
    assert out["BASE_PRICE"].notna().all()


def test_base_source_and_fallback_flags(tx_raw):
    out = clean_transactions(tx_raw).set_index(["STORE_NUM", "UPC", "WEEK_END_DATE"])
    assert out.loc[(1, 10, pd.Timestamp("2009-01-21")), "BASE_SOURCE"] == "ffill"
    fb = out.loc[(2, 20, pd.Timestamp("2009-01-14"))]
    assert fb["BASE_SOURCE"] == "fallback_price" and fb["BASE_FALLBACK"] and fb["BASE_IMPUTED"]
    assert out.loc[(2, 20, pd.Timestamp("2009-01-21")), "BASE_PRICE"] == 3.5  # no earlier base: price again
    assert out.loc[(1, 10, pd.Timestamp("2009-01-14")), "BASE_SOURCE"] == "observed"
    assert (out["BASE_IMPUTED"] == (out["BASE_SOURCE"] != "observed")).all()
    assert (out["BASE_FALLBACK"] == (out["BASE_SOURCE"] == "fallback_price")).all()
    feats = add_features(clean_transactions(tx_raw)).set_index(["STORE_NUM", "UPC", "WEEK_END_DATE"])
    assert (feats.loc[feats["BASE_FALLBACK"], "DISCOUNT"] == 0).all()


def test_first_week_missing_base_uses_price_not_later_week():
    w = pd.to_datetime(["2009-01-14", "2009-01-21"])
    cols = [
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
    tx = pd.DataFrame(
        [(1, 10, w[0], 1, 1, 1, 1.5, 1.5, np.nan, 0, 0, 1), (1, 10, w[1], 1, 1, 1, 2.0, 2.0, 2.0, 0, 0, 0)],
        columns=cols,
    )
    out = clean_transactions(tx).sort_values("WEEK_END_DATE")
    assert out["BASE_PRICE"].tolist() == [1.5, 2.0]  # the later 2.0 is never used backwards
    assert out["BASE_SOURCE"].tolist() == ["fallback_price", "observed"]


def test_cleaning_is_causal_per_row(tx_raw):
    base = clean_transactions(tx_raw).set_index(["STORE_NUM", "UPC", "WEEK_END_DATE"])
    last = tx_raw["WEEK_END_DATE"].max()
    alt = tx_raw.copy()
    later = alt["WEEK_END_DATE"] >= pd.Timestamp("2009-01-21")
    alt.loc[later, "BASE_PRICE"] = 99.0
    alt.loc[later & alt["BASE_PRICE"].isna(), "BASE_PRICE"] = np.nan
    extra = alt.iloc[[0]].assign(WEEK_END_DATE=last + pd.Timedelta(days=7), BASE_PRICE=np.nan)
    alt = pd.concat([alt, extra], ignore_index=True)
    new = clean_transactions(alt).set_index(["STORE_NUM", "UPC", "WEEK_END_DATE"])
    early = base.index[base.index.get_level_values("WEEK_END_DATE") < pd.Timestamp("2009-01-21")]
    pd.testing.assert_frame_equal(
        base.loc[early, ["BASE_PRICE", "BASE_SOURCE"]], new.loc[early, ["BASE_PRICE", "BASE_SOURCE"]]
    )


def test_outlier_flags(tx_raw):
    out = clean_transactions(tx_raw)
    assert out["OUTLIER_UPV"].sum() == 1
    assert out["OUTLIER_VPH"].sum() == 0  # boundary 3.0 is not an outlier
    assert out["PRICE_ABOVE_BASE"].sum() == 1


def test_panel_features(panel):
    assert np.allclose(panel["LOG_PRICE"], np.log(panel["PRICE"]))
    assert panel["WEEK"].min() == 0 and panel["WEEK"].max() == 1
    assert panel["IN_MODEL"].sum() == len(panel) - 1
    assert panel.loc[panel["UPC"] == 10, "PRIVATE_LABEL"].all()
    assert panel["STORE_NAME"].notna().all()


def test_discount_positive_under_tpr(tx_raw):
    out = add_features(clean_transactions(tx_raw))
    assert (out.loc[out["TPR_ONLY"] == 1, "DISCOUNT"] > 0).all()
