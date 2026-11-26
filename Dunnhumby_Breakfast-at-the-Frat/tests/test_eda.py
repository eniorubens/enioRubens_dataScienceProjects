import numpy as np
import pandas as pd
import pytest

from retail_multilevel import eda

N_WEEKS = 8
FIRST_TEST = 5


def make_panel(store_effect=None, upc_effect=None):
    """3 stores x 2 UPCs x 8 weeks; pair 3_20 misses weeks 6 and 7."""
    dates = pd.date_range("2020-01-01", periods=N_WEEKS, freq="7D")
    rows = []
    for s in (1, 2, 3):
        for u in (10, 20):
            for w in range(N_WEEKS):
                if s == 3 and u == 20 and w >= 6:
                    continue
                y = (store_effect or {}).get(s, 0.0) + (upc_effect or {}).get(u, 0.0)
                rows.append({"WEEK_END_DATE": dates[w], "WEEK": w, "STORE_NUM": s, "UPC": u, "LOG_UNITS": y})
    df = pd.DataFrame(rows)
    df["UNITS"] = 10 * df["STORE_NUM"] + df["WEEK"]
    df["PAIR_ID"] = df["STORE_NUM"].astype(str) + "_" + df["UPC"].astype(str)
    df["BASE_PRICE"] = 2.0
    df["PRICE"] = 2.0
    df["FEATURE"] = 0
    df["DISPLAY"] = 0
    df["TPR_ONLY"] = 0
    return df


def with_promos(df):
    """FEATURE in week 1, DISPLAY in week 2, TPR_ONLY (with a price cut) in week 3."""
    df = df.copy()
    df.loc[df["WEEK"] == 1, "FEATURE"] = 1
    df.loc[df["WEEK"] == 2, "DISPLAY"] = 1
    df.loc[df["WEEK"] == 3, "TPR_ONLY"] = 1
    df.loc[df["WEEK"] == 3, "PRICE"] = 1.5
    df["ANY_PROMO"] = df[["FEATURE", "DISPLAY", "TPR_ONLY"]].sum(axis=1) > 0
    df["LOG_PRICE"] = np.log(df["PRICE"])
    return df


def test_panel_overview():
    out = eda.panel_overview(make_panel())
    assert list(out.index) == ["rows", "stores", "upcs", "pairs", "weeks", "first_week", "last_week"]
    assert out["rows"] == 6 * 8 - 2
    assert (out["stores"], out["upcs"], out["pairs"], out["weeks"]) == (3, 2, 6, 8)
    assert out["first_week"] == pd.Timestamp("2020-01-01")
    assert out["last_week"] == pd.Timestamp("2020-02-19")


def test_promo_rates_overall():
    df = with_promos(make_panel())
    out = eda.promo_rates(df)
    n = len(df)
    assert list(out.columns) == ["FEATURE", "DISPLAY", "TPR_ONLY", "ANY_PROMO", "BELOW_BASE", "n"]
    assert list(out.index) == ["all"]
    assert out.loc["all", "n"] == n
    assert out.loc["all", "FEATURE"] == pytest.approx(6 / n)
    assert out.loc["all", "ANY_PROMO"] == pytest.approx(18 / n)
    assert out.loc["all", "BELOW_BASE"] == pytest.approx(6 / n)
    shares = out.drop(columns="n")
    assert ((shares >= 0) & (shares <= 1)).all().all()


def test_promo_rates_by():
    df = with_promos(make_panel())
    by_store = eda.promo_rates(df, by="STORE_NUM")
    assert list(by_store.index) == [1, 2, 3]
    assert by_store["n"].tolist() == [16, 16, 14]
    by_two = eda.promo_rates(df, by=["STORE_NUM", "UPC"])
    assert len(by_two) == 6
    assert by_two.index.names == ["STORE_NUM", "UPC"]
    assert by_two["n"].sum() == len(df)


def test_promo_rates_keeps_nan_group():
    df = with_promos(make_panel())
    df["SEGMENT"] = np.where(df["STORE_NUM"] == 1, None, "A")
    out = eda.promo_rates(df, by="SEGMENT")
    assert len(out) == 2
    assert out["n"].sum() == len(df)


def test_weekly_units():
    out = eda.weekly_units(make_panel(), first_test_week=FIRST_TEST)
    assert list(out.columns) == ["WEEK_END_DATE", "WEEK", "UNITS", "IS_TEST"]
    assert out["WEEK"].tolist() == list(range(N_WEEKS))
    assert out.loc[0, "UNITS"] == 2 * (10 + 20 + 30)
    # week 7: pair 3_20 is missing
    assert out.loc[7, "UNITS"] == 2 * (17 + 27) + 37
    assert out["IS_TEST"].tolist() == [False] * 5 + [True] * 3


def test_pair_coverage():
    out = eda.pair_coverage(make_panel(), first_test_week=FIRST_TEST)
    assert list(out.columns) == ["PAIR_ID", "STORE_NUM", "UPC", "n_weeks_train", "n_weeks_test"]
    assert len(out) == 6
    row = out.set_index("PAIR_ID")
    assert row.loc["1_10", ["n_weeks_train", "n_weeks_test"]].tolist() == [5, 3]
    assert row.loc["3_20", ["n_weeks_train", "n_weeks_test"]].tolist() == [5, 1]
    assert row.loc["3_20", "STORE_NUM"] == 3
    assert row.loc["3_20", "UPC"] == 20


def test_pair_coverage_zero_when_absent():
    df = make_panel()
    df = df[~((df["PAIR_ID"] == "1_10") & (df["WEEK"] >= FIRST_TEST))]
    out = eda.pair_coverage(df, first_test_week=FIRST_TEST).set_index("PAIR_ID")
    assert out.loc["1_10", "n_weeks_test"] == 0
    assert out["n_weeks_train"].dtype.kind == "i"


def test_price_variation():
    df = with_promos(make_panel())
    df = df[~((df["PAIR_ID"] == "1_10") & (df["WEEK"] > 0))]  # single-week pair
    res = eda.price_variation(df)
    assert list(res.columns) == ["PAIR_ID", "n_weeks", "sd_log_price", "share_below_base"]
    out = res.set_index("PAIR_ID")
    assert out.index.is_unique
    assert len(out) == 6
    assert out.loc["1_10", "n_weeks"] == 1
    assert np.isnan(out.loc["1_10", "sd_log_price"])
    lp = np.full(8, np.log(2.0))
    lp[3] = np.log(1.5)
    assert out.loc["2_10", "sd_log_price"] == pytest.approx(lp.std(ddof=1))
    assert out.loc["2_10", "share_below_base"] == pytest.approx(1 / 8)


def test_variance_shares_store_effect_only():
    df = make_panel(store_effect={1: 0.0, 2: 1.0, 3: 2.0})
    df = df[df["WEEK"] < 6]  # balanced design
    res = eda.variance_shares(df)
    assert list(res.columns) == ["group", "n_groups", "between_share"]
    assert res["group"].tolist() == ["STORE_NUM", "UPC", "PAIR_ID"]
    res = res.set_index("group")
    assert res.loc["STORE_NUM", "between_share"] == pytest.approx(1.0)
    assert res.loc["UPC", "between_share"] == pytest.approx(0.0, abs=1e-12)
    assert res.loc["PAIR_ID", "between_share"] == pytest.approx(1.0)
    assert res["n_groups"].tolist() == [3, 2, 6]


def test_variance_shares_custom_groups_known_value():
    df = make_panel(upc_effect={10: 0.0, 20: 2.0})
    df = df[df["WEEK"] < 6]
    only_upc = eda.variance_shares(df, groups=["UPC"])
    assert only_upc["group"].tolist() == ["UPC"]
    assert only_upc.loc[0, "between_share"] == pytest.approx(1.0)
    # UPC effect 0/1 plus a +-0.5 week-parity noise: between var 0.25, total var 0.5
    noise = np.where(df["WEEK"] % 2 == 0, 0.5, -0.5)
    df = df.assign(Y=np.where(df["UPC"] == 10, 0.0, 1.0) + noise)
    res = eda.variance_shares(df, y="Y", groups=("UPC",))
    assert res.loc[0, "between_share"] == pytest.approx(0.5)


def test_raw_promo_lift_recovers_planted_lift():
    df = with_promos(make_panel())
    df["LOG_UNITS"] = 0.5 * (df["FEATURE"] == 1)
    out = eda.raw_promo_lift(df)
    assert list(out.index) == ["FEATURE", "DISPLAY", "TPR_ONLY"]
    assert out.index.name == "promo"
    assert list(out.columns) == ["lift_log", "lift_pct", "n_weeks"]
    # pooled demeaning is exact up to the small pair-length imbalance of pair 3_20
    assert out.loc["FEATURE", "lift_log"] == pytest.approx(0.5, abs=0.02)
    assert out.loc["DISPLAY", "lift_log"] == pytest.approx(0.0, abs=0.02)
    assert out.loc["TPR_ONLY", "lift_log"] == pytest.approx(0.0, abs=0.02)
    assert out["lift_pct"].tolist() == pytest.approx(np.expm1(out["lift_log"]).tolist())
    assert out["n_weeks"].tolist() == [6, 6, 6]


def test_raw_promo_lift_exact_with_balanced_pairs():
    df = with_promos(make_panel())
    df = df[df["PAIR_ID"] != "3_20"]
    df["LOG_UNITS"] = 0.5 * (df["FEATURE"] == 1) - 0.25 * (df["DISPLAY"] == 1)
    out = eda.raw_promo_lift(df)
    assert out.loc["FEATURE", "lift_log"] == pytest.approx(0.5)
    assert out.loc["DISPLAY", "lift_log"] == pytest.approx(-0.25)
    assert out.loc["DISPLAY", "lift_pct"] == pytest.approx(np.exp(-0.25) - 1)
