import numpy as np
import pandas as pd
import pytest

from retail_multilevel import week_audit as wa


def _panel(n_weeks=12, n_stores=6, n_upc=4, drop_week=6, drop=0.4, seed=1, missing=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_weeks):
        for s in range(n_stores):
            for u in range(n_upc):
                units = max(1, int(rng.poisson(20) * (drop if w == drop_week else 1.0)))
                price = 2.0 + 0.5 * u
                rows.append(
                    {
                        "WEEK": w,
                        "STORE_NUM": s,
                        "UPC": u,
                        "PAIR_ID": f"{s}_{u}",
                        "UNITS": units,
                        "VISITS": units,
                        "HHS": units,
                        "SPEND": units * price,
                        "PRICE": price,
                        "FEATURE": 0,
                        "DISPLAY": 0,
                        "TPR_ONLY": 0,
                        "STATE": "OH" if s < 3 else "TX",
                        "CATEGORY": "A" if u < 2 else "B",
                    }
                )
    df = pd.DataFrame(rows)
    if missing:
        gone = df[(df["WEEK"] == drop_week) & (df["STORE_NUM"] == 0)].index[:missing]
        df = df.drop(index=gone).reset_index(drop=True)
    return df


def test_neighbour_weeks_excludes_focal_and_missing():
    assert wa.neighbour_weeks([0, 1, 2, 4, 5], 3, 2) == [1, 2, 4, 5]
    assert wa.neighbour_weeks([0, 1, 2], 0, 4) == [1, 2]


def test_profile_and_comparison_flag_the_dropped_week():
    df = _panel()
    prof = wa.week_profile(df)
    assert prof.loc[prof["WEEK"] == 3, "rows"].item() == 24
    cmp_ = wa.compare_to_neighbours(prof, 6, 4).set_index("metric")
    assert cmp_.loc["units", "ratio"] == pytest.approx(0.4, abs=0.08)
    assert cmp_.loc["stores", "ratio"] == 1.0
    assert cmp_.loc["units_per_visit", "ratio"] == pytest.approx(1.0)
    assert cmp_.loc["units", "z"] < -5


def test_group_ratio_and_deficit_shares_sum_to_one():
    df = _panel()
    out = wa.group_ratio(df, 6, "STATE", 4)
    assert set(out["group"]) == {"OH", "TX"}
    assert (out["ratio"] < 0.6).all()
    assert out["deficit_share"].sum() == pytest.approx(1.0)
    assert out["deficit"].is_monotonic_decreasing


def test_pair_presence_and_missing_pairs():
    df = _panel(missing=3)
    pres = wa.pair_presence(df, 6)
    assert pres["n_pairs"] == 24 - 3 and pres["missing_both"] == 3 and pres["new_vs_both"] == 0
    mp = wa.missing_pairs(df, 6)
    assert len(mp) == 3 and set(mp["STORE_NUM"]) == {0}


def test_pair_ratio_summary_separates_focal_from_placebo_weeks():
    out = wa.pair_ratio_summary(_panel(), [3, 6], 4).set_index("WEEK")
    assert out.loc[6, "ratio_median"] < 0.5 < out.loc[3, "ratio_median"]
    assert out.loc[6, "share_below_0.5"] > out.loc[3, "share_below_0.5"]


def test_integrity_counts_planted_corruption():
    df = _panel()
    df.loc[0, "SPEND"] = 999.0  # spend inconsistent with units x price
    df.loc[1, "UNITS"] = 0
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # duplicate key
    out = wa.integrity_by_week(df, [0, 1]).set_index("WEEK")
    assert out.loc[0, "spend_mismatch"] >= 1 and out.loc[0, "nonpositive_units"] == 1
    assert out.loc[0, "duplicate_keys"] == 2
    assert out.loc[1, ["duplicate_keys", "nonpositive_units", "spend_mismatch"]].sum() == 0


def test_with_week_index_builds_week_and_pair():
    tx = pd.DataFrame(
        {
            "WEEK_END_DATE": pd.to_datetime(["2009-01-14", "2009-01-21", "2009-02-04"]),
            "STORE_NUM": [1, 1, 2],
            "UPC": [10, 10, 11],
        }
    )
    out = wa.with_week_index(tx)
    assert out["WEEK"].tolist() == [0, 1, 3] and out["PAIR_ID"].tolist() == ["1_10", "1_10", "2_11"]


def test_classify_week_is_conservative():
    df = _panel()
    prof = wa.week_profile(df)
    cmp_ = wa.compare_to_neighbours(prof, 6, 4)
    integ = wa.integrity_by_week(df)
    assert wa.classify_week(cmp_, integ, 6)["category"] == "unexplained shock"  # a big drop alone proves nothing
    short = df[~((df["WEEK"] == 6) & (df["STORE_NUM"] == 5))]  # a whole store disappears
    verdict = wa.classify_week(wa.compare_to_neighbours(wa.week_profile(short), 6, 4), wa.integrity_by_week(short), 6)
    assert verdict["category"] == "proven data error" and verdict["reasons"]
