import numpy as np
import pandas as pd
import pytest

from retail_multilevel.cleaning import add_features, clean_transactions
from retail_multilevel.features import FOURIER_TERMS, add_model_features, pair_means, upc_means


def make_panel(n_weeks=104):
    rng = np.random.default_rng(1)
    rows = []
    for pair, base in (("1_10", 1.0), ("1_11", 2.0), ("2_10", 0.5)):
        for w in range(n_weeks):
            lb = base + rng.normal(0, 0.05)
            rows.append(
                {
                    "PAIR_ID": pair,
                    "WEEK": w,
                    "WEEK_OF_YEAR": w % 52 + 1,
                    "LOG_BASE_PRICE": lb,
                    "LOG_PRICE": lb - (0.2 if w % 5 == 0 else 0.0),
                    "FEATURE": w % 3 == 0,
                    "DISPLAY": w % 4 == 0,
                    "TPR_ONLY": w % 5 == 0,
                }
            )
    return pd.DataFrame(rows)


def test_fourier_terms_names():
    assert FOURIER_TERMS(2) == ["SIN_1", "COS_1", "SIN_2", "COS_2"]
    assert len(FOURIER_TERMS(3)) == 6


def test_log_disc_sign_and_values():
    out = add_model_features(make_panel(), pair_means(make_panel()))
    assert (out["LOG_DISC"] <= 0).all()
    assert np.allclose(out.loc[out["TPR_ONLY"] == 1, "LOG_DISC"], -0.2)
    assert np.allclose(out.loc[out["TPR_ONLY"] == 0, "LOG_DISC"], 0.0)


def test_pair_means_only_from_given_frame():
    panel = make_panel()
    train = panel[panel["WEEK"] < 52]
    m_train = pair_means(train).set_index("PAIR_ID")["LOG_BASE_MEAN"]
    m_all = pair_means(panel).set_index("PAIR_ID")["LOG_BASE_MEAN"]
    expected = train.groupby("PAIR_ID")["LOG_BASE_PRICE"].mean()
    assert np.allclose(m_train.loc[expected.index], expected)
    assert not np.allclose(m_train, m_all)
    assert list(pair_means(train).columns) == ["PAIR_ID", "LOG_BASE_MEAN"]


def test_base_dev_mean_zero_within_pair_on_train():
    train = make_panel()
    out = add_model_features(train, pair_means(train))
    assert np.allclose(out.groupby("PAIR_ID")["LOG_BASE_DEV"].mean(), 0.0, atol=1e-12)


def test_unseen_pairs_get_nan_and_input_not_mutated():
    panel = make_panel()
    train = panel[panel["PAIR_ID"] != "2_10"]
    before = panel.copy()
    out = add_model_features(panel, pair_means(train))
    assert out.loc[out["PAIR_ID"] == "2_10", ["LOG_BASE_MEAN", "LOG_BASE_DEV"]].isna().all().all()
    assert out.loc[out["PAIR_ID"] != "2_10", "LOG_BASE_DEV"].notna().all()
    pd.testing.assert_frame_equal(panel, before)


def test_fourier_bounded_and_periodic():
    panel = make_panel()
    out = add_model_features(panel, pair_means(panel), n_fourier=3)
    cols = FOURIER_TERMS(3)
    assert set(cols) <= set(out.columns)
    assert out[cols].abs().max().max() <= 1.0 + 1e-12
    a = out[(out["PAIR_ID"] == "1_10") & (out["WEEK"] == 3)].iloc[0]
    b = out[(out["PAIR_ID"] == "1_10") & (out["WEEK"] == 55)].iloc[0]
    assert a["WEEK_OF_YEAR"] == b["WEEK_OF_YEAR"]
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float))
    assert out.loc[out["WEEK_OF_YEAR"] == 1, "SIN_1"].iloc[0] == pytest.approx(np.sin(2 * np.pi / 52.18))
    assert "SIN_4" not in add_model_features(panel, pair_means(panel), n_fourier=2).columns


def test_trend_and_flags_are_ints():
    panel = make_panel()
    out = add_model_features(panel, pair_means(panel))
    assert out["TREND"].iloc[0] == 0.0
    assert out.loc[out["WEEK"] == 52, "TREND"].iloc[0] == pytest.approx(1.0)
    for col in ("FEATURE", "DISPLAY", "TPR_ONLY"):
        assert out[col].dtype.kind == "i"
        assert set(out[col].unique()) <= {0, 1}


def test_mean_source_three_paths():
    panel = make_panel().assign(UPC=lambda d: d["PAIR_ID"].str.split("_").str[1].astype(int))
    fit = panel[panel["PAIR_ID"] != "2_10"]  # pair 2_10 unseen, UPC 10 seen in store 1
    extra = panel[panel["PAIR_ID"] == "1_11"].assign(PAIR_ID="3_99", UPC=99)  # UPC 99 unseen
    scored = pd.concat([panel, extra], ignore_index=True)
    out = add_model_features(scored, pair_means(fit), upc_means(fit))
    src = out.groupby("PAIR_ID")["MEAN_SOURCE"].first()
    assert src["1_10"] == "pair" and src["2_10"] == "upc" and src["3_99"] == "none"
    expected_upc = fit.loc[fit["UPC"] == 10, "LOG_BASE_PRICE"].mean()
    r = out[out["PAIR_ID"] == "2_10"]
    assert np.allclose(r["LOG_BASE_MEAN"], expected_upc)
    assert np.allclose(r["LOG_BASE_DEV"], r["LOG_BASE_PRICE"] - expected_upc)
    assert out.loc[out["PAIR_ID"] == "3_99", ["LOG_BASE_MEAN", "LOG_BASE_DEV"]].isna().all().all()
    no_upc = add_model_features(scored, pair_means(fit))
    assert set(no_upc.loc[no_upc["PAIR_ID"] == "2_10", "MEAN_SOURCE"]) == {"none"}
    assert list(upc_means(fit).columns) == ["UPC", "LOG_BASE_MEAN"]


def _raw_tx(n_weeks=30):
    rng = np.random.default_rng(7)
    weeks = pd.date_range("2009-01-14", periods=n_weeks, freq="7D")
    rows = []
    for store, upc, base in ((1, 10, 2.0), (1, 11, 3.0), (2, 10, 2.2)):
        for i, wk in enumerate(weeks):
            b = base + 0.1 * rng.random()
            promo = int(i % 4 == 0)
            price = b * (0.8 if promo else 1.0)
            units = int(rng.integers(1, 20))
            miss_base = rng.random() < 0.1 or (i == 0 and store == 2)
            rows.append(
                (store, upc, wk, units, units, units, units * price, price, np.nan if miss_base else b, 0, 0, promo)
            )
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
    return pd.DataFrame(rows, columns=cols)


def _features_for_window(tx, cutoff):
    panel = add_features(clean_transactions(tx))
    panel["ANY_PROMO"] = panel["ANY_PROMO"].astype(bool)
    fit = panel[panel["WEEK"] < cutoff]
    return add_model_features(fit, pair_means(fit), upc_means(fit))


def test_fit_window_invariant_to_data_after_cutoff():
    tx = _raw_tx()
    cutoff = 20
    cutoff_date = tx["WEEK_END_DATE"].drop_duplicates().sort_values().iloc[cutoff]
    ref = _features_for_window(tx, cutoff)
    late = tx["WEEK_END_DATE"] >= cutoff_date
    alt = tx.copy()
    alt.loc[late, "PRICE"] *= 3.0
    alt.loc[late, "BASE_PRICE"] = alt.loc[late, "BASE_PRICE"] * 5.0
    alt.loc[late & (alt["STORE_NUM"] == 2), "BASE_PRICE"] = np.nan
    alt = alt[~(late & (alt["UPC"] == 11))]  # deletions after the cutoff
    late = alt["WEEK_END_DATE"] >= cutoff_date
    new_pair = alt[late & (alt["STORE_NUM"] == 1) & (alt["UPC"] == 10)].assign(STORE_NUM=9)
    new_upc = alt[late & (alt["STORE_NUM"] == 1) & (alt["UPC"] == 10)].assign(UPC=77)
    alt = pd.concat([alt, new_pair, new_upc], ignore_index=True)
    new = _features_for_window(alt, cutoff)
    key = ["STORE_NUM", "UPC", "WEEK_END_DATE"]
    # PAIR_N_WEEKS / FULL_PANEL count presence over the whole panel (retrospective sensitivity
    # columns, not model features), so they legitimately change with later data.
    retro = ["PAIR_N_WEEKS", "FULL_PANEL"]
    a = ref.drop(columns=retro).sort_values(key).reset_index(drop=True)
    b = new.drop(columns=retro).sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert set(a["MEAN_SOURCE"]) == {"pair"}
    assert {"observed", "ffill", "fallback_price"} <= set(a["BASE_SOURCE"])
