import numpy as np
import pandas as pd
import pytest

from retail_multilevel import diagnostics as dg


def _ar1_frame(rho=0.5, n_pairs=300, n_weeks=40, seed=1):
    rng = np.random.default_rng(seed)
    e = np.zeros((n_pairs, n_weeks))
    e[:, 0] = rng.normal(size=n_pairs)
    for w in range(1, n_weeks):
        e[:, w] = rho * e[:, w - 1] + np.sqrt(1 - rho**2) * rng.normal(size=n_pairs)
    return pd.DataFrame(
        {
            "PAIR_ID": np.repeat(np.arange(n_pairs).astype(str), n_weeks),
            "WEEK": np.tile(np.arange(1, n_weeks + 1), n_pairs),
            "CATEGORY": np.repeat(np.where(np.arange(n_pairs) % 2 == 0, "A", "B"), n_weeks),
            "resid": e.ravel(),
        }
    )


def test_ar1_is_recovered_by_lag():
    out = dg.lag_correlations(_ar1_frame(rho=0.5), max_lag=3)
    assert out["lag"].tolist() == [1, 2, 3]
    assert out["corr"].to_numpy() == pytest.approx([0.5, 0.25, 0.125], abs=0.05)
    assert out["n_pairs_of_obs"].tolist() == [300 * 39, 300 * 38, 300 * 37]


def test_noise_has_no_correlation():
    df = _ar1_frame(rho=0.0)
    assert dg.lag_correlations(df, max_lag=4)["corr"].abs().max() < 0.04
    assert abs(dg.consecutive_correlation(df)["corr"]) < 0.04


def test_gaps_calendar_lag_excludes_but_consecutive_includes():
    df = pd.DataFrame({"PAIR_ID": ["p"] * 4, "WEEK": [1, 2, 5, 6], "resid": [1.0, 2.0, 4.0, 3.0]})
    assert dg.lag_correlations(df, max_lag=1).loc[0, "n_pairs_of_obs"] == 2  # (1,2) and (5,6) only
    assert dg.consecutive_correlation(df)["n"] == 3  # also the gap 2 -> 5
    gappy = _ar1_frame(rho=0.4, n_pairs=200, n_weeks=30).query("WEEK % 3 != 0")
    assert dg.consecutive_correlation(gappy)["n"] > dg.lag_correlations(gappy, 1)["n_pairs_of_obs"].iloc[0]


def test_lag1_by_category():
    out = dg.lag1_by_category(_ar1_frame(rho=0.6))
    assert out["CATEGORY"].tolist() == ["A", "B"]
    assert out["corr"].to_numpy() == pytest.approx([0.6, 0.6], abs=0.04)


def test_planted_week_shock_is_detected():
    rng = np.random.default_rng(3)
    n_pairs, n_weeks = 400, 60
    shock = rng.normal(0, 0.5, n_weeks)
    noise = rng.normal(0, 1.0, (n_pairs, n_weeks))
    df = pd.DataFrame(
        {
            "PAIR_ID": np.repeat(np.arange(n_pairs).astype(str), n_weeks),
            "WEEK": np.tile(np.arange(1, n_weeks + 1), n_pairs),
            "CATEGORY": "A",
            "resid": (noise + shock).ravel(),
        }
    )
    weeks, s = dg.week_shocks(df)
    assert len(weeks) == n_weeks
    planted = shock.var() / (shock.var() + 1.0)
    assert s["share_week"] == pytest.approx(planted, abs=0.05)
    assert s["variance_ratio"] > 20
    assert s["share_week_expected"] == pytest.approx((n_weeks - 1) / len(df))
    assert s["share_category_week"] == pytest.approx(s["share_week"])


def test_independent_noise_has_no_week_shock():
    _, s = dg.week_shocks(_ar1_frame(rho=0.0, n_pairs=400, n_weeks=50))
    assert s["variance_ratio"] == pytest.approx(1.0, abs=0.5)
    assert s["share_week"] == pytest.approx(s["share_week_expected"], abs=0.002)
    assert abs(s["autocorr_week_means"]) < 0.4


def test_duplicate_pair_week_raises():
    df = pd.DataFrame({"PAIR_ID": ["p", "p"], "WEEK": [1, 1], "resid": [0.1, 0.2]})
    with pytest.raises(ValueError, match="uniquely"):
        dg.lag_correlations(df)


def test_residual_diagnostics_bundle():
    out = dg.residual_diagnostics(_ar1_frame(), max_lag=2)
    assert set(out) == {"lag_corr", "consecutive", "week_shocks", "week_summary", "lag1_by_category"}
