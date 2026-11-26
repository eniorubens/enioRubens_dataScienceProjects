import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as f_dist

from retail_multilevel.contrasts import (
    ContrastSpec,
    assert_training_only,
    build_design,
    build_population,
    fit_contrast,
    holm,
    level_support,
    wald_joint,
)
from retail_multilevel.sensitivity import two_way_fe_ols


def _panel(seed=0, n_pairs=90, n_weeks=30, deltas=(0.0, -0.3, 0.4), base=-1.2, const_level=None):
    """Synthetic panel: pairs nested in three levels, planted LOG_DISC slopes by level, week shocks."""
    rng = np.random.default_rng(seed)
    pair = np.repeat(np.arange(n_pairs), n_weeks)
    week = np.tile(np.arange(n_weeks), n_pairs)
    lev_of_pair = np.arange(n_pairs) % 3
    lev = np.array(["A", "B", "C"])[lev_of_pair][pair]
    disc = rng.normal(size=len(pair))
    if const_level is not None:  # no within-pair discount variation in this level
        m = lev == const_level
        disc[m] = rng.normal(size=n_pairs)[pair[m]]
    slope = np.array(deltas)[lev_of_pair][pair] + base
    y = (
        slope * disc
        + 0.5 * rng.normal(size=n_pairs)[pair]
        + 0.3 * rng.normal(size=n_weeks)[week]
        + rng.normal(0, 0.2, len(pair))
    )
    df = pd.DataFrame({"PAIR_ID": pair, "WEEK": week, "MOD": lev, "LOG_DISC": disc, "LOG_UNITS": y})
    df["LOG_BASE_PRICE"] = rng.normal(size=len(df))
    for c in ("FEATURE", "DISPLAY", "TPR_ONLY"):
        df[c] = (rng.random(len(df)) < 0.2).astype(int)
    return df


SPEC = ContrastSpec("mod", "MOD", "A")


def test_design_reference_and_names():
    df = _panel(n_pairs=6, n_weeks=4)
    x, terms, inter, others = build_design(df, "MOD", "B")
    assert others == ["A", "C"] and inter == ["LOG_DISC:MOD[A]", "LOG_DISC:MOD[C]"]
    assert x.shape == (len(df), 7) and terms[0] == "LOG_DISC"
    assert np.allclose(x[:, 5], df["LOG_DISC"] * (df["MOD"] == "A"))
    assert np.allclose(x[:, 6], df["LOG_DISC"] * (df["MOD"] == "C"))
    with pytest.raises(ValueError):
        build_design(df, "MOD", "Z")


def test_recovers_planted_slope_differences_and_matches_two_way_fe():
    df = _panel(seed=1)
    res = fit_contrast(df, SPEC)
    sl = res["slopes"].set_index("level")
    assert sl.loc["A", "slope"] == pytest.approx(-1.2, abs=0.05)
    assert sl.loc["B", "diff_vs_ref"] == pytest.approx(-0.3, abs=0.05)
    assert sl.loc["C", "diff_vs_ref"] == pytest.approx(0.4, abs=0.05)
    assert sl.loc["C", "slope"] == pytest.approx(-0.8, abs=0.05)
    assert (sl["ci_low"] < sl["slope"]).all() and (sl["slope"] < sl["ci_high"]).all()
    assert (sl["interval"] == "pointwise").all()
    # same estimates and SEs as the public two-way function on the same design
    x, terms, _, _ = build_design(df, "MOD", "A")
    d = pd.DataFrame(x, columns=terms)
    d[["LOG_UNITS", "PAIR_ID", "WEEK"]] = df[["LOG_UNITS", "PAIR_ID", "WEEK"]].to_numpy()
    tab, _ = two_way_fe_ols(d, "LOG_UNITS", terms)
    co = res["coefficients"].query("model == 'interaction'")
    assert np.allclose(co["estimate"], tab["estimate"]) and np.allclose(co["se_twoway"], tab["se_twoway"])
    assert res["wald"]["available"] and res["wald"]["p_raw"] < 1e-6


def test_restricted_uses_same_sample_and_null_is_not_rejected():
    df = _panel(seed=2, deltas=(0.0, 0.0, 0.0))
    res = fit_contrast(df, SPEC)
    co = res["coefficients"]
    assert set(co.loc[co["model"] == "restricted", "term"]) == {
        "LOG_DISC",
        "LOG_BASE_PRICE",
        "FEATURE",
        "DISPLAY",
        "TPR_ONLY",
    }
    assert res["wald"]["nobs"] == res["wald"]["nobs_restricted"] == len(df)
    assert res["wald"]["df1"] == 2 and res["wald"]["df2"] == min(90, 30) - 1
    assert res["wald"]["p_raw"] > 0.001


def test_wald_matches_manual_computation():
    delta = np.array([0.5, -0.2])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    out = wald_joint(delta, cov, g_pair=50, g_week=20)
    w = delta @ np.linalg.inv(cov) @ delta
    assert out["W"] == pytest.approx(w) and out["F"] == pytest.approx(w / 2)
    assert out["df1"] == 2 and out["df2"] == 19
    assert out["p_raw"] == pytest.approx(f_dist.sf(w / 2, 2, 19))
    assert out["available"] and out["reason"] == ""


def test_wald_unavailable_for_singular_or_illconditioned_block():
    sing = wald_joint(np.array([1.0, 1.0]), np.array([[1.0, 1.0], [1.0, 1.0]]), 10, 10)
    assert not sing["available"] and "positive definite" in sing["reason"] and np.isnan(sing["p_raw"])
    ill = wald_joint(np.array([1.0, 1.0]), np.diag([1.0, 1e-14]), 10, 10)
    assert not ill["available"] and "ill-conditioned" in ill["reason"]


def test_rank_deficient_level_is_unavailable_with_reason_and_rest_estimable():
    df = _panel(seed=4, const_level="C")
    res = fit_contrast(df, SPEC)
    w = res["wald"]
    assert not w["available"] and w["rank_deficient"] and "LOG_DISC:MOD[C]" in w["reason"]
    assert np.isnan(w["p_raw"])
    sl = res["slopes"].set_index("level")
    assert np.isfinite(sl.loc["A", "slope"]) and np.isfinite(sl.loc["B", "slope"])
    assert sl.loc["B", "diff_vs_ref"] == pytest.approx(-0.3, abs=0.06)
    sup = res["support"].set_index("level")
    assert sup.loc["C", "n_pairs_with_variation"] == 0 and sup.loc["A", "n_pairs_with_variation"] > 0
    assert sup.loc["C", "within_pair_sd_log_disc"] == pytest.approx(0.0, abs=1e-12)


def test_level_support_counts():
    df = _panel(n_pairs=6, n_weeks=5)
    sup = level_support(df, "MOD").set_index("level")
    assert sup["n_rows"].tolist() == [10, 10, 10] and sup["n_pairs"].tolist() == [2, 2, 2]
    assert (sup["within_pair_sd_log_disc"] > 0).all()


def test_holm_correctness_including_unavailable_position():
    adj, rej = holm([0.01, 0.04, 0.03], [True, True, True])
    assert np.allclose(adj, [0.03, 0.06, 0.06]) and rej.tolist() == [True, False, False]
    adj, rej = holm([0.01, 0.5, 0.001], [True, False, True])
    assert adj[0] == pytest.approx(0.02) and adj[2] == pytest.approx(0.003)
    assert np.isnan(adj[1]) and rej.tolist() == [True, False, True]
    adj, rej = holm([0.02, np.nan, 0.5], [True, False, True])  # unavailable counts as p = 1 in the family
    assert adj[0] == pytest.approx(0.06) and not rej.any()
    again, _ = holm([0.01, 0.04, 0.03], [True, True, True])
    assert np.allclose(again, [0.03, 0.06, 0.06])  # pure: same input, same output


def test_population_exclusion_counts():
    df = pd.DataFrame(
        {
            "PAIR_ID": [1, 1, 2, 3, 4],
            "WEEK": [0, 1, 0, 0, 1],
            "SEGMENT": ["MAINSTREAM", "MAINSTREAM", "VALUE", None, None],
            "SEG_AMBIGUOUS": [False, False, False, True, False],
        }
    )
    pop, info = build_population(df, ContrastSpec("segment", "SEGMENT", "MAINSTREAM", drop_ambiguous=True))
    assert info["n_rows"] == 3 and info["n_excluded"] == 2 and info["n_pairs"] == 2
    assert "SEG_AMBIGUOUS=1" in info["reason"] and "missing SEGMENT=1" in info["reason"]
    pop, info = build_population(df, ContrastSpec("x", "PAIR_ID", 1))
    assert info["n_excluded"] == 0


def test_temporal_guard_rejects_test_weeks():
    assert_training_only(pd.DataFrame({"WEEK": [0, 103]}))
    with pytest.raises(ValueError):
        assert_training_only(pd.DataFrame({"WEEK": [0, 104]}))
    with pytest.raises(ValueError):
        build_population(pd.DataFrame({"WEEK": [104], "PAIR_ID": [1], "MOD": ["A"]}), SPEC)
