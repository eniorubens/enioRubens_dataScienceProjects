import math

import numpy as np
import pandas as pd
import pytest

from retail_multilevel import interpret
from retail_multilevel.i18n import LANGS, t


def _fixef():
    terms = ["(Intercept)", "LOG_DISC", "LOG_BASE_DEV", "FEATURE", "DISPLAY", "TPR_ONLY", "TREND"]
    est = [3.0, -1.0, -2.0, math.log(2), math.log(1.5), math.log(0.5), 0.0]
    se = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    df = pd.DataFrame({"term": terms, "estimate": est, "std_error": se})
    df["ci_low"] = df["estimate"] - 1.96 * df["std_error"]
    df["ci_high"] = df["estimate"] + 1.96 * df["std_error"]
    return df


def test_pct_effect_of_price_change_hand_values():
    assert interpret.pct_effect_of_price_change(-1.0, -0.5) == pytest.approx(1.0)
    assert interpret.pct_effect_of_price_change(2.0, 0.1) == pytest.approx(0.21)
    assert interpret.pct_effect_of_price_change(-1.686, -0.10) == pytest.approx(0.194, abs=1e-3)
    assert interpret.pct_effect_of_price_change(0.0, 0.3) == 0.0


def test_pct_effect_accepts_arrays():
    out = interpret.pct_effect_of_price_change(np.array([-1.0, 2.0]), 0.1)
    assert out == pytest.approx([1 / 1.1 - 1, 0.21])


def test_pct_effect_of_flag_hand_values():
    assert interpret.pct_effect_of_flag(math.log(2)) == pytest.approx(1.0)
    assert interpret.pct_effect_of_flag(0.0) == 0.0
    assert interpret.pct_effect_of_flag(math.log(0.5)) == pytest.approx(-0.5)


@pytest.mark.parametrize("lang", LANGS)
def test_effects_table_values_and_labels(lang):
    tab = interpret.effects_table(_fixef(), lang)
    assert list(tab.columns) == [
        t(k, lang) for k in ("effects_col_effect", "effects_col_estimate", "effects_col_low", "effects_col_high")
    ]
    assert list(tab.iloc[:, 0]) == [
        t(k, lang)
        for k in ("effect_disc10", "effect_base10", "effect_feature", "effect_display", "effect_tpr", "effect_trend")
    ]
    est = tab.iloc[:, 1].to_numpy()
    assert est == pytest.approx([100 / 9, (1.1**-2 - 1) * 100, 100.0, 50.0, -50.0, 0.0])
    # a 10% deeper discount: coefficient -1 -> 1/0.9 - 1; limits come from the transformed CI, ordered
    assert tab.iloc[0, 2] == pytest.approx((0.9 ** (-1 + 0.196) - 1) * 100)
    assert tab.iloc[0, 3] == pytest.approx((0.9 ** (-1 - 0.196) - 1) * 100)
    assert (tab.iloc[:, 2] <= tab.iloc[:, 1]).all() and (tab.iloc[:, 1] <= tab.iloc[:, 3]).all()
    # a higher regular price with a negative elasticity: the interval is also ordered
    assert tab.iloc[1, 2] < tab.iloc[1, 3]


def test_method_comparison_table_stacks_and_labels():
    ols = pd.DataFrame(
        {
            "model": ["ols_pooled", "ols_upc_fe"],
            "term": ["LOG_DISC"] * 2,
            "estimate": [-0.4, -1.4],
            "std_error": [0.02, 0.01],
        }
    )
    out = interpret.method_comparison_table(ols, _fixef())
    assert list(out.columns) == ["method", "term", "estimate", "std_error"]
    assert list(out["method"].unique()) == ["ols_pooled", "ols_upc_fe", "multilevel"]
    assert len(out) == 2 + len(_fixef())


def test_seasonal_profile_is_centred_and_matches_one_harmonic():
    fx = pd.DataFrame(
        {
            "term": ["SIN_1", "COS_1", "SIN_2", "COS_2", "SIN_3", "COS_3"],
            "estimate": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    prof = interpret.seasonal_profile(fx)
    assert list(prof["week_of_year"]) == list(range(1, 53))
    angle = 2 * np.pi * np.arange(1, 53) / 52.18
    curve = 0.1 * np.sin(angle)
    assert prof["effect_pct"].to_numpy() == pytest.approx(np.expm1(curve - curve.mean()) * 100)
    assert np.log1p(prof["effect_pct"] / 100).mean() == pytest.approx(0.0, abs=1e-12)


def _scenario_fixef():
    return pd.DataFrame(
        {
            "term": ["LOG_DISC", "FEATURE", "DISPLAY", "TPR_ONLY"],
            "estimate": [-1.0, math.log(2), math.log(1.5), math.log(0.5)],
        }
    )


def test_scenario_effects_hand_values():
    scen = [
        ("regular", 0.0, 0, 0, 0),
        ("tpr20", 0.2, 0, 0, 1),
        ("feat", 0.0, 1, 0, 0),
        ("both50", 0.5, 1, 1, 0),
    ]
    out = interpret.scenario_effects(_scenario_fixef(), scen)
    assert list(out.columns) == ["scenario", "discount", "feature", "display", "tpr", "effect_pct"]
    assert list(out["scenario"]) == ["regular", "tpr20", "feat", "both50"]
    # regular week: no change; TPR at 20%: 0.8 ** -1 * 0.5 - 1; feature alone: +100%; both with 50%: 2 * 2 * 1.5 - 1
    assert out["effect_pct"].to_numpy() == pytest.approx([0.0, (1 / 0.8 * 0.5 - 1) * 100, 100.0, 500.0])


def test_scenario_effects_interval_only_with_covariance():
    scen = [("a", 0.2, 1, 0, 0)]
    assert "ci_low" not in interpret.scenario_effects(_scenario_fixef(), scen).columns
    terms = ["LOG_DISC", "FEATURE", "DISPLAY", "TPR_ONLY"]
    cov = pd.DataFrame(np.diag([0.01, 0.04, 0.0, 0.0]), index=terms, columns=terms)
    cov.loc["LOG_DISC", "FEATURE"] = cov.loc["FEATURE", "LOG_DISC"] = 0.005
    out = interpret.scenario_effects(_scenario_fixef(), scen, cov=cov)
    x = np.array([math.log(0.8), 1.0, 0.0, 0.0])
    sd = math.sqrt(x @ cov.to_numpy() @ x)
    eta = -1.0 * math.log(0.8) + math.log(2)
    assert out["ci_low"].iloc[0] == pytest.approx(math.expm1(eta - 1.96 * sd) * 100)
    assert out["ci_high"].iloc[0] == pytest.approx(math.expm1(eta + 1.96 * sd) * 100)
    assert out["ci_low"].iloc[0] < out["effect_pct"].iloc[0] < out["ci_high"].iloc[0]


def test_scenario_effects_rejects_invalid_discount():
    with pytest.raises(ValueError):
        interpret.scenario_effects(_scenario_fixef(), [("bad", 1.0, 0, 0, 0)])
    with pytest.raises(ValueError):
        interpret.scenario_effects(_scenario_fixef(), [("bad", -0.1, 0, 0, 0)])


def _support_frame():
    # (discount, feature, display, tpr): 4 regular rows, 3 TPR near 10%, 2 TPR near 30%, 1 feature near 20%
    rows = [(0.0, 0, 0, 0)] * 4 + [(0.09, 0, 0, 1)] * 2 + [(0.11, 0, 0, 1)] + [(0.30, 0, 0, 1)] * 2 + [(0.21, 1, 0, 0)]
    df = pd.DataFrame(rows, columns=["DISCOUNT", "FEATURE", "DISPLAY", "TPR_ONLY"])
    df["PAIR_ID"] = [1, 1, 2, 2, 1, 2, 3, 3, 3, 4]
    return df


def test_scenario_support_counts_flags_exactly_and_discount_within_tolerance():
    scen = [
        ("regular", 0.0, 0, 0, 0),
        ("tpr10", 0.10, 0, 0, 1),
        ("tpr20", 0.20, 0, 0, 1),
        ("feat20", 0.20, 1, 0, 0),
        ("feat0", 0.0, 1, 0, 0),
        ("display10", 0.10, 0, 1, 0),
    ]
    out = interpret.scenario_support(_support_frame(), scen, tol=0.03)
    assert list(out["scenario"]) == [s[0] for s in scen]
    assert list(out["n_rows"]) == [4, 3, 0, 1, 0, 0]
    assert out["share"].to_numpy() == pytest.approx([0.4, 0.3, 0.0, 0.1, 0.0, 0.0])
    assert list(out["n_pairs"]) == [2, 3, 0, 1, 0, 0]
    wide = interpret.scenario_support(_support_frame(), [("tpr20", 0.20, 0, 0, 1)], tol=0.101)
    assert wide["n_rows"].iloc[0] == 3  # 0.11 and the two 0.30 rows are inside; 0.09 is not


def test_scenario_support_without_pair_id_and_empty_frame():
    df = _support_frame().drop(columns="PAIR_ID")
    assert "n_pairs" not in interpret.scenario_support(df, [("r", 0.0, 0, 0, 0)]).columns
    empty = interpret.scenario_support(df.iloc[0:0], [("r", 0.0, 0, 0, 0)])
    assert empty["n_rows"].iloc[0] == 0 and empty["share"].iloc[0] == 0.0


@pytest.mark.parametrize("lang", LANGS)
def test_scenario_table_translates_known_names_and_keeps_others(lang):
    scen = [*interpret.DEFAULT_SCENARIOS[:2], ("custom", 0.15, 0, 1, 0)]
    eff = interpret.scenario_effects(_scenario_fixef(), scen)
    sup = interpret.scenario_support(_support_frame(), scen)
    tab = interpret.scenario_table(eff, sup, lang)
    assert list(tab.columns) == [
        t(k, lang) for k in ("scenario_col_name", "scenario_col_discount", "scenario_col_effect", "scenario_col_rows")
    ] + [t("scenario_col_share", lang)]
    assert list(tab.iloc[:, 0]) == [t("scenario_regular", lang), t("scenario_tpr10", lang), "custom"]
    assert tab.iloc[:, 1].to_numpy() == pytest.approx([0.0, 10.0, 15.0])
    assert tab.iloc[:, 3].tolist() == sup["n_rows"].tolist()
    assert tab.iloc[:, 4].to_numpy() == pytest.approx(sup["share"].to_numpy() * 100)


def test_default_scenarios_are_valid_and_have_translations():
    out = interpret.scenario_effects(_scenario_fixef(), interpret.DEFAULT_SCENARIOS)
    assert len(out) == 8 and out["effect_pct"].iloc[0] == 0.0
    for name, *_ in interpret.DEFAULT_SCENARIOS:
        for lang in LANGS:
            assert t(name, lang)


@pytest.mark.parametrize("lang", LANGS)
def test_promo_by_discount_band_counts_each_row_once(lang):
    df = pd.DataFrame(
        {
            "DISCOUNT": [0.0, 0.0, 0.0, 0.04, 0.12, 0.25, 0.50, 0.07],
            "FEATURE": [0, 0, 1, 1, 0, 1, 1, 0],
            "DISPLAY": [0, 0, 0, 0, 1, 1, 1, 0],
            "TPR_ONLY": [0, 0, 0, 0, 0, 0, 0, 1],
        }
    )
    tab = interpret.promo_by_discount_band(df, lang)
    assert tab.to_numpy().sum() == len(df)
    kinds = ("kind_none", "kind_tpr", "kind_feature", "kind_display", "kind_feature_display")
    assert list(tab.index) == [t(k, lang) for k in kinds]
    bands = ("band_none", "band_0_5", "band_5_10", "band_10_20", "band_20_30", "band_30_plus")
    assert list(tab.columns) == [t(k, lang) for k in bands]
    assert tab.loc[t("kind_none", lang), t("band_none", lang)] == 2
    assert tab.loc[t("kind_tpr", lang), t("band_5_10", lang)] == 1
    assert tab.loc[t("kind_feature", lang), t("band_0_5", lang)] == 1
    assert tab.loc[t("kind_display", lang), t("band_10_20", lang)] == 1
    assert tab.loc[t("kind_feature_display", lang), t("band_30_plus", lang)] == 1


def test_compare_estimates_renames_terms_and_computes_differences():
    fx = pd.DataFrame(
        {"term": ["LOG_DISC", "LOG_BASE_DEV", "FEATURE"], "estimate": [-1.0, -2.0, 0.5], "std_error": [0.1, 0.2, 0.05]}
    )
    fe = pd.DataFrame(
        {
            "term": ["LOG_DISC", "LOG_BASE_PRICE", "FEATURE", "OTHER"],
            "estimate": [-1.1, -1.9, 0.5, 9.0],
            "se_cluster": [0.3, 0.2, 0.1, 1.0],
        }
    )
    out = interpret.compare_estimates(fx, fe, se_col="se_cluster", term_map={"LOG_BASE_PRICE": "LOG_BASE_DEV"})
    assert list(out["term"]) == ["LOG_DISC", "LOG_BASE_DEV", "FEATURE"]
    assert list(out.columns) == [
        "term",
        "estimate",
        "std_error",
        "estimate_other",
        "se_other",
        "difference",
        "se_ratio",
    ]
    assert out["difference"].to_numpy() == pytest.approx([-0.1, 0.1, 0.0])
    assert out["se_ratio"].to_numpy() == pytest.approx([3.0, 1.0, 2.0])
    same = interpret.compare_estimates(fx, fx)
    assert same["difference"].abs().max() == 0.0 and (same["se_ratio"] == 1.0).all()


def _p1_metrics():
    rows = []

    def add(model, fold, population, stratum, n_pairs, n_rows, pair, row, rmse=0.5):
        rows.append(
            {
                "model": model,
                "fold": fold,
                "population": population,
                "stratum": stratum,
                "n_pairs": n_pairs,
                "n_rows": n_rows,
                "mae_pair_mean": pair,
                "mae_row": row,
                "rmse": rmse,
            }
        )

    for fold, c1_shift in (("A", 0.01), ("B", -0.01)):
        for stratum, n_pairs in (("all", 10), ("<26", 4), (">=78", 0)):
            for model, base in (("C0", 0.40), ("C1", 0.40 + c1_shift), ("OLS_pooled", 0.90)):
                nan = np.nan
                add(
                    model,
                    fold,
                    "a_seen",
                    stratum,
                    n_pairs,
                    10 * n_pairs,
                    base if n_pairs else nan,
                    base if n_pairs else nan,
                )
    add("C0", "final", "a_seen", "all", 20, 200, 0.4541, 0.4446)
    add("GPBoost", "final", "a_seen", "all", 20, 200, 0.4520, 0.4478)
    add("C0", "final", "c_unseen", "all", 5, 30, 0.626, 0.640)
    add("GPBoost", "final", "c_unseen", "all", 5, 30, 0.688, 0.650)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("lang", LANGS)
def test_metric_table_by_fold_and_stratum(lang):
    out = interpret.metric_table(_p1_metrics(), "a_seen", lang, models=["C0", "C1", "OLS_pooled"], fold="A")
    assert list(out.columns) == [
        t("col_stratum", lang),
        t("col_pairs", lang),
        t("col_rows", lang),
        t("model_c0", lang),
        t("model_c1", lang),
        t("model_ols_pooled", lang),
    ]
    assert list(out[t("col_stratum", lang)]) == [t("all_rows", lang), t("stratum_lt26", lang)]
    assert out[t("col_pairs", lang)].tolist() == [10, 4]
    assert out[t("model_c1", lang)].tolist() == pytest.approx([0.41, 0.41])
    folds = _p1_metrics().query("fold != 'final'")
    both = interpret.metric_table(folds, "a_seen", lang, models=["C0", "C1"])
    assert list(both[t("col_fold", lang)]) == ["A", "A", "B", "B"]
    assert both[t("model_c1", lang)].tolist() == pytest.approx([0.41, 0.41, 0.39, 0.39])


def test_metric_table_adds_fold_label_and_uses_requested_metric():
    out = interpret.metric_table(_p1_metrics(), "a_seen", "en", value="mae_row", models=["C0", "GPBoost"], fold="final")
    assert out["C0"].tolist() == [0.4446] and out["GPBoost"].tolist() == [0.4478]


def test_relative_difference_signs_and_fold_requirement():
    m = _p1_metrics()
    gain = interpret.relative_difference(m, "a_seen", "all", "GPBoost", "C0", fold="final")
    assert gain == pytest.approx(100 * (0.4520 - 0.4541) / 0.4541)
    assert gain < 0
    with pytest.raises(ValueError):
        interpret.relative_difference(m, "a_seen", "all", "C1", "C0")
    assert interpret.relative_difference(m, "a_seen", "all", "C1", "C0", fold="A") == pytest.approx(2.5)


@pytest.mark.parametrize("lang", LANGS)
def test_model_comparison_cells_and_relative_column(lang):
    specs = [("a_seen", "all", "mae_pair_mean"), ("a_seen", "all", "mae_row"), ("c_unseen", "all", "mae_pair_mean")]
    out = interpret.model_comparison(_p1_metrics(), lang, specs, ["C0", "GPBoost"], rel=("GPBoost", "C0"), fold="final")
    assert list(out.columns) == [
        t("col_measure", lang),
        t("model_c0", lang),
        t("model_gpboost", lang),
        t("col_difference_pct", lang),
    ]
    assert out[t("model_c0", lang)].tolist() == [0.4541, 0.4446, 0.626]
    assert out[t("col_difference_pct", lang)].iloc[2] == pytest.approx(100 * (0.688 - 0.626) / 0.626)
    assert t("pop_c_unseen", lang) in out[t("col_measure", lang)].iloc[2]
    assert t("metric_mae_row", lang) in out[t("col_measure", lang)].iloc[1]


@pytest.mark.parametrize("lang", LANGS)
def test_selection_table_converts_gains_to_percent_and_translates_decisions(lang):
    log = pd.DataFrame(
        {
            "step": [1, 2],
            "candidate": ["C1", "C2"],
            "compared_to": ["C0", "C0"],
            "mae_fold_A": [0.43, np.nan],
            "mae_fold_A_incumbent": [0.42, np.nan],
            "mae_fold_B": [0.41, np.nan],
            "mae_fold_B_incumbent": [0.42, np.nan],
            "rel_gain_A": [-0.015, np.nan],
            "rel_gain_B": [0.005, np.nan],
            "mean_rel_gain": [-0.005, np.nan],
            "decision": ["rejected", "not fitted"],
        }
    )
    out = interpret.selection_table(log, lang)
    assert out[t("sel_gain_a", lang)].iloc[0] == pytest.approx(-1.5)
    assert out[t("sel_gain_mean", lang)].iloc[0] == pytest.approx(-0.5)
    assert list(out[t("sel_decision", lang)]) == [t("decision_rejected", lang), t("decision_not_fitted", lang)]
    assert len(out) == 2


@pytest.mark.parametrize("lang", LANGS)
def test_weekly_summary_table_counts_weeks_in_favour(lang):
    weeks = np.arange(104, 108)
    frame = pd.DataFrame(
        {
            "fold": "final",
            "candidate": "C0",
            "compared_to": ["GPBoost"] * 4,
            "WEEK": weeks,
            "mae_diff": [-0.02, 0.01, -0.01, 0.02],
        }
    )
    out = interpret.weekly_summary_table(frame, lang)
    assert out[t("wk_weeks", lang)].tolist() == [4]
    assert out[t("wk_better", lang)].tolist() == [2]
    assert out[t("wk_mean", lang)].iloc[0] == pytest.approx(0.0)
    assert out[t("wk_min", lang)].iloc[0] == pytest.approx(-0.02)
    assert out[t("col_fold", lang)].iloc[0] == t("fold_final", lang)
    assert out[t("col_reference", lang)].iloc[0] == t("model_gpboost", lang)


def test_gpboost_fold_table_joins_folds_and_computes_difference():
    fold_metrics = pd.DataFrame(
        {
            "fold": ["A", "B", "A"],
            "population": ["seen", "seen", "unseen"],
            "stratum": ["all", "all", "all"],
            "mae_pair_mean": [0.40, 0.42, 0.9],
        }
    )
    out = interpret.gpboost_fold_table(_p1_metrics(), fold_metrics, "en")
    assert out["Fold"].tolist() == ["A", "B"]
    assert out["C0, per-pair MAE"].tolist() == pytest.approx([0.40, 0.40])
    assert out["GPBoost vs. C0 (%)"].tolist() == pytest.approx([0.0, 5.0])


def _coverage():
    return pd.DataFrame(
        {
            "fold": ["A", "final"],
            "n_seen_pairs": [10, 20],
            "n_seen_rows": [100, 200],
            "n_seen_pairs_with_ols": [9, 19],
            "n_covered_pairs": [8, 18],
            "n_covered_rows": [90, 190],
            "pct_pairs_covered": [80.0, 90.0],
            "pct_rows_covered": [90.0, 95.0],
            "n_unseen_pairs": [1, 2],
            "n_unseen_rows": [5, 6],
            "n_none_rows": [0, 3],
            "n_pairs_not_estimated_LOG_DISC": [1, 2],
            "n_pairs_not_estimated_FEATURE": [3, 4],
        }
    )


@pytest.mark.parametrize("lang", LANGS)
def test_coverage_table_localises_headers_and_keeps_counts(lang):
    out = interpret.coverage_table(_coverage(), lang)
    assert list(out.columns)[1] == t("cov_seen_pairs", lang)
    assert list(out[t("col_fold", lang)]) == ["A", t("fold_final", lang)]
    assert list(out[t("cov_covered_rows", lang)]) == [90, 190]
    assert list(out[t("cov_pct_pairs", lang)]) == [80.0, 90.0]
    assert len(out.columns) == 11


def test_term_gap_table_has_one_column_per_term():
    out = interpret.term_gap_table(_coverage(), "en")
    assert list(out.columns) == [t("col_fold", "en"), "LOG_DISC", "FEATURE"]
    assert list(out["FEATURE"]) == [3, 4]
