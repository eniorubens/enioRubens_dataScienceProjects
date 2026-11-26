import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from retail_multilevel import viz  # noqa: E402
from retail_multilevel.i18n import LANGS, TEXTS, t  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _weekly():
    dates = pd.date_range("2009-01-14", periods=20, freq="7D")
    return pd.DataFrame(
        {
            "WEEK_END_DATE": dates,
            "WEEK": range(1, 21),
            "UNITS": np.arange(20) * 10 + 100,
            "IS_TEST": [False] * 16 + [True] * 4,
        }
    )


def _promo_rates(index):
    n = len(index)
    data = {c: np.linspace(0.05, 0.4, n) for c in ["FEATURE", "DISPLAY", "TPR_ONLY", "ANY_PROMO", "BELOW_BASE"]}
    return pd.DataFrame({**data, "n": [100] * n}, index=index)


def _variance():
    return pd.DataFrame({"group": ["STORE_NUM", "UPC", "PAIR_ID", "Residual"], "share": [0.1, 0.3, 0.2, 0.4]})


def _icc():
    rows = []
    for model in ["m0", "m1"]:
        for g, s in zip(["STORE_NUM", "UPC", "Residual"], [0.2, 0.5, 0.3], strict=True):
            rows.append({"model": model, "group": g, "share": s})
    return pd.DataFrame(rows)


def _blups(n=10):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"level": [str(i) for i in range(n)], "term": "(Intercept)", "blup": rng.normal(size=n), "cond_sd": 0.1}
    )


def _coverage():
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame(
        {
            "PAIR_ID": [f"{i}:{i}" for i in range(n)],
            "STORE_NUM": range(n),
            "UPC": range(n),
            "n_weeks_train": rng.integers(10, 100, n),
            "n_weeks_test": rng.integers(1, 20, n),
        }
    )


def _lift():
    return pd.DataFrame(
        {"lift_log": [0.3, 0.2, 0.1], "lift_pct": [0.35, 0.22, 0.105], "n_weeks": [50, 40, 90]},
        index=pd.Index(["FEATURE", "DISPLAY", "TPR_ONLY"], name="promo"),
    )


CASES = {
    "weekly": (viz.plot_weekly_units, _weekly, "weekly_units_title"),
    "rates_all": (viz.plot_promo_rates, lambda: _promo_rates(["all"]), "promo_rates_title"),
    "rates_multi": (
        viz.plot_promo_rates,
        lambda: _promo_rates(pd.MultiIndex.from_tuples([("CEREAL", "A"), ("PASTA", "B")])),
        "promo_rates_title",
    ),
    "variance": (viz.plot_variance_shares, _variance, "variance_share_title"),
    "icc": (viz.plot_icc_by_model, _icc, "icc_by_model_title"),
    "coverage": (viz.plot_pair_coverage, _coverage, "pair_coverage_title"),
    "lift": (viz.plot_raw_promo_lift, _lift, "raw_lift_title"),
}


def _texts(fig):
    return [x.get_text() for x in fig.findobj(Text) if x.get_text().strip()]


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("name", list(CASES))
def test_plot_is_translated(name, lang):
    func, make, title_key = CASES[name]
    fig = func(make(), lang)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == t(title_key, lang)
    texts = _texts(fig)
    assert not any(raw in TEXTS for raw in texts)
    assert not any(key in text for key in TEXTS for text in texts)


@pytest.mark.parametrize("lang", LANGS)
def test_weekly_legend_and_labels(lang):
    ax = viz.plot_weekly_units(_weekly(), lang).axes[0]
    legend = [x.get_text() for x in ax.get_legend().get_texts()]
    assert legend == [t("legend_train", lang), t("legend_test", lang)]
    assert ax.get_ylabel() == t("weekly_units_ylabel", lang)


@pytest.mark.parametrize("lang", LANGS)
def test_promo_rates_legend_and_flattened_labels(lang):
    idx = pd.MultiIndex.from_tuples([("CEREAL", "A"), ("PASTA", "B")])
    ax = viz.plot_promo_rates(_promo_rates(idx), lang).axes[0]
    legend = [x.get_text() for x in ax.get_legend().get_texts()]
    assert legend == [viz.promo_label(c, lang) for c in viz.PROMO_ORDER]
    assert [x.get_text() for x in ax.get_yticklabels()] == ["CEREAL / A", "PASTA / B"]
    single = viz.plot_promo_rates(_promo_rates(["all"]), lang).axes[0]
    assert [x.get_text() for x in single.get_yticklabels()] == [t("all_rows", lang)]


@pytest.mark.parametrize("lang", LANGS)
def test_variance_shares_labels(lang):
    ax = viz.plot_variance_shares(_variance(), lang).axes[0]
    assert [x.get_text() for x in ax.get_yticklabels()] == [
        t("group_store", lang),
        t("group_product", lang),
        t("group_pair", lang),
        t("group_residual", lang),
    ]
    assert "40.0%" in _texts(ax.figure)
    other = viz.plot_variance_shares(_variance(), lang, title_key="icc_by_model_title").axes[0]
    assert other.get_title() == t("icc_by_model_title", lang)


@pytest.mark.parametrize("lang", LANGS)
def test_icc_legend_and_axis(lang):
    ax = viz.plot_icc_by_model(_icc(), lang).axes[0]
    legend = [x.get_text() for x in ax.get_legend().get_texts()]
    assert legend == [group_ for group_ in (viz.group_label(g, lang) for g in ["STORE_NUM", "UPC", "Residual"])]
    assert ax.get_xlim() == (0, 1)
    assert [x.get_text() for x in ax.get_yticklabels()] == ["m0", "m1"]


@pytest.mark.parametrize("lang", LANGS)
def test_raw_lift_caption_and_labels(lang):
    fig = viz.plot_raw_promo_lift(_lift(), lang)
    assert t("raw_lift_caption", lang) in _texts(fig)
    assert [x.get_text() for x in fig.axes[0].get_yticklabels()] == [
        viz.promo_label(c, lang) for c in ["FEATURE", "DISPLAY", "TPR_ONLY"]
    ]


@pytest.mark.parametrize("lang", LANGS)
def test_raw_lift_uses_fractions(lang):
    ax = viz.plot_raw_promo_lift(_lift(), lang).axes[0]
    assert ax.xaxis.get_major_formatter().xmax == 1.0
    assert "35%" in _texts(ax.figure)


@pytest.mark.parametrize("lang", LANGS)
def test_variance_shares_share_col(lang):
    df = _variance().rename(columns={"share": "between_share"})
    fig = viz.plot_variance_shares(df, lang, share_col="between_share")
    assert fig.axes[0].get_title() == t("variance_share_title", lang)


@pytest.mark.parametrize("lang", LANGS)
def test_caterpillar_title_and_labels(lang):
    fig = viz.plot_caterpillar(_blups(), lang, "STORE_NUM", labels={"3": "Three"})
    ax = fig.axes[0]
    assert ax.get_title() == t("caterpillar_title", lang).format(group=viz.group_label("STORE_NUM", lang))
    ticks = [x.get_text() for x in ax.get_yticklabels()]
    assert len(ticks) == 10
    assert "Three" in ticks
    assert ax.get_xlabel() == t("caterpillar_xlabel", lang)


@pytest.mark.parametrize("lang", LANGS)
def test_caterpillar_many_levels_hides_ticks_and_truncates(lang):
    ax = viz.plot_caterpillar(_blups(100), lang, "UPC").axes[0]
    assert len(ax.get_yticks()) == 0
    ax = viz.plot_caterpillar(_blups(100), lang, "UPC", max_levels=20).axes[0]
    assert len(ax.lines) >= 1 and len(ax.collections) >= 0


@pytest.mark.parametrize("lang", LANGS)
def test_pair_coverage_median_annotation(lang):
    df = _coverage()
    fig = viz.plot_pair_coverage(df, lang)
    median = float(np.median(df["n_weeks_train"]))
    assert t("pair_coverage_median", lang).format(value=median) in _texts(fig)


def test_draws_on_given_axes():
    fig, ax = plt.subplots()
    assert viz.plot_pair_coverage(_coverage(), "en", ax=ax) is fig


def test_group_label_spellings():
    for name in ["PAIR_ID", "STORE_NUM:UPC", "UPC:STORE_NUM"]:
        assert viz.group_label(name, "en") == "Store × product"
        assert viz.group_label(name, "pt") == "Loja × produto"
    assert viz.group_label("STORE_NUM", "pt") == "Loja"
    assert viz.group_label("UPC", "en") == "Product"
    assert viz.group_label("Residual", "pt") == "Resíduo"
    assert viz.group_label("OTHER", "pt") == "OTHER"


def test_promo_label_mapping():
    assert viz.promo_label("FEATURE", "pt") == "Encarte"
    assert viz.promo_label("TPR_ONLY", "en") == "Temporary price reduction (TPR)"
    assert viz.promo_label("BELOW_BASE", "pt") == "Preço abaixo do base"
    assert viz.promo_label("ANY_PROMO", "en") == "Any promotion"
    assert viz.promo_label("DISPLAY", "pt") == "Display na loja"
    assert viz.promo_label("UNKNOWN", "en") == "UNKNOWN"


def test_save_figure(tmp_path):
    fig = viz.plot_pair_coverage(_coverage(), "pt")
    out = viz.save_figure(fig, tmp_path / "sub" / "dir" / "fig.png")
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_figure_is_retina_density_by_default(tmp_path):
    from PIL import Image

    fig = viz.plot_pair_coverage(_coverage(), "pt")
    hi = viz.save_figure(fig, tmp_path / "hi.png")
    lo = viz.save_figure(fig, tmp_path / "lo.png", dpi=150)
    with Image.open(hi) as a, Image.open(lo) as b:
        assert viz.EXPORT_DPI == 300
        assert a.size[0] >= 1.9 * b.size[0] and a.size[1] >= 1.9 * b.size[1]


def test_use_retina_is_a_no_op_outside_ipython():
    assert viz.use_retina() is False


def _fixef_plot():
    terms = [
        "(Intercept)", "LOG_DISC", "LOG_BASE_DEV", "LOG_BASE_MEAN", "FEATURE", "DISPLAY", "TPR_ONLY", "TREND",
        "SIN_1", "COS_1",
    ]  # fmt: skip
    est = np.linspace(-1.5, 0.5, len(terms))
    return pd.DataFrame({"term": terms, "estimate": est, "ci_low": est - 0.1, "ci_high": est + 0.1})


def _pseudo():
    return pd.DataFrame(
        {
            "group": ["STORE_NUM:UPC", "STORE_NUM", "UPC", "Residual"],
            "reduction": [-0.04, -0.03, 0.29, 0.34],
        }
    )


def _methods():
    rows = []
    for method, shift in [("ols_pooled", 0.5), ("ols_upc_fe", 0.2), ("multilevel", 0.0)]:
        for term in ["LOG_DISC", "LOG_BASE_DEV", "FEATURE", "DISPLAY", "TPR_ONLY"]:
            rows.append({"method": method, "term": term, "estimate": shift - 1.0, "std_error": 0.05})
    return pd.DataFrame(rows)


def _season():
    weeks = np.arange(1, 53)
    return pd.DataFrame({"week_of_year": weeks, "effect_pct": 3 * np.sin(weeks / 8)})


CASES.update(
    {
        "coef": (viz.plot_coefficients, _fixef_plot, "coef_title"),
        "pseudo": (viz.plot_pseudo_r2, _pseudo, "pseudo_r2_title"),
        "methods": (viz.plot_method_comparison, _methods, "method_comparison_title"),
        "season": (viz.plot_seasonality, _season, "seasonality_title"),
    }
)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("name", ["coef", "pseudo", "methods", "season"])
def test_fixed_effects_plots_are_translated(name, lang):
    func, make, title_key = CASES[name]
    fig = func(make(), lang)
    assert fig.axes[0].get_title() == t(title_key, lang)
    texts = _texts(fig)
    assert not any(key in text for key in TEXTS for text in texts)


@pytest.mark.parametrize("lang", LANGS)
def test_coefficients_hide_intercept_and_fourier_by_default(lang):
    ax = viz.plot_coefficients(_fixef_plot(), lang).axes[0]
    ticks = [x.get_text() for x in ax.get_yticklabels()]
    assert ticks == [
        viz.term_label(x, lang)
        for x in ["LOG_DISC", "LOG_BASE_DEV", "LOG_BASE_MEAN", "FEATURE", "DISPLAY", "TPR_ONLY", "TREND"]
    ]
    chosen = viz.plot_coefficients(_fixef_plot(), lang, terms=["SIN_1", "FEATURE"]).axes[0]
    expected = [viz.term_label("SIN_1", lang), viz.promo_label("FEATURE", lang)]
    assert [x.get_text() for x in chosen.get_yticklabels()] == expected


@pytest.mark.parametrize("lang", LANGS)
def test_pseudo_r2_shows_negative_values_and_labels(lang):
    fig = viz.plot_pseudo_r2(_pseudo(), lang)
    texts = _texts(fig)
    assert "-4.0%" in texts and "34.0%" in texts
    assert [x.get_text() for x in fig.axes[0].get_yticklabels()] == [
        viz.group_label(g, lang) for g in ["STORE_NUM:UPC", "STORE_NUM", "UPC", "Residual"]
    ]
    assert fig.axes[0].get_xlim()[0] < -0.04


@pytest.mark.parametrize("lang", LANGS)
def test_method_comparison_legend_and_terms(lang):
    ax = viz.plot_method_comparison(_methods(), lang).axes[0]
    legend = [x.get_text() for x in ax.get_legend().get_texts()]
    assert legend == [t(k, lang) for k in ("method_ols_pooled", "method_ols_upc_fe", "method_multilevel")]
    assert [x.get_text() for x in ax.get_yticklabels()] == [viz.term_label(x, lang) for x in viz.COMPARISON_TERMS]


def test_term_label_mapping():
    assert viz.term_label("LOG_DISC", "en") == t("term_log_disc", "en")
    assert viz.term_label("FEATURE", "pt") == "Encarte"
    assert viz.term_label("SIN_2", "en").endswith(" 2") and viz.term_label("COS_3", "pt").endswith(" 3")
    assert viz.term_label("OTHER", "pt") == "OTHER"


def _lag_df():
    return pd.DataFrame({"lag": [1, 2, 3, 4], "corr": [0.2, 0.1, 0.05, -0.01], "n_pairs_of_obs": [100, 90, 80, 70]})


def _shock_df():
    weeks = np.arange(1, 21)
    return pd.DataFrame({"WEEK": weeks, "mean_resid": 0.01 * np.sin(weeks), "n": 100, "expected_sd": 0.01})


CASES.update(
    {
        "resid_lag": (viz.plot_residual_lag_corr, _lag_df, "resid_lag_title"),
        "week_shocks": (viz.plot_week_shocks, _shock_df, "week_shocks_title"),
    }
)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("name", ["resid_lag", "week_shocks"])
def test_residual_plots_are_translated(name, lang):
    func, make, title_key = CASES[name]
    fig = func(make(), lang)
    assert fig.axes[0].get_title() == t(title_key, lang)
    assert not any(key in text for key in TEXTS for text in _texts(fig))


@pytest.mark.parametrize("lang", LANGS)
def test_week_shocks_band_legend_and_lag_bars(lang):
    ax = viz.plot_week_shocks(_shock_df(), lang).axes[0]
    assert [x.get_text() for x in ax.get_legend().get_texts()] == [t("week_shocks_band", lang)]
    bars = viz.plot_residual_lag_corr(_lag_df(), lang).axes[0]
    assert [round(p.get_height(), 2) for p in bars.patches] == [0.2, 0.1, 0.05, -0.01]


def _p1_metrics():
    rows = []
    for fold in ("A", "B"):
        for stratum, n_pairs in (("all", 10), ("<26", 3), (">=78", 0)):
            for k, model in enumerate(("C0", "C1", "OLS_pooled")):
                rows.append(
                    {
                        "model": model,
                        "fold": fold,
                        "population": "a_seen",
                        "stratum": stratum,
                        "n_pairs": n_pairs,
                        "n_rows": 10 * n_pairs,
                        "mae_pair_mean": np.nan if n_pairs == 0 else 0.4 + 0.01 * k,
                        "mae_row": 0.4,
                        "rmse": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def _weekly_diff():
    weeks = np.arange(104, 112)
    frames = [
        pd.DataFrame({"WEEK": weeks, "compared_to": ref, "mae_diff": shift + 0.01 * np.cos(weeks)})
        for ref, shift in (("OLS_pooled", -0.4), ("OLS_pair", -0.03), ("GPBoost", 0.0))
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("lang", LANGS)
def test_plot_fold_mae_groups_models_and_skips_empty_strata(lang):
    fig = viz.plot_fold_mae(_p1_metrics(), lang)
    ax = fig.axes[0]
    assert ax.get_title() == t("fold_mae_title", lang)
    assert ax.get_ylabel() == t("fold_mae_ylabel", lang)
    assert len(ax.patches) == 3 * 4  # two folds x two non-empty strata x three models
    assert [x.get_text() for x in ax.get_legend().get_texts()] == [
        t("model_c0", lang),
        t("model_c1", lang),
        t("model_ols_pooled", lang),
    ]
    assert t("fold_label", lang).format(fold="A") in ax.get_xticklabels()[0].get_text()
    assert not any(key in text for key in TEXTS for text in _texts(fig))


@pytest.mark.parametrize("lang", LANGS)
def test_plot_weekly_error_diff_draws_one_line_per_reference_and_zero_line(lang):
    fig = viz.plot_weekly_error_diff(_weekly_diff(), lang)
    ax = fig.axes[0]
    assert ax.get_title() == t("weekly_diff_title", lang)
    assert ax.get_ylabel() == t("weekly_diff_ylabel", lang)
    assert [x.get_text() for x in ax.get_legend().get_texts()] == [
        t("model_ols_pooled", lang),
        t("model_ols_pair", lang),
        t("model_gpboost", lang),
    ]
    zero_lines = [ln for ln in ax.lines if set(ln.get_ydata()) == {0}]
    assert len(zero_lines) == 1 and len(ax.lines) == 4
    assert not any(key in text for key in TEXTS for text in _texts(fig))


def test_p1_plots_draw_on_a_given_axes():
    fig, axes = plt.subplots(1, 2)
    assert viz.plot_fold_mae(_p1_metrics(), "en", ax=axes[0]) is fig
    assert viz.plot_weekly_error_diff(_weekly_diff(), "pt", ax=axes[1]) is fig
