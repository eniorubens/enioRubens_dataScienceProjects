"""Synthetic tests of the plots and labels of notebooks 05 and 06."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from retail_multilevel import viz  # noqa: E402
from retail_multilevel.i18n import (  # noqa: E402
    LANGS,
    TEXTS,
    contrast_label,
    format_p,
    level_label,
    localize_columns,
    scenario_label,
    t,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _texts(fig):
    return [x.get_text() for x in fig.findobj(Text) if x.get_text().strip()]


def _slopes():
    rows = [
        ("segment", "MAINSTREAM", True, -1.7),
        ("segment", "VALUE", False, -1.6),
        ("private_label", "False", True, -1.9),
        ("private_label", "True", False, -1.4),
    ]
    df = pd.DataFrame(rows, columns=["test", "level", "is_reference", "slope"])
    return df.assign(ci_low=df["slope"] - 0.2, ci_high=df["slope"] + 0.2)


def _fixef_compare():
    return pd.DataFrame(
        {
            "term": ["LOG_DISC", "FEATURE", "TREND"],
            "est_ref": [-1.7, 0.5, -0.02],
            "se_ref": [0.01, 0.004, 0.002],
            "est_catweek": [-1.73, 0.51, -0.025],
            "se_catweek": [0.011, 0.004, 0.013],
        }
    )


def _lag_corr():
    return pd.DataFrame(
        {
            "model": ["F3_ref"] * 4 + ["F3_catweek"] * 4,
            "lag": [1, 2, 3, 4] * 2,
            "corr": [0.20, 0.12, 0.09, 0.10, 0.19, 0.12, 0.10, 0.09],
        }
    )


def _scenarios(outside_last=False):
    df = pd.DataFrame(
        {
            "scenario": ["scenario_regular", "scenario_tpr20", "custom"],
            "effect_pct": [0.0, 39.8, 10.0],
            "ci_low_pct": [0.0, 39.0, 9.0],
            "ci_high_pct": [0.0, 40.5, 11.0],
            "share": [0.7, 0.03, 0.0],
            "n_rows": [700, 30, 0],
            "n_rows_fallback": [0, 2, 0],
            "outside_support": [False, False, outside_last],
        }
    )
    return df


@pytest.mark.parametrize("lang", LANGS)
def test_plot_moderator_slopes_panels_levels_and_brand_labels(lang):
    fig = viz.plot_moderator_slopes(_slopes(), lang)
    assert len(fig.axes) == 2
    assert fig.get_suptitle() == t("slopes_title", lang)
    assert fig.axes[-1].get_xlabel() == t("slopes_xlabel", lang)
    assert fig.axes[0].get_title(loc="left") == contrast_label("segment", lang)
    labels = [x.get_text() for x in fig.axes[1].get_yticklabels()]
    assert labels == [f"{t('brand_national', lang)} ({t('slopes_reference', lang)})", t("brand_private", lang)]
    assert not any(key in text for key in TEXTS for text in _texts(fig))


@pytest.mark.parametrize("lang", LANGS)
def test_plot_catweek_compare_draws_two_series_and_legend(lang):
    fig = viz.plot_catweek_compare(_fixef_compare(), lang)
    ax = fig.axes[0]
    assert ax.get_title() == t("catweek_compare_title", lang)
    assert [x.get_text() for x in ax.get_legend().get_texts()] == [t("model_f3_ref", lang), t("model_f3_catweek", lang)]
    assert len(ax.get_yticklabels()) == 3
    only = viz.plot_catweek_compare(_fixef_compare(), lang, terms=["LOG_DISC"])
    assert len(only.axes[0].get_yticklabels()) == 1


@pytest.mark.parametrize("lang", LANGS)
def test_plot_lag_corr_compare_one_bar_per_model_and_lag(lang):
    fig = viz.plot_lag_corr_compare(_lag_corr(), lang)
    ax = fig.axes[0]
    assert len(ax.patches) == 8
    assert ax.get_title() == t("lag_compare_title", lang)
    assert [x.get_text() for x in ax.get_legend().get_texts()] == [t("model_f3_ref", lang), t("model_f3_catweek", lang)]
    fig2, axes = plt.subplots(1, 2)
    assert viz.plot_lag_corr_compare(_lag_corr(), lang, ax=axes[1]) is fig2


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("outside", [False, True])
def test_plot_scenario_effects_keeps_unsupported_scenarios(lang, outside):
    fig = viz.plot_scenario_effects(_scenarios(outside), lang)
    ax_eff, ax_sup = fig.axes
    assert ax_eff.get_title() == t("scen_effects_title", lang)
    assert ax_sup.get_title() == t("scen_support_title", lang)
    labels = [x.get_text() for x in ax_eff.get_yticklabels()]
    assert labels == [t("scenario_regular", lang), t("scenario_tpr20", lang), "custom"]
    texts = _texts(fig)
    assert (t("scen_outside", lang) in texts) is outside
    assert not any(key in text for key in TEXTS for text in texts)


def test_format_p_never_prints_a_bare_zero():
    assert format_p(0.0) == "< 1e-300"
    assert format_p(2.6e-16) == "2.60e-16"
    assert format_p(0.0031) == "0.0031"
    assert format_p(float("nan")) == "n/a"


def test_level_and_scenario_labels():
    assert level_label("private_label", "False", "pt") == t("brand_national", "pt")
    assert level_label("private_label", True, "en") == t("brand_private", "en")
    assert level_label("segment", "VALUE", "en") == "VALUE"
    assert contrast_label("category", "pt") == "Categoria"
    assert contrast_label("other", "pt") == "other"
    assert scenario_label("scenario_tpr10", "en") == t("scenario_tpr10", "en")
    assert scenario_label("custom", "en") == "custom"


@pytest.mark.parametrize("lang", LANGS)
def test_localize_columns_renames_known_columns_only(lang):
    df = pd.DataFrame({"n_rows": [1], "n_pairs": [2], "unknown_col": [3], "p_raw": [0.1]})
    out = localize_columns(df, lang)
    assert list(out.columns) == [t("f_n_rows", lang), t("f_n_pairs", lang), "unknown_col", t("f_p_raw", lang)]
    assert np.array_equal(out.to_numpy(), df.to_numpy())


@pytest.mark.parametrize("lang", LANGS)
def test_plot_scenario_effects_annotates_fallback_rows(lang):
    fig = viz.plot_scenario_effects(_scenarios(), lang)
    assert any(t("scen_fallback", lang).format(n=2) in text for text in _texts(fig))
