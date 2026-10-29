"""Tests for the chart helpers (drawn off screen)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from marketing_campaign import plots  # noqa: E402


@pytest.fixture
def figures(monkeypatch):
    """Capture every figure that a helper shows, then close it."""
    shown = []
    monkeypatch.setattr(plots.plt, "show", lambda: shown.append(plt.gcf()))
    yield shown
    plt.close("all")


def _texts(figure) -> list[str]:
    """Return the titles, subtitles and axis labels of a figure."""
    found = []
    for axis in figure.axes:
        found.append(axis.get_title(loc="left"))
        found.append(axis.get_xlabel())
        found.extend(text.get_text() for text in axis.texts)
    return found


def test_chart_texts_exist_in_both_languages():
    """The ``plots.*`` messages are defined for English and Portuguese."""
    assert set(plots._texts("en")) == set(plots._texts("pt"))
    assert "z_title" in plots._texts("en")


def test_set_corporate_theme_applies_the_style_and_display_options():
    """Spines are hidden and pandas shows wide tables."""
    plots.set_corporate_theme()

    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.titlesize"] == 22
    assert pd.get_option("display.max_columns") == 500


def test_add_titles_writes_title_and_subtitle(figures):
    """The title is left aligned and the subtitle sits above the plot."""
    _, ax = plt.subplots()

    plots.add_titles(ax, "Title", subtitle="Subtitle")

    assert ax.get_title(loc="left") == "Title"
    assert [t.get_text() for t in ax.texts] == ["Subtitle"]


@pytest.mark.parametrize(
    ("lang", "banner", "verdict"),
    [
        ("en", "Z-TEST RESULT", "There is not enough evidence"),
        ("pt", "RESULTADO DO TESTE Z", "Não há evidências suficientes"),
    ],
)
def test_z_test_plot_reports_the_result_in_each_language(
    lang, banner, verdict, figures, capsys
):
    """A non-significant z prints the banner and the matching verdict."""
    plots.plot_z_test_distribution(z_stat=0.0779, p_value=0.94, lang=lang)

    printed = capsys.readouterr().out
    assert banner in printed
    assert verdict in printed
    assert "1.96" in printed
    assert len(figures) == 1


def test_z_test_plot_rejects_h0_when_the_p_value_is_small(figures, capsys):
    """A significant z prints the rejection and the critical value."""
    plots.plot_z_test_distribution(z_stat=3.1, lang="en")

    printed = capsys.readouterr().out
    assert "statistically significant" in printed
    assert "P-value: 0.001935" in printed


@pytest.mark.parametrize(
    ("alternative", "critical"),
    [("greater", "1.64"), ("less", "-1.64"), ("two-sided", "1.96")],
)
def test_z_test_plot_critical_value_follows_the_alternative(
    alternative, critical, figures, capsys
):
    """The printed critical value matches the tail being tested."""
    plots.plot_z_test_distribution(z_stat=0.5, alternative=alternative)

    assert f"Critical Value: {critical}" in capsys.readouterr().out


def test_z_test_plot_marks_a_statistic_outside_the_scale(figures):
    """Z beyond the axis is clipped and explained by an annotation."""
    plots.plot_z_test_distribution(z_stat=6.0, p_value=0.0, lang="pt")

    assert any("fora da escala" in t for t in _texts(figures[0]))


def test_z_test_plot_rejects_unknown_alternatives_in_each_language():
    """The error message follows the chosen language."""
    with pytest.raises(ValueError, match="deve ser"):
        plots.plot_z_test_distribution(1.0, alternative="x", lang="pt")
    with pytest.raises(ValueError, match="should be"):
        plots.plot_z_test_distribution(1.0, alternative="x", lang="en")


@pytest.mark.parametrize(
    ("lang", "title"),
    [("en", "Confusion Matrix"), ("pt", "Matriz de Confusão")],
)
def test_retention_diagnostics_draw_matrix_and_roc(lang, title, figures):
    """Two panels: the confusion matrix and the ROC curve."""
    y_true = pd.Series([0, 0, 1, 1, 1, 0])
    plots.plot_retention_diagnostics(
        y_true,
        np.array([0, 1, 1, 1, 0, 0]),
        np.array([0.2, 0.7, 0.8, 0.6, 0.4, 0.1]),
        roc_auc=0.78,
        lang=lang,
    )

    assert len(figures[0].axes) == 2
    assert figures[0].axes[0].get_title(loc="left") == title


def test_retention_diagnostics_explain_a_missing_roc_auc(figures):
    """Without an AUC the panel says so instead of drawing a curve."""
    plots.plot_retention_diagnostics(
        pd.Series([1, 1]), np.array([1, 1]), np.array([0.9, 0.8]),
        roc_auc=float("nan"), lang="pt",
    )

    assert "ROC AUC indisponível" in _texts(figures[0])


def _experiments() -> pd.DataFrame:
    """Return two scenarios as the experiment table has them."""
    return pd.DataFrame(
        {
            "experiment": ["A | with converted", "B | without converted"],
            "roc_auc": [0.96, 0.7],
            "recall_not_retained": [0.96, 0.5],
            "f1_not_retained": [0.53, 0.2],
        }
    )


def test_experiment_comparison_draws_two_panels(figures):
    """AUC and recall are compared side by side."""
    plots.plot_retention_experiment_comparison(_experiments(), lang="pt")

    titles = [axis.get_title(loc="left") for axis in figures[0].axes]
    assert titles == [
        "ROC AUC por Experimento",
        "Recall de Não Retido por Experimento",
    ]


def test_experiment_comparison_handles_an_empty_table(figures, capsys):
    """An empty table prints a notice and draws nothing."""
    plots.plot_retention_experiment_comparison(_experiments().iloc[0:0])

    assert "No experiments available" in capsys.readouterr().out
    assert figures == []


def test_model_signal_summary_shows_localised_coefficients(
    result, figures, monkeypatch
):
    """The logistic model gets a coefficient table and chart in PT."""
    tables = []
    monkeypatch.setattr(plots, "display", tables.append)

    plots.plot_model_signal_summary(
        result.model.pipeline, "LogisticRegression", lang="pt"
    )

    assert tables[0].columns.tolist() == [
        "feature",
        "coeficiente",
        "coeficiente_abs",
    ]
    assert len(tables[0]) == 10
    assert figures[0].axes[0].get_xlabel() == "Coeficiente absoluto"


def test_model_signal_summary_uses_importances_when_available(
    result, figures, monkeypatch
):
    """Tree models expose importances instead of coefficients."""
    tables = []
    monkeypatch.setattr(plots, "display", tables.append)
    preprocessor = result.model.pipeline.named_steps["preprocessor"]
    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    tree.fit(
        preprocessor.transform(result.model.X_test), result.model.y_test
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", tree)])

    plots.plot_model_signal_summary(pipeline, "Tree", lang="en")

    assert tables[0].columns.tolist() == ["feature", "importance"]
    assert figures[0].axes[0].get_xlabel() == "Importance"


def test_model_signal_summary_explains_when_nothing_is_exposed(
    capsys, monkeypatch
):
    """Estimators without coefficients or importances print a notice."""
    class Opaque:
        """Estimator exposing neither coefficients nor importances."""

    pipeline = type("P", (), {"named_steps": {"classifier": Opaque()}})()

    plots.plot_model_signal_summary(pipeline, "Opaque", lang="pt")

    assert "nao expõe" in capsys.readouterr().out


def test_feature_names_fall_back_to_generic_names():
    """A size mismatch or a missing method gives feature_0, feature_1..."""
    class Broken:
        """Preprocessor without ``get_feature_names_out``."""

    pipeline = type("P", (), {"named_steps": {"preprocessor": Broken()}})()

    names = plots.get_transformed_feature_names(pipeline, 3)

    assert names.tolist() == ["feature_0", "feature_1", "feature_2"]
    assert len(plots.get_transformed_feature_names(pipeline)) == 0


def _all_texts(figure) -> str:
    """Join every title, label and annotation of a figure."""
    parts = []
    for axis in figure.axes:
        parts += [axis.get_title(loc="left"), axis.get_xlabel(),
                  axis.get_ylabel()]
        parts += [text.get_text() for text in axis.texts]
        legend = axis.get_legend()
        if legend:
            parts += [text.get_text() for text in legend.get_texts()]
    return "\n".join(parts)


@pytest.fixture
def small(result):
    """Return a few real tables to feed the chart functions."""
    return result


@pytest.mark.parametrize(
    ("lang", "needle"), [("en", "Funnel: From Impact"), ("pt", "Funil:")]
)
def test_funnel_chart_follows_the_language(small, figures, lang, needle):
    """Title and stage names come from the catalog."""
    plots.plot_funnel(small.funnel, lang)

    texts = _all_texts(figures[0])
    assert needle in texts
    assert "{:,}".format(int(small.funnel.iloc[0, 1])) in texts


@pytest.mark.parametrize(
    ("lang", "needle"),
    [
        ("en", "Conversion Rate by Channel"),
        ("pt", "Taxa de Conversão por Canal"),
    ],
)
def test_channel_chart_has_one_bar_per_channel(small, figures, lang, needle):
    """Two panels with a bar for every marketing channel."""
    plots.plot_channel_performance(small.channels, lang)

    assert needle in _all_texts(figures[0])
    assert len(figures[0].axes[0].patches) == len(small.channels)


def test_k_diagnostics_and_boxplots_use_segment_names(small, figures):
    """The boxplot x labels are the segment names of the language."""
    from marketing_campaign.i18n import SEGMENT_LABELS
    from marketing_campaign.segmentation import (
        assign_segments, build_rfm, evaluate_k,
    )

    rfm = assign_segments(
        build_rfm(small.exposures), labels=SEGMENT_LABELS["pt"]
    )
    plots.plot_k_diagnostics(evaluate_k(rfm, range(2, 4)), "pt")
    plots.plot_rfm_boxplots(rfm, "pt")

    assert "Inercia" in _all_texts(figures[0])
    labels = [t.get_text() for t in figures[1].axes[0].get_xticklabels()]
    assert labels == ["Alto Valor", "Médio Valor", "Baixo Valor"]


def test_ltv_chart_marks_the_global_average(small, figures):
    """The legend carries the global mean value."""
    plots.plot_ltv_by_segment(small.ltv.by_segment, small.ltv.global_ltv, "en")

    assert "Global average LTV" in _all_texts(figures[0])


def test_exploration_chart_skips_missing_columns(small, figures):
    """Only the panels whose column exists get a title."""
    plots.plot_exploration(small.users[["converted_flag", "variant"]], "en")

    titles = [a.get_title(loc="left") for a in figures[0].axes]
    assert titles == [
        "Conversion Distribution", "", "", "Distribution by A/B Variant",
    ]


def test_language_chart_translates_the_bar_labels(small, figures):
    """Bars are labelled with the language of the chart."""
    from marketing_campaign.diagnostics import analyze_language_performance

    summary = analyze_language_performance(small.users)
    plots.plot_language_performance(summary, "pt")

    labels = [t.get_text() for t in figures[0].axes[0].get_xticklabels()]
    assert labels == ["Idioma Correto", "Idioma Incorreto"]


def test_power_curve_chart_marks_the_current_sample(figures):
    """A reference line and its label show the experiment sample."""
    plots.plot_power_curve([100, 200, 300], [0.2, 0.5, 0.9], 250, "pt")

    assert "Amostra atual por grupo (250)" in _all_texts(figures[0])


def test_variant_and_roi_charts_draw_two_panels(small, figures):
    """Variant rates and ROI/profit both have two panels."""
    plots.plot_variant_rates(small.users, "en")
    plots.plot_roi(small.roi, "pt")

    assert [len(f.axes) for f in figures] == [2, 2]
    assert "ROI Incremental por Canal" in _all_texts(figures[1])


def test_sample_adequacy_chart_paints_short_variants_red(figures):
    """Variants below the requirement are drawn in red."""
    counts = pd.Series({"control": 100, "personalization": 5000})

    plots.plot_sample_adequacy(counts, 3916, "en")

    bars = figures[0].axes[0].patches
    assert bars[0].get_facecolor()[:3] != bars[1].get_facecolor()[:3]
    assert "Required (3,916)" in _all_texts(figures[0])


def test_false_positive_bias_chart_and_finding_agree(figures):
    """The chart compares base and false positives per category."""
    from marketing_campaign.diagnostics import compare_false_positives
    from marketing_campaign.narrative import bias_finding

    base = pd.DataFrame({"group": ["a"] * 60 + ["b"] * 40})
    errors = pd.DataFrame({"group": ["a"] * 2 + ["b"] * 8})
    comparison = compare_false_positives(base, errors, "group")

    plots.plot_false_positive_bias(comparison, "group", "pt")

    assert "Base total vs Falsos Positivos: group" in _all_texts(figures[0])
    assert bias_finding(comparison, "group", "en").startswith(
        "b represents 80.0% of false positives, but only 40.0%"
    )
    assert bias_finding(comparison.iloc[1:], "group", "pt").startswith(
        "Maior diferença em group"
    )
