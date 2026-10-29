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


def test_both_languages_define_the_same_keys():
    """Every label exists in English and in Portuguese."""
    assert set(plots.TEXT["en"]) == set(plots.TEXT["pt"])


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
