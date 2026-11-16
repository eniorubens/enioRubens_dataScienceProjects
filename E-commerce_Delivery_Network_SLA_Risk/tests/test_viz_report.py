import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecom_delivery_sla_viz.report import (  # noqa: E402
    evidence_table,
    get_report_labels,
    optuna_study_table,
    plot_monthly_prevalence,
    plot_reliability_curves,
)

LANGUAGES = ["pt", "en"]


@pytest.fixture
def reliability():
    rows = []
    for model, shift in (("A_f1_frozen", 0.05), ("B_retrain", -0.02)):
        for index in range(1, 5):
            rows.append(
                {
                    "model": model,
                    "bin": index,
                    "n": 100 * index,
                    "mean_predicted": 0.1 * index,
                    "observed_rate": max(0.1 * index + shift, 0.0),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def monthly():
    return pd.DataFrame(
        {
            "purchase_month": ["2018-04", "2018-05", "2018-07", "2018-08"],
            "split": ["train", "train", "test", "test"],
            "orders": [6798, 6749, 6156, 6351],
            "breaches": [305, 445, 200, 300],
            "prevalence": [0.045, 0.066, 0.032, 0.047],
        },
        index=[10, 11, 12, 13],
    )


@pytest.mark.parametrize("language", LANGUAGES)
def test_reliability_curves_have_one_trace_per_model_plus_diagonal(reliability, language):
    before = reliability.copy(deep=True)
    figure = plot_reliability_curves(reliability, language=language)

    assert isinstance(figure, go.Figure)
    assert figure.to_json()
    assert len(figure.data) == 3
    assert [trace.name for trace in figure.data[:2]] == ["A_f1_frozen", "B_retrain"]
    assert figure.data[-1].name == get_report_labels(language)["reliability"]["ideal"]
    pd.testing.assert_frame_equal(reliability, before)


def test_reliability_curves_accept_custom_title_and_missing_counts(reliability):
    figure = plot_reliability_curves(reliability.drop(columns="n"), language="en", title="August")
    assert figure.layout.title.text == "August"
    assert plot_reliability_curves(reliability, "pt").layout.title.text != plot_reliability_curves(
        reliability, "en"
    ).layout.title.text


@pytest.mark.parametrize("language", LANGUAGES)
def test_monthly_prevalence_has_split_bars_and_a_line(monthly, language):
    before = monthly.copy(deep=True)
    figure = plot_monthly_prevalence(monthly, language=language)

    assert figure.to_json()
    assert [type(trace).__name__ for trace in figure.data] == ["Bar", "Bar", "Scatter"]
    assert list(figure.data[0].x) == ["2018-04", "2018-05"]
    assert list(figure.data[1].x) == ["2018-07", "2018-08"]
    assert figure.layout.title.text == get_report_labels(language)["prevalence"]["title"]
    pd.testing.assert_frame_equal(monthly, before)


def _summary(name, direction, n_trials, value):
    best = None if value is None else SimpleNamespace(value=value)
    return SimpleNamespace(
        study_name=name,
        n_trials=n_trials,
        direction=SimpleNamespace(name=direction),
        best_trial=best,
    )


def test_optuna_study_table_filters_version_and_ranks_by_direction():
    summaries = [
        _summary("classification_CatBoost_V4_folds", "MAXIMIZE", 100, 0.172),
        _summary("classification_LightGBM_V4_folds", "MAXIMIZE", 100, 0.174),
        _summary("regression_Ridge_V4_folds", "MINIMIZE", 100, 9.2),
        _summary("regression_CatBoost_V4_folds", "MINIMIZE", 100, 8.8),
        _summary("regression_CatBoost_V3", "MINIMIZE", 100, 7.2),
        _summary("classification_XGBoost_V4_folds", "MAXIMIZE", 0, None),
    ]
    table = optuna_study_table(summaries, version="V4_folds")

    classification = table[table["task"] == "classification"]
    assert classification.loc[classification["model"] == "LightGBM", "rank"].item() == 1
    assert classification.loc[classification["model"] == "CatBoost", "rank"].item() == 2
    regression = table[table["task"] == "regression"]
    assert list(regression["model"]) == ["CatBoost", "Ridge"]
    assert set(table["direction"]) == {"maximize", "minimize"}
    assert "V3" not in " ".join(table["model"])
    assert len(table) == 5
    assert np.isnan(table.loc[table["model"] == "XGBoost", "best_value"].item())
    assert set(table["metric"]) == {"validation_average_precision", "validation_rmse"}


def test_optuna_study_table_empty_when_no_version_matches():
    table = optuna_study_table([_summary("regression_Ridge_V3", "MINIMIZE", 5, 1.0)], version="V4_folds")
    assert table.empty
    assert "best_value" in table.columns


@pytest.fixture
def evidence_inputs():
    summary = pd.DataFrame(
        {
            "model": ["CatBoost", "baseline_logistic_promise_plus_state"],
            "n_folds": [9, 9],
            "average_precision_mean": [0.175, 0.154],
            "average_precision_gain_vs_baseline": [0.139, np.nan],
            "average_precision_folds_won": [8.0, np.nan],
            "average_precision_best_baseline": ["baseline_logistic_promise_plus_state", ""],
        }
    )
    f1_metrics = pd.DataFrame(
        {
            "model": ["CatBoost", "baseline_logistic_promise_plus_state", "CatBoost"],
            "metric": ["average_precision", "average_precision", "roc_auc"],
            "estimate": [0.082, 0.084, 0.61],
            "ci_low": [0.056, 0.056, 0.56],
            "ci_high": [0.134, 0.116, 0.66],
        }
    )
    f1_paired = pd.DataFrame(
        {
            "model": ["CatBoost"],
            "reference": ["baseline_logistic_promise_plus_state"],
            "difference": [-0.002],
            "ci_low": [-0.033],
            "ci_high": [0.064],
            "outcome": ["inconclusivo"],
            "outcome_en": ["inconclusive"],
        }
    )
    f2_metrics = pd.DataFrame(
        {
            "model": ["A_f1_frozen", "D_recalibration", "baseline_logistic_promise_plus_state"],
            "metric": ["average_precision"] * 3,
            "estimate": [0.069, np.nan, 0.094],
            "ci_low": [0.054, np.nan, 0.071],
            "ci_high": [0.084, np.nan, 0.121],
            "status": ["avaliada", "reprovada no gate", "avaliada"],
        }
    )
    f2_paired = pd.DataFrame(
        {
            "model": ["A_f1_frozen", "D_recalibration"],
            "reference": ["baseline_logistic_promise_plus_state"] * 2,
            "difference": [-0.025, np.nan],
            "ci_low": [-0.044, np.nan],
            "ci_high": [-0.011, np.nan],
            "reading": ["exploratório", "exploratório"],
            "status": ["avaliada", "reprovada no gate"],
        }
    )
    return summary, f1_metrics, f1_paired, f2_metrics, f2_paired


@pytest.mark.parametrize("language", LANGUAGES)
def test_evidence_table_collects_three_stages_without_changing_inputs(evidence_inputs, language):
    before = [frame.copy(deep=True) for frame in evidence_inputs]
    table = evidence_table(*evidence_inputs, language=language)

    labels = get_report_labels(language)["evidence"]
    assert list(table.columns) == ["stage", "item", "estimate", "ci_low", "ci_high", "note"]
    assert set(table["stage"]) == {labels["development"], labels["f1"], labels["f2"]}
    gain = table[table["item"] == labels["gain"]].iloc[0]
    assert gain["estimate"] == pytest.approx(0.139)
    assert gain["note"] == labels["folds_won"].format(won=8, total=9)
    f1_rows = table[table["stage"] == labels["f1"]]
    assert len(f1_rows) == 3  # two AP rows and one paired difference; ROC AUC is not listed
    reading = "inconclusive" if language == "en" else "inconclusivo"
    assert f1_rows.iloc[-1]["note"] == reading
    gated = table[table["item"].str.contains("D_recalibration")]
    assert set(gated["note"]) == {"reprovada no gate"}
    assert gated["estimate"].isna().all()
    for frame, original in zip(evidence_inputs, before, strict=True):
        pd.testing.assert_frame_equal(frame, original)


def test_report_labels_cover_both_languages_and_reject_others():
    assert set(get_report_labels("pt")) == set(get_report_labels("en"))
    for section in get_report_labels("pt"):
        assert set(get_report_labels("pt")[section]) == set(get_report_labels("en")[section])
    with pytest.raises(ValueError):
        get_report_labels("es")


def test_report_module_does_not_import_training_or_optuna_packages():
    code = (
        "import sys; sys.path.insert(0, 'src'); import ecom_delivery_sla_viz.report; "
        "print('ecom_delivery_sla' in sys.modules, 'optuna' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False False"
    source = (SRC / "ecom_delivery_sla_viz" / "report.py").read_text(encoding="utf-8")
    assert "from ecom_delivery_sla " not in source
    assert "import ecom_delivery_sla\n" not in source
