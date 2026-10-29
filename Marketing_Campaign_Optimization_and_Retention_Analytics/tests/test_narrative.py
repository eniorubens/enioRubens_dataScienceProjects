"""Tests for the narrative helpers."""
import numpy as np
import pandas as pd
import pytest

from marketing_campaign import i18n, narrative
from marketing_campaign.retention_model import make_metrics_table

METRICS = {
    "accuracy": 0.9,
    "precision": 0.99,
    "recall": 0.9,
    "f1": 0.94,
    "roc_auc": 0.956,
    "precision_not_retained": 0.2,
    "recall_not_retained": 0.96,
    "f1_not_retained": 0.53,
    "false_positives": 2,
    "false_negatives": 100,
}


@pytest.mark.parametrize(
    ("value", "expected"),
    [("3.5", 3.5), (2, 2.0), (None, 0.0), ("abc", 0.0), (np.nan, 0.0)],
)
def test_safe_float_falls_back_to_the_default(value, expected):
    """Numbers convert; anything else becomes the default."""
    assert narrative.safe_float(value) == expected


def test_safe_float_uses_the_given_default():
    """The fallback is configurable."""
    assert narrative.safe_float("x", default=-1) == -1.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, 1),
        (False, 0),
        (1, 1),
        (0, 0),
        (2, np.nan),
        (1.0, 1),
        (0.5, np.nan),
        ("Yes", 1),
        (" sim ", 1),
        ("No", 0),
        ("nao", 0),
        ("maybe", np.nan),
        (None, np.nan),
    ],
)
def test_normalize_language_flag_reads_many_spellings(value, expected):
    """Booleans, numbers and English or Portuguese words are understood."""
    result = narrative.normalize_language_flag(value)

    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


def test_normalize_text_removes_accents_and_case():
    """Comparison keys are ASCII and lower case."""
    assert narrative.normalize_text("  Métrica Não ") == "metrica nao"


@pytest.mark.parametrize("lang", ["en", "pt"])
def test_metric_helpers_work_on_both_languages(lang):
    """The same lookups succeed on the English and Portuguese tables."""
    table = i18n.localize(make_metrics_table(METRICS), lang)

    assert narrative.metric_column_name(table) in {"Metric", "Métrica"}
    assert narrative.get_metric_value(table, "ROC AUC") == 0.956
    assert narrative.get_metric_value(table, "recall") == 0.9
    assert narrative.get_metric_value(table, "Recall - Not Retained") == 0.96
    assert narrative.get_metric_value(table, "Missing", default=-1) == -1


@pytest.mark.parametrize("lang", ["en", "pt"])
def test_recall_without_exact_match_uses_the_not_retained_class(lang):
    """A vague "recall" name falls back to the not-retained recall."""
    table = i18n.localize(make_metrics_table(METRICS), lang)
    table = table[~table.iloc[:, 0].isin(["Recall"])]

    assert narrative.get_metric_value(table, "recall (class 0)") == 0.96


def test_metric_column_falls_back_to_the_first_column():
    """A table without a metric header uses its first column."""
    table = pd.DataFrame({"name": ["a"], "number": [1]})

    assert narrative.metric_column_name(table) == "name"
