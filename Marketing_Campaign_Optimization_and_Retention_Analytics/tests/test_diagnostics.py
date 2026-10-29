"""Tests for the language and error diagnostics."""
import numpy as np
import pandas as pd
import pytest

from marketing_campaign import diagnostics


def _users() -> pd.DataFrame:
    """Return six users: four with the right language and two without."""
    return pd.DataFrame(
        {
            "user_id": list("abcdef"),
            "converted_flag": [1, 1, 0, 0, 1, 0],
            "retained_flag": [1, 0, 0, 0, 1, 0],
            "is_correct_lang": ["Yes", "Yes", "Yes", "Yes", "No", "No"],
        }
    )


def test_language_performance_rates_use_the_documented_denominators():
    """Conversion over all users, retention over converted users only."""
    summary = diagnostics.analyze_language_performance(_users())
    by_label = summary.set_index("label")

    right = by_label.loc["Correct Language"]
    wrong = by_label.loc["Wrong Language"]
    assert right["user_id"] == 4
    assert right["conversion_rate_%"] == 50.0
    assert right["retention_rate_%"] == 50.0
    assert wrong["user_id"] == 2
    assert wrong["conversion_rate_%"] == 50.0
    assert wrong["retention_rate_%"] == 100.0


def test_language_performance_needs_every_required_column():
    """A missing column yields None instead of a partial table."""
    assert diagnostics.analyze_language_performance(
        _users().drop(columns="retained_flag")
    ) is None


def test_language_performance_reports_unknown_flags():
    """Values outside the known spellings are labelled "Not informed"."""
    users = _users()
    users.loc[0, "is_correct_lang"] = "maybe"

    labels = diagnostics.analyze_language_performance(users)["label"]

    assert "Not informed" in labels.tolist()


def test_real_language_split_matches_the_dataset(result):
    """The real base has 6,524 correct and 325 wrong-language users."""
    summary = diagnostics.analyze_language_performance(result.users)
    by_label = summary.set_index("label")

    assert by_label.loc["Correct Language", "user_id"] == 6524
    assert by_label.loc["Wrong Language", "user_id"] == 325
    assert (summary["retention_rate_%"] <= 100).all()


def _predictions():
    """Return four test rows: one of each outcome."""
    features = pd.DataFrame(
        {"marketing_channel": ["Email", "Email", "Push", "Push"]},
        index=[10, 11, 12, 13],
    )
    y_true = pd.Series([1, 0, 0, 1], index=features.index)
    y_pred = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.1, 0.2])
    return features, y_true, y_pred, proba


def test_classify_errors_splits_false_positives_and_negatives():
    """FP: true 0 predicted 1; FN: true 1 predicted 0."""
    errors, false_positives, false_negatives = diagnostics.classify_errors(
        *_predictions()
    )

    assert len(errors) == 4
    assert false_positives["original_index"].tolist() == [11]
    assert false_negatives["original_index"].tolist() == [13]
    assert errors["prob_retention"].tolist() == [0.9, 0.8, 0.1, 0.2]


def test_error_distribution_counts_categories_and_missing_values():
    """Shares are over the errors; missing values become "Missing"."""
    errors = pd.DataFrame({"channel": ["Email", "Email", None, "Push"]})

    table = diagnostics.error_distribution(errors, "channel")

    assert dict(zip(table["channel"], table["quantity"])) == {
        "Email": 2,
        "Missing": 1,
        "Push": 1,
    }
    assert table["percentage"].sum() == pytest.approx(100.0)
    assert diagnostics.error_distribution(errors, "absent") is None


def _populations():
    """Return a base of 100 users and 10 false positives."""
    base = pd.DataFrame({"group": ["a"] * 60 + ["b"] * 40})
    false_positives = pd.DataFrame({"group": ["a"] * 2 + ["b"] * 8})
    return base, false_positives


def test_false_positive_bias_flags_over_and_under_representation():
    """b is 40% of the base but 80% of the errors: +40 p.p."""
    base, false_positives = _populations()

    table = diagnostics.compare_false_positives(base, false_positives, "group")
    by_category = table.set_index("category")

    assert table["category"].tolist() == ["b", "a"]  # largest diff first
    assert by_category.loc["b", "diff_pp"] == 40.0
    assert by_category.loc["b", "interpretation"] == (
        diagnostics.OVER_REPRESENTED
    )
    assert by_category.loc["a", "diff_pp"] == -40.0
    assert by_category.loc["a", "interpretation"] == (
        diagnostics.UNDER_REPRESENTED
    )


def test_false_positive_bias_threshold_is_inclusive():
    """A difference equal to the threshold is already flagged."""
    base, false_positives = _populations()

    strict = diagnostics.compare_false_positives(
        base, false_positives, "group", threshold_pp=40.0
    )
    loose = diagnostics.compare_false_positives(
        base, false_positives, "group", threshold_pp=40.01
    )

    assert strict["interpretation"].tolist() == [
        diagnostics.OVER_REPRESENTED,
        diagnostics.UNDER_REPRESENTED,
    ]
    assert set(loose["interpretation"]) == {diagnostics.PROPORTIONAL}


def test_false_positive_bias_handles_empty_inputs():
    """Two empty frames give an empty table with the output columns."""
    empty = pd.DataFrame({"group": []})

    table = diagnostics.compare_false_positives(empty, empty, "group")

    assert table.empty
    assert table.columns.tolist() == diagnostics.BIAS_OUTPUT_COLUMNS


def test_analyze_false_positive_bias_ignores_unknown_columns():
    """Only variables present in both frames are compared."""
    base, false_positives = _populations()

    results = diagnostics.analyze_false_positive_bias(
        base, false_positives, ["group", "absent"]
    )

    assert list(results) == ["group"]
    assert diagnostics.analyze_false_positive_bias(
        base, false_positives, []
    ) == {}
