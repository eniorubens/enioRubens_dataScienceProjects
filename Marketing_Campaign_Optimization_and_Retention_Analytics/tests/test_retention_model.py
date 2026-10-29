"""Tests for the retention model and its metrics."""
import numpy as np
import pandas as pd
import pytest

from marketing_campaign import retention_model as rm


def test_to_binary_flag_understands_text_and_numbers():
    """Yes/No, true/false, 1/0 and sim/nao map to 1 and 0."""
    series = pd.Series(["Yes", "no", True, False, 1, 0, "sim", "nao", "?"])

    assert rm.to_binary_flag(series).tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 0]


def test_prepare_frame_derives_missing_flags_and_month():
    """Flags and the month are created from the raw columns."""
    users = pd.DataFrame(
        {
            "converted": [True, False],
            "is_retained": [False, True],
            "is_correct_lang": ["Yes", "No"],
            "date_served": ["2018-03-05", "2018-04-07"],
        }
    )

    frame = rm.prepare_retention_model_frame(users)

    assert frame["converted_flag"].tolist() == [1, 0]
    assert frame["retained_flag"].tolist() == [0, 1]
    assert frame["is_correct_lang_flag"].tolist() == [1, 0]
    assert frame["month"].tolist() == [3, 4]


def test_feature_sets_depend_on_the_conversion_flag():
    """``converted_flag`` is only offered when requested."""
    frame = pd.DataFrame(columns=["marketing_channel", "month",
                                  "converted_flag", "is_correct_lang_flag"])

    with_conv = rm.get_feature_sets(frame, include_converted=True)
    without = rm.get_feature_sets(frame, include_converted=False)

    assert with_conv == (["marketing_channel"],
                         ["is_correct_lang_flag", "month", "converted_flag"])
    assert "converted_flag" not in without[1]


def test_metrics_for_the_not_retained_class():
    """Not-retained precision, recall and the error counts are right."""
    truth = pd.Series([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 0])
    proba = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.4])

    metrics = rm.calculate_retention_metrics(truth, pred, proba)

    assert metrics["recall_not_retained"] == pytest.approx(0.6667, abs=1e-4)
    assert metrics["precision_not_retained"] == pytest.approx(0.6667, abs=1e-4)
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["roc_auc"] == pytest.approx(0.8889, abs=1e-4)


@pytest.mark.filterwarnings("ignore:Only one class")
def test_roc_auc_is_nan_when_only_one_class_is_present():
    """A single-class test split does not crash the metrics."""
    metrics = rm.calculate_retention_metrics(
        pd.Series([1, 1]), np.array([1, 1]), np.array([0.9, 0.8])
    )

    assert np.isnan(metrics["roc_auc"])


def test_build_model_requires_a_target_and_both_classes():
    """Missing targets or a single class raise a clear error."""
    with pytest.raises(ValueError, match="retained_flag"):
        rm.build_retention_model(pd.DataFrame({"month": [1, 2]}))
    one_class = pd.DataFrame(
        {"retained_flag": [1] * 10, "month": range(10),
         "marketing_channel": ["Email"] * 10}
    )
    with pytest.raises(ValueError, match="both classes"):
        rm.build_retention_model(one_class)


def test_real_model_matches_the_report(result):
    """ROC AUC 0.956, not-retained recall 0.959, F1 0.525, 2 FP."""
    metrics = result.model.metrics

    assert metrics["roc_auc"] == pytest.approx(0.956, abs=0.001)
    assert metrics["recall_not_retained"] == pytest.approx(0.959, abs=0.001)
    assert metrics["f1_not_retained"] == pytest.approx(0.525, abs=0.001)
    assert metrics["false_positives"] == 2
    assert int((result.users["retained_flag"] == 0).sum()) == 246


def test_real_experiments_rank_the_model_with_conversion_first(result):
    """Including ``converted_flag`` is the selected scenario."""
    experiments = result.experiments

    assert len(experiments) == 2
    assert bool(experiments.iloc[0]["include_converted"]) is True
    assert experiments.iloc[0]["roc_auc"] > experiments.iloc[1]["roc_auc"]
    assert experiments.iloc[1]["roc_auc"] == pytest.approx(0.707, abs=0.005)
