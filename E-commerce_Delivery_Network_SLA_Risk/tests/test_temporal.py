import pandas as pd
import pytest

from ecom_delivery_sla.temporal import (
    expanding_fold_summary,
    expanding_window_folds,
    plot_temporal_split,
    right_censoring_sensitivity,
    temporal_monthly_summary,
    temporal_split,
    temporal_split_summary,
)


def test_expanding_folds_respect_embargo_and_holdout_boundary():
    data = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "order_purchase_timestamp": [
                "2017-08-01", "2017-09-15", "2017-10-15", "2018-07-02"
            ],
        }
    )

    folds = expanding_window_folds(
        data,
        first_val_month="2017-10",
        last_val_month="2017-10",
        embargo_days=30,
    )

    assert len(folds) == 1
    fold = folds[0]
    assert set(fold) == {"fold", "train", "validation"}
    assert fold["validation"]["order_id"].tolist() == ["c"]
    assert fold["train"]["order_id"].tolist() == ["a"]
    assert not (pd.to_datetime(fold["train"]["order_purchase_timestamp"]) >= pd.Timestamp("2017-09-15")).any()
    assert not (pd.to_datetime(fold["validation"]["order_purchase_timestamp"]) >= pd.Timestamp("2018-07-01")).any()


def test_right_censoring_sensitivity_reports_scenario_counts_and_prevalence():
    orders = pd.DataFrame(
        {
            "order_status": ["shipped", "delivered", "delivered", "shipped", "shipped", "delivered"],
            "order_purchase_timestamp": [
                "2018-07-02", "2018-07-03", "2018-07-05", "2018-07-20", "2018-09-02", "2018-06-10",
            ],
            "order_delivered_customer_date": [
                None, "2018-07-10", "2018-07-25", None, None, "2018-06-20",
            ],
            "order_estimated_delivery_date": [
                "2018-07-20", "2018-07-15", "2018-07-15", "2018-08-01", "2018-09-30", "2018-06-30",
            ],
        }
    )

    result = right_censoring_sensitivity(orders).set_index("scenario")

    # In the period: 2 delivered (1 late), 2 censored shipped; the 09-02 row
    # and the June delivery are outside [2018-07-01, 2018-09-01).
    exclude = result.loc["exclude"]
    assert (exclude["supervised_orders"], exclude["breaches"], exclude["censored_orders"]) == (2, 1, 2)
    assert exclude["prevalence"] == 0.5
    include = result.loc["include_as_breach"]
    assert (include["supervised_orders"], include["breaches"], include["censored_orders"]) == (4, 3, 2)
    assert include["prevalence"] == 0.75
    assert set(result.columns) == {"supervised_orders", "breaches", "censored_orders", "prevalence"}


def test_temporal_split_is_ordered_and_exhaustive():
    data = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "order_purchase_timestamp": [
                "2018-01-01",
                "2018-02-01",
                "2018-03-01",
                "2018-04-01",
            ],
        }
    )

    splits = temporal_split(data, "order_purchase_timestamp", "2018-03-01", "2018-04-01")
    summary = temporal_split_summary(splits)

    assert list(summary["split"]) == ["train", "validation", "test"]
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [2, 1, 1]
    assert sum(len(frame) for frame in splits.values()) == len(data)


def test_temporal_monthly_summary_has_cumulative_share():
    data = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2018-01-01",
                "2018-01-15",
                "2018-02-01",
            ]
        }
    )

    summary = temporal_monthly_summary(data)

    assert list(summary["orders"]) == [2, 1]
    assert summary["cumulative_share"].iloc[-1] == 1


def test_temporal_split_rejects_invalid_timestamp_and_cutoffs():
    data = pd.DataFrame({"order_purchase_timestamp": ["not-a-date"]})

    with pytest.raises(ValueError, match="invalid or missing"):
        temporal_split(data, "order_purchase_timestamp", "2018-01-01", "2018-02-01")

    valid = pd.DataFrame({"order_purchase_timestamp": ["2018-01-01", "2018-02-01"]})
    with pytest.raises(ValueError, match="earlier"):
        temporal_split(valid, "order_purchase_timestamp", "2018-02-01", "2018-01-01")


def test_plot_temporal_split_contains_one_shape_per_partition():
    data = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2018-01-01",
                "2018-02-01",
                "2018-03-01",
                "2018-04-01",
            ]
        }
    )
    splits = temporal_split(data, "order_purchase_timestamp", "2018-03-01", "2018-04-01")

    figure = plot_temporal_split(splits)

    assert len(figure.data) == 3
    assert figure.layout.title.text == "Divisão cronológica para modelagem"


def test_expanding_fold_summary_reports_periods_sizes_and_prevalence():
    data = pd.DataFrame(
        {
            "order_id": list("abcdef"),
            "order_purchase_timestamp": [
                "2017-07-01", "2017-08-01", "2017-09-01", "2017-10-05", "2017-10-20", "2017-11-05",
            ],
            "sla_breach": [True, False, False, True, False, True],
        }
    )
    folds = expanding_window_folds(
        data, first_val_month="2017-10", last_val_month="2017-11", embargo_days=30
    )

    summary = expanding_fold_summary(folds)

    assert list(summary["fold"]) == ["2017-10", "2017-11"]
    first = summary.iloc[0]
    assert (first["train_orders"], first["validation_orders"]) == (2, 2)
    assert first["validation_prevalence"] == 0.5
    assert first["train_end"] < first["validation_start"]
