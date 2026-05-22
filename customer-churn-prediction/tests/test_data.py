"""Tests for churn_project/data.py."""
import numpy as np
import pandas as pd
import pytest

from churn_project.data import compute_class_ratio, split_telecom_dataset


class TestReadTelecomData:
    def test_returns_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)

    def test_has_churn_column(self, df):
        assert "Churn" in df.columns

    def test_row_count(self, df):
        assert len(df) > 100


class TestSplitTelecomDataset:
    def test_returns_required_keys(self, splits):
        expected = {
            "data", "X", "y",
            "train_data", "test_data",
            "train_features", "test_features",
            "train_labels", "test_labels",
        }
        assert expected.issubset(splits.keys())

    def test_test_size_fraction(self, splits):
        total     = len(splits["train_features"]) + len(splits["test_features"])
        test_frac = len(splits["test_features"]) / total
        assert abs(test_frac - 0.2) < 0.01

    def test_labels_are_binary(self, splits):
        assert set(np.unique(splits["train_labels"])).issubset({0, 1})
        assert set(np.unique(splits["test_labels"])).issubset({0, 1})

    def test_no_customer_id_in_features(self, splits):
        assert "customerID" not in splits["train_features"].columns

    def test_no_churn_in_features(self, splits):
        assert "Churn" not in splits["train_features"].columns


class TestComputeClassRatio:
    def test_ratio_value(self):
        labels = np.array([0, 0, 0, 1])
        assert compute_class_ratio(labels) == pytest.approx(3.0)

    def test_equal_classes(self):
        labels = np.array([0, 1, 0, 1])
        assert compute_class_ratio(labels) == pytest.approx(1.0)

    def test_raises_on_no_positive(self):
        with pytest.raises(ValueError, match="Positive class count is zero"):
            compute_class_ratio(np.zeros(5))
