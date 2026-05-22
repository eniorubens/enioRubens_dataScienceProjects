"""Tests for the winning sklearn Pipeline behaviour."""
import numpy as np
import pytest
from sklearn.metrics import recall_score


class TestPipeline:
    def test_predict_proba_shape(self, trained_pipeline, splits):
        proba = trained_pipeline.predict_proba(splits["test_features"])
        assert proba.shape == (len(splits["test_features"]), 2)

    def test_proba_in_unit_interval(self, trained_pipeline, splits):
        proba = trained_pipeline.predict_proba(splits["test_features"])[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_returns_binary(self, trained_pipeline, splits):
        preds = trained_pipeline.predict(splits["test_features"])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_output_length_matches_input(self, trained_pipeline, splits):
        X     = splits["test_features"]
        preds = trained_pipeline.predict(X)
        assert len(preds) == len(X)

    def test_lower_threshold_more_positives(self, trained_pipeline, splits):
        proba    = trained_pipeline.predict_proba(splits["test_features"])[:, 1]
        preds_52 = (proba >= 0.52).astype(int)
        preds_30 = (proba >= 0.30).astype(int)
        assert preds_30.sum() >= preds_52.sum()

    def test_lower_threshold_increases_churn_recall(self, trained_pipeline, splits):
        # Class-1 (churn) recall is guaranteed non-decreasing as threshold drops
        y_test    = splits["test_labels"]
        proba     = trained_pipeline.predict_proba(splits["test_features"])[:, 1]
        recall_52 = recall_score(y_test, (proba >= 0.52).astype(int))
        recall_30 = recall_score(y_test, (proba >= 0.30).astype(int))
        assert recall_30 >= recall_52

    def test_recall_above_chance(self, trained_pipeline, splits):
        proba  = trained_pipeline.predict_proba(splits["test_features"])[:, 1]
        preds  = (proba >= 0.52).astype(int)
        recall = recall_score(splits["test_labels"], preds, average="macro")
        assert recall > 0.5
