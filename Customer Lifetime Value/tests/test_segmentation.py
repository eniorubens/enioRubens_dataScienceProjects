"""
test_segmentation.py — Tests for CustomerSegmenter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.segmentation import (
    ACTION_ORDER,
    SEGMENT_ORDER,
    CustomerSegmenter,
)


@pytest.fixture
def cltv_df() -> pd.DataFrame:
    """Synthetic CLTV DataFrame with 40 customers."""
    rng = np.random.default_rng(99)
    n = 40
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(n)],
            "cltv_180d": rng.uniform(10, 5000, size=n),
            "probability_alive": rng.uniform(0.3, 1.0, size=n),
        }
    )


def test_segment_adds_required_columns(cltv_df):
    """Four new columns must be present after segmentation."""
    segmenter = CustomerSegmenter()
    result, _ = segmenter.segment(cltv_df, cltv_col="cltv_180d")
    for col in ("cltv_segment", "probability_alive_group", "value_group", "marketing_action"):
        assert col in result.columns, f"Missing column: {col}"


def test_segment_exactly_four_segments(cltv_df):
    """Exactly the four SEGMENT_ORDER labels must appear."""
    segmenter = CustomerSegmenter()
    result, _ = segmenter.segment(cltv_df, cltv_col="cltv_180d")
    found = set(result["cltv_segment"].astype(str).unique())
    assert found == set(SEGMENT_ORDER)


def test_segment_exactly_four_actions(cltv_df):
    """Exactly the four ACTION_ORDER labels must appear."""
    segmenter = CustomerSegmenter()
    result, _ = segmenter.segment(cltv_df, cltv_col="cltv_180d")
    found = set(result["marketing_action"].unique())
    assert found == set(ACTION_ORDER)


def test_segment_no_nulls_in_marketing_action(cltv_df):
    """marketing_action must not contain NaN or 'Review' (fallback)."""
    segmenter = CustomerSegmenter()
    result, _ = segmenter.segment(cltv_df, cltv_col="cltv_180d")
    assert result["marketing_action"].isna().sum() == 0
    assert (result["marketing_action"] != "Review").all()


def test_decision_matrix_shape():
    """build_decision_matrix() must return a 2×2 DataFrame."""
    segmenter = CustomerSegmenter()
    matrix = segmenter.build_decision_matrix()
    assert matrix.shape == (2, 2)


def test_segment_summary_indexed_by_action_order(cltv_df):
    """segment_summary() must be indexed exactly by ACTION_ORDER."""
    segmenter = CustomerSegmenter()
    result, _ = segmenter.segment(cltv_df, cltv_col="cltv_180d")
    summary = segmenter.segment_summary(result)
    assert list(summary.index) == ACTION_ORDER


def test_custom_alive_threshold(cltv_df):
    """A fixed alive_threshold must be respected and returned."""
    threshold = 0.75
    segmenter = CustomerSegmenter(alive_threshold=threshold)
    _, used_threshold = segmenter.segment(cltv_df, cltv_col="cltv_180d")
    assert abs(used_threshold - threshold) < 1e-9
    assert abs(segmenter._fitted_threshold - threshold) < 1e-9
