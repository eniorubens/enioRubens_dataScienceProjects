"""Tests for the RFM-proxy segmentation."""
import pandas as pd
import pytest
from helpers import make_exposures

from marketing_campaign import data, segmentation


def _exposures() -> pd.DataFrame:
    """Return exposures of three well separated groups of users."""
    rows = []
    for index in range(10):  # high: many exposures, all converted, recent
        for day in (28, 29, 30):
            rows.append({"user_id": f"high{index}",
                         "date_served": f"2018-01-{day}",
                         "converted": True})
    for index in range(10):  # medium: one converted exposure, a bit older
        rows.append({"user_id": f"mid{index}", "date_served": "2018-01-20",
                     "converted": True})
    for index in range(10):  # low: one exposure, never converted, old
        rows.append({"user_id": f"low{index}", "date_served": "2018-01-01"})
    return data.clean_data(make_exposures(rows))


def test_build_rfm_computes_recency_frequency_and_monetary():
    """Recency is days since the last exposure; monetary is revenue."""
    rfm = segmentation.build_rfm(_exposures()).set_index("user_id")

    assert rfm.loc["high0", ["Recency", "Frequency", "Monetary"]].tolist() == [
        0,
        3,
        135.0,
    ]
    assert rfm.loc["mid0", "Recency"] == 10
    assert rfm.loc["low0", "Monetary"] == 0
    assert rfm.loc["low0", "Recency"] == 29


def test_segments_are_named_by_monetary_value():
    """The highest-spending cluster is High Value, the lowest Low Value."""
    segmented = segmentation.assign_segments(
        segmentation.build_rfm(_exposures())
    ).set_index("user_id")

    assert (segmented.loc[[f"high{i}" for i in range(10)], "Segment"]
            == "High Value").all()
    assert (segmented.loc[[f"mid{i}" for i in range(10)], "Segment"]
            == "Medium Value").all()
    assert (segmented.loc[[f"low{i}" for i in range(10)], "Segment"]
            == "Low Value").all()


def test_segmentation_is_deterministic():
    """The same seed gives the same clusters."""
    rfm = segmentation.build_rfm(_exposures())

    first = segmentation.assign_segments(rfm)["Segment"]
    second = segmentation.assign_segments(rfm)["Segment"]

    assert first.equals(second)


def test_more_clusters_than_names_is_rejected():
    """Only three segment names exist."""
    rfm = segmentation.build_rfm(_exposures())

    with pytest.raises(ValueError):
        segmentation.assign_segments(rfm, n_clusters=4)


def test_attach_segments_adds_columns_without_changing_rows():
    """The join keeps one row per user and replaces old segment columns."""
    users = pd.DataFrame({"user_id": ["a", "b"], "Segment": ["old", "old"]})
    segmented = pd.DataFrame(
        {"user_id": ["a", "b"], "RFM_Cluster": [0, 1],
         "Segment": ["High Value", "Low Value"]}
    )

    attached = segmentation.attach_segments(users, segmented)

    assert attached["Segment"].tolist() == ["High Value", "Low Value"]
    assert attached["RFM_Cluster"].tolist() == [0, 1]


def test_real_segments_match_the_report(result):
    """688 High, 3,889 Medium and 2,272 Low Value users."""
    counts = result.users["Segment"].value_counts()

    assert counts["High Value"] == 688
    assert counts["Medium Value"] == 3889
    assert counts["Low Value"] == 2272
