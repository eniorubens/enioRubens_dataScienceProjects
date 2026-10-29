"""Tests for the funnel and the channel performance."""
import pandas as pd
import pytest

from marketing_campaign import funnel


def _users() -> pd.DataFrame:
    """Return a tiny user-level base with two channels."""
    return pd.DataFrame(
        {
            "user_id": ["a", "b", "c", "d"],
            "marketing_channel": ["Email", "Email", "Push", "Push"],
            "converted_flag": [1, 1, 1, 0],
            "retained_flag": [1, 0, 1, 1],
            "date_canceled": [None, "2018-02-01", None, None],
        }
    )


def test_funnel_counts_conversions_and_retention_among_converted():
    """Retention only counts users that converted."""
    summary = funnel.build_funnel(_users()).set_index("Metric")["Value"]

    assert summary["Impacted Users"] == 4
    assert summary["Conversions"] == 3
    assert summary["Retained (among converted)"] == 2
    assert summary["Conversion Rate (%)"] == 75.0
    assert summary["Retention Rate (%)"] == pytest.approx(66.67)


def test_channel_performance_rates_and_ordering():
    """Channels are ranked by conversion and retention is per converter."""
    channels = funnel.analyze_channel_performance(_users())

    assert channels["marketing_channel"].tolist() == ["Email", "Push"]
    email = channels.set_index("marketing_channel").loc["Email"]
    assert email["conversion_rate_%"] == 100.0
    assert email["retention_rate_%"] == 50.0
    assert email["cancellations"] == 1


def test_retention_is_zero_for_a_channel_without_conversions():
    """A channel with no conversions has 0% retention, not NaN."""
    users = _users()
    users["converted_flag"] = 0

    channels = funnel.analyze_channel_performance(users)

    assert (channels["retention_rate_%"] == 0).all()


def test_real_funnel_matches_the_report(result):
    """703 conversions (10.26%) and 463 retained (65.86%)."""
    summary = result.funnel.set_index("Metric")["Value"]

    assert summary["Impacted Users"] == 6849
    assert summary["Conversions"] == 703
    assert summary["Retained (among converted)"] == 463
    assert summary["Conversion Rate (%)"] == 10.26
    assert summary["Retention Rate (%)"] == 65.86


def test_real_channel_table_matches_the_report(result):
    """Email converts 37.9% and House Ads has the most impacted users."""
    channels = result.channels.set_index("marketing_channel")

    assert channels.loc["Email", "impacted_users"] == 343
    assert channels.loc["Email", "conversions"] == 130
    assert channels.loc["Email", "conversion_rate_%"] == 37.90
    assert channels.loc["Email", "retention_rate_%"] == 76.15
    assert channels.loc["House Ads", "impacted_users"] == 3225
    assert channels.loc["House Ads", "conversions"] == 374
    assert channels["conversion_rate_%"].is_monotonic_decreasing
