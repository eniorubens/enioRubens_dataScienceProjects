"""Tests for customer value by segment and the simulated ROI."""
import pandas as pd
import pytest

from marketing_campaign import economics, funnel


def _users() -> pd.DataFrame:
    """Return a segmented user base with known revenue and cost."""
    return pd.DataFrame(
        {
            "user_id": ["a", "b", "c", "d"],
            "marketing_channel": ["Email", "Email", "Push", "Mystery"],
            "converted_flag": [1, 0, 1, 0],
            "retained_flag": [1, 0, 1, 0],
            "Segment": ["High Value", "Low Value", "High Value",
                        "Low Value"],
        }
    )


def test_ltv_is_revenue_minus_channel_cost():
    """Each user's value is 45 per conversion minus the channel cost."""
    ltv = economics.estimate_ltv(
        _users(), cost_per_impression={"Email": 0.8, "Push": 1.0}
    )

    values = ltv.by_customer.set_index("user_id")["historical_ltv"]
    assert values["a"] == pytest.approx(45 - 0.8)
    assert values["b"] == pytest.approx(-0.8)
    assert values["c"] == pytest.approx(45 - 1.0)
    assert values["d"] == pytest.approx(-1.2)  # unknown channel: default


def test_segment_value_map_and_global_mean():
    """The segment map holds the mean value; the global one is the mean."""
    ltv = economics.estimate_ltv(
        _users(), cost_per_impression={"Email": 0.8, "Push": 1.0}
    )

    assert ltv.segment_ltv_map["High Value"] == pytest.approx(44.1)
    assert ltv.segment_ltv_map["Low Value"] == pytest.approx(-1.0)
    assert ltv.global_ltv == pytest.approx(
        ltv.by_customer["historical_ltv"].mean()
    )
    assert ltv.by_segment["ltv_mean"].is_monotonic_decreasing


def test_roi_formulas_for_one_channel():
    """Revenue, profit and incremental ROI follow the documented rules."""
    users = _users()
    channels = funnel.analyze_channel_performance(users)
    ltv = economics.LtvSummary(
        by_customer=pd.DataFrame(),
        by_segment=pd.DataFrame(),
        global_ltv=10.0,
        segment_ltv_map={"High Value": 40.0, "Low Value": -1.0},
    )

    roi = economics.simulate_roi(
        channels, users, ltv, cost_per_impression={"Email": 1.0}
    ).set_index("marketing_channel")

    email = roi.loc["Email"]
    assert email["total_cost"] == 2.0  # 2 users x cost 1.0
    # one conversion and one retained user, majority segment per mode
    assert email["estimated_revenue"] == pytest.approx(
        1 * 45 + 1 * email["ltv_reference"] * 0.15
    )
    assert email["estimated_profit"] == pytest.approx(
        email["estimated_revenue"] - 2.0
    )
    organic = 2 * users["converted_flag"].mean() * 0.20 * 45
    assert email["incremental_revenue"] == pytest.approx(
        email["estimated_revenue"] - organic
    )


def test_roi_of_a_zero_cost_channel_is_zero_not_infinite():
    """A free channel reports 0% ROI instead of dividing by zero."""
    users = _users()
    channels = funnel.analyze_channel_performance(users)
    ltv = economics.estimate_ltv(users)

    roi = economics.simulate_roi(
        channels, users, ltv, cost_per_impression={"Email": 0.0}
    ).set_index("marketing_channel")

    assert roi.loc["Email", "total_cost"] == 0
    assert roi.loc["Email", "roi_%"] == 0
    assert roi.loc["Email", "roi_incremental_%"] == 0


def test_real_segment_values_match_the_report(result):
    """High 43.98, Medium -0.92 and Low -1.09 per customer."""
    by_segment = result.ltv.by_segment.set_index("segment")

    assert by_segment.loc["High Value", "ltv_mean"] == pytest.approx(
        43.98, abs=0.005
    )
    assert by_segment.loc["Medium Value", "ltv_mean"] == pytest.approx(
        -0.92, abs=0.005
    )
    assert by_segment.loc["Low Value", "ltv_mean"] == pytest.approx(
        -1.09, abs=0.005
    )


def test_real_roi_matches_the_report(result):
    """Incremental ROI by channel and the estimated profit of Email."""
    roi = result.roi.set_index("marketing_channel")

    assert roi.loc["Email", "roi_incremental_%"] == 1911.45
    assert roi.loc["House Ads", "roi_incremental_%"] == 328.50
    assert roi.loc["Facebook", "roi_incremental_%"] == 128.32
    assert roi.loc["Push", "roi_incremental_%"] == -21.46
    assert roi.loc["Instagram", "roi_incremental_%"] == -31.27
    assert roi.loc["Email", "estimated_profit"] == pytest.approx(
        5561.87, abs=0.005
    )
