"""Tests for the experiment tables and messages."""
import math

import pandas as pd
import pytest

from marketing_campaign import data, experiment, reports


def test_design_table_describes_the_experiment_in_each_language():
    """Eleven rows; rates, sample and runtime come from the arguments."""
    english = reports.design_table(0.1026, 3916, 14, "en")
    portuguese = reports.design_table(0.1026, 3916, 14, "pt")

    assert english.columns.tolist() == ["Parameter", "Value"]
    assert portuguese.columns.tolist() == ["Parametro", "Valor"]
    assert len(english) == len(portuguese) == 11
    values = english.set_index("Parameter")["Value"]
    assert values["Baseline conversion rate"] == "10.26%"
    assert values["Expected treatment rate"] == "12.26%"
    assert values["Sample size per group"] == "3,916 users"
    assert values["Estimated runtime"].startswith("14 days")
    assert portuguese.iloc[8, 1] == "3,916 usuarios"


@pytest.mark.parametrize(
    ("available", "key"),
    [(8000, "Sufficient"), (7000, "Insufficient")],
)
def test_adequacy_message_compares_with_both_groups(available, key):
    """The base must cover the required users of both groups."""
    message = reports.adequacy_message(available, 3916, "en")

    assert message.startswith(key)
    assert "7,832" in message


def test_power_summary_table_flags_a_power_below_80_percent():
    """The interpretation row follows the empirical power."""
    adequate = reports.power_summary_table(0.1, 0.12, 3916, 10000, 0.8, "en")
    weak = reports.power_summary_table(0.1, 0.12, 3916, 10000, 0.5, "pt")

    assert adequate.iloc[-1, 1] == "Adequate (>= 80%)"
    assert weak.iloc[-1, 1].startswith("Insuficiente")
    assert adequate.iloc[5, 1] == "80.0%"


def test_adequacy_table_estimates_power_in_proportion_to_the_sample():
    """A group with half the sample gets half the simulated power."""
    counts = pd.Series({"control": 1958, "personalization": 3916})

    table = reports.adequacy_table(counts, 3916, 0.8, "en")

    assert table["Estimated power"].tolist() == ["40.0%", "80.0%"]
    assert table["Sufficient?"].tolist() == ["✗ No", "✓ Yes"]
    assert table["Gap (users)"].tolist() == [1958, 0]


def test_adequacy_table_without_a_power_analysis_says_not_available():
    """Required 0 means the power analysis did not run."""
    counts = pd.Series({"control": 10})

    table = reports.adequacy_table(counts, 0, float("nan"), "pt")

    assert table.iloc[0].tolist() == [
        "control", 10, "N/D", "✓ Sim", "N/D", "N/D",
    ]


def test_pretest_decision_accepts_a_sufficient_sample(capsys):
    """No warning is raised and the next step is suggested."""
    counts = pd.Series({"control": 4000, "personalization": 4200})

    reports.pretest_decision(counts, 3916, 0.8, 10, "en")

    printed = capsys.readouterr().out
    assert "Sufficient sample across all variants" in printed
    assert "Smallest group: 4,000 users (required: 3,916)" in printed
    assert "Proceed with run_ab_test" in printed


def test_pretest_decision_warns_about_an_insufficient_sample(capsys):
    """The deficit, the options and a UserWarning are produced."""
    counts = pd.Series({"control": 3000, "personalization": 3225})

    with pytest.warns(UserWarning, match="Real sample"):
        reports.pretest_decision(counts, 3916, 0.8, 10, "en")

    printed = capsys.readouterr().out
    assert "deficit: 916" in printed
    assert "~61.3%" in printed
    assert "3. Proceed acknowledging reduced statistical power" in printed


def test_pretest_decision_in_portuguese_when_the_analysis_did_not_run(capsys):
    """Required 0 explains what is missing, in Portuguese."""
    reports.pretest_decision(pd.Series({"a": 5}), 0, float("nan"), 0, "pt")

    assert "required_n_per_group ausente" in capsys.readouterr().out


def test_missing_values_table_orders_columns_by_incompleteness():
    """Shares are in percent of the rows, largest first."""
    frame = pd.DataFrame({"a": [1, None, None, 4], "b": [1, 2, 3, None],
                          "c": [1, 2, 3, 4]})

    table = data.missing_values_table(frame)

    assert table["Column"].tolist() == ["a", "b", "c"]
    assert table["% Missing"].tolist() == [50.0, 25.0, 0.0]


def test_estimate_runtime_days_uses_the_recent_daily_pace():
    """100 users a day need 80 days for 4,000 per group; never below 7."""
    days = pd.date_range("2018-01-01", periods=40)
    frame = pd.DataFrame(
        {
            "date_served": days.repeat(100),
            "user_id": [f"u{i}" for i in range(4000)],
        }
    )

    assert experiment.estimate_runtime_days(frame, 4000) == 80
    assert experiment.estimate_runtime_days(frame, 10) == 7


def test_simulate_power_curve_is_reproducible_and_grows_with_the_sample():
    """The same generator seed gives the same curve; power rises."""
    import numpy as np

    first = experiment.simulate_power_curve(
        0.1, [200, 3000], 2.0, n_simulations=200, rng=np.random.default_rng(1)
    )
    second = experiment.simulate_power_curve(
        0.1, [200, 3000], 2.0, n_simulations=200, rng=np.random.default_rng(1)
    )

    assert first == second
    assert first[1] > first[0]
    assert not any(math.isnan(value) for value in first)
