"""Tests for the A/B test, the sample-size design and the power."""
import pandas as pd
import pytest

from marketing_campaign import experiment


def _two_variants(conv_a, n_a, conv_b, n_b) -> pd.DataFrame:
    """Return a user base with the given conversions per variant."""
    rows = []
    for variant, conversions, size in (
        ("control", conv_a, n_a),
        ("personalization", conv_b, n_b),
    ):
        flags = [1] * conversions + [0] * (size - conversions)
        rows += [{"variant": variant, "converted_flag": f} for f in flags]
    return pd.DataFrame(rows)


def test_ab_test_computes_rates_difference_and_interval():
    """Rates, the B-versus-A difference and its interval are reported."""
    users = _two_variants(100, 1000, 150, 1000)

    row = experiment.run_ab_test(users).iloc[0]

    assert (row["Variant_A"], row["Variant_B"]) == (
        "control",
        "personalization",
    )
    assert row["Conv_A_%"] == 10.0
    assert row["Conv_B_%"] == 15.0
    assert row["Diff_B_vs_A_pp"] == 5.0
    assert row["Uplift_B_vs_A_%"] == 50.0
    assert row["Significance"] == "YES"
    assert row["Z_Stat"] < 0  # control is below personalization


def test_ab_test_is_not_significant_for_identical_rates():
    """Equal conversion gives z = 0 and p = 1."""
    row = experiment.run_ab_test(_two_variants(50, 500, 50, 500)).iloc[0]

    assert row["Z_Stat"] == 0
    assert row["P_Value"] == 1
    assert row["Significance"] == "NO"


def test_significance_label_follows_the_alpha_argument():
    """A p-value near 0.28 is not significant at 5% but is at 50%."""
    users = _two_variants(100, 1000, 115, 1000)

    strict = experiment.run_ab_test(users).iloc[0]
    loose = experiment.run_ab_test(users, alpha=0.5).iloc[0]

    assert 0.05 < strict["P_Value"] < 0.5
    assert strict["Significance"] == "NO"
    assert loose["Significance"] == "YES"


def test_ab_test_needs_two_variants():
    """A single variant yields an empty result instead of an error."""
    single = pd.DataFrame({"variant": ["control"], "converted_flag": [1]})

    assert experiment.run_ab_test(single).empty


def test_required_sample_grows_as_the_lift_shrinks():
    """Detecting a smaller lift needs more users per group."""
    large = experiment.required_sample_size(0.10, minimum_lift_pp=4.0)
    small = experiment.required_sample_size(0.10, minimum_lift_pp=2.0)

    assert small > large


def test_simulated_power_follows_the_effect_size():
    """A large lift is almost always detected; a tiny one rarely is."""
    strong = experiment.simulate_power(
        0.10, 2000, minimum_lift_pp=8.0, n_simulations=200
    )
    weak = experiment.simulate_power(
        0.10, 200, minimum_lift_pp=0.5, n_simulations=200
    )

    assert strong > 0.95
    assert weak < 0.2


def test_real_ab_test_matches_the_report(result):
    """z = 0.0779, p = 0.937870 and the interval [-1.50, 1.38] p.p."""
    row = result.ab_test.iloc[0]

    assert row["Conv_A_%"] == 10.29
    assert row["Conv_B_%"] == 10.24
    assert row["Z_Stat"] == pytest.approx(0.0779, abs=1e-4)
    assert row["P_Value"] == pytest.approx(0.937870, abs=1e-5)
    assert row["Diff_B_vs_A_pp"] == -0.06
    assert row["IC95_Diff_B_vs_A_pp"] == "[-1.50, 1.38]"
    assert row["Significance"] == "NO"


def test_real_variant_sizes_are_below_the_required_sample(result):
    """3,225 and 3,624 users per variant against 3,916 required."""
    sizes = result.users["variant"].value_counts()
    baseline = result.users["converted_flag"].mean()

    assert sizes["control"] == 3225
    assert sizes["personalization"] == 3624
    assert experiment.required_sample_size(baseline) == 3916
    assert sizes.min() < 3916


def test_monte_carlo_power_at_the_required_sample_is_80_percent(result):
    """10,000 simulations at 3,916 per group estimate 80.0% power."""
    baseline = result.users["converted_flag"].mean()

    power = experiment.simulate_power(baseline, 3916)

    assert round(power, 3) == pytest.approx(0.80, abs=0.01)
