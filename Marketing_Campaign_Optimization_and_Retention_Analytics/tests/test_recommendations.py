"""Tests for the recommendations and the exports."""
import pandas as pd
import pytest

from marketing_campaign import recommendations as rec


def _row(**values) -> pd.Series:
    """Return a scored user with positive profit and correct language."""
    base = {"Segment": "High Value", "expected_profit": 5.0,
            "is_correct_lang": 1}
    return pd.Series({**base, **values})


@pytest.mark.parametrize(
    ("segment", "action"),
    [
        ("High Value", "Loyalty Campaign"),
        ("Medium Value", "Promotional Offer"),
        ("Low Value", "Win-Back Campaign or No Action"),
        ("Unknown segment", "Win-Back Campaign or No Action"),
    ],
)
def test_the_segment_decides_the_action(segment, action):
    """Each segment maps to its action; unknown ones use the low one."""
    assert rec.recommend_action(_row(Segment=segment)) == action


@pytest.mark.parametrize("profit", [0.0, -0.1])
def test_no_action_when_the_expected_profit_is_not_positive(profit):
    """Zero or negative expected profit overrides the segment action."""
    assert rec.recommend_action(_row(expected_profit=profit)) == (
        "No Action - Negative Incremental Profit"
    )


def test_numeric_wrong_language_flag_is_sent_to_a_language_fix():
    """``is_correct_lang == 0`` routes the user to a language fix first."""
    action = rec.recommend_action(_row(is_correct_lang=0))

    assert action == "Fix Language -> then Loyalty Campaign"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation reproduced from the notebooks: the dataset stores "
        "the flag as 'Yes'/'No', which `== 0` never matches."
    ),
)
def test_text_wrong_language_flag_should_also_be_sent_to_a_language_fix():
    """Intended behaviour: the string "No" is a wrong-language user."""
    action = rec.recommend_action(_row(is_correct_lang="No"))

    assert action.startswith("Fix Language")


def _scored() -> pd.DataFrame:
    """Return two scored users for the recommendation tests."""
    return pd.DataFrame(
        {
            "user_id": ["a", "b"],
            "marketing_channel": ["Email", "Push"],
            "Segment": ["High Value", "Medium Value"],
            "prob_retention": [0.5, 0.2],
            "expected_profit": [10.0, 1.0],
            "is_correct_lang": ["Yes", "Yes"],
        }
    )


def test_incremental_profit_discounts_the_organic_baseline():
    """Incremental profit = profit - prob x global value x organic rate."""
    recs = rec.generate_recommendations(_scored(), global_ltv=40.0)

    first = recs.set_index("user_id").loc["a"]
    assert first["expected_profit_no_action"] == pytest.approx(0.5 * 40 * 0.2)
    assert first["incremental_profit"] == pytest.approx(10.0 - 4.0)


def test_recommendations_are_sorted_and_carry_incentive_and_rationale():
    """Rows come best first, with the segment's incentive and rationale."""
    recs = rec.generate_recommendations(_scored(), global_ltv=40.0)

    assert recs["user_id"].tolist() == ["a", "b"]
    assert recs.iloc[0]["max_incentive"] == 15.0
    assert recs.iloc[1]["max_incentive"] == 8.0
    assert recs.iloc[0]["strategy_rationale"].startswith("High-value")


def test_export_crm_writes_one_row_per_user(tmp_path):
    """The CRM file is written to the requested folder."""
    recs = rec.generate_recommendations(_scored(), global_ltv=40.0)

    path = rec.export_crm(recs, tmp_path / "out")

    assert path.name == "crm_next_best_action.csv"
    assert pd.read_csv(path)["user_id"].tolist() == ["a", "b"]


def test_export_results_skips_empty_tables(tmp_path):
    """Empty tables are not exported, whichever format is available."""
    sheets = {"1_Campaign_Summary": pd.DataFrame({"x": [1]}),
              "4_AB_Test_Results": pd.DataFrame()}

    target = rec.export_results(sheets, tmp_path)

    assert target.exists()
    if target.is_dir():
        assert [p.name for p in target.iterdir()] == ["1_Campaign_Summary.csv"]
    else:
        names = pd.ExcelFile(target).sheet_names
        assert names == ["1_Campaign_Summary"]


def test_real_recommendations_cover_every_user_once(result):
    """One row per stable user and a positive-profit action subset."""
    recs = result.recommendations

    assert len(recs) == 6849
    assert recs["user_id"].is_unique
    assert recs["expected_profit"].is_monotonic_decreasing
    acted = recs[recs["recommended_action"]
                 != "No Action - Negative Incremental Profit"]
    assert (acted["expected_profit"] > 0).all()


def test_real_action_distribution_matches_the_exported_file(result):
    """About 580 users get the loyalty action; the rest get none.

    The split depends on logistic-regression probabilities near the
    break-even point, which move by a few users across library versions.
    """
    counts = result.recommendations["recommended_action"].value_counts()

    assert set(counts.index) == {
        "No Action - Negative Incremental Profit",
        "Loyalty Campaign",
    }
    assert counts["Loyalty Campaign"] == pytest.approx(580, abs=10)
    assert counts["No Action - Negative Incremental Profit"] == pytest.approx(
        6269, abs=10
    )
