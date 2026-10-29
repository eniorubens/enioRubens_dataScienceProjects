"""Tests for loading, cleaning and consolidating the campaign data."""
import numpy as np
import pytest
from helpers import make_exposures

from marketing_campaign import data


def test_load_data_rejects_a_missing_file(tmp_path):
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.load_data(tmp_path / "missing.csv")


def test_clean_data_drops_duplicate_user_and_date():
    """Two rows for the same user and date keep a single exposure."""
    raw = make_exposures(
        [
            {"user_id": "u1", "date_served": "2018-01-01"},
            {"user_id": "u1", "date_served": "2018-01-01"},
            {"user_id": "u1", "date_served": "2018-01-02"},
        ]
    )

    assert len(data.clean_data(raw)) == 2


def test_clean_data_derives_flags_and_calendar_fields():
    """Boolean columns become 0/1 flags and dates yield calendar fields."""
    raw = make_exposures(
        [
            {"date_served": "2018-01-02", "converted": True,
             "is_retained": True},
            {"user_id": "u2", "date_served": "2018-02-03"},
        ]
    )

    cleaned = data.clean_data(raw)

    assert cleaned["converted_flag"].tolist() == [1, 0]
    assert cleaned["retained_flag"].tolist() == [1, 0]
    assert cleaned["month"].tolist() == [1, 2]
    assert cleaned["campaign_period"].tolist() == ["2018-01", "2018-02"]
    assert cleaned["day_of_week"].iloc[0] == "Tuesday"


def test_clean_data_strips_text_and_turns_blanks_into_missing():
    """Whitespace is trimmed and blank or "nan" text becomes missing."""
    raw = make_exposures(
        [
            {"marketing_channel": "  Email  "},
            {"user_id": "u2", "marketing_channel": ""},
            {"user_id": "u3", "marketing_channel": "nan"},
        ]
    )

    channels = data.clean_data(raw)["marketing_channel"]

    assert channels.iloc[0] == "Email"
    assert channels.iloc[1:].isna().all()


def test_clean_data_derives_correct_language_when_missing():
    """Without ``is_correct_lang`` it compares displayed and preferred."""
    raw = make_exposures(
        [
            {"language_displayed": "English", "language_preferred": "English"},
            {"user_id": "u2", "language_displayed": "Spanish",
             "language_preferred": "English"},
        ]
    )

    assert data.clean_data(raw)["is_correct_lang"].tolist() == [1, 0]


def test_stable_users_exclude_variant_switchers_and_use_ever_flags():
    """A user shown two variants is dropped; flags use the maximum."""
    raw = make_exposures(
        [
            {"user_id": "stable", "date_served": "2018-01-01",
             "converted": False},
            {"user_id": "stable", "date_served": "2018-01-05",
             "converted": True},
            {"user_id": "switcher", "variant": "control"},
            {"user_id": "switcher", "variant": "personalization",
             "date_served": "2018-01-02"},
        ]
    )

    users, switchers = data.aggregate_stable_users(data.clean_data(raw))

    assert switchers == 1
    assert users["user_id"].tolist() == ["stable"]
    assert users["converted_flag"].tolist() == [1]
    assert users["converted"].dtype == bool
    assert np.issubdtype(users["date_served"].dtype, np.datetime64)


def test_real_dataset_reaches_the_published_counts(result):
    """10,037 raw rows -> 9,757 exposures -> 6,849 stable users."""
    assert result.raw_rows == 10037
    assert len(result.exposures) == 9757
    assert len(result.users) == 6849
    assert result.switchers == 460
    assert result.users["user_id"].is_unique
