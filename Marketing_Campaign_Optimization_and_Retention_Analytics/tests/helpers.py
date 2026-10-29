"""Small builders for synthetic campaign data used by the tests."""
import pandas as pd


def make_exposures(rows: list[dict]) -> pd.DataFrame:
    """Build a small raw-exposure frame with sensible defaults.

    Parameters
    ----------
    rows : list of dict
        Values to override per row; ``user_id`` and ``date_served`` are
        usually given.

    Returns
    -------
    pd.DataFrame
        Frame with the raw campaign columns.
    """
    defaults = {
        "user_id": "u1",
        "date_served": "2018-01-01",
        "marketing_channel": "Email",
        "variant": "control",
        "converted": False,
        "language_displayed": "English",
        "language_preferred": "English",
        "age_group": "19-24 years",
        "date_subscribed": None,
        "date_canceled": None,
        "subscribing_channel": None,
        "is_retained": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])
