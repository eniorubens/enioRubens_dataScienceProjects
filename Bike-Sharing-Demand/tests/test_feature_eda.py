"""Tests for src/feature_eda.py (notebook-03 report/plot builders)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import feature_eda as fe  # noqa: E402

INDEX = "Demand_Index"


@pytest.fixture
def cat_frame() -> pd.DataFrame:
    """Synthetic frame with every engineered category + Seasons + the index."""
    rng = np.random.default_rng(0)
    n = 400
    seasons = np.array(["Winter", "Spring", "Summer", "Autumn"])
    rows = {
        "Seasons": rng.choice(seasons, n),
        INDEX: rng.uniform(0.2, 1.8, n),
        "Sunshine (hr)": rng.uniform(0, 1, n),
        "Cloud Cover (oktas)": rng.uniform(0, 8, n),
        "Ground Temp(C)": rng.uniform(-10, 40, n),
    }
    for col, levels in fe.CAT_ORDERS.items():
        rows[col] = rng.choice(levels, n)
    return pd.DataFrame(rows)


def test_rush_hour_audit_counts():
    df = pd.DataFrame(
        {
            "Functioning Day": ["Yes"] * 4,
            "Holiday": ["No Holiday"] * 4,
            "Hour": [8, 8, 17, 12],
            "WeekStatus": ["Weekend", "Weekday", "Weekend", "Weekday"],
            "Rush_Hour": ["No Rush", "Rush", "No Rush", "No Rush"],
        }
    )
    out = fe.rush_hour_audit(df)
    # legacy rule flags the two weekend rush-hours (rows 0 and 2); revised flags none
    assert out["legacy_weekend"] == 2
    assert out["revised_weekend"] == 0


def test_created_features_inventory_pt(cat_frame):
    inv = fe.created_features_inventory(cat_frame.assign(Date=1, Month=1, Year=2020))
    assert list(inv.columns) == ["Variável", "Presente", "Tipo", "Valores distintos", "Ausentes"]
    assert len(inv) == len(fe.CREATED_FEATURES)


def test_missing_features_summary(cat_frame):
    df = cat_frame.copy()
    df.loc[:9, "Cloud Cover (oktas)"] = np.nan
    out = fe.missing_features_summary(df)
    assert "Percentual" in out.columns
    assert out.loc["Cloud Cover (oktas)", "Ausentes"] == 10


def test_category_effect_summary_columns_no_mutation(cat_frame):
    before = cat_frame.copy()
    effect_summary, category_effects = fe.category_effect_summary(cat_frame, index_col=INDEX)
    assert "Amplitude" in effect_summary.columns
    assert effect_summary["Amplitude"].is_monotonic_decreasing  # sorted desc
    assert category_effects.index.names == ["Feature", "Nível"]
    pd.testing.assert_frame_equal(cat_frame, before)


def test_mean_encoding_and_within_season(cat_frame):
    enc = fe.mean_encoding_by_season(cat_frame, index_col=INDEX)
    assert "Overall" in enc.columns and "Amplitude entre estações" in enc.columns
    wss = fe.within_season_spread(cat_frame, index_col=INDEX)
    assert list(wss.columns) == fe.SEASON_ORDER
    assert wss.index.name == "Feature"
    # Overall reproduces a real MeanEncoder in-sample for the demo column.
    ver = fe.meanencoder_verification(cat_frame, enc, index_col=INDEX, demo_col="Rush_Period")
    assert float(ver["Diferença absoluta"].max()) < 1e-9


def test_new_weather_summary(cat_frame):
    out = fe.new_weather_summary(
        cat_frame, ["Sunshine (hr)", "Cloud Cover (oktas)", "Ground Temp(C)"], INDEX
    )
    assert "Spearman com o índice" in out.columns
    assert out.index.name == "Variável"


def test_plots_return_fig(cat_frame):
    df = cat_frame.assign(Date=pd.Timestamp("2020-06-01"))
    fig1, _ = fe.plot_category_distribution(df)
    fig2, _ = fe.plot_category_boxplots(cat_frame, index_col=INDEX)
    fig3, _ = fe.plot_weather_relationships(
        cat_frame, ["Sunshine (hr)", "Cloud Cover (oktas)", "Ground Temp(C)"], INDEX
    )
    for f in (fig1, fig2, fig3):
        assert isinstance(f, plt.Figure)
        plt.close(f)


def test_relationship_summary_bins():
    df = pd.DataFrame({"x": np.arange(100), "y": np.random.default_rng(0).uniform(0, 2, 100)})
    out = fe._relationship_summary(df, "x", "y")
    assert {"n", "Media", "Mediana", "Q1", "Q3", "x"} <= set(out.columns)
    assert out["x"].is_monotonic_increasing


def test_localize_feature_table_renames_season_columns_and_overall(cat_frame):
    enc = fe.mean_encoding_by_season(cat_frame, index_col=INDEX)
    before = enc.copy()

    display_enc = fe.localize_feature_table(enc)

    assert "Geral" in display_enc.columns and "Overall" not in display_enc.columns
    for season_pt in ("Inverno", "Primavera", "Verão", "Outono"):
        assert season_pt in display_enc.columns
    assert display_enc.index.names == ["Variável", "Nível"]
    # numeric values are unchanged by localization
    assert display_enc["Geral"].equals(before["Overall"])
    pd.testing.assert_frame_equal(enc, before)  # internal schema untouched


def test_localize_feature_table_renames_plain_feature_index(cat_frame):
    wss = fe.within_season_spread(cat_frame, index_col=INDEX)
    display_wss = fe.localize_feature_table(wss)
    assert display_wss.index.name == "Variável"
    assert list(display_wss.columns) == ["Inverno", "Primavera", "Verão", "Outono"]
    assert display_wss["Inverno"].equals(wss["Winter"])  # numbers preserved


def test_localize_feature_table_renames_effect_summary_column(cat_frame):
    effect_summary, category_effects = fe.category_effect_summary(cat_frame, index_col=INDEX)
    display_summary = fe.localize_feature_table(effect_summary)
    assert "Variável" in display_summary.columns and "Feature" not in display_summary.columns
    assert display_summary["Amplitude"].equals(effect_summary["Amplitude"])

    display_effects = fe.localize_feature_table(category_effects)
    assert display_effects.index.names == ["Variável", "Nível"]
