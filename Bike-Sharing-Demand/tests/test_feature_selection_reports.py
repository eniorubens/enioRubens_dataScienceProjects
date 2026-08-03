"""Tests for the nb02 sampling / phik / VIF / split report helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import feature_selection as fs


def test_stratified_sample_by_caps_per_group():
    df = pd.DataFrame({"x": np.arange(100)})
    by = pd.Series(np.repeat([2018, 2019, 2020, 2021], 25), index=df.index)
    sample = fs.stratified_sample_by(df, by, n_per_group=10, seed=42)
    counts = by.loc[sample.index].value_counts()
    assert (counts == 10).all()
    assert len(sample) == 40
    # reproducible
    again = fs.stratified_sample_by(df, by, n_per_group=10, seed=42)
    assert list(sample.index) == list(again.index)


def test_filter_significant_phik_pairs():
    pairs = pd.DataFrame(
        {
            "var1": ["a", "c", "e", "g"],
            "var2": ["b", "d", "f", "h"],
            "phik": [0.9, 0.4, 0.55, 0.51],
            "significance": [5.0, 9.0, 2.0, 4.0],
        }
    )
    out = fs.filter_significant_phik_pairs(pairs, 0.50, 3.0)
    # only rows with phik>=.50 AND significance>=3 survive: (0.9,5) and (0.51,4)
    assert list(out["phik"]) == [0.9, 0.51]
    assert list(out.columns) == ["var1", "var2", "phik", "significance"]


def test_phik_pairs_with_target_pt_column():
    pairs = pd.DataFrame(
        {
            "var1": ["Rented Bike Count", "x", "Rented Bike Count"],
            "var2": ["Hour", "y", "Temp"],
            "phik": [0.5, 0.2, 0.4],
            "significance": [4.0, 1.0, 3.0],
        }
    )
    out = fs.phik_pairs_with_target(pairs, "Rented Bike Count")
    assert "Outra variável" in out.columns
    assert set(out["Outra variável"]) == {"Hour", "Temp"}


def test_compute_vif_report_adds_r2():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a * 2 + rng.normal(scale=0.01, size=200)  # near-collinear
    c = rng.normal(size=200)
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    report = fs.compute_vif_report(df, ["a", "b", "c"])
    assert "R2_auxiliar" in report.columns
    assert "VIF" in report.columns
    # near-collinear pair has high VIF / high auxiliary R2
    assert report["VIF"].max() > 10


def test_mfa_split_report_localized():
    mfa = fs.MultivariateFeatureAnalysis(target_col="y")
    mfa.X_train_ = pd.DataFrame({"is_anomalous_2020": [0, 1, 0]})
    mfa.X_test_ = pd.DataFrame({"is_anomalous_2020": [0, 0]})
    mfa.y_train_ = pd.Series([1.0, 2.0, 3.0])
    mfa.y_test_ = pd.Series([4.0, 5.0])
    years = pd.Series([2018, 2019, 2020, 2021, 2022])
    rep = mfa.split_report(years)
    assert list(rep["Divisão"]) == ["Treino", "Validação temporal"]
    assert rep.loc[0, "N"] == 3 and rep.loc[1, "N"] == 2
    assert rep.loc[0, "is_anomalous_2020 = 1"] == 1


def test_phik_pairs_with_target_localizes_significance_column():
    pairs = pd.DataFrame(
        {
            "var1": ["Rented Bike Count", "x"],
            "var2": ["Hour", "y"],
            "phik": [0.5, 0.2],
            "significance": [4.0, 1.0],
        }
    )
    out = fs.phik_pairs_with_target(pairs, "Rented Bike Count")
    assert "significância" in out.columns
    assert "significance" not in out.columns
    assert out["significância"].iloc[0] == 4.0  # numbers preserved


def test_localize_phik_report_renames_without_mutating_input():
    pairs = pd.DataFrame({"var1": ["a"], "var2": ["b"], "phik": [0.9], "significance": [5.0]})
    before = pairs.copy()
    out = fs.localize_phik_report(pairs)
    assert list(out.columns) == ["Variável 1", "Variável 2", "phik", "significância"]
    assert out["phik"].iloc[0] == 0.9
    pd.testing.assert_frame_equal(pairs, before)


def test_localize_vif_report_renames_feature_column():
    vif_df = pd.DataFrame({"feature": ["a", "b"], "VIF": [1.5, 30.0], "R2_auxiliar": [0.3, 0.97]})
    out = fs.localize_vif_report(vif_df)
    assert list(out.columns) == ["Variável", "VIF", "R2_auxiliar"]
    assert list(out["VIF"]) == [1.5, 30.0]


def test_localize_importance_and_ablation_tables():
    importance = pd.DataFrame(
        {
            "feature": ["Hour"],
            "impurity_importance": [0.33],
            "permutation_importance_mean": [0.43],
            "permutation_importance_std": [0.02],
            "mean_abs_shap": [0.5],
        }
    )
    out = fs.localize_importance_table(importance)
    assert list(out.columns) == [
        "Variável",
        "Importância por impureza",
        "Importância por permutação (média)",
        "Importância por permutação (desvio-padrão)",
        "|SHAP| médio",
    ]
    assert out["Importância por impureza"].iloc[0] == 0.33

    ablation = pd.DataFrame(
        {
            "removed_feature": ["Hour"],
            "rmse_without_feature": [0.42],
            "mae_without_feature": [0.25],
            "r2_without_feature": [0.79],
            "delta_rmse": [0.046],
            "delta_mae": [0.01],
            "delta_r2": [-0.01],
        }
    )
    out_ablation = fs.localize_ablation_table(ablation)
    assert "Variável removida" in out_ablation.columns
    assert out_ablation["Δ RMSE"].iloc[0] == 0.046


def test_prepare_rf_sample_fills_categorical_missing_and_stratifies():
    df = pd.DataFrame(
        {
            "cat": pd.array(["x", None, "y", "x"], dtype="object"),
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    years = pd.Series([2020, 2020, 2021, 2021])
    sample, sample_years, categorical_missing = fs.prepare_rf_sample(
        df, years, n_per_group=2, seed=42
    )
    assert categorical_missing == 1
    assert not sample.isna().any().any()
    assert "Missing" in sample["cat"].values
    assert len(sample) == len(sample_years) == 4
