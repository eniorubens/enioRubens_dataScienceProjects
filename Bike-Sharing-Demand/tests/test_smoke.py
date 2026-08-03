"""Smoke tests for visualization/statistical modules.

Uses non-interactive matplotlib (Agg backend) and the synthetic
`seoul_df` / `preprocessed_df` fixtures.  All plt.show() calls are
replaced by the Agg backend no-op.

Modules covered:
  src/utils.py, src/plotting.py, src/eda.py, src/outliers.py,
  src/stats_tests.py, src/feature_selection.py, src/tracking.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must come before any other matplotlib/pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / inline fixtures
# ---------------------------------------------------------------------------


def _make_condition_df(n: int = 100) -> pd.DataFrame:
    """Minimal df for plot_boxplot_comparation ('Condition', 'Bike Count')."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Condition": np.where(rng.random(n) < 0.5, "Rain", "No Rain"),
            "Bike Count": rng.integers(0, 800, n).astype(float),
        }
    )


@pytest.fixture
def multi_season_df():
    """Seoul-schema df spanning all 4 seasons (1 year, no preprocessing).

    plot_iqr_outliers_by_season runs np.percentile per season, so each
    seasonal bucket must be non-empty.
    """
    rng = np.random.default_rng(7)
    n = 8760  # 1 full year of hourly data
    dates = pd.date_range("2018-01-01", periods=n, freq="h")
    hours = dates.hour
    months = dates.month
    seasons = np.where(
        months.isin([12, 1, 2]),
        "Winter",
        np.where(
            months.isin([3, 4, 5]), "Spring", np.where(months.isin([6, 7, 8]), "Summer", "Autumn")
        ),
    )
    return pd.DataFrame(
        {
            "DateTime": dates,
            "Date": dates.strftime("%d/%m/%Y"),
            "Hour": hours,
            "Rented Bike Count": rng.integers(0, 1000, n).astype(float),
            "Temperature(C)": rng.uniform(-10, 35, n),
            "Humidity(%)": rng.integers(10, 100, n).astype(float),
            "Wind speed (m/s)": rng.uniform(0, 8, n),
            "Visibility (10m)": rng.integers(100, 2000, n).astype(float),
            "Dew point temperature(C)": rng.uniform(-20, 25, n),
            "Solar Radiation (MJ/m2)": rng.uniform(0, 3.5, n),
            "Rainfall(mm)": np.where(rng.random(n) < 0.1, rng.uniform(0.1, 20, n), 0.0),
            "Snowfall (cm)": np.where(rng.random(n) < 0.05, rng.uniform(0.1, 5, n), 0.0),
            "Seasons": seasons,
            "Holiday": np.where(rng.random(n) < 0.05, "Holiday", "No Holiday"),
            "Functioning Day": np.where(rng.random(n) < 0.02, "No", "Yes"),
        }
    )


# ── src/utils.py ──────────────────────────────────────────────────────────


class TestUtils:
    def test_setup_logging_returns_logger(self):
        import logging
        from src.utils import setup_logging

        logger = setup_logging("WARNING")
        assert isinstance(logger, logging.Logger)

    def test_set_global_seed_reproducible(self):
        from src.utils import set_global_seed

        set_global_seed(0)
        val1 = np.random.rand()
        set_global_seed(0)
        val2 = np.random.rand()
        assert val1 == val2

    def test_project_root_is_dir(self):
        from src.utils import PROJECT_ROOT

        assert PROJECT_ROOT.is_dir()

    def test_constants_defined(self):
        from src.utils import RANDOM_STATE, DATASET_DIR, MODELS_DIR, MLRUNS_DIR

        assert RANDOM_STATE == 42
        assert DATASET_DIR is not None
        assert MODELS_DIR is not None


# ── src/plotting.py ───────────────────────────────────────────────────────


class TestPlotting:
    def test_set_graph_parameters_runs(self):
        import matplotlib as mpl
        from src.plotting import set_graph_parameters

        set_graph_parameters()
        assert mpl.rcParams["figure.figsize"] == [16, 6]

    def test_idempotent(self):
        from src.plotting import set_graph_parameters

        set_graph_parameters()
        set_graph_parameters()  # must not raise


# ── src/eda.py ────────────────────────────────────────────────────────────


class TestEdaSmoke:
    def test_skewness_measure(self, preprocessed_df):
        from src.eda import skewness_measure

        direction, intensity, text = skewness_measure(preprocessed_df, "Rented Bike Count", "All")
        assert direction in ("Left", "Right", "Neutral")
        assert isinstance(text, str)

    def test_distribution_on_target(self, preprocessed_df):
        from src.eda import distribution_on_target

        fig, axes = distribution_on_target(preprocessed_df.copy())
        plt.close("all")
        assert fig is not None

    def test_distribution_by_season_on_weekday(self, preprocessed_df):
        from src.eda import distribution_by_season_on_weekday

        fig, ax = distribution_by_season_on_weekday(preprocessed_df.copy())
        plt.close("all")
        assert fig is not None

    def test_plot_pointplot(self, preprocessed_df):
        from src.eda import plot_pointplot

        fig, ax = plot_pointplot(preprocessed_df.copy())
        plt.close("all")
        assert fig is not None

    def test_distribution_pointplot(self, preprocessed_df):
        from src.eda import distribution_pointplot

        fig, axes = distribution_pointplot(preprocessed_df.copy())
        plt.close("all")
        assert fig is not None

    def test_plot_boxplot_comparation(self):
        from src.eda import plot_boxplot_comparation

        cond_df = _make_condition_df()
        fig, ax = plot_boxplot_comparation(cond_df, title="Rain vs No Rain")
        plt.close("all")
        assert fig is not None


# ── src/outliers.py ───────────────────────────────────────────────────────


class TestOutliersSmoke:
    def test_iqr_outliers_returns_series(self, preprocessed_df):
        from src.outliers import iqr_outliers

        result = iqr_outliers(preprocessed_df["Rented Bike Count"])
        assert isinstance(result, pd.Series)

    def test_rainfall_event_returns_positives_only(self, seoul_df):
        from src.outliers import rainfall_event

        result = rainfall_event(seoul_df["Rainfall(mm)"])
        assert (result > 0).all()

    def test_plot_outliers_integer_index(self, preprocessed_df):
        # Tests plot_outliers directly with integer-indexed series (no DateTime).
        # Avoids pandas DatetimeIndex + second-plot-on-Agg issue in
        # plot_iqr_outliers_by_season which requires set_index("DateTime").
        from src.outliers import iqr_outliers, plot_outliers

        data = preprocessed_df["Rented Bike Count"].reset_index(drop=True)
        temp = preprocessed_df["Temperature(C)"].reset_index(drop=True)
        outliers = iqr_outliers(data)
        fig, ax = plt.subplots()
        plot_outliers(outliers=outliers, data=data, temperature=temp, ax=ax)
        plt.close("all")
        assert ax is not None

    def test_iqr_outliers_by_season_year(self):
        from src.outliers import (
            iqr_outlier_summary_by_season_year,
            plot_iqr_outliers_by_season_year,
        )

        timestamp = pd.date_range("2020-12-01", periods=24 * 100, freq="h")
        dataframe = pd.DataFrame(
            {
                "DateTime": timestamp.normalize(),
                "Hour": timestamp.hour,
                "Seasons": np.select(
                    [timestamp.month.isin([12, 1, 2]), timestamp.month.isin([3, 4, 5])],
                    ["Winter", "Spring"],
                    default="Summer",
                ),
                "Rented Bike Count": np.r_[np.ones(len(timestamp) - 1) * 10, 1000],
            }
        )

        summary = iqr_outlier_summary_by_season_year(dataframe)
        assert {"Meteorological year", "Season", "Outliers", "Outlier rate (%)"}.issubset(
            summary.columns
        )
        assert summary["Outliers"].sum() >= 1

        fig, axes, year_summary = plot_iqr_outliers_by_season_year(dataframe, 2021)
        plt.close(fig)
        assert len(axes) == 5
        assert not year_summary.empty

    def test_plot_rainfall_by_season(self, multi_season_df):
        # plot_rainfall_by_season needs Seasons, Rented Bike Count,
        # Temperature(C), Rainfall(mm) — all present in multi_season_df.
        from src.outliers import plot_rainfall_by_season

        fig, ax = plot_rainfall_by_season(multi_season_df.copy())
        plt.close("all")
        assert fig is not None


# ── src/stats_tests.py ────────────────────────────────────────────────────


class TestStatsTestsSmoke:
    def test_seasonal_hypothesis_tests(self, preprocessed_df):
        from src.stats_tests import seasonal_hypothesis_tests

        result = seasonal_hypothesis_tests(preprocessed_df.copy(), alpha=0.05)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "Decision" in result.columns
        assert {"Effect measure", "Effect size", "Holm p-value", "Holm Decision"}.issubset(
            result.columns
        )
        valid = result["p-value"].notna()
        assert (result.loc[valid, "Holm p-value"] >= result.loc[valid, "p-value"]).all()
        assert result.loc[~valid, "Holm Decision"].eq("Not evaluated").all()

    def test_plot_seasonal_hypothesis_tests(self, preprocessed_df):
        from src.stats_tests import seasonal_hypothesis_tests, plot_seasonal_hypothesis_tests

        results_df = seasonal_hypothesis_tests(preprocessed_df.copy())
        fig, axes = plot_seasonal_hypothesis_tests(results_df)
        plt.close("all")
        assert fig is not None

    def test_normality_tests(self, preprocessed_df):
        # normality_tests expects Dict[str, Series]
        from src.stats_tests import normality_tests

        samples = {"all_seasons": preprocessed_df["Rented Bike Count"]}
        results_df, fig, axes = normality_tests(samples, alpha=0.05)
        plt.close("all")
        assert isinstance(results_df, pd.DataFrame)
        assert "Shapiro Decision" in results_df.columns
        assert {"Median", "IQR", "Skewness", "Excess kurtosis"}.issubset(results_df.columns)

    def test_normality_tests_multiple_groups(self, preprocessed_df):
        from src.stats_tests import normality_tests

        bike = preprocessed_df["Rented Bike Count"]
        samples = {
            "group_a": bike.iloc[: len(bike) // 2],
            "group_b": bike.iloc[len(bike) // 2 :],
        }
        results_df, fig, axes = normality_tests(samples)
        plt.close("all")
        assert len(results_df) == 2

    def test_chi_square_feature_selection(self, preprocessed_df):
        from src.stats_tests import chi_square_feature_selection

        feature_scores, fig, ax = chi_square_feature_selection(
            preprocessed_df.copy(), target_col="Rented Bike Count"
        )
        plt.close("all")
        assert isinstance(feature_scores, pd.DataFrame)
        assert "Chi Squared Score" in feature_scores.columns
        assert {"Cramer's V", "df", "Min expected count", "Holm Decision"}.issubset(
            feature_scores.columns
        )
        assert "Decision" in feature_scores.columns
        assert set(feature_scores["Decision"]).issubset({"Reject H0", "Fail to reject H0"})

    def test_chi_square_is_invariant_to_category_labels(self):
        from src.stats_tests import chi_square_feature_selection

        dataframe = pd.DataFrame(
            {
                "segment": ["a", "b", "c", "a", "b"] * 4,
                "Rented Bike Count": np.arange(1.0, 21.0),
            }
        )
        relabeled = dataframe.assign(
            segment=dataframe["segment"].map({"a": "z", "b": "x", "c": "y"})
        )

        original, fig_original, _ = chi_square_feature_selection(dataframe)
        renamed, fig_renamed, _ = chi_square_feature_selection(relabeled)
        plt.close(fig_original)
        plt.close(fig_renamed)

        assert original.loc["segment", "Chi Squared Score"] == pytest.approx(
            renamed.loc["segment", "Chi Squared Score"]
        )
        assert original.loc["segment", "Cramer's V"] == pytest.approx(
            renamed.loc["segment", "Cramer's V"]
        )

    def test_anova_feature_selection(self, preprocessed_df):
        # anova_feature_selection takes (df, feature_list, target_series)
        from src.stats_tests import anova_feature_selection

        num_features = [
            c
            for c in ["Temperature(C)", "Humidity(%)", "Wind speed (m/s)"]
            if c in preprocessed_df.columns
        ]
        target = preprocessed_df["Rented Bike Count"]
        feature_scores, fig, ax = anova_feature_selection(
            preprocessed_df.copy(), num_features, target=target
        )
        plt.close("all")
        assert isinstance(feature_scores, pd.DataFrame)
        assert "ANOVA Score" in feature_scores.columns
        assert {"Eta squared", "Holm p-value", "Holm Decision"}.issubset(feature_scores.columns)

    def test_f_regression_feature_selection(self, preprocessed_df):
        from src.stats_tests import f_regression_feature_selection

        feature_scores, fig, ax = f_regression_feature_selection(
            preprocessed_df.copy(),
            target_col="Rented Bike Count",
            top_n=5,
        )
        plt.close(fig)

        assert isinstance(feature_scores, pd.DataFrame)
        assert {"F Score", "Univariate R2", "Relative F share (%)", "Holm Decision"}.issubset(
            feature_scores.columns
        )
        assert feature_scores["Univariate R2"].between(0, 1).all()

    def test_kruskal_weather_test(self, preprocessed_df):
        from src.stats_tests import kruskal_weather_test

        results_df, fig, axes = kruskal_weather_test(preprocessed_df.copy(), alpha=0.05)
        plt.close("all")
        assert isinstance(results_df, pd.DataFrame)
        assert "H observed" in results_df.columns
        assert {"N", "Group summary", "Epsilon squared", "Holm p-value", "Holm Decision"}.issubset(
            results_df.columns
        )
        assert results_df["Epsilon squared"].between(0, 1).all()
        assert (results_df["Holm p-value"] >= results_df["p-value"]).all()

    def test_mannwhitney_ab_decision_keys(self, preprocessed_df):
        from src.stats_tests import mannwhitney_ab_decision

        data = preprocessed_df["Rented Bike Count"]
        half = len(data) // 2
        result = mannwhitney_ab_decision(data.iloc[:half], data.iloc[half:], label="test")
        assert isinstance(result, dict)
        assert "U observed" in result
        assert "Decision" in result

    def test_plot_mannwhitney_critical_regions(self, preprocessed_df):
        from src.stats_tests import mannwhitney_ab_decision, plot_mannwhitney_critical_regions

        data = preprocessed_df["Rented Bike Count"]
        half = len(data) // 2
        result = mannwhitney_ab_decision(data.iloc[:half], data.iloc[half:], label="AB")
        results_df = pd.DataFrame([result])
        fig, axes = plot_mannwhitney_critical_regions(results_df)
        plt.close("all")
        assert fig is not None
        assert len(axes) == 1


# ── src/feature_selection.py ──────────────────────────────────────────────


class TestFeatureSelectionSmoke:
    def test_compute_vif(self, preprocessed_df):
        from src.feature_selection import compute_vif

        features = [
            c
            for c in ["Temperature(C)", "Humidity(%)", "Wind speed (m/s)"]
            if c in preprocessed_df.columns
        ]
        result = compute_vif(preprocessed_df.copy(), features)
        assert isinstance(result, pd.DataFrame)
        assert "VIF" in result.columns
        assert len(result) == len(features)

    def test_compute_vif_uses_an_intercept(self):
        from src.feature_selection import compute_vif

        frame = pd.DataFrame(
            {
                "x": [10.0, 11.0, 12.0, 14.0, 17.0, 21.0],
                "y": [30.0, 29.0, 33.0, 31.0, 38.0, 35.0],
            }
        )
        correlation = frame["x"].corr(frame["y"])
        expected_vif = 1.0 / (1.0 - correlation**2)

        result = compute_vif(frame, ["x", "y"])

        assert np.allclose(result["VIF"], expected_vif)

    def test_build_phik_significance_df(self, preprocessed_df):
        import phik  # registers pandas extensions (df.phik_matrix, etc.)
        from src.feature_selection import build_phik_significance_df

        result = build_phik_significance_df(preprocessed_df.copy())
        assert isinstance(result, pd.DataFrame)
        assert "phik" in result.columns
        assert "significance" in result.columns

    def test_multivariate_feature_analysis_fit(self, preprocessed_df):
        from src.feature_selection import MultivariateFeatureAnalysis

        mfa = MultivariateFeatureAnalysis(
            target_col="Rented Bike Count",
            n_estimators=5,
            n_repeats=2,
            max_shap_samples=50,
        )
        mfa.fit(preprocessed_df.copy())
        assert mfa.model_ is not None
        assert not mfa.impurity_importance_df_.empty

    def test_get_feature_importance_table(self, preprocessed_df):
        from src.feature_selection import MultivariateFeatureAnalysis

        mfa = MultivariateFeatureAnalysis(
            target_col="Rented Bike Count", n_estimators=5, n_repeats=2
        )
        mfa.fit(preprocessed_df.copy())
        table = mfa.get_feature_importance_table(top_n=5)
        assert isinstance(table, pd.DataFrame)
        assert len(table) <= 5

    def test_plot_feature_importance_impurity(self, preprocessed_df):
        from src.feature_selection import MultivariateFeatureAnalysis

        mfa = MultivariateFeatureAnalysis(
            target_col="Rented Bike Count", n_estimators=5, n_repeats=2
        )
        mfa.fit(preprocessed_df.copy())
        result = mfa.plot_feature_importance(kind="impurity", top_n=5)
        plt.close("all")
        # returns chart_df (a DataFrame slice), not a Figure
        assert result is not None

    def test_plot_ablation(self, preprocessed_df):
        from src.feature_selection import MultivariateFeatureAnalysis

        mfa = MultivariateFeatureAnalysis(
            target_col="Rented Bike Count", n_estimators=5, n_repeats=2
        )
        mfa.fit(preprocessed_df.copy())
        result = mfa.plot_ablation(top_n=3)
        plt.close("all")
        assert result is not None


# ── src/tracking.py ───────────────────────────────────────────────────────


class TestTrackingSmoke:
    """Use tempfile.TemporaryDirectory instead of tmp_path to avoid
    Windows permission issues with the pytest-of-<user> temp folder."""

    def test_setup_mlflow_sets_uri(self):
        import tempfile
        import mlflow
        from src.tracking import setup_mlflow

        with tempfile.TemporaryDirectory() as td:
            setup_mlflow(tracking_uri=f"file:{td}/mlruns")
            assert mlflow.get_tracking_uri().startswith("file:")

    def test_log_estimator_run_returns_run_id(self):
        import tempfile
        from src.tracking import log_estimator_run, setup_mlflow

        with tempfile.TemporaryDirectory() as td:
            setup_mlflow(tracking_uri=f"file:{td}/mlruns")
            run_id = log_estimator_run(
                estimator_name="TestEstimator",
                params={"alpha": 0.1, "max_depth": 5},
                inner_cv_metrics={"r2_mean": 0.80, "rmse_mean": 50.0, "mae_mean": 35.0},
                holdout_metrics={"r2": 0.82, "rmse": 48.0, "mae": 33.0},
                artifacts_dir=td,
            )
            assert isinstance(run_id, str) and len(run_id) > 0

    def test_log_estimator_run_with_loso(self):
        import tempfile
        from src.tracking import log_estimator_run, setup_mlflow

        loso_df = pd.DataFrame(
            {
                "Model": ["XGB"] * 4,
                "Season": ["Winter", "Spring", "Summer", "Autumn"],
                "MAE": [80.0, 70.0, 65.0, 75.0],
                "RMSE": [110.0, 95.0, 90.0, 100.0],
                "R2": [0.75, 0.80, 0.82, 0.78],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            setup_mlflow(tracking_uri=f"file:{td}/mlruns")
            run_id = log_estimator_run(
                estimator_name="XGBRegressor",
                params={"n_estimators": 100},
                inner_cv_metrics={"r2_mean": 0.83},
                holdout_metrics={"r2": 0.85},
                loso_df=loso_df,
                artifacts_dir=td,
            )
            assert isinstance(run_id, str)

    def test_log_loso_run_returns_run_id(self):
        import tempfile
        from src.tracking import log_loso_run, setup_mlflow

        loso_df = pd.DataFrame(
            {
                "Model": ["XGB"] * 4,
                "Season": ["Winter", "Spring", "Summer", "Autumn"],
                "MAE": [80.0, 70.0, 65.0, 75.0],
                "RMSE": [110.0, 95.0, 90.0, 100.0],
                "R2": [0.75, 0.80, 0.82, 0.78],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            setup_mlflow(tracking_uri=f"file:{td}/mlruns")
            run_id = log_loso_run("XGBRegressor", loso_df)
            assert isinstance(run_id, str)


# ── src/mlflow_integration.py ─────────────────────────────────────────────────


class TestMLflowIntegrationSmoke:
    """Smoke tests for src/mlflow_integration.py (class-based API).

    Uses tempfile.TemporaryDirectory instead of tmp_path to avoid
    Windows permission issues with the pytest-of-<user> temp folder.
    """

    def _make_loso_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Model": ["XGB"] * 4,
                "Season": ["Winter", "Spring", "Summer", "Fall"],
                "MAE": [80.0, 70.0, 65.0, 75.0],
                "RMSE": [110.0, 95.0, 90.0, 100.0],
                "R2": [0.75, 0.80, 0.82, 0.78],
            }
        )

    def test_experiment_config_defaults(self):
        from src.mlflow_integration import ExperimentConfig

        cfg = ExperimentConfig()
        assert cfg.experiment_name == "bike_sharing_demand_v3"
        assert cfg.project_name == "bike_sharing_demand_v3"
        assert "project_type" in cfg.tags

    def test_experiment_config_validation(self):
        from src.mlflow_integration import ExperimentConfig

        with pytest.raises(ValueError, match="project_name"):
            ExperimentConfig(project_name="")

    def test_regression_model_metrics_to_dict_prefixes(self):
        from src.mlflow_integration import RegressionModelMetrics

        m = RegressionModelMetrics(
            estimator_name="XGBRegressor",
            params={"n_estimators": 100},
            cv_metrics={"r2_mean": 0.80, "rmse_mean": 95.0, "mae_mean": 63.0},
            holdout_metrics={"r2": 0.82, "rmse": 90.0, "mae": 60.0},
        )
        d = m.to_dict()
        assert d["cv_r2_mean"] == pytest.approx(0.80)
        assert d["cv_rmse_mean"] == pytest.approx(95.0)
        assert d["holdout_r2"] == pytest.approx(0.82)
        assert d["holdout_rmse"] == pytest.approx(90.0)

    def test_to_dict_excludes_loso_df(self):
        from src.mlflow_integration import RegressionModelMetrics

        m = RegressionModelMetrics(
            estimator_name="XGBRegressor",
            params={},
            cv_metrics={"r2_mean": 0.80},
            holdout_metrics={"r2": 0.82},
            loso_df=self._make_loso_df(),
        )
        keys = list(m.to_dict().keys())
        assert not any(k.startswith("loso_") for k in keys)

    def test_mlflow_tracker_setup_experiment(self):
        import tempfile
        import mlflow
        from src.mlflow_integration import ExperimentConfig, MLflowTracker

        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(tracking_uri=f"file:{td}/mlruns")
            tracker = MLflowTracker(cfg)
            exp_id = tracker.setup_experiment()
            assert isinstance(exp_id, str) and len(exp_id) > 0
            experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
            assert experiment is not None

    def test_mlflow_tracker_log_model_run(self):
        import tempfile
        import mlflow
        from src.mlflow_integration import (
            ExperimentConfig,
            MLflowTracker,
            RegressionModelMetrics,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(tracking_uri=f"file:{td}/mlruns")
            tracker = MLflowTracker(cfg)
            tracker.setup_experiment()

            metrics = RegressionModelMetrics(
                estimator_name="XGBRegressor",
                params={"n_estimators": 100, "max_depth": 6},
                cv_metrics={"r2_mean": 0.80, "rmse_mean": 95.0, "mae_mean": 63.0},
                holdout_metrics={"r2": 0.82, "rmse": 90.0, "mae": 60.0},
                loso_df=self._make_loso_df(),
            )
            run_id = tracker.log_model_run(metrics, artifacts_dir=td)

            assert isinstance(run_id, str) and len(run_id) > 0
            runs = mlflow.search_runs(experiment_names=[cfg.experiment_name])
            assert run_id in runs["run_id"].values

    def test_get_best_model_ascending(self):
        import tempfile
        from src.mlflow_integration import (
            ExperimentConfig,
            MLflowTracker,
            RegressionModelMetrics,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(tracking_uri=f"file:{td}/mlruns")
            tracker = MLflowTracker(cfg)
            tracker.setup_experiment()

            for name, rmse in [("ModelA", 120.0), ("ModelB", 80.0)]:
                m = RegressionModelMetrics(
                    estimator_name=name,
                    params={},
                    cv_metrics={"r2_mean": 0.75},
                    holdout_metrics={"r2": 0.76, "rmse": rmse, "mae": rmse * 0.7},
                )
                tracker.log_model_run(m, artifacts_dir=td)

            best = tracker.get_best_model(metric_name="holdout_rmse", ascending=True)
            assert best["estimator"] == "ModelB"
            assert best["metric_value"] == pytest.approx(80.0)

    def test_get_experiment_summary_columns(self):
        import tempfile
        from src.mlflow_integration import (
            ExperimentConfig,
            MLflowTracker,
            RegressionModelMetrics,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(tracking_uri=f"file:{td}/mlruns")
            tracker = MLflowTracker(cfg)
            tracker.setup_experiment()

            m = RegressionModelMetrics(
                estimator_name="XGBRegressor",
                params={"n_estimators": 50},
                cv_metrics={"r2_mean": 0.78, "rmse_mean": 100.0, "mae_mean": 70.0},
                holdout_metrics={"r2": 0.80, "rmse": 95.0, "mae": 65.0},
            )
            tracker.log_model_run(m, artifacts_dir=td)

            summary = tracker.get_experiment_summary()
            assert not summary.empty
            for col in ["run_id", "estimator", "holdout_rmse", "holdout_r2", "cv_rmse"]:
                assert col in summary.columns, f"Coluna '{col}' ausente do summary"

    def test_get_experiment_summary_excludes_loso_runs(self):
        import tempfile
        from src.tracking import log_loso_run
        from src.mlflow_integration import (
            ExperimentConfig,
            MLflowTracker,
            RegressionModelMetrics,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(tracking_uri=f"file:{td}/mlruns")
            tracker = MLflowTracker(cfg)
            tracker.setup_experiment()

            m = RegressionModelMetrics(
                estimator_name="XGBRegressor",
                params={},
                cv_metrics={"r2_mean": 0.80},
                holdout_metrics={"r2": 0.82, "rmse": 90.0, "mae": 60.0},
            )
            tracker.log_model_run(m, artifacts_dir=td)
            log_loso_run("XGBRegressor", self._make_loso_df())

            summary = tracker.get_experiment_summary()
            if "evaluation" in summary.columns:
                assert "LOSO" not in summary["evaluation"].values

    def test_orchestrator_run_full_experiment(self):
        import tempfile
        from src.mlflow_integration import (
            ExperimentConfig,
            MLflowTracker,
            MLflowExperimentOrchestrator,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(tracking_uri=f"file:{td}/mlruns")
            tracker = MLflowTracker(cfg)
            orchestrator = MLflowExperimentOrchestrator(tracker)

            configs = [
                {
                    "estimator_name": "XGBRegressor",
                    "params": {"n_estimators": 100},
                    "cv_metrics": {"r2_mean": 0.80, "rmse_mean": 95.0},
                    "holdout_metrics": {"r2": 0.82, "rmse": 90.0, "mae": 60.0},
                },
                {
                    "estimator_name": "LGBMRegressor",
                    "params": {"num_leaves": 31},
                    "cv_metrics": {"r2_mean": 0.81, "rmse_mean": 93.0},
                    "holdout_metrics": {"r2": 0.83, "rmse": 88.0, "mae": 58.0},
                },
            ]

            summary_df, run_ids = orchestrator.run_full_experiment(
                estimator_configs=configs, artifacts_dir=td
            )

            assert set(run_ids.keys()) == {"XGBRegressor", "LGBMRegressor"}
            assert all(isinstance(rid, str) for rid in run_ids.values())
            assert not summary_df.empty
