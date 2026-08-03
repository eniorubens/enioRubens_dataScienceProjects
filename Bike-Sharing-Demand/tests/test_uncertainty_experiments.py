from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from sklearn.base import clone

from src import uncertainty_reports as reports
from src.feature_engineering import (
    EXPERIMENTAL_CATEGORICAL_FEATURES,
    HourOfWeekTransformer,
    SelectiveWeatherInteractionTransformer,
    build_preprocessing_pipeline,
)
from src.probabilistic_modeling import (
    CatBoostResidualUncertaintyRegressor,
    RobustTrendProbabilisticRegressor,
    lognormal_demand_distribution,
)
from src.uncertainty_experiments import (
    UncertaintyExperimentConfig,
    _apply_e4_scale_model,
    _selected_folds,
    _residual_lag_frame,
    build_experiment_pipeline,
    frozen_artifact_hashes,
    probabilistic_fold_metrics,
    scale_diagnostics,
)


def _minimal_manifest() -> dict:
    return {
        "champion": {
            "best_params": {
                "target_strategy": "robust_trend_residual",
                "random_strength": "1",
                "encoder": "OrdinalEncoder",
                "loss_function": "MAE",
                "modeler_name": "Periodic_Spline",
                "depth": "4",
                "boosting_budget_strategy": "fixed_iterations",
                "trend_extrapolation_damping": "0.0",
                "learning_rate": "0.1",
                "l2_leaf_reg": "1.0",
                "selector": "NoSelector",
                "fixed_iterations": "100",
                "border_count": "32",
                "bagging_temperature": "0.0",
            },
            "selector": "NoSelector",
            "modeler_name": "Periodic_Spline",
            "encoder": "OrdinalEncoder",
        }
    }


def test_hour_of_week_maps_monday_zero_and_sunday_167():
    frame = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(["2024-01-01", "2024-01-07"]),
            "Hour": [0, 23],
        }
    )

    out = HourOfWeekTransformer().fit_transform(frame)

    assert list(out["HourOfWeek"].astype(int)) == [0, 167]
    assert out["HourOfWeek"].min() >= 0
    assert out["HourOfWeek"].max() <= 167


def test_hour_of_week_does_not_mutate_and_is_cloneable():
    frame = pd.DataFrame({"DateTime": pd.to_datetime(["2024-01-02"]), "Hour": [5]})
    original = frame.copy(deep=True)
    transformer = clone(HourOfWeekTransformer())

    out = transformer.fit_transform(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert "HourOfWeek" in out.columns


def test_selective_interactions_are_target_free_and_categorical(seoul_df):
    X = seoul_df.drop(columns=["Rented Bike Count"]).head(72)
    preprocessed = build_preprocessing_pipeline().fit_transform(X)
    with_hour = HourOfWeekTransformer().fit_transform(preprocessed)

    out = SelectiveWeatherInteractionTransformer().fit_transform(with_hour)

    assert "Rented Bike Count" not in out.columns
    for column in EXPERIMENTAL_CATEGORICAL_FEATURES:
        assert column in out.columns
        assert str(out[column].dtype) == "category"


def test_lognormal_distribution_has_monotonic_intervals():
    frame = lognormal_demand_distribution(
        [np.log1p(10.0), np.log1p(100.0)],
        [0.04, 0.09],
        coverages=(0.5, 0.9),
    )

    assert np.allclose(frame["demand_median"], [10.0, 100.0])
    assert (frame["demand_mean"] > frame["demand_median"]).all()
    assert (frame["lower_50"] <= frame["demand_median"]).all()
    assert (frame["demand_median"] <= frame["upper_50"]).all()
    assert (frame["lower_90"] <= frame["lower_50"]).all()
    assert (frame["upper_50"] <= frame["upper_90"]).all()


def test_catboost_uncertainty_predict_is_1d_and_distribution_has_schema():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.sin(np.arange(20, dtype=float) / 3.0)
    model = CatBoostResidualUncertaintyRegressor(iterations=2, depth=2, learning_rate=0.1)

    model.fit(X, y)
    pred = model.predict(X)
    dist = model.predict_distribution(X)

    assert pred.shape == (20,)
    assert list(dist.columns) == [
        "log_residual_location",
        "log_residual_variance",
        "log_residual_sigma",
    ]
    assert np.isfinite(dist.to_numpy()).all()
    assert (dist["log_residual_variance"] >= 0).all()


def test_build_experiment_pipeline_replays_dynamic_contract():
    config = UncertaintyExperimentConfig(
        run_mode="smoke",
        smoke_iterations=2,
        log_to_mlflow=False,
    )

    pipeline, spec = build_experiment_pipeline(_minimal_manifest(), "E1", config)

    assert spec["extra"]["experiment_id"] == "E1"
    assert spec["extra"]["uses_hour_of_week"] is True
    assert isinstance(pipeline, RobustTrendProbabilisticRegressor) is False
    assert "residual_uncertainty_features" in pipeline.estimator.steps[1][0]


def test_probabilistic_pipeline_uses_adapter_without_changing_predict_contract():
    config = UncertaintyExperimentConfig(
        run_mode="smoke",
        smoke_iterations=2,
        log_to_mlflow=False,
    )

    pipeline, spec = build_experiment_pipeline(_minimal_manifest(), "E2", config)

    assert isinstance(pipeline, RobustTrendProbabilisticRegressor)
    assert spec["extra"]["probabilistic_loss"] is True


def test_residual_lags_are_joined_by_exact_timestamp_not_row_shift():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                    "2024-01-01 03:00",
                    "2024-01-02 00:00",
                ]
            ),
            "standardized_abs_error": [10.0, 20.0, 30.0, 40.0],
        }
    )

    out = _residual_lag_frame(frame, lags=(1, 24))

    assert out.loc[1, "standardized_abs_error_lag_1"] == 10.0
    assert pd.isna(out.loc[2, "standardized_abs_error_lag_1"])
    assert out.loc[3, "standardized_abs_error_lag_24"] == 10.0


def test_residual_lags_do_not_cross_the_excluded_regime_boundary():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-12-31 23:00",
                    "2021-01-01 00:00",
                    "2021-01-01 01:00",
                ]
            ),
            "standardized_abs_error": [99.0, 10.0, 20.0],
            "selection_eligible": [False, True, True],
        }
    )

    out = _residual_lag_frame(frame, lags=(1,))

    assert pd.isna(out.loc[0, "standardized_abs_error_lag_1"])
    assert pd.isna(out.loc[1, "standardized_abs_error_lag_1"])
    assert out.loc[2, "standardized_abs_error_lag_1"] == 10.0


def test_smoke_limit_counts_normal_selection_folds_not_stress_folds():
    class DummySplitter:
        def split(self, X, y):
            for fold in range(5):
                yield np.arange(fold + 2), np.array([fold + 2])

    development = SimpleNamespace(
        X_dev=pd.DataFrame({"x": range(7)}),
        y_dev=pd.Series(range(7)),
        splitter=DummySplitter(),
        config=SimpleNamespace(test_years=(2019, 2020, 2021, 2022, 2023)),
    )
    config = UncertaintyExperimentConfig(
        run_mode="smoke",
        smoke_fold_limit=2,
        log_to_mlflow=False,
    )

    observed = [test_year for _, test_year, _, _ in _selected_folds(development, config)]

    assert observed == [2019, 2021]


def test_e4_resets_state_and_uses_fallback_until_all_lags_mature():
    prior_timestamps = pd.date_range("2019-01-01", periods=400, freq="h")
    excluded_timestamps = pd.date_range("2020-12-31", periods=24, freq="h")
    current_timestamps = pd.date_range("2021-01-01", periods=200, freq="h")
    timestamps = prior_timestamps.append(excluded_timestamps).append(current_timestamps)
    n_rows = len(timestamps)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "fold": [1] * 400 + [3] * 224,
            "selection_eligible": [True] * 400 + [False] * 24 + [True] * 200,
            "y_true": np.full(n_rows, 101.0),
            "y_pred": np.full(n_rows, 100.0),
            "demand_std": np.full(n_rows, 10.0),
            "log_location": np.full(n_rows, np.log1p(100.0)),
            "log_variance": np.full(n_rows, 0.04),
            "log_sigma": np.full(n_rows, 0.2),
            "hour": timestamps.hour,
            "weekday": timestamps.dayofweek,
        }
    )
    config = UncertaintyExperimentConfig(run_mode="smoke", log_to_mlflow=False)

    out = _apply_e4_scale_model(frame, config)
    current = out.loc[out["fold"].eq(3) & out["selection_eligible"]].reset_index(drop=True)
    excluded = out.loc[out["fold"].eq(3) & ~out["selection_eligible"]]

    assert excluded["residual_state_age_hours"].eq(-1).all()
    assert current.loc[:167, "fallback_used"].all()
    assert current.loc[168, "fallback_used"] == np.False_
    assert current.loc[0, "residual_state_age_hours"] == 0
    assert current.loc[168, "residual_state_age_hours"] == 168


def test_frozen_artifact_hashes_skips_holdout_predictions(tmp_path):
    safe = tmp_path / "manifest.json"
    forbidden = tmp_path / "holdout_predictions.csv"
    safe.write_text(json.dumps({"ok": True}), encoding="utf-8")
    forbidden.write_text("do not hash", encoding="utf-8")

    frame = frozen_artifact_hashes((tmp_path,))

    assert list(frame["path"]) == [str(safe)]


def test_probabilistic_fold_metrics_use_only_normal_selection_rows():
    config = UncertaintyExperimentConfig(
        run_mode="smoke",
        interval_coverages=(0.9,),
        log_to_mlflow=False,
    )
    predictions = pd.DataFrame(
        {
            "experiment_id": ["E2", "E2", "E2"],
            "fold": [1, 1, 2],
            "test_year": [2019, 2019, 2020],
            "fold_role": ["selection", "selection", "stress"],
            "selection_eligible": [True, False, False],
            "y_true": [10.0, 20.0, 30.0],
            "log_location": [1.0, 1.0, 1.0],
            "lower_90": [8.0, 18.0, 28.0],
            "upper_90": [12.0, 22.0, 32.0],
        }
    )

    frame = probabilistic_fold_metrics(predictions, config)

    assert list(frame["test_year"]) == [2019]
    assert list(frame["n"]) == [1]
    assert frame["empirical_coverage"].iloc[0] == 1.0


def test_probabilistic_fold_metrics_winkler_score_small_example():
    config = UncertaintyExperimentConfig(
        run_mode="smoke",
        interval_coverages=(0.9,),
        log_to_mlflow=False,
    )
    predictions = pd.DataFrame(
        {
            "experiment_id": ["E3", "E3"],
            "fold": [1, 1],
            "test_year": [2019, 2019],
            "fold_role": ["selection", "selection"],
            "selection_eligible": [True, True],
            "y_true": [10.0, 20.0],
            "log_location": [1.0, 1.0],
            "lower_90": [8.0, 22.0],
            "upper_90": [12.0, 24.0],
        }
    )

    frame = probabilistic_fold_metrics(predictions, config)

    assert frame["empirical_coverage"].iloc[0] == 0.5
    assert frame["mean_width"].iloc[0] == 3.0
    assert frame["winkler_score"].iloc[0] == pytest.approx(23.0)


def test_scale_diagnostics_floor_and_fallback_rates():
    predictions = pd.DataFrame(
        {
            "experiment_id": ["E4", "E4", "E4", "E4", "E4"],
            "fold": [1, 1, 1, 2, 2],
            "test_year": [2019, 2019, 2019, 2020, 2020],
            "fold_role": ["selection", "selection", "selection", "stress", "stress"],
            "selection_eligible": [True, True, True, False, False],
            "scale_multiplier": [0.25, 0.30, 0.25, 5.0, 5.0],
            "fallback_used": [True, False, False, False, False],
        }
    )

    frame = scale_diagnostics(predictions)

    assert list(frame["test_year"]) == [2019]
    assert frame["floor_rate"].iloc[0] == pytest.approx(2 / 3)
    assert frame["fallback_rate"].iloc[0] == pytest.approx(1 / 3)


def _report_results():
    predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-12-05", periods=168 * 4, freq="h").tolist() * 2,
            "weekday": ([0] * (168 * 4)) * 2,
            "hour": list(range(24)) * 28 * 2,
            "Rush_Period": (["Non-Rush"] * (168 * 4)) * 2,
            "Seasons": (["Winter"] * 168 + ["Spring"] * 168 + ["Summer"] * 168 + ["Autumn"] * 168)
            * 2,
            "Rainfall Cat": (["No Rain"] * (168 * 4)) * 2,
            "experiment_id": ["E0"] * (168 * 4) + ["E4"] * (168 * 4),
            "fold": [5] * (168 * 8),
            "test_year": [2023] * (168 * 8),
            "fold_role": ["selection"] * (168 * 8),
            "selection_eligible": [True] * (168 * 8),
            "y_true": np.linspace(100, 200, 168 * 8),
            "y_pred": np.linspace(90, 190, 168 * 8),
            "demand_median": np.linspace(95, 195, 168 * 8),
            "lower_90": np.linspace(50, 150, 168 * 8),
            "upper_90": np.linspace(150, 250, 168 * 8),
            "scale_multiplier": [np.nan] * (168 * 4) + [0.25] * (168 * 4),
            "fallback_used": [False] * (168 * 8),
        }
    )
    fold_metrics = pd.DataFrame(
        {
            "experiment_id": ["E0", "E1", "E2", "E3", "E4"] * 2,
            "fold": [1] * 5 + [3] * 5,
            "test_year": [2019] * 5 + [2021] * 5,
            "fold_role": ["selection"] * 10,
            "selection_mae": [1.0, 2.0, 3.0, 2.5, 2.5] * 2,
            "selection_r2": [0.9, 0.8, 0.7, 0.75, 0.75] * 2,
        }
    )
    aggregate = pd.DataFrame(
        {
            "experiment_id": ["E0", "E1", "E2", "E3", "E4"],
            "cv_mae_weighted": [1.0, 2.0, 3.0, 2.5, 2.5],
            "cv_r2_weighted": [0.9, 0.8, 0.7, 0.75, 0.75],
            "cv_r2_mean": [0.88, 0.79, 0.69, 0.74, 0.74],
        }
    )
    residual = pd.DataFrame(
        {
            "experiment_id": ["E0", "E1", "E2", "E3", "E4"],
            "residual_acf_lag_1": [0.1, 0.2, 0.3, 0.4, 0.4],
            "residual_acf_lag_24": [0.1, 0.2, 0.3, 0.4, 0.4],
            "residual_acf_lag_168": [0.1, 0.2, 0.3, 0.4, 0.4],
            "squared_residual_acf_lag_1": [0.1, 0.2, 0.3, 0.4, 0.4],
            "squared_residual_acf_lag_24": [0.1, 0.2, 0.3, 0.4, 0.4],
            "squared_residual_acf_lag_168": [0.1, 0.2, 0.3, 0.4, 0.4],
            "arch_per_obs_lag_24": [0.1, 0.2, 0.3, 0.4, 0.4],
            "arch_per_obs_lag_168": [0.1, 0.2, 0.3, 0.4, 0.4],
        }
    )
    probabilistic_by_fold = pd.DataFrame(
        {
            "experiment_id": ["E2", "E3", "E4"],
            "fold": [1, 1, 1],
            "test_year": [2019, 2019, 2019],
            "coverage": [0.9, 0.9, 0.9],
            "empirical_coverage": [0.99, 0.99, 0.81],
            "coverage_error": [0.09, 0.09, -0.09],
            "mean_width": [16000.0, 16400.0, 4500.0],
            "winkler_score": [17000.0, 16400.0, 6500.0],
            "n": [100, 100, 100],
        }
    )
    scale = pd.DataFrame(
        {
            "experiment_id": ["E4"],
            "fold": [1],
            "test_year": [2019],
            "n": [100],
            "fallback_rate": [0.1],
            "floor_rate": [0.8],
            "p10": [0.25],
            "p25": [0.25],
            "median": [0.25],
            "p75": [0.4],
            "p90": [0.7],
        }
    )
    segment = pd.DataFrame(
        {
            "experiment_id": ["E4", "E4", "E4", "E4"],
            "segment": ["Seasons", "Rush_Period", "Rainfall Cat", "predicted_demand_decile"],
            "segment_value": ["Winter", "Non-Rush", "No Rain", 0],
            "coverage_90": [0.8, 0.85, 0.82, 0.88],
            "n": [10, 10, 10, 10],
        }
    )
    return SimpleNamespace(
        config=UncertaintyExperimentConfig(run_mode="smoke", log_to_mlflow=False),
        is_smoke=False,
        predictions=predictions,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        residual_metrics=residual,
        interval_metrics=pd.DataFrame(
            {
                "experiment_id": ["E4"],
                "coverage": [0.9],
                "empirical_coverage": [0.81],
                "mean_width": [4501.5],
                "winkler_score": [6530.4],
            }
        ),
        probabilistic_fold_metrics=probabilistic_by_fold,
        scale_diagnostics=scale,
        segment_metrics=segment,
    )


@pytest.mark.parametrize(
    "factory",
    [
        reports.plot_point_metrics_by_fold,
        reports.plot_residual_diagnostics_heatmap,
        reports.plot_probabilistic_metrics_by_fold,
        reports.plot_scale_diagnostics,
        reports.plot_representative_interval_windows,
        reports.plot_segment_coverage,
    ],
)
def test_new_uncertainty_plots_return_figures_and_do_not_mutate(factory):
    result = _report_results()
    before = result.predictions.copy(deep=True)

    fig = factory(result)

    assert fig.__class__.__name__ == "Figure"
    pd.testing.assert_frame_equal(result.predictions, before)
    plt.close(fig)


def test_successor_message_does_not_call_e0_a_successor():
    result = _report_results()

    message = reports.successor_message(result)
    synthesis = reports.synthesis_report(result)

    assert "Nenhum sucessor pontual" in message
    assert "Nenhum sucessor pontual" in synthesis


@pytest.mark.parametrize(
    "path",
    [
        Path("src/uncertainty_experiments.py"),
        Path("src/probabilistic_modeling.py"),
        Path("src/uncertainty_reports.py"),
    ],
)
def test_new_modules_do_not_call_final_validation_workflow(path):
    source = path.read_text(encoding="utf-8")
    forbidden = (
        "holdout_predictions.csv",
        "materialize_final_holdout",
        "run_final_validation",
        "FinalValidationResults",
    )
    for token in forbidden:
        assert token not in source
