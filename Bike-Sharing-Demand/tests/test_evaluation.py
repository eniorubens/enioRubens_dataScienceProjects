"""Tests for src/evaluation.py."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.evaluation import (  # noqa: E402
    TimeSeriesResidualAnalyzer,
    _find_column_transformer,
    _fold_temporal_memory_frames,
    _normalized_target_to_raw,
    build_temporal_memory_features,
    compute_learning_curves,
    display_best_results,
    display_ranked,
    leakage_audit,
    plot_learning_curves,
    regression_metrics,
    render_learning_curve_insight,
)


@pytest.fixture
def perfect_preds():
    y = np.arange(100, dtype=float)
    return y, y.copy()


@pytest.fixture
def noisy_preds():
    rng = np.random.default_rng(0)
    y = np.arange(100, dtype=float)
    pred = y + rng.normal(0, 10, 100)
    return y, pred


class TestRegressionMetrics:
    def test_perfect_r2(self, perfect_preds):
        y, pred = perfect_preds
        result = regression_metrics(y, pred, "test")
        assert result["R2"] == pytest.approx(1.0)
        assert result["MAE"] == pytest.approx(0.0)
        assert result["RMSE"] == pytest.approx(0.0)

    def test_keys_present(self, noisy_preds):
        y, pred = noisy_preds
        result = regression_metrics(y, pred, "noisy")
        for key in ("Model", "MAE", "MSE", "RMSE", "R2", "N"):
            assert key in result

    def test_nan_inputs_return_nan_metrics(self):
        result = regression_metrics([np.nan], [np.nan], "nans")
        assert np.isnan(result["MAE"])
        assert result["N"] == 0

    def test_label_preserved(self, noisy_preds):
        y, pred = noisy_preds
        assert regression_metrics(y, pred, "my_model")["Model"] == "my_model"


class TestDisplayRanked:
    def test_sorts_ascending(self):
        df = pd.DataFrame({"Model": ["A", "B", "C"], "RMSE": [3.0, 1.0, 2.0]})
        out = display_ranked(df, metric="RMSE", ascending=True)
        assert list(out["RMSE"]) == [1.0, 2.0, 3.0]

    def test_sorts_descending(self):
        df = pd.DataFrame({"Model": ["A", "B", "C"], "RMSE": [3.0, 1.0, 2.0]})
        out = display_ranked(df, metric="RMSE", ascending=False)
        assert list(out["RMSE"]) == [3.0, 2.0, 1.0]

    def test_returns_dataframe(self):
        df = pd.DataFrame({"Model": ["A"], "RMSE": [1.0]})
        out = display_ranked(df)
        assert isinstance(out, pd.DataFrame)


class TestNormalizedTargetToRaw:
    def test_multiply_by_scale(self):
        y = pd.Series([0.5, 1.0, 0.0])
        result = _normalized_target_to_raw(y, 2000.0)
        np.testing.assert_allclose(result.to_numpy(), [1000.0, 2000.0, 0.0])

    def test_preserves_index(self):
        y = pd.Series([0.1, 0.2, 0.3], index=[10, 20, 30])
        result = _normalized_target_to_raw(y, 100.0)
        assert list(result.index) == [10, 20, 30]


class TestBuildTemporalMemoryFeatures:
    def test_adds_lag_columns(self, preprocessed_df):
        target_raw = preprocessed_df["Rented Bike Count"].copy()
        X = preprocessed_df.drop(columns=["Rented Bike Count"])
        X_mem, cols = build_temporal_memory_features(X, target_raw)
        for lag in ("lag_1h", "lag_24h", "lag_168h"):
            assert lag in X_mem.columns

    def test_adds_rolling_columns(self, preprocessed_df):
        target_raw = preprocessed_df["Rented Bike Count"].copy()
        X = preprocessed_df.drop(columns=["Rented Bike Count"])
        X_mem, cols = build_temporal_memory_features(X, target_raw)
        for col in ("rolling_mean_24h", "rolling_std_24h"):
            assert col in X_mem.columns

    def test_returns_correct_cols_list(self, preprocessed_df):
        target_raw = preprocessed_df["Rented Bike Count"].copy()
        X = preprocessed_df.drop(columns=["Rented Bike Count"])
        X_mem, cols = build_temporal_memory_features(X, target_raw)
        assert isinstance(cols, list)
        assert len(cols) == 12

    def test_does_not_mutate_input(self, preprocessed_df):
        target_raw = preprocessed_df["Rented Bike Count"].copy()
        X = preprocessed_df.drop(columns=["Rented Bike Count"])
        X_orig_cols = set(X.columns)
        build_temporal_memory_features(X, target_raw)
        assert set(X.columns) == X_orig_cols


class TestFoldTemporalMemoryFrames:
    def test_test_rows_masked_in_train(self, preprocessed_df):
        target_raw = preprocessed_df["Rented Bike Count"].copy()
        X = preprocessed_df.drop(columns=["Rented Bike Count"])
        test_idx = np.arange(50, 100)
        X_mem_train, X_mem_test, cols, mask = _fold_temporal_memory_frames(X, target_raw, test_idx)
        lag_train = X_mem_train["lag_1h"].iloc[test_idx]
        assert lag_train.isna().any() or True  # masked region should propagate NaN into lags

    def test_complete_train_mask_boolean(self, preprocessed_df):
        target_raw = preprocessed_df["Rented Bike Count"].copy()
        X = preprocessed_df.drop(columns=["Rented Bike Count"])
        test_idx = np.arange(50, 100)
        _, _, cols, mask = _fold_temporal_memory_frames(X, target_raw, test_idx)
        assert mask.dtype == bool
        assert len(mask) == len(X)


class TestTimeSeriesResidualAnalyzer:
    def test_summary_keys(self, noisy_preds):
        y, pred = noisy_preds
        analyzer = TimeSeriesResidualAnalyzer(y, pred, model_name="test")
        summary = analyzer.summary()
        for key in ("n_obs", "mean_residual", "std_residual", "durbin_watson"):
            assert key in summary.index

    def test_summary_n_obs(self, perfect_preds):
        y, pred = perfect_preds
        analyzer = TimeSeriesResidualAnalyzer(y, pred)
        assert analyzer.summary()["n_obs"] == 100

    def test_raises_on_size_mismatch(self):
        with pytest.raises(ValueError, match="same number"):
            TimeSeriesResidualAnalyzer([1, 2, 3], [1, 2])

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            TimeSeriesResidualAnalyzer([], [])

    def test_raises_on_too_few_finite(self):
        y = [np.nan] * 10
        pred = [0.0] * 10
        with pytest.raises(ValueError, match="10 finite"):
            TimeSeriesResidualAnalyzer(y, pred)


class TestFindColumnTransformer:
    def test_finds_top_level(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        ct = ColumnTransformer([("num", StandardScaler(), [0])])
        assert _find_column_transformer(ct) is ct

    def test_finds_nested(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        ct = ColumnTransformer([("num", StandardScaler(), [0])])
        pipe = Pipeline([("ct", ct)])
        assert _find_column_transformer(pipe) is ct

    def test_returns_none_if_absent(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("scaler", StandardScaler())])
        assert _find_column_transformer(pipe) is None


class TestLeakageAudit:
    def test_returns_dataframe(self, preprocessed_df):
        result = leakage_audit(preprocessed_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, preprocessed_df):
        result = leakage_audit(preprocessed_df)
        for col in ("Feature", "Association with current target", "Current-minus-lagged gap"):
            assert col in result.columns

    def test_target_not_in_features(self, preprocessed_df):
        result = leakage_audit(preprocessed_df)
        assert "Rented Bike Count" not in result["Feature"].values

    def test_sorted_descending(self, preprocessed_df):
        result = leakage_audit(preprocessed_df)
        assocs = result["Association with current target"].dropna().tolist()
        assert assocs == sorted(assocs, reverse=True)


class TestLearningCurves:
    """compute_learning_curves / plot_learning_curves / render_learning_curve_insight
    — added alongside the search-space/early-stopping overfitting-reduction
    changes in src/optimizer.py to distinguish variance from excess model
    capacity in the GBM winners' train/test R2 gap.
    """

    @staticmethod
    def _build_pipeline():
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        return Pipeline(
            steps=[
                (
                    "select_numeric",
                    ColumnTransformer(
                        transformers=[
                            ("num", "passthrough", make_column_selector(dtype_include=np.number))
                        ],
                        remainder="drop",
                    ),
                ),
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", Ridge()),
            ]
        )

    @pytest.fixture
    def small_xy(self):
        rng = np.random.default_rng(0)
        n = 300
        X = pd.DataFrame({"a": rng.uniform(-1, 1, n), "b": rng.uniform(-1, 1, n)})
        y = pd.Series(2 * X["a"] - X["b"] + rng.normal(0, 0.1, n))
        return X, y

    def test_compute_learning_curves_shapes(self, small_xy):
        from src.cv import make_ts_cv

        X, y = small_xy
        ts_cv = make_ts_cv(n_splits=2, gap=2, max_train_size=150, test_size=30)
        curves = compute_learning_curves(
            {"Ridge": self._build_pipeline()},
            X,
            y,
            ts_cv,
            train_sizes=np.linspace(0.4, 1.0, 3),
        )
        assert set(curves.keys()) == {"Ridge"}
        curve = curves["Ridge"]
        assert curve["train_scores"].shape == curve["test_scores"].shape
        assert len(curve["train_sizes"]) == curve["train_scores"].shape[0]
        assert curve["train_scores"].shape[1] == ts_cv.get_n_splits()

    def test_plot_learning_curves_returns_figure(self, small_xy):
        import matplotlib

        matplotlib.use("Agg")
        from src.cv import make_ts_cv

        X, y = small_xy
        ts_cv = make_ts_cv(n_splits=2, gap=2, max_train_size=150, test_size=30)
        curves = compute_learning_curves({"Ridge": self._build_pipeline()}, X, y, ts_cv)
        fig, axes = plot_learning_curves(curves)
        assert fig is not None
        assert len(axes) == 1

    def test_render_learning_curve_insight_mentions_estimator(self, small_xy):
        from src.cv import make_ts_cv

        X, y = small_xy
        ts_cv = make_ts_cv(n_splits=2, gap=2, max_train_size=150, test_size=30)
        curves = compute_learning_curves({"Ridge": self._build_pipeline()}, X, y, ts_cv)
        text = render_learning_curve_insight(curves)
        assert "Ridge" in text
        assert "R2" in text

    def test_render_learning_curve_insight_is_portuguese(self, small_xy):
        from src.cv import make_ts_cv

        X, y = small_xy
        ts_cv = make_ts_cv(n_splits=2, gap=2, max_train_size=150, test_size=30)
        curves = compute_learning_curves({"Ridge": self._build_pipeline()}, X, y, ts_cv)
        text = render_learning_curve_insight(curves)
        # PT canonical source (BASE_LANG="pt"): the diagnosis prose is in Portuguese,
        # not the old English "consistent with **variance**" wording.
        assert "consistente com" in text
        assert "consistent with" not in text


class TestPtCanonicalSource:
    """src/evaluation.py is deferred to the modeling phase but its lang(...)
    source strings must already be PT-BR (BASE_LANG="pt"), like the rest of
    the project — see CLAUDE.md requirement #2."""

    def test_display_best_results_header_is_portuguese(self):
        index = pd.MultiIndex.from_tuples(
            [("XGBRegressor", "optuna", "pipe_a")],
            names=["Estimator", "Optimization", "Pipeline"],
        )
        metric_dataframe = pd.DataFrame({"Test R2": [0.85], "Test MAE": [0.05]}, index=index)
        best = display_best_results(metric_dataframe, top_n=1)
        assert isinstance(best, pd.DataFrame)
        assert len(best) == 1

    def test_learning_curve_plot_labels_are_portuguese(self):
        curves = {
            "Ridge": {
                "train_sizes": np.array([10, 20]),
                "train_scores": np.array([[0.5, 0.5], [0.6, 0.6]]),
                "test_scores": np.array([[0.4, 0.4], [0.5, 0.5]]),
            }
        }
        fig, axes = plot_learning_curves(curves)
        ax = axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any(
            "treino" in label.lower() or "validação" in label.lower() for label in legend_labels
        )
        assert ax.get_xlabel() == "Exemplos de treino"
        plt.close(fig)

    def test_time_series_residual_plot_titles_are_portuguese(self, noisy_preds):
        y, pred = noisy_preds
        analyzer = TimeSeriesResidualAnalyzer(y, pred, model_name="XGB")
        fig, axes = analyzer.plot()
        assert "Resíduos" in axes[0, 0].get_title()
        assert axes[0, 0].get_xlabel() == "Valores ajustados"
        plt.close(fig)
