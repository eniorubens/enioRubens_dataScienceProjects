"""Tests for src/modeling_pipeline.py — the dynamic, estimator-conditioned contract.

These cover the architectural guarantees rather than the numbers: that the
representation space depends on the estimator family, that every strategy is
buildable and fittable, that changing the strategy really changes the
pipeline's structure, and that every trainable step — imputer, scaler,
normalizer, encoder, selector, target transformer — sees the training fold
and nothing else.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import FeatureUnion

import src.modeling_pipeline as mp
from src.modeling_pipeline import (
    ALL_MODELERS,
    CATEGORICAL_FEATURES,
    HIGH_DIMENSIONAL_MODELERS,
    NUMERICAL_FEATURES,
    KernelExpansion,
    PipelineSpec,
    build_dynamic_pipeline,
    build_encoder,
    estimator_family,
    get_feature_selection,
    get_target_transformer,
    modeler_space,
    selector_space,
    validate_combination,
    weekday_indicator,
)


def _fixed(params: dict) -> optuna.trial.FixedTrial:
    return optuna.trial.FixedTrial(params)


def _ridge_params(**overrides) -> dict:
    params = {
        "modeler_name": "linear_modeling",
        "encoder": "OrdinalEncoder",
        "normalizer": "QuantileUniform",
        "alpha": 1.0,
        "selector": "NoSelector",
        "kbest_k": 8,
        "kbest_score_func": "f_regression",
        "target_transform": "none",
    }
    params.update(overrides)
    return params


@pytest.fixture(scope="module")
def small_fold(request):
    """A short train/test slice of the shared synthetic v4 frame."""
    X_dev, y_dev, splitter = request.getfixturevalue("dev_split_v4")
    train_idx, test_idx = next(splitter.split(X_dev))
    return X_dev, y_dev, train_idx[-600:], test_idx[:200]


# ---------------------------------------------------------------------------
# (a) Different estimator families receive different modeler_name spaces
# ---------------------------------------------------------------------------


class TestModelerSpacesPerFamily:
    def test_linear_and_tree_families_get_different_spaces(self):
        linear = modeler_space("Ridge")
        xgboost = modeler_space("XGBRegressor")
        baseline = modeler_space("DummyRegressor")
        assert linear != xgboost
        assert xgboost != baseline
        assert linear != baseline

    def test_linear_family_receives_every_strategy(self):
        assert set(modeler_space("Ridge")) == set(ALL_MODELERS)

    def test_tree_families_exclude_expansion_strategies(self):
        for estimator in (
            "XGBRegressor",
            "LGBMRegressor",
            "HistGradientBoostingRegressor",
            "RandomForestRegressor",
            "CatBoostRegressor",
        ):
            space = set(modeler_space(estimator))
            assert not space & HIGH_DIMENSIONAL_MODELERS
            assert space == {
                "linear_modeling",
                "Sin_Cos",
                "Time_steps_as_categories",
                "Periodic_Spline",
            }

    def test_families_are_declared_explicitly(self):
        assert estimator_family("Ridge") == mp.FAMILY_LINEAR
        assert estimator_family("XGBRegressor") == mp.FAMILY_XGBOOST
        with pytest.raises(ValueError, match="Unknown estimator"):
            estimator_family("SomeFutureRegressor")

    def test_scaler_space_is_also_family_conditioned(self):
        _, linear_name = mp.get_standardization(_fixed({}), "Ridge")
        assert linear_name == "StandardScaler"  # pinned, no sampling at all
        _, tree_name = mp.get_standardization(
            _fixed({"standardizer": "RobustScaler"}), "XGBRegressor"
        )
        assert tree_name == "RobustScaler"


# ---------------------------------------------------------------------------
# (b) Every strategy builds and fits a minimal pipeline
# ---------------------------------------------------------------------------


class TestEveryStrategyIsBuildable:
    @pytest.mark.parametrize("modeler_name", list(ALL_MODELERS))
    def test_strategy_builds_fits_and_predicts(self, small_fold, modeler_name):
        X_dev, y_dev, train_idx, test_idx = small_fold
        trial = _fixed(_ridge_params(modeler_name=modeler_name, selector="SelectKBest"))
        pipeline, spec = build_dynamic_pipeline(trial, "Ridge")

        pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        predictions = pipeline.predict(X_dev.iloc[test_idx])

        assert spec.modeler_name == modeler_name
        assert len(predictions) == len(test_idx)
        assert np.isfinite(predictions).all()

    @pytest.mark.parametrize("encoder_name", list(mp.ENCODER_SPACE))
    def test_every_encoder_survives_unseen_categories(self, small_fold, encoder_name):
        """A temporal split guarantees categories absent from the training
        fold; no encoder may answer that with NaN, because nothing downstream
        imputes after the ColumnTransformer any more."""
        X_dev, y_dev, train_idx, test_idx = small_fold
        trial = _fixed(_ridge_params(encoder=encoder_name))
        pipeline, _ = build_dynamic_pipeline(trial, "Ridge")
        pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        assert np.isfinite(pipeline.predict(X_dev.iloc[test_idx])).all()


# ---------------------------------------------------------------------------
# (c) The pipeline changes structurally when modeler_name changes
# ---------------------------------------------------------------------------


class TestStructureFollowsModelerName:
    def _modeling_step(self, modeler_name):
        trial = _fixed(_ridge_params(modeler_name=modeler_name, selector="SelectKBest"))
        pipeline, _ = build_dynamic_pipeline(trial, "Ridge")
        return pipeline.named_steps["modeling"]

    def test_branch_counts_and_types_differ(self):
        linear = self._modeling_step("linear_modeling")
        periodic = self._modeling_step("Periodic_Spline")
        sin_cos = self._modeling_step("Sin_Cos")
        pairwise = self._modeling_step("Pairwise_Interactions")
        kernels = self._modeling_step("Interactions_with_Kernels")

        assert isinstance(linear, ColumnTransformer) and len(linear.transformers) == 2
        assert isinstance(periodic, ColumnTransformer) and len(periodic.transformers) == 4
        assert isinstance(sin_cos, ColumnTransformer) and len(sin_cos.transformers) == 6
        assert isinstance(pairwise, FeatureUnion)
        assert isinstance(kernels, KernelExpansion)

    def test_polynomial_adds_a_polynomial_step_to_the_numeric_branch(self):
        polynomial = self._modeling_step("Polynomial")
        branches = {name: transformer for name, transformer, _ in polynomial.transformers}
        numeric_branch = branches["numeric"]
        assert "polynomial" in numeric_branch.named_steps
        assert "normalizer" in numeric_branch.named_steps

    def test_time_steps_as_categories_routes_month_and_hour_to_the_encoder(self):
        step = self._modeling_step("Time_steps_as_categories")
        categorical_columns = dict((name, cols) for name, _, cols in step.transformers)[
            "categorical"
        ]
        assert "Hour" in categorical_columns and "Month" in categorical_columns
        numeric_columns = dict((name, cols) for name, _, cols in step.transformers)["numeric"]
        assert "Hour" not in numeric_columns and "Month" not in numeric_columns

    def test_weather_columns_survive_every_periodic_strategy(self):
        """The v3 branches declared remainder='drop' while listing only the
        categorical columns and the re-encoded time, silently discarding every
        weather measurement."""
        for modeler_name in ("Sin_Cos", "Time_steps_as_categories", "Periodic_Spline"):
            step = self._modeling_step(modeler_name)
            numeric_columns = dict((name, cols) for name, _, cols in step.transformers)["numeric"]
            assert "Temperature(C)" in numeric_columns
            assert "Humidity(%)" in numeric_columns


# ---------------------------------------------------------------------------
# Combination validation stays out of the Optuna distributions
# ---------------------------------------------------------------------------


class TestCombinationValidation:
    def test_selector_space_does_not_depend_on_modeler_name(self):
        """Narrowing the selector list by the already-sampled representation
        would give Optuna a per-trial candidate list, which it rejects as a
        dynamic value space."""
        import inspect

        source = inspect.signature(selector_space)
        assert list(source.parameters) == ["estimator_name", "search_profile"]

    def test_high_dimensional_representation_rejects_no_selector(self):
        spec = PipelineSpec(
            estimator="Ridge",
            family="linear",
            modeler_name="Polynomial",
            encoder="OrdinalEncoder",
            scaler="StandardScaler",
            selector="NoSelector",
            target_transform="none",
        )
        assert "NoSelector is not allowed" in validate_combination(spec)

    def test_ordinary_combination_is_accepted(self):
        spec = PipelineSpec(
            estimator="Ridge",
            family="linear",
            modeler_name="linear_modeling",
            encoder="OrdinalEncoder",
            scaler="StandardScaler",
            selector="NoSelector",
            target_transform="none",
        )
        assert validate_combination(spec) is None


# ---------------------------------------------------------------------------
# (g) Feature selection: capability-aware spaces and a temporal inner CV
# ---------------------------------------------------------------------------


class TestFeatureSelection:
    def test_sequential_selector_uses_a_temporal_inner_cv(self):
        selector, name = get_feature_selection(
            _fixed({"selector": "SequentialFeatureSelector", "sfs_n_features": 5}),
            "Ridge",
            Ridge(),
        )
        assert name == "SequentialFeatureSelector"
        assert isinstance(selector, SequentialFeatureSelector)
        assert isinstance(selector.cv, TimeSeriesSplit)
        assert selector.cv.gap == 48

    def test_model_based_selectors_are_withheld_from_hgb(self):
        """HistGradientBoostingRegressor exposes neither coef_ nor
        feature_importances_, so RFE/SelectFromModel cannot read it."""
        space = selector_space("HistGradientBoostingRegressor")
        assert "RFE" not in space and "SelectFromModel" not in space
        assert space == ["NoSelector", "SelectKBest"]

    def test_model_based_selectors_are_offered_where_supported(self):
        assert "RFE" in selector_space("Ridge")
        assert "SelectFromModel" in selector_space("XGBRegressor")

    def test_baseline_has_nothing_to_select(self):
        assert selector_space("DummyRegressor") == ["NoSelector"]


# ---------------------------------------------------------------------------
# (e)/(f) Every trainable step is fitted on the training fold only
# ---------------------------------------------------------------------------


class TestFittedOnTrainingFoldOnly:
    def test_numeric_scaler_sees_exactly_the_training_rows(self, small_fold):
        X_dev, y_dev, train_idx, _ = small_fold
        trial = _fixed(_ridge_params(target_transform="standard"))
        pipeline, _ = build_dynamic_pipeline(trial, "Ridge")
        pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])

        scaler = (
            pipeline.named_steps["modeling"].named_transformers_["numeric"].named_steps["scaler"]
        )
        assert int(np.atleast_1d(scaler.n_samples_seen_)[0]) == len(train_idx)

    def test_numeric_imputer_uses_the_training_fold_median(self, small_fold):
        X_dev, y_dev, train_idx, _ = small_fold
        trial = _fixed(_ridge_params())
        pipeline, _ = build_dynamic_pipeline(trial, "Ridge")
        pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])

        imputer = (
            pipeline.named_steps["modeling"].named_transformers_["numeric"].named_steps["imputer"]
        )
        column = "Sunshine (hr)"
        position = NUMERICAL_FEATURES.index(column)
        train_median = X_dev.iloc[train_idx][column].median()
        full_median = X_dev[column].median()

        assert imputer.statistics_[position] == pytest.approx(train_median)
        assert imputer.statistics_[position] != pytest.approx(full_median)

    def test_target_transformer_is_fitted_on_the_training_fold(self, small_fold):
        X_dev, y_dev, train_idx, test_idx = small_fold
        trial = _fixed(_ridge_params(target_transform="standard"))
        pipeline, _ = build_dynamic_pipeline(trial, "Ridge")
        pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])

        fitted = pipeline.named_steps["regressor"].transformer_
        assert fitted.mean_[0] == pytest.approx(y_dev.iloc[train_idx].mean())
        assert fitted.mean_[0] != pytest.approx(y_dev.iloc[test_idx].mean())

    def test_normalizer_sees_exactly_the_training_rows(self, small_fold):
        X_dev, y_dev, train_idx, _ = small_fold
        trial = _fixed(_ridge_params(modeler_name="Normalizers", selector="SelectKBest"))
        pipeline, _ = build_dynamic_pipeline(trial, "Ridge")
        pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])

        normalizer = (
            pipeline.named_steps["modeling"]
            .named_transformers_["numeric"]
            .named_steps["normalizer"]
        )
        assert normalizer.n_quantiles_ <= len(train_idx)
        assert normalizer.references_.size > 0

    def test_selector_is_fitted_inside_the_pipeline(self, small_fold):
        X_dev, y_dev, train_idx, _ = small_fold
        trial = _fixed(_ridge_params(selector="SelectKBest", kbest_k=7))
        pipeline, _ = build_dynamic_pipeline(trial, "Ridge")
        pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])

        selector = pipeline.named_steps["selector"]
        assert selector.get_support().sum() == 7

    def test_mean_encoder_ignores_the_validation_fold_target(self, small_fold):
        """Perturbing y only outside the training fold must leave the encoder's
        learned mapping bit-for-bit identical."""
        X_dev, y_dev, train_idx, test_idx = small_fold
        trial_params = _ridge_params(encoder="MeanEncoder")

        y_high = y_dev.copy()
        y_high.iloc[test_idx] = 99999.0
        y_low = y_dev.copy()
        y_low.iloc[test_idx] = -99999.0

        mappings = []
        for perturbed in (y_high, y_low):
            pipeline, _ = build_dynamic_pipeline(_fixed(trial_params), "Ridge")
            pipeline.fit(X_dev.iloc[train_idx], perturbed.iloc[train_idx])
            encoder = (
                pipeline.named_steps["modeling"]
                .named_transformers_["categorical"]
                .named_steps["encoder"]
            )
            mappings.append(encoder.encoder_dict_)

        assert mappings[0] == mappings[1]


# ---------------------------------------------------------------------------
# Branch-local imputation and the corrected weekday indicator
# ---------------------------------------------------------------------------


class TestBranchImputation:
    def test_categorical_branch_fills_with_the_english_missing_token(self):
        branch = mp._categorical_branch(build_encoder("OrdinalEncoder", ["cat_feat"]))
        # NaN, not None: pandas Categorical -> .astype("object") yields a float
        # NaN, and SimpleImputer's default missing_values is np.nan.
        frame = pd.DataFrame({"cat_feat": ["A", np.nan, "B"]})
        imputed = branch[:2].fit_transform(frame)
        assert imputed["cat_feat"].tolist() == ["A", "Missing", "B"]

    def test_numeric_branch_fills_with_the_median(self):
        from sklearn.preprocessing import StandardScaler

        branch = mp._numeric_branch(StandardScaler())
        frame = pd.DataFrame({"num_feat": [1.0, np.nan, 3.0]})
        imputed = branch[:1].fit_transform(frame)
        assert imputed["num_feat"].tolist() == [1.0, 2.0, 3.0]

    def test_no_global_imputer_after_the_column_transformer(self, small_fold):
        X_dev, y_dev, train_idx, _ = small_fold
        pipeline, _ = build_dynamic_pipeline(_fixed(_ridge_params()), "Ridge")
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == [
            "feature_engineering",
            "elapsed_hours",
            "modeling",
            "selector",
            "regressor",
        ]


class TestWeekdayIndicator:
    def test_reads_week_status_not_the_day_name_column(self):
        """The v3 helper compared the Weekday column — which holds day *names*
        — with the literal 'Weekday', so the indicator was always zero."""
        frame = pd.DataFrame(
            {
                "WeekStatus": ["Weekday", "Weekend", "Weekday"],
                "Weekday": ["Monday", "Sunday", "Tuesday"],
            }
        )
        result = weekday_indicator(frame)
        assert result.iloc[:, 0].tolist() == [1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Target transformer stays independent of the feature scaler
# ---------------------------------------------------------------------------


class TestTargetTransformerIndependence:
    def test_target_and_feature_transforms_are_sampled_separately(self):
        trial = _fixed({"target_transform": "log1p", "standardizer": "RobustScaler"})
        target, target_name = get_target_transformer(trial)
        scaler, scaler_name = mp.get_standardization(trial, "XGBRegressor")
        assert target_name == "log1p" and scaler_name == "RobustScaler"
        assert target.func is np.log1p and target.inverse_func is np.expm1
        assert target is not scaler

    def test_none_is_an_identity_target_transform(self):
        transformer, name = get_target_transformer(_fixed({"target_transform": "none"}))
        assert transformer is None and name == "none"

    def test_refined_catboost_alone_can_use_the_raw_target(self):
        params = {
            "modeler_name": "Periodic_Spline",
            "encoder": "OrdinalEncoder",
            "loss_function": "RMSE",
            "boosting_budget_strategy": "fixed_iterations",
            "fixed_iterations": 283,
            "depth": 10,
            "learning_rate": 0.1,
            "random_strength": 10,
            "bagging_temperature": 0.5,
            "l2_leaf_reg": 1.0,
            "border_count": 64,
            "selector": "NoSelector",
            "target_transform": "none",
        }
        _, spec = build_dynamic_pipeline(
            _fixed(params),
            "CatBoostRegressor",
            search_profile=mp.SEARCH_PROFILE_REFINED,
            target_strategy=mp.TARGET_STRATEGY_DIRECT,
        )
        assert spec.target_transform == "none"

    def test_refined_non_catboost_direct_target_remains_log1p(self):
        params = {
            "modeler_name": "linear_modeling",
            "encoder": "JamesSteinEncoder",
            "loss_function": "reg:squarederror",
            "boosting_budget_strategy": "temporal_early_stopping",
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "min_child_weight": 2.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "selector": "NoSelector",
        }
        _, spec = build_dynamic_pipeline(
            _fixed(params),
            "XGBRegressor",
            search_profile=mp.SEARCH_PROFILE_REFINED,
            target_strategy=mp.TARGET_STRATEGY_DIRECT,
        )
        assert spec.target_transform == "log1p"


# ---------------------------------------------------------------------------
# PipelineSpec is the record written to MLflow and to the manifest
# ---------------------------------------------------------------------------


class TestPipelineSpec:
    def test_spec_records_every_dynamic_choice(self):
        trial = _fixed(
            _ridge_params(
                modeler_name="Normalizers",
                encoder="MeanEncoder",
                selector="SelectKBest",
                target_transform="log1p",
            )
        )
        _, spec = build_dynamic_pipeline(trial, "Ridge")
        tags = spec.as_tags()
        assert tags["modeler_name"] == "Normalizers"
        assert tags["encoder"] == "MeanEncoder"
        assert tags["scaler"] == "StandardScaler"
        assert tags["normalizer"] == "QuantileUniform"
        assert tags["selector"] == "SelectKBest"
        assert tags["target_transform"] == "log1p"
        assert tags["family"] == "linear"

    def test_normalizer_is_absent_when_the_strategy_does_not_use_one(self):
        _, spec = build_dynamic_pipeline(_fixed(_ridge_params()), "Ridge")
        assert spec.normalizer is None
        assert "normalizer" not in spec.as_tags()

    def test_default_feature_lists_are_the_shared_v4_lists(self):
        _, spec = build_dynamic_pipeline(_fixed(_ridge_params()), "Ridge")
        assert "Elapsed_Hours" in NUMERICAL_FEATURES
        assert "Rush_Period" in CATEGORICAL_FEATURES
        assert "Year" not in NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        assert "is_anomalous_2020" not in NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        assert spec.estimator == "Ridge"


class TestRefinedTrendResidualProfile:
    def test_auto_target_can_build_and_fit_the_trend_residual_branch(self, small_fold):
        X, y, train_idx, test_idx = small_fold
        params = {
            "modeler_name": "Periodic_Spline",
            "encoder": "MeanEncoder",
            "loss_function": "absolute_error",
            "learning_rate": 0.08,
            "max_iter": 80,
            "max_leaf_nodes": 31,
            "max_depth": 5,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
            "selector": "NoSelector",
            "target_strategy": "robust_trend_residual",
            "trend_extrapolation_damping": 0.5,
        }
        pipeline, spec = build_dynamic_pipeline(
            _fixed(params),
            "HistGradientBoostingRegressor",
            search_profile="refined",
            target_strategy="auto",
        )

        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        prediction = pipeline.predict(X.iloc[test_idx])

        assert isinstance(pipeline, mp.RobustTrendResidualRegressor)
        assert spec.target_transform == "robust_trend_residual"
        assert spec.selector == "NoSelector"
        assert np.isfinite(prediction).all()
        assert (prediction >= 0).all()
