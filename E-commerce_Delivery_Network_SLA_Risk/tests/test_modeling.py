import numpy as np
import optuna
import pandas as pd
import pytest

import ecom_delivery_sla.modeling as modeling
from ecom_delivery_sla.modeling import (
    build_classification_tournament,
    build_modeling_frame,
    build_regression_tournament,
    classification_baseline_scores,
    classification_metrics,
    fit_logistic_classification,
    fit_ols_regression,
    fit_quantile_regressors,
    fit_ridge_regression,
    predict_ols,
    prepare_model_matrices,
    regression_baseline_predictions,
    regression_metrics,
    run_classification_tournament,
    run_classification_tournament_folds,
    run_regression_tournament,
    run_regression_tournament_folds,
    tournament_summary,
)
from ecom_delivery_sla.network import NETWORK_FEATURE_COLUMNS
from ecom_delivery_sla.optimization import build_optuna_pipeline, run_optuna_study


def _feature_frame() -> pd.DataFrame:
    frame = _base_feature_frame()
    for number, column in enumerate(NETWORK_FEATURE_COLUMNS, start=1):
        frame[column] = np.arange(4, dtype=float) * 0.1 * number + number
    frame.loc[3, "seller_days_since_last"] = np.nan  # exercised by the imputer
    return frame


def _base_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "promised_lead_days": [10.0, 12.0, 8.0, 15.0],
            "item_count": [1, 2, 1, 3],
            "distinct_product_count": [1, 2, 1, 2],
            "distinct_category_count": [1, 1, 1, 2],
            "seller_count": [1, 1, 1, 2],
            "origin_state_count": [1, 1, 1, 2],
            "total_price": [10.0, 20.0, 15.0, 40.0],
            "total_freight_value": [2.0, 3.0, 2.0, 5.0],
            "mean_item_price": [10.0, 10.0, 15.0, 13.3],
            "payment_count": [1, 1, 1, 1],
            "payment_installments_max": [1, 2, 1, 3],
            "primary_payment_type": ["credit_card", "boleto", "credit_card", "boleto"],
            "distance_mean_km": [100.0, 200.0, 150.0, 300.0],
            "total_weight_g": [500.0, 1000.0, 700.0, 1500.0],
            "max_weight_g": [500.0, 700.0, 700.0, 1000.0],
            "total_volume_cm3": [1000.0, 2000.0, 1500.0, 3000.0],
            "freight_per_kg": [4.0, 3.0, 2.86, 3.33],
            "seller_state": ["SP", "RJ", "SP", "MG"],
            "same_state_share": [1.0, 0.0, 1.0, 0.5],
            "customer_is_capital": [1, 0, 1, 0],
            "purchase_month": [1, 1, 1, 1],
            "purchase_dayofweek": [0, 1, 2, 3],
            "purchase_hour": [10, 11, 12, 13],
            "days_to_black_friday": [327, 326, 325, 324],
            "days_to_christmas": [358, 357, 356, 355],
            "customer_state": ["SP", "RJ", "SP", "MG"],
            "order_purchase_timestamp": pd.date_range("2018-01-01", periods=4),
        }
    )


def test_modeling_frame_attaches_targets_without_changing_grain():
    features = _feature_frame()
    targets = pd.DataFrame(
        {
            "order_id": features["order_id"],
            "delivery_delay_days": [1.0, -2.0, 3.0, 0.0],
            "sla_breach": [True, False, True, False],
        }
    )

    result = build_modeling_frame(features, targets)

    assert len(result) == 4
    assert result["order_id"].is_unique
    assert {"delivery_delay_days", "sla_breach"}.issubset(result.columns)


def test_preprocessing_fits_and_transforms_temporal_splits():
    features = _feature_frame()
    splits = {
        "train": features.iloc[:2],
        "validation": features.iloc[2:3],
        "test": features.iloc[3:],
    }

    matrices, preprocessor = prepare_model_matrices(splits)

    assert set(matrices) == {"train", "validation", "test"}
    assert matrices["train"].shape[1] == matrices["test"].shape[1]
    assert matrices["train"].shape[1] < len(features.columns)
    assert preprocessor is not None


def test_baseline_models_and_metrics_return_expected_values():
    X_train = pd.DataFrame(
        {
            "feature_a": np.arange(20, dtype=float),
            "feature_b": np.linspace(0.0, 1.0, 20),
            "feature_c": np.sin(np.arange(20, dtype=float)),
        }
    )
    y_reg = pd.Series(0.5 * X_train["feature_a"] + X_train["feature_b"])
    y_clf = pd.Series([0, 1] * 10)

    ridge = fit_ridge_regression(X_train, y_reg)
    logistic = fit_logistic_classification(X_train, y_clf)
    regression = regression_metrics(y_reg, ridge.predict(X_train))
    classification = classification_metrics(y_clf, logistic.predict_proba(X_train)[:, 1])

    assert ridge.predict(X_train.iloc[:1]).shape == (1,)
    assert set(regression) == {"mae", "rmse", "r2"}
    assert {"roc_auc", "average_precision", "balanced_accuracy", "brier", "ece_10bins"}.issubset(classification)
    assert np.isfinite(list(regression.values())).all()


def test_baseline_predictions_and_quantiles_use_train_contract():
    features = _feature_frame().copy()
    features["actual_lead_days"] = [8.0, 10.0, 6.0, 14.0]
    features["delivery_delay_days"] = [-2.0, -2.0, -2.0, -1.0]
    features["sla_breach"] = [False, True, False, False]
    reg = regression_baseline_predictions(features.iloc[:3], features.iloc[3:])
    clf = classification_baseline_scores(features.iloc[:3], features.iloc[3:])
    quantiles = fit_quantile_regressors(
        features[["promised_lead_days", "item_count"]].iloc[:3],
        features["actual_lead_days"].iloc[:3],
        quantiles=(0.5,),
    )

    assert set(reg) == {"train_mean", "median_actual_minus_promise", "state_median_actual_minus_promise"}
    assert set(clf) == {"prevalence", "negative_promised_lead_days", "logistic_promise_plus_state"}
    assert set(quantiles) == {0.5}


def test_ols_wrapper_adds_intercept_and_delegates(monkeypatch):
    captured = {}

    class FakeResult:
        pass

    class FakeOLS:
        def __init__(self, endog, exog):
            captured["endog"] = endog
            captured["exog"] = exog

        def fit(self):
            return FakeResult()

    monkeypatch.setattr(modeling.sm, "OLS", FakeOLS)
    X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y_train = pd.Series([0.0, 1.0, 2.0])

    result = fit_ols_regression(X_train, y_train)

    assert isinstance(result, FakeResult)
    assert "const" in captured["exog"].columns


def test_predict_ols_uses_the_same_intercept_convention():
    class FakeModel:
        def predict(self, design):
            assert "const" in design.columns
            return np.ones(len(design))

    predictions = predict_ols(FakeModel(), pd.DataFrame({"feature": [1.0, 2.0]}))

    assert predictions.tolist() == [1.0, 1.0]


def test_tournament_factories_expose_main_estimators():
    regression_names = set(build_regression_tournament())
    classification_names = set(
        build_classification_tournament(pd.Series([0, 0, 1, 1]))
    )

    assert {
        "Ridge",
        "ElasticNet",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    }.issubset(regression_names)
    assert {
        "LogisticRegression",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    }.issubset(classification_names)


def test_tournaments_block_test_evaluation_by_default():
    splits = {
        "train": _feature_frame(),
        "validation": _feature_frame(),
        "test": _feature_frame(),
    }

    with pytest.raises(ValueError, match="test evaluation is blocked"):
        run_regression_tournament(splits, evaluate_splits=("test",))
    with pytest.raises(ValueError, match="test evaluation is blocked"):
        run_classification_tournament(splits, evaluate_splits=("test",))


def _synthetic_modeling_frame(rows: int = 90, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with every model feature, targets and lead times."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {name: rng.normal(size=rows) for name in modeling.NUMERIC_FEATURES}
    )
    frame["promised_lead_days"] = rng.uniform(8, 30, size=rows)
    frame["customer_state"] = rng.choice(["SP", "RJ", "MG"], size=rows)
    frame["seller_state"] = rng.choice(["SP", "RJ"], size=rows)
    frame["primary_payment_type"] = rng.choice(["credit_card", "boleto"], size=rows)
    frame["order_id"] = [f"o{i}" for i in range(rows)]
    frame["order_purchase_timestamp"] = pd.date_range("2018-01-01", periods=rows)
    noise = rng.normal(scale=2.0, size=rows)
    frame["actual_lead_days"] = 12 + 2 * frame["item_count"] + noise
    frame["delivery_delay_days"] = frame["actual_lead_days"] - frame["promised_lead_days"]
    median_delay = frame["delivery_delay_days"].median()
    frame["sla_breach"] = (frame["delivery_delay_days"] > median_delay).astype(int)
    return frame


def _synthetic_folds(frame: pd.DataFrame) -> list[dict]:
    return [
        {"fold": "f1", "train": frame.iloc[:45], "validation": frame.iloc[45:65]},
        {"fold": "f2", "train": frame.iloc[:65], "validation": frame.iloc[65:]},
    ]


def test_tournaments_can_run_over_development_folds_without_test_partition():
    folds = _synthetic_folds(_synthetic_modeling_frame())

    regression, _ = run_regression_tournament_folds(folds)
    classification, _ = run_classification_tournament_folds(folds)

    assert set(regression["fold"]) == {"f1", "f2"}
    assert set(classification["fold"]) == {"f1", "f2"}
    assert set(regression["split"]) == {"validation"}


def test_fold_tournaments_include_baselines_and_timings():
    folds = _synthetic_folds(_synthetic_modeling_frame())

    regression, _ = run_regression_tournament_folds(folds)
    classification, _ = run_classification_tournament_folds(folds)

    for results, expected in (
        (
            regression,
            {
                "baseline_train_mean",
                "baseline_median_actual_minus_promise",
                "baseline_state_median_actual_minus_promise",
            },
        ),
        (
            classification,
            {
                "baseline_prevalence",
                "baseline_negative_promised_lead_days",
                "baseline_logistic_promise_plus_state",
            },
        ),
    ):
        baselines = results[results["model"].str.startswith("baseline_")]
        assert set(baselines["model"]) == expected
        assert baselines.groupby("model")["fold"].nunique().eq(2).all()
        trained = results[~results["model"].str.startswith("baseline_")]
        assert trained["fit_seconds"].ge(0).all()
        assert trained["predict_seconds"].ge(0).all()

    negative_name = "baseline_negative_promised_lead_days"
    negative = classification[classification["model"] == negative_name]
    assert negative["brier"].isna().all()
    assert negative["ece_10bins"].isna().all()
    assert negative["average_precision"].notna().all()
    assert classification[classification["model"] != negative_name]["brier"].notna().all()


def test_classification_policy_has_no_class_weights_and_uses_isotonic_calibration():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler

    candidates = build_classification_tournament(pd.Series([0, 0, 1, 1]))

    for estimator in candidates.values():
        params = estimator.get_params(deep=True)
        assert params.get("class_weight") is None
        assert params.get("scale_pos_weight") is None
        assert params.get("auto_class_weights") is None
    assert isinstance(candidates["LogisticRegression"].named_steps["scaler"], StandardScaler)

    frame = _synthetic_modeling_frame()
    splits = {"train": frame.iloc[:60], "validation": frame.iloc[60:]}
    _, models = run_classification_tournament(splits)
    for model in models.values():
        assert isinstance(model, CalibratedClassifierCV)
        assert model.method == "isotonic"
        assert model.cv == 3


def test_classification_metrics_nan_for_non_probability_scores():
    y = np.array([0, 1, 0, 1, 1, 0])
    scores = -np.array([10.0, 3.0, 9.0, 4.0, 5.0, 12.0])

    metrics = classification_metrics(y, scores, probabilistic=False)

    assert np.isnan(metrics["brier"])
    assert np.isnan(metrics["ece_10bins"])
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_tournament_summary_reports_gain_and_wins_over_best_baseline():
    results = pd.DataFrame(
        {
            "fold": ["a", "b", "c"] * 3,
            "model": ["M"] * 3 + ["baseline_x"] * 3 + ["baseline_y"] * 3,
            "mae": [1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0],
            "rmse": [2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 9.0, 9.0, 9.0],
            "r2": [0.3] * 9,
            "fit_seconds": [1.0] * 3 + [np.nan] * 6,
            "predict_seconds": [0.1] * 3 + [np.nan] * 6,
        }
    )

    summary = tournament_summary(results).set_index("model")

    assert summary.loc["M", "mae_mean"] == pytest.approx(2.0)
    assert summary.loc["M", "mae_std"] == pytest.approx(1.0)
    assert summary.loc["M", "mae_best_baseline"] == "baseline_x"
    # Best baseline MAE mean is 2.0 and the model mean is 2.0: no gain.
    assert summary.loc["M", "mae_gain_vs_baseline"] == pytest.approx(0.0)
    # RMSE: baseline_x mean 4.0 versus model mean 3.0 -> 25% gain.
    assert summary.loc["M", "rmse_gain_vs_baseline"] == pytest.approx(0.25)
    assert summary.loc["M", "mae_folds_won"] == 1  # only fold "a" (1.0 < 2.0)
    assert summary.loc["M", "rmse_folds_won"] == 2  # a and b win, c ties
    assert np.isnan(summary.loc["baseline_x", "rmse_gain_vs_baseline"])
    assert summary.loc["M", "n_folds"] == 3


def test_tournament_summary_classification_higher_is_better():
    results = pd.DataFrame(
        {
            "fold": ["a", "b", "a", "b"],
            "model": ["M", "M", "baseline_prevalence", "baseline_prevalence"],
            "average_precision": [0.30, 0.20, 0.10, 0.25],
            "roc_auc": [0.7, 0.6, 0.5, 0.5],
            "brier": [0.1, 0.1, 0.1, np.nan],
            "ece_10bins": [0.01, 0.02, 0.0, np.nan],
        }
    )

    summary = tournament_summary(results).set_index("model")

    assert summary.loc["M", "average_precision_gain_vs_baseline"] == pytest.approx(
        (0.25 - 0.175) / 0.175
    )
    assert summary.loc["M", "average_precision_folds_won"] == 1


def test_optuna_pipeline_contains_preprocessing_selector_and_model():
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    pipeline = build_optuna_pipeline(trial, "regression", "ElasticNet")

    assert list(pipeline.named_steps) == ["preprocessor", "selector", "model"]
    assert "preprocessing_scaler" in trial.params
    assert "feature_selector" in trial.params
    assert set(trial.distributions["preprocessing_scaler"].choices) == {
        "standard",
        "robust",
    }


def test_optuna_scaler_only_for_linear_models():
    cases = (
        ("regression", ["ElasticNet", "Ridge"], ["XGBoost", "LightGBM", "CatBoost"]),
        ("classification", ["LogisticRegression"], ["XGBoost", "LightGBM", "CatBoost"]),
    )
    for task, linear, trees in cases:
        for model_name in linear:
            trial = optuna.create_study().ask()
            build_optuna_pipeline(trial, task, model_name)
            assert "preprocessing_scaler" in trial.params
        for model_name in trees:
            trial = optuna.create_study().ask()
            pipeline = build_optuna_pipeline(trial, task, model_name)
            assert "preprocessing_scaler" not in trial.params
            numeric = dict(
                (name, step) for name, step, _ in pipeline.named_steps["preprocessor"].transformers
            )["numeric"]
            assert list(numeric.named_steps) == ["imputer"]


def test_optuna_classifiers_use_no_class_weights_and_are_calibrated():
    from sklearn.calibration import CalibratedClassifierCV

    for model_name in ("LogisticRegression", "XGBoost", "LightGBM", "CatBoost"):
        trial = optuna.create_study().ask()
        pipeline = build_optuna_pipeline(trial, "classification", model_name)
        model = pipeline.named_steps["model"]
        assert isinstance(model, CalibratedClassifierCV)
        assert model.method == "isotonic"
        assert model.cv == 3
        params = model.estimator.get_params(deep=True)
        assert params.get("class_weight") is None
        assert params.get("auto_class_weights") is None
        assert params.get("scale_pos_weight") is None
        assert "scale_pos_weight" not in trial.params


def _optuna_inputs():
    frame = _synthetic_modeling_frame()
    frame["sla_breach"] = frame["sla_breach"].astype(bool)  # same dtype as real data
    splits = {"train": frame.iloc[:60], "validation": frame.iloc[60:]}
    folds = [
        {"train": frame.iloc[:45], "validation": frame.iloc[45:65]},
        {"train": frame.iloc[:65], "validation": frame.iloc[65:]},
    ]
    return splits, folds


def test_optuna_reports_each_development_fold_with_two_trials():
    splits, folds = _optuna_inputs()

    study = run_optuna_study(
        splits,
        task="regression",
        model_name="Ridge",
        n_trials=2,
        progress=False,
        folds=folds,
    )

    assert len(study.trials) == 2
    assert all(set(trial.intermediate_values) == {0, 1} for trial in study.trials)


def test_optuna_classification_runs_two_trials_on_synthetic_folds():
    splits, folds = _optuna_inputs()

    study = run_optuna_study(
        splits,
        task="classification",
        model_name="LogisticRegression",
        n_trials=2,
        progress=False,
        folds=folds,
    )

    assert len(study.trials) == 2
    assert all(trial.value is not None and 0 <= trial.value <= 1 for trial in study.trials)


def test_optuna_timeout_is_forwarded_to_optimize(monkeypatch):
    captured = {}
    original = optuna.Study.optimize

    def spy(self, *args, **kwargs):
        captured.update(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(optuna.Study, "optimize", spy)
    splits, _ = _optuna_inputs()

    run_optuna_study(
        splits,
        task="regression",
        model_name="Ridge",
        n_trials=1,
        progress=False,
        timeout_seconds=120.0,
    )

    assert captured["timeout"] == 120.0


def test_numeric_features_override_changes_matrix_columns_and_default_is_unchanged():
    features = _feature_frame()
    features["distance_max_km"] = [120.0, 210.0, 150.0, 400.0]
    splits = {"train": features.iloc[:3], "validation": features.iloc[3:]}

    default, _ = prepare_model_matrices(splits)
    swapped_features = [
        "distance_max_km" if name == "distance_mean_km" else name
        for name in modeling.NUMERIC_FEATURES
    ]
    swapped, _ = prepare_model_matrices(splits, numeric_features=swapped_features)

    assert "distance_mean_km" in default["train"].columns
    assert "distance_max_km" not in default["train"].columns
    assert "distance_max_km" in swapped["train"].columns
    assert "distance_mean_km" not in swapped["train"].columns
    assert default["train"].shape == swapped["train"].shape
    assert modeling.make_preprocessor("standard").transformers[0][2] == (
        modeling.NUMERIC_FEATURES
    )
    assert modeling.NUMERIC_FEATURES.count("distance_mean_km") == 1


def test_tournament_models_argument_filters_estimators_and_keeps_default():
    folds = _synthetic_folds(_synthetic_modeling_frame())

    regression, _ = run_regression_tournament_folds(
        folds, models=("ElasticNet",), include_baselines=False
    )
    classification, _ = run_classification_tournament_folds(
        folds, models=("LogisticRegression",), include_baselines=False
    )
    split_data = {"train": folds[0]["train"], "validation": folds[0]["validation"]}
    full, fitted = run_regression_tournament(split_data, models=("OLS", "Ridge"))

    assert set(regression["model"]) == {"ElasticNet"}
    assert set(classification["model"]) == {"LogisticRegression"}
    assert set(full["model"]) == {"OLS", "Ridge"} == set(fitted)
    with pytest.raises(KeyError, match="unknown models"):
        run_regression_tournament(split_data, models=("Nope",))
    default, _ = run_regression_tournament(split_data)
    assert len(default) == len(build_regression_tournament()) + 1


def _ablation_folds(multi_share: float = 0.6, rows: int = 220) -> list[dict]:
    frame = _synthetic_modeling_frame(rows=rows, seed=3)
    rng = np.random.default_rng(5)
    frame["seller_count"] = np.where(rng.random(rows) < multi_share, 2, 1)
    frame["distance_max_km"] = frame["distance_mean_km"] + 1.0
    frame["distance_range_km"] = rng.uniform(0, 5, size=rows)
    return [
        {"fold": "f1", "train": frame.iloc[:100], "validation": frame.iloc[100:160]},
        {"fold": "f2", "train": frame.iloc[:160], "validation": frame.iloc[160:]},
    ]


def test_distance_ablation_variants_replace_only_the_mean_distance():
    variants = modeling.distance_ablation_variants()

    assert set(variants) == {"mean", "max", "max_plus_range"}
    assert variants["mean"] == modeling.NUMERIC_FEATURES
    assert "distance_mean_km" not in variants["max"]
    assert "distance_max_km" in variants["max"]
    assert len(variants["max"]) == len(modeling.NUMERIC_FEATURES)
    assert {"distance_max_km", "distance_range_km"} <= set(variants["max_plus_range"])
    assert len(variants["max_plus_range"]) == len(modeling.NUMERIC_FEATURES) + 1


def test_distance_ablation_regression_reports_overall_and_multi_seller_metrics():
    folds = _ablation_folds()

    results = modeling.run_distance_ablation(
        folds, "regression", models=("Ridge",), random_state=1
    )

    assert len(results) == 3 * 2 * 1
    assert set(results["variant"]) == {"mean", "max", "max_plus_range"}
    assert {"mae", "multi_seller_mae", "multi_seller_n"} <= set(results.columns)
    expected_n = int((folds[0]["validation"]["seller_count"] > 1).sum())
    assert results.loc[results["fold"] == "f1", "multi_seller_n"].eq(expected_n).all()
    assert results["multi_seller_mae"].notna().all()
    # the subgroup metric is computed only on the multi-seller rows
    assert not np.allclose(results["mae"], results["multi_seller_mae"])


def test_distance_ablation_subgroup_is_nan_when_small_or_single_class():
    only_mean = {"mean": modeling.NUMERIC_FEATURES}
    small_results = modeling.run_distance_ablation(
        _ablation_folds(multi_share=0.05),
        "regression",
        variants=only_mean,
        models=("Ridge",),
    )
    assert small_results["multi_seller_mae"].isna().all()
    assert small_results["multi_seller_n"].lt(30).all()

    folds = _ablation_folds()
    for fold in folds:
        validation = fold["validation"].copy()
        validation.loc[validation["seller_count"] > 1, "sla_breach"] = 0
        fold["validation"] = validation
    one_class = modeling.run_distance_ablation(
        folds, "classification", variants=only_mean, models=("LogisticRegression",)
    )
    assert one_class["multi_seller_n"].ge(30).all()
    assert one_class["multi_seller_average_precision"].isna().all()
    assert one_class["average_precision"].notna().all()


def test_ablation_summary_computes_delta_and_folds_won_with_known_values():
    rows = []
    for variant, maes, sub in [
        ("mean", [2.0, 4.0], [3.0, 5.0]),
        ("max", [1.0, 5.0], [2.0, 4.0]),
    ]:
        for fold, value, sub_value in zip(["f1", "f2"], maes, sub, strict=True):
            rows.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "model": "Ridge",
                    "mae": value,
                    "rmse": value,
                    "r2": 0.0,
                    "multi_seller_n": 40,
                    "multi_seller_mae": sub_value,
                }
            )
    summary = modeling.ablation_summary(pd.DataFrame(rows), "mean", task="regression")

    mean_row = summary.loc[summary["variant"] == "mean"].iloc[0]
    max_row = summary.loc[summary["variant"] == "max"].iloc[0]
    assert mean_row["mae_mean"] == 3.0
    assert np.isnan(mean_row["mae_delta_vs_mean"])
    assert max_row["mae_mean"] == 3.0
    assert max_row["mae_delta_vs_mean"] == 0.0
    assert max_row["mae_folds_won"] == 1
    assert max_row["multi_seller_mae_mean"] == 3.0
    assert max_row["multi_seller_mae_folds_won"] == 2
    assert max_row["multi_seller_mae_delta_vs_mean"] == pytest.approx(0.25)
    assert max_row["multi_seller_n_folds"] == 2


def test_ablation_summary_classification_end_to_end():
    results = modeling.run_distance_ablation(
        _ablation_folds(),
        "classification",
        models=("LogisticRegression",),
        random_state=1,
    )

    summary = modeling.ablation_summary(results, "mean", task="classification")

    assert len(summary) == 3
    assert {
        "average_precision_mean",
        "average_precision_delta_vs_mean",
        "multi_seller_average_precision_folds_won",
    } <= set(summary.columns)
    with pytest.raises(KeyError):
        modeling.ablation_summary(results, "absent", task="classification")
