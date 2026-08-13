import hashlib
import json
import re
import shutil

import joblib
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest

import src.policy_reports as policy_reports_module
import src.reports as reports_module
from src.compat import patch_sklearn_matplotlib_support
from src.config import (
    ARMS,
    BIN_VARS,
    CAT_VARS,
    CONT_VARS,
    CONTROL_ARM,
    FEATURE_COLS,
    POOLED_TREATMENT_COL,
    PROJECT_ROOT,
    SEED,
    TEST_FRAC,
    TRAIN_FRAC,
    TREATED_ARMS,
    TREATMENT_COL,
    VAL_FRAC,
)
from src.data import add_pooled_treatment, variable_dictionary
from src.eda import (
    build_smd_table,
    formal_balance_tests,
    outcome_summary_by_arm,
    smd_binary,
    smd_continuous,
    treatment_distribution,
)
from src.effects import ate_binary, ate_continuous, ate_table
from src.evaluation import (
    _stratified_bootstrap_indices,
    bootstrap_qini_comparison,
    evaluate_multiple_rankings,
    evaluate_ranking,
    gates_by_cate_quintile,
    gates_delta_bootstrap,
    paired_deltas,
    permutation_noise_floor,
    repeated_holdout_summary,
    repeated_stratified_holdout,
    spearman_ranking_correlation,
)
from src.features import (
    EXTENDED_BIN_VARS,
    EXTENDED_CAT_VARS,
    EXTENDED_CONT_VARS,
    add_engineered_features,
)
from src.i18n import BASE_LANG, make_lang, resolve_lang
from src.learners import (
    _regularized_base_learner,
    _uplift_treatment_labels,
    build_meta_learner_encoder,
    encode_meta_learner_features,
    fit_causal_forest,
    fit_meta_learners,
    fit_meta_learners_regularized,
    fit_propensity_baseline,
    fit_s_learner_linear_interaction,
    fit_single_meta_learner,
    fit_t_learner_quick,
    fit_uplift_random_forest,
    fit_uplift_tree,
    get_meta_learners,
    load_meta_learners,
    predict_causal_forest_uplift,
    predict_meta_learners_uplift,
    predict_propensity_score,
    predict_s_learner_linear_interaction_uplift,
    predict_single_meta_learner,
    predict_t_learner_uplift,
    predict_uplift_random_forest_uplift,
    predict_uplift_tree_uplift,
    save_meta_learners,
)
from src.policy import (
    all_arm_actions,
    binary_ipw_incremental_value,
    budget_mask,
    evaluate_binary_policies,
    evaluate_three_way_policies,
    fit_three_way_surrogate,
    make_binary_policy_masks,
    multi_arm_ipw_incremental_value,
    roi_metrics,
    stratified_bootstrap_indices,
    three_way_actions,
    three_way_net_gains,
)
from src.policy_reports import (
    build_s8_policy_report,
    display_s8_policy_report,
    plot_s8_policy_report,
    save_s8_policy_artifacts,
)
from src.reports import (
    add_score_quantiles,
    build_s7_heterogeneity_report,
    fit_high_uplift_surrogate,
    funnel_ranking_summary,
    top_bottom_profile,
)
from src.robustness_reports import (
    build_s9_report,
    display_s9_report,
    load_s9_inputs,
    plot_s9_report,
    save_s9_artifacts,
)
from src.splits import (
    MANIFEST_PATH,
    dataset_fingerprint,
    get_train_val,
    load_sealed_test,
    make_splits,
)
from src.viz import (
    add_chart_footer,
    add_chart_header,
    apply_plot_style,
    plot_ate_forest,
    plot_gates_bars,
    plot_love_plot,
    plot_outcomes_by_arm,
    plot_permutation_noise_floor,
    plot_split_overview,
    plot_univariate_categorical,
    plot_univariate_continuous,
    plot_uplift_distributions,
)

# ---- data.py ----------------------------------------------------------


def test_load_hillstrom_shape(df):
    assert df.shape == (64000, 12)


def test_load_hillstrom_treatment_categories(df):
    assert set(df[TREATMENT_COL].cat.categories) == set(ARMS)


def test_load_hillstrom_no_missing_in_key_columns(df):
    key_cols = ["recency", "history", "mens", "womens", "newbie", "visit", "conversion", "spend"]
    assert df[key_cols].isna().sum().sum() == 0


def test_add_pooled_treatment_does_not_mutate_input(df):
    original_columns = list(df.columns)
    add_pooled_treatment(df)
    assert list(df.columns) == original_columns


def test_add_pooled_treatment_matches_treated_arms(df):
    pooled = add_pooled_treatment(df)
    n_treated_pooled = pooled[POOLED_TREATMENT_COL].sum()
    n_treated_arms = pooled[TREATMENT_COL].isin(TREATED_ARMS).sum()
    n_control = pooled[TREATMENT_COL].eq(CONTROL_ARM).sum()
    assert n_treated_pooled == n_treated_arms
    assert n_treated_pooled + n_control == len(pooled)


def test_variable_dictionary_covers_all_columns(df):
    var_dict = variable_dictionary()
    assert set(var_dict["variavel"]) == set(df.columns)


# ---- splits.py ----------------------------------------------------------


@pytest.fixture()
def split_result(df, tmp_path, monkeypatch):
    import src.splits as splits_mod

    monkeypatch.setattr(splits_mod, "SPLITS_DIR", tmp_path)
    result = splits_mod.make_splits(df, splits_dir=tmp_path)
    return result, tmp_path


def test_splits_sum_to_full_dataset(df, split_result):
    result, splits_dir = split_result
    test_df = load_sealed_test(df, unlock=True, splits_dir=splits_dir)
    total = len(result["train_idx"]) + len(result["val_idx"]) + len(test_df)
    assert total == len(df)


def test_splits_have_no_overlap(df, split_result):
    result, splits_dir = split_result
    test_df = load_sealed_test(df, unlock=True, splits_dir=splits_dir)
    all_idx = np.concatenate([result["train_idx"], result["val_idx"], test_df.index.values])
    assert len(all_idx) == len(set(all_idx))


def test_splits_proportions_match_target(df, split_result):
    result, splits_dir = split_result
    test_df = load_sealed_test(df, unlock=True, splits_dir=splits_dir)
    n = len(df)
    assert result["train_idx"].size / n == pytest.approx(TRAIN_FRAC, abs=0.01)
    assert result["val_idx"].size / n == pytest.approx(VAL_FRAC, abs=0.01)
    assert len(test_df) / n == pytest.approx(TEST_FRAC, abs=0.01)


def test_splits_preserve_arm_proportions(df, split_result):
    result, splits_dir = split_result
    test_df = load_sealed_test(df, unlock=True, splits_dir=splits_dir)
    overall = df[TREATMENT_COL].value_counts(normalize=True)
    for idx, name in [(result["train_idx"], "train"), (result["val_idx"], "val")]:
        partition_dist = df.loc[idx, TREATMENT_COL].value_counts(normalize=True)
        for arm in ARMS:
            assert partition_dist[arm] == pytest.approx(overall[arm], abs=0.01), name
    test_dist = test_df[TREATMENT_COL].value_counts(normalize=True)
    for arm in ARMS:
        assert test_dist[arm] == pytest.approx(overall[arm], abs=0.01)


def test_splits_preserve_visit_rate(df, split_result):
    result, splits_dir = split_result
    test_df = load_sealed_test(df, unlock=True, splits_dir=splits_dir)
    overall_rate = df["visit"].mean()
    for idx, name in [(result["train_idx"], "train"), (result["val_idx"], "val")]:
        partition_rate = df.loc[idx, "visit"].mean()
        assert partition_rate == pytest.approx(overall_rate, abs=0.01), name
    assert test_df["visit"].mean() == pytest.approx(overall_rate, abs=0.01)


def test_load_sealed_test_requires_unlock(df, split_result):
    _, splits_dir = split_result
    with pytest.raises(PermissionError):
        load_sealed_test(df, unlock=False, splits_dir=splits_dir)


def test_load_sealed_test_without_prior_split_raises(df, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sealed_test(df, unlock=True, splits_dir=tmp_path)


def test_make_splits_is_reproducible_given_seed(df, tmp_path):
    import src.splits as splits_mod

    result_a = splits_mod.make_splits(df, seed=42, splits_dir=tmp_path / "a")
    result_b = splits_mod.make_splits(df, seed=42, splits_dir=tmp_path / "b")
    assert np.array_equal(np.sort(result_a["train_idx"]), np.sort(result_b["train_idx"]))
    assert np.array_equal(np.sort(result_a["val_idx"]), np.sort(result_b["val_idx"]))


def test_get_train_val_matches_make_splits(df, tmp_path):
    import src.splits as splits_mod

    pooled = add_pooled_treatment(df)
    expected = splits_mod.make_splits(pooled, persist_test=False, splits_dir=tmp_path)
    train_df, val_df = get_train_val(pooled, persist_test=False, splits_dir=tmp_path)
    assert np.array_equal(np.sort(train_df.index.values), np.sort(expected["train_idx"]))
    assert np.array_equal(np.sort(val_df.index.values), np.sort(expected["val_idx"]))


def test_get_train_val_does_not_persist_test_by_default(df, tmp_path):
    pooled = add_pooled_treatment(df)
    get_train_val(pooled, splits_dir=tmp_path)
    assert not (tmp_path / "sealed_test_index.parquet").exists()


def test_make_splits_persists_train_val_test_and_manifest(df, tmp_path):
    import src.splits as splits_mod

    pooled = add_pooled_treatment(df)
    splits_mod.make_splits(pooled, splits_dir=tmp_path)
    assert (tmp_path / "train_index.parquet").exists()
    assert (tmp_path / "validation_index.parquet").exists()
    assert (tmp_path / "sealed_test_index.parquet").exists()
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert manifest["dataset_sha256"] == splits_mod.dataset_fingerprint()
    assert manifest["n_train"] + manifest["n_val"] + manifest["n_test"] == manifest["n_rows"]


def test_get_train_val_loads_persisted_manifest_instead_of_recomputing(df, tmp_path):
    import src.splits as splits_mod

    pooled = add_pooled_treatment(df)
    persisted = splits_mod.make_splits(pooled, splits_dir=tmp_path)
    train_df, val_df = get_train_val(pooled, splits_dir=tmp_path)
    assert np.array_equal(np.sort(train_df.index.values), np.sort(persisted["train_idx"]))
    assert np.array_equal(np.sort(val_df.index.values), np.sort(persisted["val_idx"]))


def test_get_train_val_raises_on_dataset_fingerprint_mismatch(df, tmp_path):
    import src.splits as splits_mod

    pooled = add_pooled_treatment(df)
    splits_mod.make_splits(pooled, splits_dir=tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        get_train_val(pooled, splits_dir=tmp_path)


def test_load_sealed_test_raises_on_dataset_fingerprint_mismatch(df, split_result):
    _, splits_dir = split_result
    manifest_path = splits_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_sealed_test(df, unlock=True, splits_dir=splits_dir)


def test_get_train_val_raises_on_incomplete_manifest_when_sealed_test_exists(df, tmp_path):
    """Estado parcial perigoso (item 1 do hardening): sealed_test_index.parquet
    já existe, mas dataset_manifest.json foi removido/corrompido — não deve
    cair no fallback silencioso de recalcular treino/validação."""
    import src.splits as splits_mod

    pooled = add_pooled_treatment(df)
    splits_mod.make_splits(pooled, splits_dir=tmp_path)
    (tmp_path / "dataset_manifest.json").unlink()
    with pytest.raises(ValueError):
        get_train_val(pooled, splits_dir=tmp_path)


def test_get_train_val_raises_when_sealed_test_exists_but_train_index_missing(df, tmp_path):
    import src.splits as splits_mod

    pooled = add_pooled_treatment(df)
    splits_mod.make_splits(pooled, splits_dir=tmp_path)
    (tmp_path / "train_index.parquet").unlink()
    with pytest.raises(ValueError):
        get_train_val(pooled, splits_dir=tmp_path)


# ---- i18n.py --------------------------------------------------------------


def test_make_lang_pt_is_pure_passthrough():
    lang = make_lang("pt")
    original = {"a": "Distribuição de recency", "b": "Balance covariáveis"}
    assert lang(original) == original


def test_make_lang_pt_does_not_write_cache_file(tmp_path, monkeypatch):
    import src.i18n as i18n_mod

    monkeypatch.setattr(i18n_mod, "CACHE_DIR", tmp_path / "cache")
    lang = i18n_mod.make_lang("pt")
    lang({"a": "texto qualquer"})
    assert not (tmp_path / "cache" / "pt_pt.json").exists()


def test_make_lang_pt_does_not_import_deep_translator(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "deep_translator":
            raise AssertionError("deep_translator should not be imported for pt->pt passthrough")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    lang = make_lang("pt")
    lang({"a": "texto qualquer"})


def test_resolve_lang_defaults_to_base_lang_passthrough():
    lang = resolve_lang(None)
    assert lang.source == BASE_LANG
    assert lang.target == BASE_LANG
    assert lang({"a": "mantém"}) == {"a": "mantém"}


def test_resolve_lang_returns_given_lang_unchanged():
    given = make_lang("pt")
    assert resolve_lang(given) is given


# ---- learners.py / evaluation.py (S3) --------------------------------


@pytest.fixture(scope="module")
def pooled_split(df):
    sample = df.sample(n=8000, random_state=SEED).reset_index(drop=True)
    pooled = add_pooled_treatment(sample)
    idx = make_splits(pooled, persist_test=False)
    train_df = pooled.loc[idx["train_idx"]]
    val_df = pooled.loc[idx["val_idx"]]
    return train_df, val_df


@pytest.fixture(scope="module")
def propensity_model(pooled_split):
    train_df, _ = pooled_split
    return fit_propensity_baseline(train_df, POOLED_TREATMENT_COL, "visit")


@pytest.fixture(scope="module")
def t_learner_models(pooled_split):
    train_df, _ = pooled_split
    return fit_t_learner_quick(train_df, POOLED_TREATMENT_COL, "visit")


def test_fit_propensity_baseline_trains_only_on_treated_rows(pooled_split):
    train_df, _ = pooled_split
    model = fit_propensity_baseline(train_df, POOLED_TREATMENT_COL, "visit")
    assert model.n_features_ == len(FEATURE_COLS)
    assert set(model.classes_) <= {0, 1}


def test_predict_propensity_score_returns_valid_probabilities(pooled_split, propensity_model):
    _, val_df = pooled_split
    scores = predict_propensity_score(propensity_model, val_df)
    assert len(scores) == len(val_df)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_fit_t_learner_quick_returns_two_fitted_models(t_learner_models):
    model_treated, model_control = t_learner_models
    assert set(model_treated.classes_) <= {0, 1}
    assert set(model_control.classes_) <= {0, 1}


def test_predict_t_learner_uplift_returns_bounded_array(pooled_split, t_learner_models):
    _, val_df = pooled_split
    uplift = predict_t_learner_uplift(t_learner_models, val_df)
    assert len(uplift) == len(val_df)
    assert (uplift >= -1).all() and (uplift <= 1).all()


def test_evaluate_ranking_returns_expected_keys(pooled_split, propensity_model):
    _, val_df = pooled_split
    score = predict_propensity_score(propensity_model, val_df)
    result = evaluate_ranking(val_df["visit"].values, score, val_df[POOLED_TREATMENT_COL].values)
    assert set(result.keys()) == {"qini_auc", "uplift_auc", "uplift_at_30pct"}
    assert all(isinstance(v, float) for v in result.values())


def test_spearman_ranking_correlation_perfect_for_identical_scores():
    scores = np.array([0.1, 0.5, 0.2, 0.9, 0.4])
    result = spearman_ranking_correlation(scores, scores)
    assert result["spearman_corr"] == pytest.approx(1.0)


def test_spearman_ranking_correlation_returns_expected_keys(pooled_split, propensity_model, t_learner_models):
    _, val_df = pooled_split
    propensity_score = predict_propensity_score(propensity_model, val_df)
    t_uplift = predict_t_learner_uplift(t_learner_models, val_df)
    result = spearman_ranking_correlation(propensity_score, t_uplift)
    assert set(result.keys()) == {"spearman_corr", "p_value"}
    assert -1.0 <= result["spearman_corr"] <= 1.0


# ---- learners.py (S4: meta-learners) / evaluation.py (S4) --------------


@pytest.fixture(scope="module")
def meta_split(df):
    sample = df.sample(n=1200, random_state=SEED).reset_index(drop=True)
    pooled = add_pooled_treatment(sample)
    idx = make_splits(pooled, persist_test=False)
    train_df = pooled.loc[idx["train_idx"]]
    val_df = pooled.loc[idx["val_idx"]]
    return train_df, val_df


@pytest.fixture(scope="module")
def meta_encoder(meta_split):
    train_df, _ = meta_split
    return build_meta_learner_encoder(train_df)


@pytest.fixture(scope="module")
def meta_models(meta_split, meta_encoder):
    train_df, _ = meta_split
    return fit_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", meta_encoder)


def test_encode_meta_learner_features_is_fully_numeric(meta_split, meta_encoder):
    train_df, _ = meta_split
    X = encode_meta_learner_features(train_df, meta_encoder)
    assert X.dtype.kind == "f"
    assert X.shape[0] == len(train_df)
    assert X.shape[1] == len(CONT_VARS) + len(BIN_VARS) + sum(train_df[c].nunique() for c in CAT_VARS)


def test_fit_meta_learners_returns_all_four(meta_models):
    assert set(meta_models.keys()) == {"S", "T", "X", "R"}


def test_fit_meta_learners_uses_custom_base_learner_factory(meta_split, meta_encoder):
    from sklearn.linear_model import ElasticNet

    train_df, _ = meta_split
    calls = []

    def factory():
        calls.append(1)
        return ElasticNet()

    models = fit_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", meta_encoder, base_learner_factory=factory)
    assert set(models.keys()) == {"S", "T", "X", "R"}
    assert len(calls) == 4  # uma instância nova por meta-learner, nenhuma compartilhada


def test_fit_meta_learners_default_behavior_unchanged_after_refactor(meta_split, meta_encoder):
    train_df, _ = meta_split
    models_a = fit_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", meta_encoder)
    models_b = fit_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", meta_encoder)
    X = encode_meta_learner_features(train_df, meta_encoder)
    for name in ("S", "T", "X", "R"):
        np.testing.assert_allclose(models_a[name].predict(X).ravel(), models_b[name].predict(X).ravel())


def test_predict_meta_learners_uplift_returns_array_per_learner(meta_split, meta_models, meta_encoder):
    _, val_df = meta_split
    cate = predict_meta_learners_uplift(meta_models, val_df, meta_encoder)
    assert set(cate.keys()) == {"S", "T", "X", "R"}
    for arr in cate.values():
        assert len(arr) == len(val_df)


def test_save_and_load_meta_learners_round_trip(meta_models, meta_encoder, tmp_path):
    path = tmp_path / "meta_learners.joblib"
    save_meta_learners(meta_models, meta_encoder, path)
    loaded_models, loaded_encoder = load_meta_learners(path)
    assert set(loaded_models.keys()) == {"S", "T", "X", "R"}
    assert loaded_encoder.categories_[0].tolist() == meta_encoder.categories_[0].tolist()


def test_get_meta_learners_trains_and_saves_when_missing(meta_split, tmp_path):
    train_df, _ = meta_split
    path = tmp_path / "meta_learners.joblib"
    assert not path.exists()
    models, _encoder = get_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", retrain=False, path=path)
    assert path.exists()
    assert set(models.keys()) == {"S", "T", "X", "R"}


def test_get_meta_learners_loads_without_retraining_when_present(meta_split, tmp_path):
    train_df, _ = meta_split
    path = tmp_path / "meta_learners.joblib"
    _models_a, _ = get_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", retrain=False, path=path)
    mtime_after_first_call = path.stat().st_mtime
    _models_b, _ = get_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", retrain=False, path=path)
    assert path.stat().st_mtime == mtime_after_first_call


def test_get_meta_learners_saves_metadata_alongside_artifact(meta_split, tmp_path):
    import joblib

    train_df, _ = meta_split
    path = tmp_path / "meta_learners.joblib"
    get_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", retrain=False, path=path)
    bundle = joblib.load(path)
    assert "metadata" in bundle
    assert set(bundle["metadata"]) == {
        "data_fingerprint", "feature_cols", "seed", "base_learner",
        "base_learner_params", "package_versions",
    }
    assert bundle["metadata"]["base_learner_params"]["max_depth"] == -1


def test_get_meta_learners_retrains_when_cached_metadata_is_stale(meta_split, tmp_path):
    import joblib

    train_df, _ = meta_split
    path = tmp_path / "meta_learners.joblib"
    get_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", retrain=False, path=path)

    bundle = joblib.load(path)
    bundle["metadata"]["data_fingerprint"] = "stale"
    joblib.dump(bundle, path)

    get_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", retrain=False, path=path)
    reloaded = joblib.load(path)
    assert reloaded["metadata"]["data_fingerprint"] != "stale"


def test_get_meta_learners_retrains_when_base_learner_params_change(meta_split, tmp_path):
    """Hardening (item 3): mudar hiperparâmetros concretos do base learner
    invalida o cache mesmo que nome/dados/versões de pacote batam — protege
    contra uma mudança interna futura em `_default_base_learner` que
    mantivesse o mesmo nome descritivo."""
    import joblib

    train_df, _ = meta_split
    path = tmp_path / "meta_learners.joblib"
    get_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", retrain=False, path=path)
    mtime_before = path.stat().st_mtime

    bundle = joblib.load(path)
    bundle["metadata"]["base_learner_params"]["max_depth"] = 4
    joblib.dump(bundle, path)

    get_meta_learners(train_df, POOLED_TREATMENT_COL, "visit", retrain=False, path=path)
    assert path.stat().st_mtime > mtime_before
    reloaded = joblib.load(path)
    assert reloaded["metadata"]["base_learner_params"]["max_depth"] == -1


def test_evaluate_multiple_rankings_returns_one_row_per_learner(meta_split, meta_models, meta_encoder):
    _, val_df = meta_split
    cate = predict_meta_learners_uplift(meta_models, val_df, meta_encoder)
    result = evaluate_multiple_rankings(val_df["visit"].values, cate, val_df[POOLED_TREATMENT_COL].values)
    assert set(result.index) == {"S", "T", "X", "R"}
    assert set(result.columns) == {"qini_auc", "uplift_auc", "uplift_at_30pct"}


def test_plot_uplift_distributions_returns_fig_and_ax(meta_split, meta_models, meta_encoder):
    _, val_df = meta_split
    cate = predict_meta_learners_uplift(meta_models, val_df, meta_encoder)
    fig, _ax = plot_uplift_distributions(cate, title="t", subtitle="s")
    plt.close(fig)


# ---- S4.4: diagnóstico de heterogeneidade --------------------------------


def test_fit_single_meta_learner_matches_fit_meta_learners_default(meta_split, meta_encoder):
    train_df, _ = meta_split
    X = encode_meta_learner_features(train_df, meta_encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df["visit"].to_numpy(dtype=float)
    model = fit_single_meta_learner("S", X, treatment, y)
    assert hasattr(model, "predict")
    assert len(model.predict(X).ravel()) == len(train_df)


def test_fit_single_meta_learner_uses_custom_base_learner_factory(meta_split, meta_encoder):
    from sklearn.linear_model import ElasticNet

    train_df, _ = meta_split
    X = encode_meta_learner_features(train_df, meta_encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df["visit"].to_numpy(dtype=float)
    model = fit_single_meta_learner("T", X, treatment, y, base_learner_factory=ElasticNet)
    assert len(model.predict(X).ravel()) == len(train_df)


def test_permutation_noise_floor_returns_n_reps_values_and_does_not_mutate_train(meta_split, meta_encoder):
    train_df, val_df = meta_split
    original_treatment = train_df[POOLED_TREATMENT_COL].copy()
    stds = permutation_noise_floor(
        train_df, val_df, POOLED_TREATMENT_COL, "visit", meta_encoder, learner_name="S", n_reps=3,
    )
    assert len(stds) == 3
    assert (stds >= 0).all()
    pd.testing.assert_series_equal(train_df[POOLED_TREATMENT_COL], original_treatment)


def test_gates_by_cate_quintile_group_sizes_sum_to_input_rows(meta_split, meta_models, meta_encoder):
    _, val_df = meta_split
    cate = predict_meta_learners_uplift(meta_models, val_df, meta_encoder)
    val_df = val_df.copy()
    val_df["s_learner_uplift"] = cate["S"]
    gates_df = gates_by_cate_quintile(val_df, "s_learner_uplift", "visit", POOLED_TREATMENT_COL, n_groups=5)
    assert gates_df["n"].sum() == len(val_df)


def test_gates_delta_bootstrap_returns_expected_keys(meta_split, meta_models, meta_encoder):
    _, val_df = meta_split
    cate = predict_meta_learners_uplift(meta_models, val_df, meta_encoder)
    val_df = val_df.copy()
    val_df["s_learner_uplift"] = cate["S"]
    result = gates_delta_bootstrap(
        val_df, "s_learner_uplift", "visit", POOLED_TREATMENT_COL, n_groups=5, n_boot=200,
    )
    assert set(result) == {
        "delta_gates", "ci_low", "ci_high", "p_value", "n_boot",
        "top_group", "bottom_group", "n_top", "n_bottom",
    }
    assert result["ci_low"] <= result["delta_gates"] <= result["ci_high"]
    assert 0.0 <= result["p_value"] <= 1.0


def test_gates_delta_bootstrap_is_reproducible_given_seed(meta_split, meta_models, meta_encoder):
    _, val_df = meta_split
    cate = predict_meta_learners_uplift(meta_models, val_df, meta_encoder)
    val_df = val_df.copy()
    val_df["s_learner_uplift"] = cate["S"]
    result_a = gates_delta_bootstrap(
        val_df, "s_learner_uplift", "visit", POOLED_TREATMENT_COL, n_boot=200, seed=123,
    )
    result_b = gates_delta_bootstrap(
        val_df, "s_learner_uplift", "visit", POOLED_TREATMENT_COL, n_boot=200, seed=123,
    )
    assert result_a["ci_low"] == result_b["ci_low"]
    assert result_a["ci_high"] == result_b["ci_high"]


def test_repeated_stratified_holdout_returns_long_frame_with_all_candidates(meta_split):
    train_df, _ = meta_split
    from sklearn.tree import DecisionTreeRegressor

    tree_factory = lambda: DecisionTreeRegressor(max_depth=4, random_state=SEED)
    candidates = {
        "X+Tree": ("meta", "X", tree_factory, False),
        "Baseline": ("propensity", None, None, False),
    }
    result = repeated_stratified_holdout(train_df, POOLED_TREATMENT_COL, "visit", candidates, n_reps=2)
    assert set(result["candidate"].unique()) == {"X+Tree", "Baseline"}
    assert set(result["rep"].unique()) == {0, 1}
    assert {"qini_auc", "uplift_auc", "uplift_at_30pct"}.issubset(result.columns)
    assert len(result) == 2 * len(candidates)


def test_repeated_stratified_holdout_reuses_same_splits_across_candidates(meta_split):
    """Mesmo split/seed por rep para todos os candidatos - pre-condicao para
    comparacao pareada (paired_deltas)."""
    train_df, _ = meta_split
    candidates = {
        "S": ("meta", "S", None, False),
        "T": ("meta", "T", None, False),
    }
    result = repeated_stratified_holdout(train_df, POOLED_TREATMENT_COL, "visit", candidates, n_reps=2)
    result_again = repeated_stratified_holdout(train_df, POOLED_TREATMENT_COL, "visit", candidates, n_reps=2)
    pd.testing.assert_frame_equal(
        result.sort_values(["rep", "candidate"]).reset_index(drop=True),
        result_again.sort_values(["rep", "candidate"]).reset_index(drop=True),
    )


def test_repeated_stratified_holdout_splits_once_per_rep_not_per_candidate(meta_split, monkeypatch):
    """Trava diretamente a propriedade central do protocolo pareado (hardening,
    item 4): o split fit/eval é sorteado uma única vez por repetição, antes do
    loop de candidatos — não uma vez por candidato. É essa chamada única por
    rep que garante fit_idx/eval_idx idênticos entre candidatos; sem isso, as
    deltas pareadas de paired_deltas não seriam válidas."""
    import src.evaluation as evaluation_mod

    train_df, _ = meta_split
    call_count = 0
    real_train_test_split = evaluation_mod.train_test_split

    def counting_train_test_split(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_train_test_split(*args, **kwargs)

    monkeypatch.setattr(evaluation_mod, "train_test_split", counting_train_test_split)

    candidates = {
        "S": ("meta", "S", None, False),
        "T": ("meta", "T", None, False),
        "Baseline": ("propensity", None, None, False),
    }
    n_reps = 3
    repeated_stratified_holdout(train_df, POOLED_TREATMENT_COL, "visit", candidates, n_reps=n_reps)
    assert call_count == n_reps


def test_repeated_holdout_summary_includes_win_rate_summing_to_one(meta_split):
    train_df, _ = meta_split
    candidates = {"S": ("meta", "S", None, False), "T": ("meta", "T", None, False)}
    result = repeated_stratified_holdout(train_df, POOLED_TREATMENT_COL, "visit", candidates, n_reps=3)
    summary = repeated_holdout_summary(result)
    assert set(summary.index) == {"S", "T"}
    assert summary["win_rate"].sum() == pytest.approx(1.0)


def test_paired_deltas_excludes_baseline_and_has_expected_columns(meta_split):
    train_df, _ = meta_split
    candidates = {
        "S": ("meta", "S", None, False),
        "T": ("meta", "T", None, False),
        "Baseline": ("propensity", None, None, False),
    }
    result = repeated_stratified_holdout(train_df, POOLED_TREATMENT_COL, "visit", candidates, n_reps=3)
    deltas = paired_deltas(result, baseline_candidate="Baseline")
    assert set(deltas.index) == {"S", "T"}
    assert set(deltas.columns) == {"delta_mean", "delta_median", "delta_std", "prop_delta_positive"}


def test_plot_permutation_noise_floor_returns_fig_and_ax():
    fig, _ax = plot_permutation_noise_floor(
        np.array([0.01, 0.02, 0.015, 0.018]), {"S": 0.036, "T": 0.07}, title="t", subtitle="s",
    )
    plt.close(fig)


def test_plot_gates_bars_returns_fig_and_ax():
    gates_df = pd.DataFrame({
        "group": [0, 1, 2], "ate": [0.01, 0.03, 0.06],
        "ci_low": [-0.01, 0.01, 0.03], "ci_high": [0.03, 0.05, 0.09],
    })
    fig, _ax = plot_gates_bars(gates_df, title="t", subtitle="s")
    plt.close(fig)


# ---- S4.5: features derivadas + regularização X/R ------------------------


def test_add_engineered_features_does_not_mutate_input(df):
    original_columns = list(df.columns)
    add_engineered_features(df)
    assert list(df.columns) == original_columns


def test_add_engineered_features_adds_expected_columns_with_correct_values():
    small = pd.DataFrame({
        "recency": [1, 3], "history": [10.0, 0.0],
        "mens": [1, 0], "womens": [1, 1],
        "newbie": [0, 1], "channel": ["Web", "Phone"], "zip_code": ["Urban", "Rural"],
    })
    out = add_engineered_features(small)
    new_cols = {"history_per_recency", "newbie_x_channel", "mens_and_womens", "zip_code_x_channel"}
    assert new_cols <= set(out.columns)
    assert out["history_per_recency"].tolist() == pytest.approx([10.0 / 2, 0.0 / 4])
    assert out["mens_and_womens"].tolist() == [1, 0]
    assert out["newbie_x_channel"].tolist() == ["0_Web", "1_Phone"]
    assert out["zip_code_x_channel"].tolist() == ["Urban_Web", "Rural_Phone"]


def test_regularized_base_learner_has_expected_hyperparameters():
    model = _regularized_base_learner(seed=SEED)
    assert model.get_params()["num_leaves"] == 15
    assert model.get_params()["max_depth"] == 5
    assert model.get_params()["min_child_samples"] == 50
    assert model.get_params()["reg_alpha"] == 1.0
    assert model.get_params()["reg_lambda"] == 1.0
    assert model.get_params()["random_state"] == SEED


def test_fit_meta_learners_regularized_returns_only_requested_learners(meta_split, meta_encoder):
    train_df, _ = meta_split
    models = fit_meta_learners_regularized(
        train_df, POOLED_TREATMENT_COL, "visit", meta_encoder, learner_names=("X",),
    )
    assert set(models.keys()) == {"X"}


def test_s_learner_linear_interaction_recovers_known_interaction_effect():
    """Dados sinteticos com CATE(x) = 3 + 4*x1 conhecido (por construcao):
    y = 5 + 2*x0 + 3*T + 4*T*x1 + ruido minimo. alpha quase zero para
    aproximar OLS (sem vies de encolhimento do ElasticNet) e verificar que a
    formula analitica recupera o efeito verdadeiro."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.uniform(-1, 1, size=(n, 2))
    treatment = rng.integers(0, 2, size=n)
    y = 5 + 2 * X[:, 0] + 3 * treatment + 4 * treatment * X[:, 1] + rng.normal(0, 0.01, size=n)

    model = fit_s_learner_linear_interaction(X, treatment, y, alpha=1e-6, l1_ratio=0.0)
    cate = predict_s_learner_linear_interaction_uplift(model, X)
    true_cate = 3 + 4 * X[:, 1]
    np.testing.assert_allclose(cate, true_cate, atol=0.1)


def test_s_learner_linear_interaction_without_true_interaction_is_not_forced_to_zero():
    """Sem interacao verdadeira no dado gerador, o CATE estimado deve ficar
    proximo da constante beta_T (nao necessariamente exatamente igual a
    zero-interacao, mas nao deve ter dependencia forte em X)."""
    rng = np.random.default_rng(1)
    n = 2000
    X = rng.uniform(-1, 1, size=(n, 2))
    treatment = rng.integers(0, 2, size=n)
    y = 5 + 2 * X[:, 0] + 3 * treatment + rng.normal(0, 0.01, size=n)

    model = fit_s_learner_linear_interaction(X, treatment, y, alpha=1e-6, l1_ratio=0.0)
    cate = predict_s_learner_linear_interaction_uplift(model, X)
    assert cate.std() < 0.2
    assert abs(cate.mean() - 3) < 0.2


def test_fit_meta_learners_regularized_accepts_extended_features(meta_split):
    train_df, _ = meta_split
    train_ext = add_engineered_features(train_df)
    encoder_ext = build_meta_learner_encoder(train_ext, cat_vars=EXTENDED_CAT_VARS)
    models = fit_meta_learners_regularized(
        train_ext, POOLED_TREATMENT_COL, "visit", encoder_ext,
        cont_vars=EXTENDED_CONT_VARS, bin_vars=EXTENDED_BIN_VARS, cat_vars=EXTENDED_CAT_VARS,
        learner_names=("R",),
    )
    X = encode_meta_learner_features(
        train_ext, encoder_ext, cont_vars=EXTENDED_CONT_VARS, bin_vars=EXTENDED_BIN_VARS, cat_vars=EXTENDED_CAT_VARS,
    )
    assert len(models["R"].predict(X).ravel()) == len(train_ext)


# ---- S4.7: ablation de propensão conhecida (X/R) --------------------------


def test_fit_x_learner_with_fixed_p_requires_p_at_predict(meta_split, meta_encoder):
    """Trava mecânica verificada empiricamente (não suposta): X-learner ajustado
    com p explícito não seta self.propensity_model (fica None, valor default
    do atributo) — predict sem p estoura TypeError ('NoneType' não é
    subscritável), não AttributeError."""
    train_df, _ = meta_split
    X = encode_meta_learner_features(train_df, meta_encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df["visit"].to_numpy(dtype=float)
    p_fixed = np.full(len(train_df), 2 / 3)
    model = fit_single_meta_learner("X", X, treatment, y, p=p_fixed)
    with pytest.raises(TypeError):
        model.predict(X)


def test_predict_single_meta_learner_x_uses_provided_p(meta_split, meta_encoder):
    train_df, _ = meta_split
    X = encode_meta_learner_features(train_df, meta_encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df["visit"].to_numpy(dtype=float)
    p_fixed = np.full(len(train_df), 2 / 3)
    model = fit_single_meta_learner("X", X, treatment, y, p=p_fixed)
    cate = predict_single_meta_learner("X", model, X, p=p_fixed)
    assert len(cate) == len(train_df)


def test_predict_single_meta_learner_r_ignores_p(meta_split, meta_encoder):
    """R-learner aceita p no predict mas o ignora — resultado deve ser idêntico
    independente do valor passado."""
    train_df, _ = meta_split
    X = encode_meta_learner_features(train_df, meta_encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df["visit"].to_numpy(dtype=float)
    model = fit_single_meta_learner("R", X, treatment, y)
    cate_no_p = predict_single_meta_learner("R", model, X, p=None)
    cate_with_p = predict_single_meta_learner("R", model, X, p=np.full(len(train_df), 0.5))
    np.testing.assert_allclose(cate_no_p, cate_with_p)


def test_fit_r_learner_with_fixed_p_changes_fitted_model(meta_split, meta_encoder):
    """R-learner usa p no fit (perda R) — p fixo != p estimado deve produzir
    um modelo tau diferente do default (mesmo treino/features)."""
    train_df, _ = meta_split
    X = encode_meta_learner_features(train_df, meta_encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df["visit"].to_numpy(dtype=float)
    model_default = fit_single_meta_learner("R", X, treatment, y, seed=SEED)
    model_fixed_p = fit_single_meta_learner("R", X, treatment, y, seed=SEED, p=np.full(len(train_df), 2 / 3))
    cate_default = predict_single_meta_learner("R", model_default, X)
    cate_fixed_p = predict_single_meta_learner("R", model_fixed_p, X)
    assert not np.allclose(cate_default, cate_fixed_p)


def test_s7_xtree_scores_passes_known_p_to_fit_and_predict(pooled_split, monkeypatch):
    """S7 must use the RCT design propensity instead of fitting a nuisance model."""
    train_df, val_df = pooled_split
    captured = []

    class FakeModel:
        pass

    def fake_fit(name, X, treatment, y, **kwargs):
        captured.append(("fit", kwargs["p"].copy()))
        return FakeModel()

    def fake_predict(name, model, X, **kwargs):
        captured.append(("predict", kwargs["p"].copy()))
        return np.zeros(len(X))

    monkeypatch.setattr(reports_module, "fit_single_meta_learner", fake_fit)
    monkeypatch.setattr(reports_module, "predict_single_meta_learner", fake_predict)
    reports_module.fit_development_xtree_scores(train_df, val_df, outcomes=("visit",))

    assert [kind for kind, _ in captured] == ["fit", "predict"]
    np.testing.assert_allclose(captured[0][1], np.full(len(train_df), 2 / 3))
    np.testing.assert_allclose(captured[1][1], np.full(len(val_df), 2 / 3))


# ---- compat.py ----------------------------------------------------------


def test_patch_sklearn_matplotlib_support_adds_attribute_if_missing(monkeypatch):
    import sklearn.utils as sklearn_utils

    monkeypatch.delattr(sklearn_utils, "check_matplotlib_support", raising=False)
    patch_sklearn_matplotlib_support()
    assert callable(sklearn_utils.check_matplotlib_support)
    assert sklearn_utils.check_matplotlib_support() is None


def test_patch_sklearn_matplotlib_support_is_noop_if_present(monkeypatch):
    import sklearn.utils as sklearn_utils

    sentinel = object()
    monkeypatch.setattr(sklearn_utils, "check_matplotlib_support", sentinel, raising=False)
    patch_sklearn_matplotlib_support()
    assert sklearn_utils.check_matplotlib_support is sentinel


# ---- eda.py ---------------------------------------------------------------


def test_treatment_distribution_shape_and_chi2(df):
    table, chi2, p = treatment_distribution(df, TREATMENT_COL, ARMS)
    assert list(table.index) == ARMS
    assert table["n"].sum() == len(df)
    assert chi2 >= 0
    assert 0.0 <= p <= 1.0


def test_smd_is_zero_for_identical_groups():
    same = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert smd_continuous(same, same) == pytest.approx(0.0)
    same_bin = pd.Series([0, 1, 1, 0])
    assert smd_binary(same_bin, same_bin) == pytest.approx(0.0)


def test_build_smd_table_row_count_and_columns(df):
    smd = build_smd_table(df, TREATMENT_COL, CONTROL_ARM, "Mens E-Mail", CONT_VARS, BIN_VARS, CAT_VARS)
    n_cat_levels = sum(df[v].nunique() for v in CAT_VARS)
    assert len(smd) == len(CONT_VARS) + len(BIN_VARS) + n_cat_levels
    assert set(smd.columns) == {"variable", "type", "mean_control", "mean_treated", "smd"}


def test_build_smd_table_same_variable_order_across_arms(df):
    smd_mens = build_smd_table(df, TREATMENT_COL, CONTROL_ARM, "Mens E-Mail", CONT_VARS, BIN_VARS, CAT_VARS)
    smd_womens = build_smd_table(df, TREATMENT_COL, CONTROL_ARM, "Womens E-Mail", CONT_VARS, BIN_VARS, CAT_VARS)
    assert list(smd_mens["variable"]) == list(smd_womens["variable"])


def test_formal_balance_tests_labels_and_flags(df):
    result = formal_balance_tests(df, TREATMENT_COL, ARMS, CONT_VARS, BIN_VARS + CAT_VARS)
    assert set(result.loc[result["variable"].isin(CONT_VARS), "test"]) == {"ANOVA F"}
    assert set(result.loc[result["variable"].isin(BIN_VARS + CAT_VARS), "test"]) == {"chi-squared"}
    assert set(result["flag"]).issubset({"ok", "WARN"})


def test_outcome_summary_by_arm_matches_manual_groupby(df):
    summary = outcome_summary_by_arm(df, TREATMENT_COL, ARMS)
    manual = df.groupby(TREATMENT_COL, observed=True)["visit"].mean()
    for arm in ARMS:
        assert summary.loc[arm, "visit_rate"] == pytest.approx(manual[arm], abs=1e-4)


# ---- effects.py -------------------------------------------------------


def test_ate_binary_ci_contains_point_estimate(df):
    result = ate_binary(df, TREATMENT_COL, "visit", "Mens E-Mail", CONTROL_ARM)
    assert result["ci_low"] < result["ate"] < result["ci_high"]
    assert result["ate"] > 0
    assert result["p_value"] < 0.05


def test_ate_continuous_ci_contains_point_estimate(df):
    result = ate_continuous(df, TREATMENT_COL, "spend", "Mens E-Mail", CONTROL_ARM)
    assert result["ci_low"] < result["ate"] < result["ci_high"]


def test_ate_table_shape_and_columns(df):
    result = ate_table(
        df, TREATMENT_COL, TREATED_ARMS, CONTROL_ARM,
        binary_outcomes=["visit", "conversion"], continuous_outcome="spend",
    )
    assert len(result) == len(TREATED_ARMS) * 3
    assert list(result.columns) == ["treatment", "outcome", "ate", "se", "ci_low", "ci_high", "p_value"]


# ---- viz.py -------------------------------------------------------------


def test_apply_plot_style_runs_without_error():
    apply_plot_style()


def test_add_chart_header_writes_title_and_subtitle():
    fig, _ = plt.subplots()
    add_chart_header(fig, title="Título", subtitle="Subtítulo")
    assert len(fig.texts) == 2
    plt.close(fig)


def test_add_chart_header_omits_subtitle_when_none():
    fig, _ = plt.subplots()
    add_chart_header(fig, title="Título")
    assert len(fig.texts) == 1
    plt.close(fig)


def test_add_chart_footer_writes_nothing_when_empty():
    fig, _ = plt.subplots()
    add_chart_footer(fig)
    assert len(fig.texts) == 0
    plt.close(fig)


def test_add_chart_footer_joins_parts():
    fig, _ = plt.subplots()
    add_chart_footer(fig, data_source="X", method="Y")
    assert len(fig.texts) == 1
    assert "X" in fig.texts[0].get_text() and "Y" in fig.texts[0].get_text()
    plt.close(fig)


def test_plot_univariate_continuous_returns_fig_and_axes(df):
    fig, axes = plot_univariate_continuous(df, CONT_VARS, titles=CONT_VARS, title="t", subtitle="s")
    assert len(axes) == len(CONT_VARS)
    plt.close(fig)


def test_plot_univariate_categorical_returns_fig_and_axes(df):
    fig, axes = plot_univariate_categorical(df, BIN_VARS, CAT_VARS, title="t", subtitle="s")
    assert axes.size >= len(BIN_VARS) + len(CAT_VARS)
    plt.close(fig)


def test_plot_love_plot_returns_fig_and_ax(df):
    smd_mens = build_smd_table(df, TREATMENT_COL, CONTROL_ARM, "Mens E-Mail", CONT_VARS, BIN_VARS, CAT_VARS)
    smd_womens = build_smd_table(df, TREATMENT_COL, CONTROL_ARM, "Womens E-Mail", CONT_VARS, BIN_VARS, CAT_VARS)
    fig, _ax = plot_love_plot(
        {"Mens": smd_mens, "Womens": smd_womens}, title="t", xlabel="x", subtitle="s",
    )
    plt.close(fig)


def test_plot_outcomes_by_arm_returns_fig_and_axes(df):
    outcomes = ["visit", "conversion", "spend"]
    fig, axes = plot_outcomes_by_arm(
        df, TREATMENT_COL, ARMS, outcomes=outcomes, ylabels=outcomes, titles=outcomes,
        title="t", subtitle="s",
    )
    assert len(axes) == len(outcomes)
    plt.close(fig)


def test_plot_split_overview_returns_fig_and_ax(pooled_split):
    train_df, val_df = pooled_split
    labels = {
        "partition_row": "Partição", "arm_row": "Braço", "outcome_row": "Outcome",
        "train": "Treino", "val": "Validação", "sealed": "Teste selado",
        "outcome_neg": "Não visitou (0)", "outcome_pos": "Visitou (1)",
    }
    fig, ax = plot_split_overview(train_df, val_df, n_test=500, treatment_col=TREATMENT_COL,
                                   arms=ARMS, outcome_col="visit", labels=labels,
                                   title="t", subtitle="s")
    assert ax.get_xlim()[1] == pytest.approx(len(train_df) + len(val_df) + 500)
    plt.close(fig)


def test_plot_ate_forest_returns_fig_and_axes(df):
    result = ate_table(
        df, TREATMENT_COL, TREATED_ARMS, CONTROL_ARM,
        binary_outcomes=["visit", "conversion"], continuous_outcome="spend",
    )
    outcomes = ["visit", "conversion", "spend"]
    fig, axes = plot_ate_forest(result, outcomes=outcomes, xlabel_prefix="ATE: ", title_prefix="",
                                 title="t", subtitle="s")
    assert len(axes) == len(outcomes)
    plt.close(fig)


# ---- learners.py / evaluation.py (S5: Causal Forest, Uplift Tree/RF) ------


def test_uplift_treatment_labels_maps_zero_one_to_control_treated():
    labels = _uplift_treatment_labels(np.array([0, 1, 1, 0]))
    assert list(labels) == ["control", "treated", "treated", "control"]


def test_fit_causal_forest_returns_uplift_with_expected_length_and_finite_values(meta_split, meta_encoder):
    train_df, val_df = meta_split
    X_train = encode_meta_learner_features(train_df, meta_encoder)
    X_val = encode_meta_learner_features(val_df, meta_encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df["visit"].to_numpy(dtype=float)
    model = fit_causal_forest(X_train, treatment, y)
    uplift = predict_causal_forest_uplift(model, X_val)
    assert len(uplift) == len(val_df)
    assert np.isfinite(uplift).all()
    assert uplift.std() > 0


def test_fit_uplift_tree_returns_uplift_with_expected_length_and_finite_values(meta_split, meta_encoder):
    train_df, val_df = meta_split
    X_train = encode_meta_learner_features(train_df, meta_encoder)
    X_val = encode_meta_learner_features(val_df, meta_encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df["visit"].to_numpy(dtype=float)
    model = fit_uplift_tree(X_train, treatment, y)
    assert model.classes_ == ["control", "treated"]
    uplift = predict_uplift_tree_uplift(model, X_val)
    assert len(uplift) == len(val_df)
    assert np.isfinite(uplift).all()


def test_fit_uplift_random_forest_returns_uplift_with_expected_length_and_finite_values(meta_split, meta_encoder):
    train_df, val_df = meta_split
    X_train = encode_meta_learner_features(train_df, meta_encoder)
    X_val = encode_meta_learner_features(val_df, meta_encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df["visit"].to_numpy(dtype=float)
    model = fit_uplift_random_forest(X_train, treatment, y)
    assert model.classes_ == ["control", "treated"]
    uplift = predict_uplift_random_forest_uplift(model, X_val)
    assert len(uplift) == len(val_df)
    assert np.isfinite(uplift).all()


@pytest.fixture(scope="module")
def synthetic_rct_with_known_heterogeneity():
    """RCT sintético com heterogeneidade conhecida por construção (efeito maior
    para x0 alto). Usado só para checar o *sentido* do uplift estimado pelos
    wrappers de S5 — não a recuperação numérica exata do efeito verdadeiro
    (modelos estocásticos não devem ser cobrados por essa precisão)."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.uniform(-1, 1, size=(n, 3))
    treatment = rng.integers(0, 2, size=n)
    true_cate = 0.10 + 0.15 * X[:, 0]
    base_p = 0.20 + 0.05 * X[:, 1]
    p1 = np.clip(base_p + true_cate, 0.01, 0.99)
    p0 = np.clip(base_p, 0.01, 0.99)
    y = rng.binomial(1, np.where(treatment == 1, p1, p0)).astype(float)
    return X, treatment, y, true_cate


def test_causal_forest_uplift_correlates_with_known_synthetic_effect(synthetic_rct_with_known_heterogeneity):
    X, treatment, y, true_cate = synthetic_rct_with_known_heterogeneity
    model = fit_causal_forest(X, treatment, y)
    uplift = predict_causal_forest_uplift(model, X)
    assert np.corrcoef(uplift, true_cate)[0, 1] > 0.3


def test_uplift_tree_uplift_correlates_with_known_synthetic_effect(synthetic_rct_with_known_heterogeneity):
    X, treatment, y, true_cate = synthetic_rct_with_known_heterogeneity
    model = fit_uplift_tree(X, treatment, y)
    uplift = predict_uplift_tree_uplift(model, X)
    assert np.corrcoef(uplift, true_cate)[0, 1] > 0.3


def test_uplift_random_forest_uplift_correlates_with_known_synthetic_effect(synthetic_rct_with_known_heterogeneity):
    X, treatment, y, true_cate = synthetic_rct_with_known_heterogeneity
    model = fit_uplift_random_forest(X, treatment, y)
    uplift = predict_uplift_random_forest_uplift(model, X)
    assert np.corrcoef(uplift, true_cate)[0, 1] > 0.3


def test_repeated_stratified_holdout_supports_new_s5_kinds_without_breaking_old_candidates(meta_split):
    train_df, _ = meta_split
    candidates = {
        "S": ("meta", "S", None, False),
        "Baseline": ("propensity", None, None, False),
        "CausalForest": ("causal_forest", None, None, False),
        "UpliftTree": ("uplift_tree", None, None, False),
        "UpliftRF": ("uplift_rf", None, None, False),
    }
    result = repeated_stratified_holdout(train_df, POOLED_TREATMENT_COL, "visit", candidates, n_reps=2)
    assert set(result["candidate"].unique()) == set(candidates)
    assert {"qini_auc", "uplift_auc", "uplift_at_30pct"}.issubset(result.columns)
    assert len(result) == 2 * len(candidates)


def test_repeated_stratified_holdout_still_splits_once_per_rep_with_s5_candidates(meta_split, monkeypatch):
    """Mesma trava do item 4 do hardening (S4), agora incluindo um candidato de
    S5 — os novos `kind` não podem introduzir uma chamada extra a
    `train_test_split` por candidato."""
    import src.evaluation as evaluation_mod

    train_df, _ = meta_split
    call_count = 0
    real_train_test_split = evaluation_mod.train_test_split

    def counting_train_test_split(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_train_test_split(*args, **kwargs)

    monkeypatch.setattr(evaluation_mod, "train_test_split", counting_train_test_split)

    candidates = {
        "S": ("meta", "S", None, False),
        "Baseline": ("propensity", None, None, False),
        "CausalForest": ("causal_forest", None, None, False),
    }
    n_reps = 2
    repeated_stratified_holdout(train_df, POOLED_TREATMENT_COL, "visit", candidates, n_reps=n_reps)
    assert call_count == n_reps





def test_notebooks_have_pt_br_and_en_us_layout():
    """Published notebooks live under explicit locale directories."""
    pt_dir = PROJECT_ROOT / "notebooks" / "pt-BR"
    en_dir = PROJECT_ROOT / "notebooks" / "en-US"
    pt_names = sorted(path.name for path in pt_dir.glob("*_PT.ipynb"))
    en_names = sorted(path.name for path in en_dir.glob("*_EN.ipynb"))
    expected_stems = [
        "01_Framing_EDA",
        "02_Baseline_Propensity",
        "03_Meta_Learners",
        "04_Causal_Forest_Uplift_Trees",
        "05_Evaluation_Sealed_Test",
        "06_Heterogeneity_Uplift_Funnel",
        "07_Policy_Learning_ROI",
        "08_Robustness_Limitations",
    ]
    assert pt_names == [f"{stem}_PT.ipynb" for stem in expected_stems]
    assert en_names == [f"{stem}_EN.ipynb" for stem in expected_stems]
    assert pt_names == [name.replace("_EN.ipynb", "_PT.ipynb") for name in en_names]
    assert not list((PROJECT_ROOT / "notebooks").glob("*_PT.ipynb"))
    assert not list((PROJECT_ROOT / "notebooks").glob("*_EN.ipynb"))


def test_notebooks_02_to_08_have_resolved_internal_tocs():
    """Every downstream notebook exposes stable links to all internal headings."""
    link_counts = {}
    for locale, suffix in (("pt-BR", "PT"), ("en-US", "EN")):
        paths = sorted((PROJECT_ROOT / "notebooks" / locale).glob(f"0[2-8]_*_{suffix}.ipynb"))
        assert len(paths) == 7

        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            toc_cells = [cell for cell in notebook.cells if cell.get("id") == "internal-toc"]
            assert len(toc_cells) == 1
            assert notebook.cells[1].get("id") == "internal-toc"

            links = re.findall(r"\]\(#([^)]+)\)", toc_cells[0].source)
            markdown = "\n".join(
                cell.source for cell in notebook.cells if cell.cell_type == "markdown"
            )
            anchors = re.findall(r'<a id="([^"]+)"></a>', markdown)
            assert links
            assert len(anchors) == len(set(anchors))
            assert links == anchors
            link_counts.setdefault(path.name[:2], []).append(len(links))

    assert all(len(counts) == 2 and counts[0] == counts[1] for counts in link_counts.values())


# ---- reports.py (S7) -------------------------------------------------------


def test_s7_score_quantiles_and_profile_are_well_formed(meta_split):
    _, val_df = meta_split
    scores = {
        "visit": np.linspace(-0.2, 0.3, len(val_df)),
        "conversion": np.linspace(0.3, -0.2, len(val_df)),
        "spend": np.sin(np.arange(len(val_df))),
    }
    scored = add_score_quantiles(val_df, scores)
    assert {"score_visit", "q_visit", "score_conversion", "q_conversion", "score_spend", "q_spend"}.issubset(scored)
    profile = top_bottom_profile(scored)
    assert {"variable", "level", "bottom_quantile", "top_quantile", "delta_top_minus_bottom"}.issubset(profile)
    assert {"recency", "history_segment", "channel", "zip_code", "newbie", "mens", "womens", "history"}.issubset(
        set(profile["variable"])
    )


def test_s7_funnel_summary_uses_each_outcome_ranking(meta_split):
    _, val_df = meta_split
    scores = {
        "visit": np.linspace(-0.2, 0.3, len(val_df)),
        "conversion": np.linspace(0.3, -0.2, len(val_df)),
        "spend": np.linspace(0.0, 1.0, len(val_df)),
    }
    scored = add_score_quantiles(val_df, scores)
    metrics, corr = funnel_ranking_summary(scored)
    assert set(metrics["outcome"]) == {"visit", "conversion", "spend"}
    assert {"qini_auc", "uplift_auc", "uplift_at_30pct"}.issubset(metrics.columns)
    assert set(corr["score_a"]) == {"visit", "conversion", "spend"}
    assert set(corr["score_b"]) == {"visit", "conversion", "spend"}


def test_s7_surrogate_is_descriptive_tree(meta_split):
    _, val_df = meta_split
    scored = add_score_quantiles(val_df, {"visit": np.linspace(-1, 1, len(val_df))})
    surrogate = fit_high_uplift_surrogate(scored)
    assert set(surrogate) == {"positive_rate", "balanced_accuracy", "rules"}
    assert 0 < surrogate["positive_rate"] < 1
    assert 0 <= surrogate["balanced_accuracy"] <= 1
    assert "class:" in surrogate["rules"]


def test_s7_report_builder_does_not_need_sealed_test(meta_split):
    train_df, val_df = meta_split
    report = build_s7_heterogeneity_report(train_df, val_df, full_df=pd.concat([train_df, val_df]))
    assert {
        "scored_validation", "profile", "quantile_outcomes", "funnel_metrics",
        "funnel_spearman", "surrogate", "conversion_positives", "conversion_rate",
    } == set(report)
    assert set(report["funnel_metrics"]["outcome"]) == {"visit", "conversion", "spend"}
    assert "q_visit" in report["scored_validation"]


def test_s7_notebooks_exist_and_do_not_use_shap():
    paths = [
        PROJECT_ROOT / "notebooks" / "pt-BR" / "06_Heterogeneity_Uplift_Funnel_PT.ipynb",
        PROJECT_ROOT / "notebooks" / "en-US" / "06_Heterogeneity_Uplift_Funnel_EN.ipynb",
    ]
    for path in paths:
        assert path.exists()
        notebook = nbformat.read(path, as_version=4)
        text = "\n".join(
            cell.source
            for cell in notebook.cells
            if cell.cell_type in {"code", "markdown"}
        ).lower()
        assert "shap" not in text


def test_s7_notebooks_preserve_code_parity_except_language_selector():
    pt_path = PROJECT_ROOT / "notebooks" / "pt-BR" / "06_Heterogeneity_Uplift_Funnel_PT.ipynb"
    en_path = PROJECT_ROOT / "notebooks" / "en-US" / "06_Heterogeneity_Uplift_Funnel_EN.ipynb"
    pt_nb = nbformat.read(pt_path, as_version=4)
    en_nb = nbformat.read(en_path, as_version=4)
    assert [c.get("id") for c in pt_nb.cells] == [c.get("id") for c in en_nb.cells]
    pt_code = [c.source for c in pt_nb.cells if c.cell_type == "code"]
    en_code = [c.source for c in en_nb.cells if c.cell_type == "code"]
    assert len(pt_code) == len(en_code)
    normalized_en = [src.replace("make_lang('en')", "make_lang('pt')") for src in en_code]
    assert pt_code == normalized_en


def test_s7_does_not_change_frozen_s6_artifacts():
    assert hashlib.sha256((S6_DIR / "preregistration.json").read_bytes()).hexdigest() == EXPECTED_PREREGISTRATION_SHA256
    assert hashlib.sha256((S6_DIR / "final_models.joblib").read_bytes()).hexdigest() == EXPECTED_FINAL_MODELS_SHA256

# ---- guardas estruturais de notebook ---------------------------------------


def test_no_notebook_touches_sealed_test_early():
    """Só o futuro 05_Evaluation_Sealed_Test_PT pode abrir o teste selado. Guarda
    contra colar acidentalmente `unlock=True` no notebook errado — risco que só
    existe porque o projeto agora tem vários arquivos pequenos em vez de um único
    lugar visível."""
    notebooks_dir = PROJECT_ROOT / "notebooks"
    allowed = {"05_Evaluation_Sealed_Test_PT.ipynb", "05_Evaluation_Sealed_Test_EN.ipynb"}
    offenders = []
    for path in sorted(notebooks_dir.glob("*/*.ipynb")):
        if path.name in allowed:
            continue
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code" and "unlock=True" in cell.get("source", ""):
                offenders.append((path.name, cell.get("id")))
    assert offenders == [], f"'unlock=True' encontrado fora de S6: {offenders}"


# ---- S6: pré-registro, bootstrap, modelos congelados, guarda do Notebook 05 ----

S6_DIR = PROJECT_ROOT / "artifacts" / "s6"
NOTEBOOK_05_PATH = PROJECT_ROOT / "notebooks" / "pt-BR" / "05_Evaluation_Sealed_Test_PT.ipynb"

S6_EXPECTED_MODELS = {"UpliftTree", "X+Tree(depth=4)", "S+LightGBM(vanilla)", "Baseline (propensão)"}


@pytest.fixture(scope="module")
def preregistration():
    return json.loads((S6_DIR / "preregistration.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def final_models_bundle():
    return joblib.load(S6_DIR / "final_models.joblib")


def test_preregistration_json_has_expected_primary_spec(preregistration):
    assert preregistration["primary_model"] == "UpliftTree"
    assert "Qini AUC" in preregistration["primary_metric"]
    assert preregistration["primary_comparison"] == "UpliftTree - Baseline (propensão)"
    assert preregistration["bootstrap_protocol"]["n_boot"] == 2000
    assert preregistration["status"] == "preregistered_before_sealed_test"
    assert preregistration["created_before_test_unlock"] is True


def test_preregistration_json_lists_all_four_models(preregistration):
    assert set(preregistration["model_specs"].keys()) == S6_EXPECTED_MODELS
    assert set(preregistration["bootstrap_protocol"]["candidates"]) == S6_EXPECTED_MODELS


def _synthetic_bootstrap_inputs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    arms = rng.choice(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], size=n)
    treatment = (arms != "No E-Mail").astype(int)
    y = rng.binomial(1, 0.10 + 0.05 * treatment, size=n).astype(float)
    return rng, arms, treatment, y


def test_bootstrap_qini_comparison_returns_finite_cis():
    rng, arms, treatment, y = _synthetic_bootstrap_inputs()
    scores = {"A": rng.normal(size=len(y)), "B": rng.normal(size=len(y))}
    result = bootstrap_qini_comparison(y, treatment, arms, scores, deltas=[("A", "B")], n_boot=50, seed=0)
    for name in ("A", "B"):
        assert np.isfinite(result["candidates"][name]["ci_low"])
        assert np.isfinite(result["candidates"][name]["ci_high"])
    assert np.isfinite(result["deltas"]["A - B"]["ci_low"])
    assert np.isfinite(result["deltas"]["A - B"]["ci_high"])


def test_bootstrap_qini_comparison_delta_is_zero_for_identical_scores():
    """Reamostra o MESMO índice bootstrap para todos os scores (pareado) — se
    não fosse assim, dois arrays de score idênticos ainda produziriam deltas
    não nulos por reamostragens independentes."""
    rng, arms, treatment, y = _synthetic_bootstrap_inputs(seed=1)
    score = rng.normal(size=len(y))
    result = bootstrap_qini_comparison(
        y, treatment, arms, {"A": score, "B": score.copy()}, deltas=[("A", "B")], n_boot=50, seed=0,
    )
    delta = result["deltas"]["A - B"]
    assert delta["point_estimate"] == 0.0
    assert delta["ci_low"] == 0.0
    assert delta["ci_high"] == 0.0


def test_stratified_bootstrap_indices_preserves_arm_sizes():
    arms = np.array(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    idx = _stratified_bootstrap_indices(arms, np.random.default_rng(3))
    boot_arms = arms[idx]
    assert (boot_arms == "a").sum() == 50
    assert (boot_arms == "b").sum() == 30
    assert (boot_arms == "c").sum() == 20
    assert len(idx) == len(arms)


def test_stratified_bootstrap_indices_does_not_stratify_by_outcome():
    """Dentro de um único braço, a fração de outcome=1 amostrada varia entre
    réplicas — se a reamostragem também estratificasse por outcome, essa
    fração ficaria travada no valor original em toda réplica."""
    arms = np.array(["a"] * 200)
    outcome = np.array([1] * 20 + [0] * 180)
    means = [outcome[_stratified_bootstrap_indices(arms, np.random.default_rng(s))].mean() for s in range(30)]
    assert np.std(means) > 0


def test_final_models_metadata_contains_preregistration_hash(final_models_bundle):
    expected_hash = hashlib.sha256((S6_DIR / "preregistration.json").read_bytes()).hexdigest()
    assert final_models_bundle["metadata"]["preregistration_sha256"] == expected_hash


def test_final_models_metadata_lists_four_model_specs(final_models_bundle):
    metadata = final_models_bundle["metadata"]
    assert set(metadata["model_names"]) == S6_EXPECTED_MODELS
    assert set(metadata["hyperparameters"].keys()) == S6_EXPECTED_MODELS
    assert set(final_models_bundle["models"].keys()) == S6_EXPECTED_MODELS


def test_final_models_metadata_contains_dataset_fingerprint(final_models_bundle):
    assert final_models_bundle["metadata"]["dataset_fingerprint"] == dataset_fingerprint()


def test_final_models_trained_only_on_train_plus_val(final_models_bundle, df):
    metadata = final_models_bundle["metadata"]
    assert metadata["n_dev"] == metadata["n_train"] + metadata["n_val"]
    train_df, val_df = get_train_val(add_pooled_treatment(df), persist_test=False)
    assert metadata["n_train"] == len(train_df)
    assert metadata["n_val"] == len(val_df)


def test_real_persisted_splits_manifest_proportions_unchanged():
    """Splits existentes (artifacts/splits/) não foram alterados nesta rodada
    — só lê dataset_manifest.json (contagens/hash), nunca o teste selado."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["n_train"] + manifest["n_val"] + manifest["n_test"] == manifest["n_rows"]
    assert manifest["n_train"] / manifest["n_rows"] == pytest.approx(TRAIN_FRAC, abs=0.001)
    assert manifest["n_val"] / manifest["n_rows"] == pytest.approx(VAL_FRAC, abs=0.001)
    assert manifest["n_test"] / manifest["n_rows"] == pytest.approx(TEST_FRAC, abs=0.001)


# ---- S6 pós-execução: o teste selado já foi aberto uma única vez, com
# autorização explícita (ver Notebook 05, seções 6.1-6.7). Os testes abaixo
# validam o estado final persistido -- nenhum deles reabre o teste selado
# nem recalcula a avaliação confirmatória; todos leem artefatos já gravados
# em disco (`artifacts/s6/`) ou a estrutura estática do notebook.

S6_RESULTS_PATH = S6_DIR / "s6_results.json"
SEALED_TEST_SCORES_PATH = S6_DIR / "sealed_test_scores.parquet"
SENTINEL_PATH = S6_DIR / "SEALED_TEST_EVALUATED.json"

EXPECTED_PREREGISTRATION_SHA256 = "eb51d102617a2a72dd4de6495de625452267ae622600d7a705b3beff0925f329"
EXPECTED_FINAL_MODELS_SHA256 = "2d07afdd33ebf6317e6dbb4975f819ac993547c2bb5ef4060797520f959f464d"
EXPECTED_S6_ARTIFACT_HASHES = {
    "final_models.joblib": EXPECTED_FINAL_MODELS_SHA256,
    "preregistration.json": EXPECTED_PREREGISTRATION_SHA256,
    "s6_results.json": "5e31f843bde49cd2f2c839959a079952378e450a1b6e5f9b146b10b3d0e2d9ec",
    "SEALED_TEST_EVALUATED.json": "6a2e1d2192cc4264be1067675b3168094c2cfc542a7dd9e4445b63072b63c8f2",
    "sealed_test_scores.parquet": "c40fa53488b23636b32cf251a38723feade776fab6ad029ba187a808e856ab70",
}


@pytest.fixture(scope="module")
def s6_results():
    return json.loads(S6_RESULTS_PATH.read_text(encoding="utf-8"))


def _notebook_05_code_cells():
    nb = nbformat.read(NOTEBOOK_05_PATH, as_version=4)
    return [c for c in nb.cells if c.cell_type == "code"]


def test_notebook_05_contains_exactly_one_unlock_call():
    """(a) Exatamente uma chamada de código com unlock=True em todo o notebook."""
    code_cells = _notebook_05_code_cells()
    unlock_cells = [c.get("id") for c in code_cells if "unlock=True" in c.get("source", "")]
    assert unlock_cells == ["s6-4-code"], (
        f"Esperada exatamente uma célula de código com unlock=True (s6-4-code), achou: {unlock_cells}"
    )


def test_notebook_05_guard_cell_precedes_unlock_call():
    """(b) A célula de guarda (id='s6-3-code') aparece antes da célula que abre
    o teste. Localiza as células por id -- não depende do texto literal
    'UNLOCK_SEALED_TEST = False', que deixou de existir no notebook após a
    execução autorizada de S6."""
    code_cells = _notebook_05_code_cells()
    ids = [c.get("id") for c in code_cells]
    assert "s6-3-code" in ids, "Célula de guarda 's6-3-code' não encontrada em Notebook 05."
    guard_idx = ids.index("s6-3-code")
    unlock_idx = next(i for i, c in enumerate(code_cells) if "unlock=True" in c.source)
    assert guard_idx < unlock_idx


def test_notebook_05_no_fit_calls_after_guard():
    """(c) Nenhuma célula depois de 's6-3-code' contém .fit( -- nenhum modelo
    foi (ou pode ser) reajustado após a guarda, antes ou depois do unlock."""
    code_cells = _notebook_05_code_cells()
    ids = [c.get("id") for c in code_cells]
    guard_idx = ids.index("s6-3-code")
    offenders = [c.get("id") for c in code_cells[guard_idx + 1:] if ".fit(" in c.source]
    assert offenders == [], f"Ajuste de modelo encontrado após a guarda: {offenders}"


def test_notebook_05_guard_cell_shows_unlock_true_post_s6():
    """Estado pós-S6: a guarda registra UNLOCK_SEALED_TEST = True como marca
    histórica da execução autorizada -- não foi revertida para False."""
    code_cells = _notebook_05_code_cells()
    guard_cell = next(c for c in code_cells if c.get("id") == "s6-3-code")
    assert "UNLOCK_SEALED_TEST = True" in guard_cell.source
    assert "UNLOCK_SEALED_TEST = False" not in guard_cell.source


def test_notebook_05_unlock_true_flag_only_in_notebook_05():
    """(d) UNLOCK_SEALED_TEST = True só é permitido no Notebook 05."""
    notebooks_dir = PROJECT_ROOT / "notebooks"
    allowed = {"05_Evaluation_Sealed_Test_PT.ipynb", "05_Evaluation_Sealed_Test_EN.ipynb"}
    offenders = []
    for path in sorted(notebooks_dir.glob("*/*.ipynb")):
        if path.name in allowed:
            continue
        nb = nbformat.read(path, as_version=4)
        if any("UNLOCK_SEALED_TEST = True" in c.get("source", "") for c in nb.cells):
            offenders.append(path.name)
    assert offenders == [], f"UNLOCK_SEALED_TEST = True encontrado fora do Notebook 05: {offenders}"


def test_s6_result_artifacts_exist_post_execution():
    """(e) Os três artefatos produzidos pela execução confirmatória existem."""
    assert SENTINEL_PATH.exists(), f"{SENTINEL_PATH} não existe -- S6 não foi executada ou o sentinel foi removido."
    assert S6_RESULTS_PATH.exists(), f"{S6_RESULTS_PATH} não existe."
    assert SEALED_TEST_SCORES_PATH.exists(), f"{SEALED_TEST_SCORES_PATH} não existe."


def test_sealed_test_evaluated_sentinel_is_true():
    """(f) O sentinel contém evaluated: true."""
    sentinel = json.loads(SENTINEL_PATH.read_text(encoding="utf-8"))
    assert sentinel["evaluated"] is True


def test_s6_results_hashes_match_frozen_artifacts(s6_results):
    """(g) Os hashes registrados em s6_results.json são exatamente os do
    pré-registro/modelos congelados usados nesta execução -- e continuam
    batendo com os artefatos hoje em disco (nenhum foi alterado depois)."""
    assert s6_results["preregistration_sha256"] == EXPECTED_PREREGISTRATION_SHA256
    assert s6_results["final_models_sha256"] == EXPECTED_FINAL_MODELS_SHA256
    assert hashlib.sha256((S6_DIR / "preregistration.json").read_bytes()).hexdigest() == EXPECTED_PREREGISTRATION_SHA256
    assert hashlib.sha256((S6_DIR / "final_models.joblib").read_bytes()).hexdigest() == EXPECTED_FINAL_MODELS_SHA256


def test_s6_results_n_test_is_12800(s6_results):
    """(h) n_test persistido continua 12800 (20% do dataset)."""
    assert s6_results["n_test"] == 12800


def test_s6_results_primary_delta_matches_persisted_execution(s6_results, preregistration):
    """(i) O resultado primário salvo é exatamente o produzido na execução
    autorizada -- valida a estrutura e os valores já persistidos em
    s6_results.json, sem reabrir o teste selado nem recalcular o bootstrap."""
    primary_key = preregistration["primary_comparison"]
    delta_primary = s6_results["bootstrap"]["deltas"][primary_key]
    assert delta_primary["point_estimate"] == pytest.approx(-0.008844719981198396, abs=1e-12)
    assert delta_primary["ci_low"] == pytest.approx(-0.04915708997452239, abs=1e-12)
    assert delta_primary["ci_high"] == pytest.approx(0.03024856926795489, abs=1e-12)
    assert delta_primary["ci_low"] < 0 < delta_primary["ci_high"], (
        "IC do delta primário deveria conter zero -- veredito registrado foi "
        "'não há evidência confirmatória de vantagem'."
    )


# ---- S8: policy learning and ROI ------------------------------------------


def test_s8_budget_mask_has_deterministic_tie_break_and_respects_cap():
    scores = np.array([1.0, 1.0, 0.5, 0.1])
    assert budget_mask(scores, 0.5).tolist() == [True, True, False, False]
    assert budget_mask(np.ones(10), 0.31).sum() == 3


def test_s8_binary_ipw_closed_endpoints():
    y = np.ones(6)
    t = np.array([1, 1, 1, 1, 0, 0])
    assert binary_ipw_incremental_value(y, t, np.zeros(6, dtype=bool)) == pytest.approx(0.0)
    assert binary_ipw_incremental_value(y, t, np.ones(6, dtype=bool)) == pytest.approx(0.0)
    assert binary_ipw_incremental_value(y, t, np.array([1, 0, 0, 0, 0, 0], dtype=bool)) == pytest.approx(0.25)


def test_s8_treat_all_is_marked_infeasible_below_full_budget():
    masks = make_binary_policy_masks({"propensity": np.arange(10, dtype=float)}, [0.3, 1.0])
    assert masks["treat_all"][0.3].all()
    val = pd.DataFrame({
        POOLED_TREATMENT_COL: [1, 0] * 5,
        TREATMENT_COL: pd.Categorical([ARMS[0], ARMS[1]] * 5, categories=ARMS),
        "visit": [0, 1] * 5,
        "conversion": [0, 0] * 5,
        "spend": [0.0, 1.0] * 5,
    })
    result = evaluate_binary_policies(val, masks, n_boot=4)
    assert not result.loc[(result.policy == "treat_all") & (result.budget == 0.3), "budget_feasible"].iloc[0]
    assert result.loc[(result.policy == "treat_all") & (result.budget == 1.0), "budget_feasible"].iloc[0]


def test_s8_bootstrap_requires_positive_n_boot():
    with pytest.raises(ValueError):
        evaluate_binary_policies(
            pd.DataFrame({POOLED_TREATMENT_COL: [0, 1], TREATMENT_COL: [ARMS[0], ARMS[1]], "visit": [0, 1], "conversion": [0, 0], "spend": [0.0, 1.0]}),
            {"no_contact": {0.5: np.array([False, False])}},
            n_boot=0,
        )


def test_s8_roi_algebra_and_zero_contact():
    result = roi_metrics(10.0, 0.2, 0.4, 1.0, n_customers=100)
    assert result["incremental_revenue"] == pytest.approx(1000.0)
    assert result["gross_profit"] == pytest.approx(400.0)
    assert result["campaign_cost"] == pytest.approx(20.0)
    assert result["net_profit"] == pytest.approx(380.0)
    assert result["roi"] == pytest.approx(19.0)
    assert result["value_per_1000"] == pytest.approx(3800.0)
    assert result["break_even_cost"] == pytest.approx(20.0)
    zero = roi_metrics(10.0, 0.0, 0.4, 1.0)
    assert np.isnan(zero["roi"])
    assert np.isnan(zero["break_even_cost"])


def test_s8_three_way_economic_gains_force_no_email_when_not_profitable():
    predictions = {
        CONTROL_ARM: {"spend": np.array([10.0, 10.0, 10.0])},
        "Mens E-Mail": {"spend": np.array([10.1, 15.0, 9.0])},
        "Womens E-Mail": {"spend": np.array([10.2, 11.0, 8.0])},
    }
    gains = three_way_net_gains(predictions, gross_margin=0.5, email_cost=1.0)
    assert three_way_actions(gains, 1.0).tolist() == [0, 1, 0]


def test_s8_three_way_surrogate_keeps_multiple_synthetic_classes():
    n_per_class = 200
    val = pd.DataFrame({
        "recency": np.repeat([1, 2, 3], n_per_class),
        "history": np.zeros(3 * n_per_class),
        "mens": np.zeros(3 * n_per_class, dtype=int),
        "womens": np.zeros(3 * n_per_class, dtype=int),
        "newbie": np.zeros(3 * n_per_class, dtype=int),
        "zip_code": ["A"] * (3 * n_per_class),
        "channel": ["B"] * (3 * n_per_class),
        "history_segment": ["C"] * (3 * n_per_class),
    })
    actions = np.repeat([0, 1, 2], n_per_class)
    surrogate = fit_three_way_surrogate(val, actions, seed=SEED)
    assert len(surrogate["action_distribution"]) == 3
    assert surrogate["balanced_accuracy"] > 0.5
    assert surrogate["rules"].count("class:") >= 2


def test_s8_multi_arm_ipw_closed_endpoints():
    observed = np.array([CONTROL_ARM, "Mens E-Mail", "Womens E-Mail"] * 2)
    y = np.ones(6)
    assert multi_arm_ipw_incremental_value(y, observed, np.zeros(6, dtype=int)) == pytest.approx(0.0)
    assert multi_arm_ipw_incremental_value(y, observed, np.ones(6, dtype=int)) == pytest.approx(0.0)
    assert multi_arm_ipw_incremental_value(y, observed, np.full(6, 2, dtype=int)) == pytest.approx(0.0)


def test_s8_three_way_bootstrap_returns_reproducible_intervals():
    val = pd.DataFrame({
        TREATMENT_COL: pd.Categorical([CONTROL_ARM, "Mens E-Mail", "Womens E-Mail"] * 4, categories=ARMS),
        "visit": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        "conversion": [0] * 12,
        "spend": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    })
    actions = {"learned": {0.5: np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])}}
    first = evaluate_three_way_policies(val, actions, n_boot=8, seed=7)
    second = evaluate_three_way_policies(val, actions, n_boot=8, seed=7)
    assert {"ci_low", "ci_high"}.issubset(first.columns)
    assert first["ci_low"].tolist() == second["ci_low"].tolist()
    assert first["ci_high"].tolist() == second["ci_high"].tolist()


def test_s8_literal_all_arm_actions_are_not_targeting():
    assert all_arm_actions(4, 1).tolist() == [1, 1, 1, 1]
    assert all_arm_actions(4, 0).tolist() == [0, 0, 0, 0]


def test_s8_stratified_bootstrap_preserves_arm_counts():
    labels = np.array([CONTROL_ARM] * 3 + ["Mens E-Mail"] * 4 + ["Womens E-Mail"] * 2)
    sampled = stratified_bootstrap_indices(labels, np.random.default_rng(42))
    assert {label: int(np.sum(labels[sampled] == label)) for label in ARMS} == {
        CONTROL_ARM: 3, "Mens E-Mail": 4, "Womens E-Mail": 2
    }


def test_s8_notebooks_are_bilingual_and_do_not_cross_confirmatory_boundary():
    pt_path = PROJECT_ROOT / "notebooks" / "pt-BR" / "07_Policy_Learning_ROI_PT.ipynb"
    en_path = PROJECT_ROOT / "notebooks" / "en-US" / "07_Policy_Learning_ROI_EN.ipynb"
    assert pt_path.exists() and en_path.exists()
    pt_nb = nbformat.read(pt_path, as_version=4)
    en_nb = nbformat.read(en_path, as_version=4)
    assert [cell.get("id") for cell in pt_nb.cells] == [cell.get("id") for cell in en_nb.cells]
    pt_code = [cell.source for cell in pt_nb.cells if cell.cell_type == "code"]
    en_code = [cell.source.replace("make_lang('en')", "make_lang('pt')") for cell in en_nb.cells if cell.cell_type == "code"]
    assert pt_code == en_code
    text = "\n".join(cell.source for cell in pt_nb.cells + en_nb.cells).lower()
    for forbidden in ("load_sealed_test", "unlock=true", "final_models", "sealed_test_scores", "shap"):
        assert forbidden not in text
    for notebook in (pt_nb, en_nb):
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        assert sum(len(source.splitlines()) for source in code) <= 35
        assert all("from src.policy import" not in source for source in code)
        assert all("import matplotlib" not in source for source in code)
        assert all("import seaborn" not in source for source in code)


def test_s8_builder_composes_domain_results_into_complete_report(monkeypatch):
    n = 4
    val = pd.DataFrame({"visit": [0, 1, 0, 1], "conversion": [0, 0, 0, 1], "spend": [0.0, 1.0, 0.0, 2.0]})
    policies = ["random", "propensity", "uplift_tree", "x_tree", "no_contact", "treat_all"]
    rows = []
    for policy in policies:
        for outcome in ("visit", "conversion", "spend"):
            rows.append({
                "policy": policy, "budget": 0.5, "outcome": outcome,
                "contact_rate": 0.5, "contacts": 2, "budget_feasible": True,
                "incremental_value": 0.1, "ci_low": 0.0, "ci_high": 0.2,
                "delta_vs_propensity": 0.0, "delta_ci_low": -0.1, "delta_ci_high": 0.1,
            })
    policy_curve = pd.DataFrame(rows)
    three_rows = []
    for policy in ("all_no_email", "all_mens", "all_womens", "random", "learned"):
        for outcome in ("visit", "conversion", "spend"):
            three_rows.append({
                "policy": policy, "budget": 0.5, "outcome": outcome,
                "no_email_rate": 0.5, "mens_rate": 0.25, "womens_rate": 0.25,
                "contact_rate": 0.5, "budget_feasible": True,
                "incremental_value": 0.1, "ci_low": 0.0, "ci_high": 0.2,
            })
    three_way_curve = pd.DataFrame(three_rows)
    monkeypatch.setattr(policy_reports_module, "fit_binary_policy_scores", lambda *a, **k: {"propensity": np.zeros(n), "uplift_tree": np.ones(n), "x_tree": np.ones(n)})
    monkeypatch.setattr(policy_reports_module, "make_binary_policy_masks", lambda *a, **k: {p: {0.5: np.array([1, 1, 0, 0], dtype=bool)} for p in policies})
    monkeypatch.setattr(policy_reports_module, "evaluate_binary_policies", lambda *a, **k: policy_curve)
    monkeypatch.setattr(policy_reports_module, "fit_three_way_models", lambda *a, **k: {"models": True})
    monkeypatch.setattr(policy_reports_module, "predict_three_way_models", lambda *a, **k: {"predictions": True})
    monkeypatch.setattr(policy_reports_module, "three_way_net_gains", lambda *a, **k: np.zeros((n, 2)))
    monkeypatch.setattr(policy_reports_module, "three_way_actions", lambda *a, **k: np.zeros(n, dtype=int))
    monkeypatch.setattr(policy_reports_module, "evaluate_three_way_policies", lambda *a, **k: three_way_curve)
    monkeypatch.setattr(policy_reports_module, "fit_three_way_surrogate", lambda *a, **k: {"fidelity": 1.0, "balanced_accuracy": 1.0, "action_distribution": {0: 1.0}, "rules": "class: 0"})
    report = build_s8_policy_report(pd.DataFrame(index=range(6)), val, budgets=(0.5,), representative_budgets=(0.5,))
    assert {"policy_curve", "binary_table", "roi_sensitivity", "economic_curve", "three_way_curve", "three_way_table", "surrogate", "assumptions"}.issubset(report)
    assert report["train_n"] == 6 and report["validation_n"] == 4
    assert report["binary_table"].shape[0] == len(policies)


def test_s8_report_persistence_display_and_plots(tmp_path, capsys):
    policy_curve = pd.DataFrame([
        {"policy": p, "budget": 0.3, "outcome": "visit", "incremental_value": 0.1, "delta_vs_propensity": 0.0, "ci_low": 0.0, "ci_high": 0.2, "budget_feasible": True, "contact_rate": 0.3}
        for p in ["random", "propensity", "uplift_tree", "x_tree", "no_contact", "treat_all"]
    ] + [
        {"policy": p, "budget": 0.3, "outcome": "spend", "incremental_value": 0.1, "delta_vs_propensity": 0.0, "ci_low": 0.0, "ci_high": 0.2, "budget_feasible": True, "contact_rate": 0.3}
        for p in ["random", "propensity", "uplift_tree", "x_tree", "no_contact", "treat_all"]
    ])
    report = {
        "train_n": 6, "validation_n": 4, "policy_curve": policy_curve,
        "policy_comparisons": policy_curve.copy(), "roi_sensitivity": pd.DataFrame([{"policy": "uplift_tree", "budget": 0.3, "gross_margin": 0.5, "email_cost": 0.05, "net_profit": 1.0}]),
        "economic_curve": pd.DataFrame([
            {"policy": p, "budget": 0.3, "budget_feasible": True, "value_per_1000": 1.0}
            for p in ["random", "propensity", "uplift_tree", "x_tree", "no_contact"]
        ]),
        "three_way_curve": pd.DataFrame([{"policy": "learned", "budget": 0.3, "outcome": "spend", "no_email_rate": 0.5, "mens_rate": 0.25, "womens_rate": 0.25, "contact_rate": 0.5, "budget_feasible": False, "incremental_value": 0.1, "ci_low": 0.0, "ci_high": 0.2}]),
        "binary_table": policy_curve[policy_curve["outcome"] == "visit"].copy(),
        "three_way_table": pd.DataFrame([{"policy": "learned", "budget": 0.3, "no_email_rate": 0.5, "mens_rate": 0.25, "womens_rate": 0.25, "incremental_value": 0.1, "ci_low": 0.0, "ci_high": 0.2, "budget_feasible": False}]),
        "surrogate": {"fidelity": 0.8, "balanced_accuracy": 0.7, "action_distribution": {0: 1.0}, "rules": "class: 0"},
        "assumptions": {"illustrative_margin": 0.5, "illustrative_email_cost": 0.05},
    }
    files = save_s8_policy_artifacts(report, tmp_path)
    assert {path.name for path in files.values()} == {"s8_policy_curve.csv", "s8_policy_comparisons.csv", "s8_roi_sensitivity.csv", "s8_three_way_policy.csv", "s8_assumptions.json"}
    display_s8_policy_report(report, lang=make_lang("pt"), artifacts_dir=tmp_path)
    assert "Artefatos S8 salvos em" in capsys.readouterr().out
    figures = plot_s8_policy_report(report, lang=make_lang("pt"), show=False)
    assert len(figures) == 4
    plt.close("all")


def test_s8_preserves_every_frozen_s6_artifact_hash():
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (S6_DIR).glob("*") if path.is_file()
    }
    assert actual == EXPECTED_S6_ARTIFACT_HASHES


# ---- S9: robustness and limitations --------------------------------------


def test_s9_curve_summaries_do_not_aggregate_estimates_or_ci_endpoints():
    report = build_s9_report(load_s9_inputs(PROJECT_ROOT / "artifacts"))
    summary = report["robustness_summary"]
    curve_rows = summary[summary["section"].isin(["binary_policy_stability", "economic_sensitivity", "three_way_learned"])]
    assert curve_rows[["estimate", "ci_low", "ci_high"]].isna().all().all()
    evidence_s8 = report["evidence_register"].query("evidence_id == 'S8_policy_validation'").iloc[0]
    assert pd.isna(evidence_s8["estimate"])
    assert pd.isna(evidence_s8["ci_low"])
    assert pd.isna(evidence_s8["ci_high"])
    assert curve_rows["positive_proportion"].between(0, 1).all()


def test_s9_first_plot_uses_comparable_proportions_and_localized_policy_labels():
    report = build_s9_report(load_s9_inputs(PROJECT_ROOT / "artifacts"))
    figures = plot_s9_report(report, lang=make_lang("pt"), show=False)
    ax = figures[0].axes[0]
    assert ax.get_ylabel() == "Proporção"
    assert len(ax.patches) == 2 * len(report["robustness_summary"].query("section == 'binary_policy_stability'"))
    assert "uplift_tree" not in " ".join(label.get_text() for label in ax.get_xticklabels())
    plt.close("all")


def test_s9_display_localizes_textual_values_and_preserves_pt_accents(capsys):
    report = build_s9_report(load_s9_inputs(PROJECT_ROOT / "artifacts"))
    display_s9_report(report, lang=make_lang("pt"))
    pt_output = capsys.readouterr().out
    assert "não pronto para implantação direta" in pt_output
    assert "não confirmada" in pt_output
    assert "surrogate descritivo da ação" in pt_output.lower()
    assert "nÃ" not in pt_output
    display_s9_report(report, lang=make_lang("en"))
    en_output = capsys.readouterr().out
    assert "not ready for direct deployment" in en_output
    assert "no confirmatory advantage" in en_output.lower()
    assert "não pronto para implantação direta" not in en_output
    assert "Hipótese primária não confirmada" not in en_output


def test_s9_persistence_and_modular_notebooks(tmp_path):
    report = build_s9_report(load_s9_inputs(PROJECT_ROOT / "artifacts"))
    files = save_s9_artifacts(report, tmp_path)
    assert {path.name for path in files.values()} == {
        "s9_evidence_register.csv", "s9_robustness_summary.csv", "s9_limitation_register.csv",
        "s9_decision_boundary.json", "s9_source_manifest.json",
    }
    pt_path = PROJECT_ROOT / "notebooks" / "pt-BR" / "08_Robustness_Limitations_PT.ipynb"
    en_path = PROJECT_ROOT / "notebooks" / "en-US" / "08_Robustness_Limitations_EN.ipynb"
    pt_nb = nbformat.read(pt_path, as_version=4)
    en_nb = nbformat.read(en_path, as_version=4)
    assert [cell.get("id") for cell in pt_nb.cells] == [cell.get("id") for cell in en_nb.cells]
    pt_code = [cell.source for cell in pt_nb.cells if cell.cell_type == "code"]
    en_code = [cell.source.replace("make_lang('en')", "make_lang('pt')") for cell in en_nb.cells if cell.cell_type == "code"]
    assert pt_code == en_code
    assert sum(len(source.splitlines()) for source in pt_code) <= 35
    text = "\n".join(cell.source for cell in pt_nb.cells + en_nb.cells).lower()
    for forbidden in ("sealed_test_scores", "final_models", "load_sealed_test", "unlock=true"):
        assert forbidden not in text


def test_s9_input_loading_fails_closed_for_missing_or_altered_allowed_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_s9_inputs(tmp_path)

    source = load_s9_inputs(PROJECT_ROOT / "artifacts")
    for path in source["paths"].values():
        relative = path.relative_to(PROJECT_ROOT / "artifacts")
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    registration = tmp_path / "s6" / "preregistration.json"
    altered = json.loads(registration.read_text(encoding="utf-8"))
    altered["status"] = "altered-for-test"
    registration.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="registration hash"):
        load_s9_inputs(tmp_path)
