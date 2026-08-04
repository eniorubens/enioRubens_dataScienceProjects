from __future__ import annotations

import json
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import conformal_uncertainty as conformal
from src import conformal_uncertainty_reports as reports


def _config(**overrides):
    base = conformal.ConformalUncertaintyConfig(
        run_mode="smoke",
        warmup_hours=4,
        rolling_window_hours=12,
        global_min_history=4,
        group_min_history=2,
        smoke_fold_limit=2,
        smoke_hours_per_fold=20,
        interval_coverages=(0.8, 0.9, 0.95),
        recency_half_life_hours=(2, 4, 8),
        aci_gamma_values=(0.01,),
        bootstrap_block_hours=2,
        smoke_bootstrap_repetitions=10,
        log_to_mlflow=False,
    )
    values = {**base.__dict__, **overrides}
    return conformal.ConformalUncertaintyConfig(**values)


def _write_replay_fixture(tmp_path: Path):
    config = _config(run_mode="full", runtime_root=tmp_path)
    audit = pd.DataFrame(
        [
            {
                "artifact": "source_predictions",
                "path": "source.csv",
                "bytes": 10,
                "sha256": "a" * 64,
            }
        ]
    )
    for filename in conformal._RESULT_TABLE_FILES.values():
        pd.DataFrame({"value": [1.0]}).to_csv(tmp_path / filename, index=False)
    pd.DataFrame(
        [
            conformal.ConformalMethodSpec(
                candidate_id="U0",
                method_id="U0",
                label="Baseline",
                calibration_window="expanding",
                nonconformity="absolute_residual",
                adaptive_alpha=False,
                uses_e4_scale=False,
                status="executed",
            ).__dict__
        ]
    ).to_csv(tmp_path / "method_specs.csv", index=False)
    manifest = {
        "code_version": conformal.CONFORMAL_CODE_VERSION,
        "config": json.loads(json.dumps(config.__dict__, default=str)),
        "input_hashes": audit.to_dict(orient="records"),
        "methods": [
            conformal.ConformalMethodSpec(
                candidate_id="U0",
                method_id="U0",
                label="Baseline",
                calibration_window="expanding",
                nonconformity="absolute_residual",
                adaptive_alpha=False,
                uses_e4_scale=False,
                status="executed",
            ).__dict__
        ],
    }
    (tmp_path / "conformal_uncertainty_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return config, audit


def test_conformal_results_replay_without_recalibration(tmp_path, monkeypatch):
    config, audit = _write_replay_fixture(tmp_path)
    monkeypatch.setattr(conformal, "require_environment", lambda: None)
    monkeypatch.setattr(conformal, "load_source_manifest", lambda path: {"run_mode": "full"})
    monkeypatch.setattr(conformal, "validate_source_manifest", lambda config, manifest: None)
    monkeypatch.setattr(conformal, "source_artifact_hashes", lambda config, manifest: audit)

    replay = conformal.load_conformal_calibration_results(config)

    assert replay.specs[0].candidate_id == "U0"
    assert replay.predictions["value"].tolist() == [1.0]
    assert replay.manifest_path.name == "conformal_uncertainty_manifest.json"


def test_conformal_results_replay_fails_closed_on_code_drift(tmp_path, monkeypatch):
    config, audit = _write_replay_fixture(tmp_path)
    manifest_path = tmp_path / "conformal_uncertainty_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code_version"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(conformal, "require_environment", lambda: None)
    monkeypatch.setattr(conformal, "load_source_manifest", lambda path: {"run_mode": "full"})
    monkeypatch.setattr(conformal, "validate_source_manifest", lambda config, manifest: None)
    monkeypatch.setattr(conformal, "source_artifact_hashes", lambda config, manifest: audit)

    with pytest.raises(ValueError, match="code_version"):
        conformal.load_conformal_calibration_results(config)


def _fold_frame(
    fold: int,
    year: int,
    role: str,
    start: str,
    residuals,
    scales=2.0,
) -> pd.DataFrame:
    residuals = np.asarray(residuals, dtype=float)
    if np.isscalar(scales):
        scales = np.repeat(float(scales), residuals.size)
    y_pred = np.repeat(100.0, residuals.size)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=residuals.size, freq="h"),
            "weekday": pd.date_range(start, periods=residuals.size, freq="h").weekday,
            "hour": pd.date_range(start, periods=residuals.size, freq="h").hour,
            "Rush_Period": np.where(np.arange(residuals.size) % 3 == 0, "Yes", "No"),
            "Seasons": "Winter",
            "Rainfall_Cat": np.where(np.arange(residuals.size) % 5 == 0, "Rain", "No Rain"),
            "fold": fold,
            "test_year": year,
            "fold_role": role,
            "selection_eligible": role == "selection",
            "y_true": y_pred + residuals,
            "y_pred": y_pred,
            "residual": residuals,
            "predicted_scale": np.asarray(scales, dtype=float),
            "scale_valid": np.isfinite(scales) & (np.asarray(scales) > 0),
        }
    )


def _two_folds() -> pd.DataFrame:
    first = _fold_frame(1, 2019, "selection", "2018-12-01", np.tile([1, -2, 3, -4], 5))
    second = _fold_frame(3, 2021, "selection", "2020-12-01", np.tile([2, -3, 4, -5], 5))
    return pd.concat([first, second], ignore_index=True)


def _source_rows(timestamp="2018-12-01", e4_y_true=101.0):
    common = {
        "timestamp": timestamp,
        "weekday": 5,
        "hour": 0,
        "Rush_Period": "No",
        "Seasons": "Winter",
        "Rainfall Cat": "No Rain",
        "fold": 1,
        "test_year": 2019,
        "fold_role": "selection",
        "selection_eligible": True,
        "y_pred": 100.0,
        "residual": 1.0,
    }
    return [
        {**common, "experiment_id": "E0", "y_true": 101.0, "predicted_scale": np.nan},
        {
            **common,
            "experiment_id": "E4",
            "y_true": e4_y_true,
            "predicted_scale": 2.0,
        },
    ]


def _write_source(tmp_path: Path, rows=None):
    predictions = tmp_path / "development_oof_predictions.csv"
    pd.DataFrame(rows or _source_rows()).to_csv(predictions, index=False)
    manifest = tmp_path / "uncertainty_experiments_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "code_version": "uncertainty_experiments_v1",
                "run_mode": "full",
                "dataset_fingerprint": "dataset",
                "regime_fingerprint": "regime",
                "cv_strategy": "ForwardMeteorologicalYearSplit",
                "cv_strategy_version": "ForwardMeteorologicalYearSplit_v3_normal_operations",
                "artifacts": {"predictions": str(predictions)},
            }
        ),
        encoding="utf-8",
    )
    return manifest, predictions


def test_source_manifest_rejects_non_full_artifact(tmp_path):
    manifest_path, predictions = _write_source(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_mode"] = "smoke"
    config = _config(source_manifest_path=manifest_path, source_predictions_path=predictions)
    with pytest.raises(ValueError, match="full"):
        conformal.validate_source_manifest(config, manifest)


def test_source_loader_aligns_exact_temporal_keys(tmp_path):
    manifest, predictions = _write_source(tmp_path)
    config = _config(source_manifest_path=manifest, source_predictions_path=predictions)
    frame = conformal.load_point_and_scale_predictions(config)
    assert len(frame) == 1
    assert frame.loc[0, "predicted_scale"] == 2.0
    assert frame.loc[0, "Rainfall_Cat"] == "No Rain"


def test_source_loader_rejects_duplicate_keys(tmp_path):
    rows = _source_rows() + [_source_rows()[0]]
    manifest, predictions = _write_source(tmp_path, rows)
    config = _config(source_manifest_path=manifest, source_predictions_path=predictions)
    with pytest.raises(ValueError, match="duplicate"):
        conformal.load_point_and_scale_predictions(config)


def test_source_loader_rejects_y_true_disagreement(tmp_path):
    manifest, predictions = _write_source(tmp_path, _source_rows(e4_y_true=999.0))
    config = _config(source_manifest_path=manifest, source_predictions_path=predictions)
    with pytest.raises(ValueError, match="y_true"):
        conformal.load_point_and_scale_predictions(config)


def test_source_hash_audit_contains_both_inputs(tmp_path):
    manifest, predictions = _write_source(tmp_path)
    config = _config(source_manifest_path=manifest, source_predictions_path=predictions)
    audit = conformal.source_artifact_hashes(config)
    assert set(audit["artifact"]) == {"source_manifest", "source_predictions"}
    assert audit["sha256"].str.len().eq(64).all()


def test_finite_conformal_quantile_uses_corrected_order_statistic():
    assert conformal._finite_conformal_quantile([1, 2, 3, 4], alpha=0.2) == 4


def test_weighted_quantile_favors_recent_high_score():
    value = conformal._weighted_quantile([1.0, 10.0], [0.01, 1.0], 0.8)
    assert value == 10.0


def test_u0_scores_first_fold_after_warmup():
    frame = _fold_frame(1, 2019, "selection", "2018-12-01", [1, 2, 3, 4, 5, 6])
    output = conformal._run_u0(frame, _config(smoke_fold_limit=1, smoke_hours_per_fold=6))
    primary = output.loc[np.isclose(output["coverage"], 0.9)]
    assert primary["interval_available"].tolist() == [False, False, False, False, True, True]
    assert primary.loc[primary["interval_available"], "calibration_size"].iloc[0] == 4


def test_current_observation_changes_only_next_u0_interval():
    config = _config(smoke_fold_limit=1, smoke_hours_per_fold=7)
    frame_a = _fold_frame(1, 2019, "selection", "2018-12-01", [1, 1, 1, 1, 1, 1, 1])
    frame_b = frame_a.copy()
    frame_b.loc[4, ["y_true", "residual"]] = [1100.0, 1000.0]
    out_a = conformal._run_u0(frame_a, config)
    out_b = conformal._run_u0(frame_b, config)
    row_a = out_a.loc[np.isclose(out_a["coverage"], 0.9)].reset_index(drop=True)
    row_b = out_b.loc[np.isclose(out_b["coverage"], 0.9)].reset_index(drop=True)
    assert row_a.loc[4, "upper"] == row_b.loc[4, "upper"]
    assert row_a.loc[5, "upper"] != row_b.loc[5, "upper"]


def test_state_resets_at_every_fold():
    output = conformal._run_u0(_two_folds(), _config())
    primary = output.loc[np.isclose(output["coverage"], 0.9)]
    starts = primary.groupby("fold", sort=True).head(4)
    assert (~starts["interval_available"]).all()


def test_stress_fold_never_feeds_later_fold():
    config = _config(run_mode="full", runtime_root=Path("unused"))
    first = _fold_frame(1, 2019, "selection", "2018-12-01", [1] * 8)
    stress = _fold_frame(2, 2020, "stress", "2019-12-01", [999] * 8)
    later = _fold_frame(3, 2021, "selection", "2020-12-01", [1] * 8)
    output = conformal._run_u0(pd.concat([first, stress, later]), config)
    later_primary = output.loc[
        output["fold"].eq(3) & np.isclose(output["coverage"], 0.9)
    ].reset_index(drop=True)
    assert later_primary.loc[4, "upper"] == pytest.approx(101.0)


def test_u1_produces_asymmetric_intervals():
    residuals = [-10, -8, 1, 2, 3, 4, 5]
    output = conformal._run_u1(
        _fold_frame(1, 2019, "selection", "2018-12-01", residuals),
        _config(smoke_fold_limit=1, smoke_hours_per_fold=7),
    )
    row = output.loc[np.isclose(output["coverage"], 0.8) & output["interval_available"]].iloc[-1]
    assert (row.y_pred - row.lower) != pytest.approx(row.upper - row.y_pred)


def test_u2_declares_every_half_life_variant():
    specs = conformal.conformal_method_specs(_config())
    assert {spec.candidate_id for spec in specs if spec.method_id == "U2"} == {
        "U2_h2",
        "U2_h4",
        "U2_h8",
    }


def test_u3_preserves_e0_point_and_varies_width_by_scale():
    frame = _fold_frame(
        1,
        2019,
        "selection",
        "2018-12-01",
        [1, 1, 1, 1, 1, 1],
        scales=[1, 1, 1, 1, 2, 4],
    )
    output = conformal._run_u3(frame, _config(smoke_fold_limit=1, smoke_hours_per_fold=6))
    primary = output.loc[np.isclose(output["coverage"], 0.9) & output["interval_available"]]
    assert primary["y_pred"].eq(100.0).all()
    assert primary.iloc[-1]["width"] > primary.iloc[0]["width"]


def test_u3_invalid_scale_uses_explicit_absolute_fallback():
    frame = _fold_frame(
        1,
        2019,
        "selection",
        "2018-12-01",
        [1, 1, 1, 1, 1, 1],
        scales=[1, 1, 1, 1, 0, 0],
    )
    frame["scale_valid"] = frame["predicted_scale"].gt(0)
    output = conformal._run_u3(frame, _config(smoke_fold_limit=1, smoke_hours_per_fold=6))
    scored = output.loc[output["interval_available"]]
    assert scored["fallback_used"].all()
    assert scored["status"].eq("fallback_absolute").all()


def test_aci_violation_reduces_alpha_for_next_observation():
    frame = _fold_frame(1, 2019, "selection", "2018-12-01", [1, 1, 1, 1, 100, 1])
    predictions, trace = conformal._run_u4(
        frame,
        _config(smoke_fold_limit=1, smoke_hours_per_fold=6),
        normalized_method=False,
        gamma=0.1,
    )
    primary = trace.loc[np.isclose(trace["coverage"], 0.9)].reset_index(drop=True)
    assert primary.loc[4, "alpha_after"] < primary.loc[4, "alpha_before"]
    assert primary.loc[5, "alpha_before"] == primary.loc[4, "alpha_after"]
    assert predictions["method_id"].eq("U4a").all()


def test_aci_covered_observation_increases_alpha_gradually():
    frame = _fold_frame(1, 2019, "selection", "2018-12-01", [10, -10, 10, -10, 0, 0])
    _, trace = conformal._run_u4(
        frame,
        _config(smoke_fold_limit=1, smoke_hours_per_fold=6),
        normalized_method=False,
        gamma=0.01,
    )
    primary = trace.loc[np.isclose(trace["coverage"], 0.9)].reset_index(drop=True)
    assert primary.loc[4, "alpha_after"] > primary.loc[4, "alpha_before"]


def test_aci_states_are_independent_by_coverage():
    frame = _fold_frame(1, 2019, "selection", "2018-12-01", [1] * 8)
    _, trace = conformal._run_u4(frame, _config(), normalized_method=False, gamma=0.01)
    initial = trace.groupby("coverage", sort=True)["alpha_before"].first()
    assert initial.loc[0.8] == pytest.approx(0.2)
    assert initial.loc[0.9] == pytest.approx(0.1)
    assert initial.loc[0.95] == pytest.approx(0.05)


def test_u5_falls_back_then_uses_specific_hierarchy():
    frame = _fold_frame(1, 2019, "selection", "2018-12-01", [1] * 10)
    frame["Rush_Period"] = "No"
    frame["Rainfall_Cat"] = "No Rain"
    frame["Seasons"] = "Winter"
    output = conformal._run_u5(frame, _config(smoke_fold_limit=1, smoke_hours_per_fold=10))
    primary = output.loc[np.isclose(output["coverage"], 0.9)].reset_index(drop=True)
    assert primary.loc[4, "hierarchy_level"] == "rush_rain_season"
    assert not bool(primary.loc[4, "fallback_used"])


def test_u5_group_selection_does_not_depend_on_current_target():
    config = _config(smoke_fold_limit=1, smoke_hours_per_fold=8)
    frame_a = _fold_frame(1, 2019, "selection", "2018-12-01", [1] * 8)
    frame_b = frame_a.copy()
    frame_b.loc[6, ["y_true", "residual"]] = [1000.0, 900.0]
    a = conformal._run_u5(frame_a, config)
    b = conformal._run_u5(frame_b, config)
    a = a.loc[np.isclose(a["coverage"], 0.9)].reset_index(drop=True)
    b = b.loc[np.isclose(b["coverage"], 0.9)].reset_index(drop=True)
    assert a.loc[6, "hierarchy_key"] == b.loc[6, "hierarchy_key"]
    assert a.loc[6, "upper"] == b.loc[6, "upper"]


def test_interval_nesting_95_is_not_narrower_than_90_or_80():
    output = conformal._run_u0(_two_folds(), _config())
    scored = output.loc[output["interval_available"]]
    pivot = scored.pivot_table(index=["fold", "timestamp"], columns="coverage", values="width")
    assert (pivot[0.95] >= pivot[0.9]).all()
    assert (pivot[0.9] >= pivot[0.8]).all()


def test_fold_metrics_exclude_warmup_and_report_miss_sides():
    output = conformal._run_u1(_two_folds(), _config())
    metrics = conformal.calibration_fold_metrics(output)
    assert metrics["n_warmup"].eq(4).all()
    assert metrics["n_scored"].eq(16).all()
    assert np.allclose(
        metrics["lower_miss_rate"] + metrics["upper_miss_rate"],
        1.0 - metrics["empirical_coverage"],
    )


def test_aggregate_metrics_keep_stress_out_of_ranking():
    config = _config(run_mode="full", runtime_root=Path("unused"))
    data = pd.concat(
        [
            _fold_frame(1, 2019, "selection", "2018-12-01", [1] * 8),
            _fold_frame(2, 2020, "stress", "2019-12-01", [999] * 8),
            _fold_frame(3, 2021, "selection", "2020-12-01", [1] * 8),
        ]
    )
    predictions = conformal._run_u0(data, config)
    folds = conformal.calibration_fold_metrics(predictions)
    aggregate = conformal.calibration_aggregate_metrics(folds, config)
    assert aggregate["n_folds_calibrated"].eq(2).all()


def test_scale_diagnostics_reports_deciles_and_spearman():
    frame = _two_folds()
    frame["predicted_scale"] = np.linspace(1, 10, len(frame))
    frame["residual"] = frame["predicted_scale"]
    diagnostics = conformal.scale_diagnostics(frame)
    overall = diagnostics.loc[diagnostics["scope"].eq("overall")]
    assert len(overall) == 10
    assert overall["spearman_scale_abs_error"].iloc[0] == pytest.approx(1.0)


def test_block_bootstrap_is_reproducible_and_fold_local():
    config = _config()
    predictions = conformal._run_u0(_two_folds(), config)
    first = conformal.block_bootstrap_coverage(predictions, config)
    second = conformal.block_bootstrap_coverage(predictions, config)
    pd.testing.assert_frame_equal(first, second)
    assert first["block_hours"].eq(2).all()


def _report_results():
    config = _config()
    base = _two_folds()
    base["predicted_scale"] = np.linspace(1.0, 10.0, len(base))
    base["scale_valid"] = True
    frames = [
        conformal._run_u0(base, config),
        conformal._run_u1(base, config),
        conformal._run_u2(base, config, 2),
        conformal._run_u3(base, config),
        conformal._run_u5(base, config),
    ]
    u4, trace = conformal._run_u4(base, config, False, 0.01)
    frames.append(u4)
    predictions = pd.concat(frames, ignore_index=True)
    folds = conformal.calibration_fold_metrics(predictions)
    aggregate = conformal.calibration_aggregate_metrics(folds, config)
    scale = conformal.scale_diagnostics(base)
    rolling = conformal.rolling_coverage_metrics(predictions, config)
    segments = conformal.segment_calibration_metrics(predictions, config)
    bootstrap = conformal.block_bootstrap_coverage(predictions, config)
    decision = conformal.experimental_decision_table(aggregate, folds, segments, config)
    return conformal.ConformalCalibrationResults(
        config=config,
        source_manifest={"code_version": "v1", "run_mode": "full"},
        input_audit=pd.DataFrame(
            [{"artifact": "source", "path": "x", "bytes": 1, "sha256": "a" * 64}]
        ),
        specs=conformal.conformal_method_specs(config),
        predictions=predictions,
        fold_metrics=folds,
        aggregate_metrics=aggregate,
        stress_metrics=folds.iloc[0:0].copy(),
        scale_diagnostics=scale,
        rolling_coverage=rolling,
        segment_metrics=segments,
        bootstrap_metrics=bootstrap,
        decision_table=decision,
        aci_alpha_trace=trace,
        manifest_path=Path("manifest.json"),
    )


@pytest.mark.parametrize(
    "factory",
    [
        reports.plot_coverage_calibration,
        reports.plot_fold_coverage_heatmap,
        reports.plot_coverage_width_pareto,
        reports.plot_scale_diagnostics,
        reports.plot_rolling_coverage,
        reports.plot_aci_alpha_trajectory,
        reports.plot_segment_coverage,
        reports.plot_representative_interval_windows,
    ],
)
def test_all_plots_return_figures_without_mutating(factory):
    results = _report_results()
    before = results.predictions.copy(deep=True)
    figure = factory(results, lang="pt")
    assert isinstance(figure, matplotlib.figure.Figure)
    pd.testing.assert_frame_equal(results.predictions, before)


def _figure_text(figure) -> str:
    text = [item.get_text() for item in figure.texts]
    for axis in figure.axes:
        text.extend(
            [
                axis.get_title(),
                axis.get_xlabel(),
                axis.get_ylabel(),
                *(item.get_text() for item in axis.get_xticklabels()),
                *(item.get_text() for item in axis.get_yticklabels()),
            ]
        )
        legend = axis.get_legend()
        if legend is not None:
            text.extend(item.get_text() for item in legend.get_texts())
    return "\n".join(filter(None, text))


def test_offline_english_catalog_covers_conformal_figures():
    results = _report_results()
    factories = (
        reports.plot_coverage_calibration,
        reports.plot_fold_coverage_heatmap,
        reports.plot_coverage_width_pareto,
        reports.plot_scale_diagnostics,
        reports.plot_rolling_coverage,
        reports.plot_aci_alpha_trajectory,
        reports.plot_segment_coverage,
        reports.plot_representative_interval_windows,
    )
    figures = [factory(results, lang="en") for factory in factories]
    rendered = "\n".join(_figure_text(figure) for figure in figures)
    try:
        for token in (
            "Cobertura",
            "Erro de cobertura",
            "Fronteira",
            "Decil da escala",
            "Trajetória",
            "regime operacional",
            "Semana representativa",
        ):
            assert token not in rendered
        for title in (
            "Nominal versus observed coverage",
            "Coverage error by candidate and normal-operation fold",
            "Calibration-width frontier",
            "E0 error by decile of the E4 predicted scale",
            "Prequential adaptive-alpha trajectory",
            "Coverage by operating regime",
            "Representative week in the most recent normal-operation fold",
        ):
            assert title in rendered
    finally:
        for figure in figures:
            plt.close(figure)


def test_decision_message_never_claims_point_successor():
    message = reports.synthesis_report(_report_results(), lang="pt")
    assert "Champion pontual E0 foi preservado" in message
    assert "nenhum sucessor pontual" in message.lower()


def test_english_synthesis_composes_stable_templates_without_language_mixing():
    results = _report_results()
    results.config.run_mode = "full"
    results.decision_table.loc[:, "coverage_gate"] = False
    winner = results.decision_table.index[0]
    results.decision_table.loc[winner, "coverage_gate"] = True
    results.decision_table.loc[winner, "experimental_rank"] = 1

    message = reports.synthesis_report(results, lang="en")

    assert "The full execution covered every declared fold." in message
    assert "Point Champion E0 was preserved" in message
    assert "most defensible experimental calibrator" in message
    for token in ("execução", "cobertura", "largura média", "holdout selado", "reajustado"):
        assert token not in message


@pytest.mark.parametrize(
    "path",
    [Path("src/conformal_uncertainty.py"), Path("src/conformal_uncertainty_reports.py")],
)
def test_new_modules_do_not_call_model_training_or_final_validation(path):
    source = path.read_text(encoding="utf-8")
    forbidden = [".fit(", ".predict(", "materialize_final_holdout", "run_final_validation"]
    assert not any(token in source for token in forbidden)


def test_method_specs_cover_u0_through_u5():
    methods = {spec.method_id for spec in conformal.conformal_method_specs(_config())}
    assert methods == {"U0", "U1", "U2", "U3", "U4a", "U4b", "U5"}
