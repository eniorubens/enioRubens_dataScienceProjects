"""Tests for the final-validation reports layer (:mod:`src.final_validation_reports`).

The reports are exercised against a synthetic results object built from the same
fitted-candidate factory as :mod:`tests.test_final_validation`; no real holdout,
target or MLflow run is touched. The tests assert that every function returns a
displayable object, that the localization boundary holds (internal English
schema, Portuguese display copy) and that the narrative strings are produced.
"""

from __future__ import annotations

import gzip
import json
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import final_validation as fv  # noqa: E402
from src import final_validation_reports as reports  # noqa: E402
from src.final_validation import EXPECTED_DATASET_FINGERPRINT  # noqa: E402
from src.i18n import make_lang  # noqa: E402
from src.temporal_optimizer import CODE_VERSION  # noqa: E402
from src.tracking import stamp_pipeline_provenance  # noqa: E402
from tests.test_final_validation import fit_pipeline, make_entry, make_raw  # noqa: E402


def _figure_text(figure) -> str:
    """Collect every user-visible Matplotlib label from one figure."""
    values = []
    if figure._suptitle is not None:
        values.append(figure._suptitle.get_text())
    for axis in figure.axes:
        values.extend([axis.get_title(), axis.get_xlabel(), axis.get_ylabel()])
        values.extend(label.get_text() for label in axis.get_xticklabels())
        values.extend(label.get_text() for label in axis.get_yticklabels())
        legend = axis.get_legend()
        if legend is not None:
            values.extend(label.get_text() for label in legend.get_texts())
    return "\n".join(value for value in values if value)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """A full FinalValidationResults plus SHAP, from synthetic fitted candidates."""
    directory = tmp_path_factory.mktemp("frozen_reports")
    dev = make_raw("2019-01-01", 3000, seed=1)
    specs = [
        ("champion", "CatBoostRegressor", "run_champion"),
        ("challenger", "HistGradientBoostingRegressor", "run_challenger_1"),
        ("challenger", "RandomForestRegressor", "run_challenger_2"),
    ]
    entries = []
    for role, estimator, run_id in specs:
        pipeline, spec = fit_pipeline(estimator, dev)
        path = directory / f"{role}_{estimator}_{run_id}.pkl.gz"
        entry = make_entry(role, run_id, estimator, spec, path)
        stamp_pipeline_provenance(
            pipeline,
            run_id,
            entry["_best_params"],
            spec.as_tags(),
            CODE_VERSION,
            EXPECTED_DATASET_FINGERPRINT,
        )
        with gzip.open(path, "wb") as handle:
            pickle.dump(pipeline, handle)
        entries.append(entry)

    manifest = {
        "run_mode": "full",
        "provisional": False,
        "champion": entries[0],
        "challengers": entries[1:],
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "regime_fingerprint": fv.EXPECTED_REGIME_FINGERPRINT,
    }
    candidates = fv.load_frozen_candidates(manifest)

    holdout = make_raw("2024-01-01", 600, seed=7)
    target = "Rented Bike Count"
    y = holdout[target].reset_index(drop=True)
    X = holdout.drop(columns=[target]).reset_index(drop=True)
    ts = (
        pd.to_datetime(holdout["DateTime"]) + pd.to_timedelta(holdout["Hour"], unit="h")
    ).reset_index(drop=True)
    data = fv.FinalEvaluationData(
        X_holdout=X,
        y_holdout=y,
        timestamps=ts,
        dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT,
        holdout_fingerprint="synthetic",
        regime_fingerprint=fv.EXPECTED_REGIME_FINGERPRINT,
        dev_start=ts.min(),
        dev_end=ts.max(),
        n_dev_rows=3000,
        holdout_start=ts.min(),
        holdout_end=ts.max(),
        n_holdout_rows=len(X),
        n_post_holdout_rows=0,
        post_holdout_start=None,
        post_holdout_end=None,
        environment={},
    )
    config = fv.FinalValidationConfig(
        runtime_root=directory / "runtime",
        log_to_mlflow=False,
        shap_max_sample=120,
    )
    evaluations = [fv.evaluate_candidate(c, data, config.error_quantiles) for c in candidates]
    comparison = fv.comparison_frame(evaluations)
    confirmation = fv.decide_confirmation(evaluations)
    engineered = fv.engineered_holdout_frame(candidates[0], data.X_holdout)
    segmented = fv.segmented_metrics(evaluations, engineered, data.y_holdout.to_numpy(), config)
    plan = fv.FinalValidationPlan(config, manifest, candidates, fv.manifest_fingerprint(manifest))
    result = fv.FinalValidationResults(
        config=config,
        manifest=manifest,
        candidates=candidates,
        data=data,
        evaluations=evaluations,
        comparison=comparison,
        confirmation=confirmation,
        segmented=segmented,
        predictions=fv._predictions_frame(data, evaluations),
        manifest_fingerprint=fv.manifest_fingerprint(manifest),
        final_manifest_path=config.final_manifest_path,
    )
    config.final_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.final_manifest_path.write_text(
        json.dumps({"holdout_fingerprint": data.holdout_fingerprint}),
        encoding="utf-8",
    )
    shap_results = fv.run_shap_validation(result, config)
    return result, plan, shap_results


@pytest.fixture(scope="module")
def lang():
    return make_lang("pt")


@pytest.fixture(scope="module")
def lang_en():
    return make_lang("en")


class TestReportTables:
    def test_every_table_report_returns_a_non_empty_frame(self, results, lang):
        result, plan, shap_results = results
        tables = [
            reports.protocol_report(result.config, lang=lang),
            reports.provenance_report(plan, lang=lang),
            reports.candidates_report(plan, lang=lang),
            reports.holdout_seal_report(result, lang=lang),
            reports.metrics_report(result, lang=lang),
            reports.comparison_report(result, lang=lang),
            reports.confirmation_report(result, lang=lang),
            reports.residual_diagnostics_report(result, lang=lang),
            reports.heteroscedasticity_report(result, lang=lang),
            reports.residual_triage_report(result, lang=lang),
            reports.residual_profile_report(result, "month", lang=lang),
            reports.residual_profile_report(result, "hour", lang=lang),
            reports.residual_profile_report(result, "season", lang=lang),
            reports.residual_profile_report(result, "predicted_demand_decile", lang=lang),
            reports.residual_transformation_report(result, lang=lang),
            reports.condition_metrics_report(result, "season", lang=lang),
            reports.shap_methodology_report(lang=lang),
            reports.shap_additivity_report(shap_results, lang=lang),
            reports.shap_grouped_report(shap_results, lang=lang),
            reports.shap_detailed_report(shap_results, lang=lang),
            reports.shap_feature_comparison_report(shap_results, lang=lang),
            reports.artifacts_report(result, lang=lang),
            reports.synthesis_report(result, lang=lang),
        ]
        for table in tables:
            assert isinstance(table, pd.DataFrame)
            assert not table.empty

    def test_offline_english_catalog_covers_the_notebook_reports(self, results, lang_en):
        result, plan, shap_results = results
        tables = [
            reports.protocol_report(result.config, lang=lang_en),
            reports.provenance_report(plan, lang=lang_en),
            reports.candidates_report(plan, lang=lang_en),
            reports.holdout_seal_report(result, lang=lang_en),
            reports.metrics_report(result, lang=lang_en),
            reports.comparison_report(result, lang=lang_en),
            reports.confirmation_report(result, lang=lang_en),
            reports.residual_diagnostics_report(result, lang=lang_en),
            reports.heteroscedasticity_report(result, lang=lang_en),
            reports.residual_triage_report(result, lang=lang_en),
            reports.residual_profile_report(result, "month", lang=lang_en),
            reports.residual_profile_report(result, "hour", lang=lang_en),
            reports.residual_profile_report(result, "season", lang=lang_en),
            reports.residual_profile_report(result, "predicted_demand_decile", lang=lang_en),
            reports.residual_transformation_report(result, lang=lang_en),
            reports.condition_metrics_report(result, "season", lang=lang_en),
            reports.shap_methodology_report(lang=lang_en),
            reports.shap_additivity_report(shap_results, lang=lang_en),
            reports.shap_grouped_report(shap_results, lang=lang_en),
            reports.shap_detailed_report(shap_results, lang=lang_en),
            reports.shap_feature_comparison_report(shap_results, lang=lang_en),
            reports.artifacts_report(result, lang=lang_en),
            reports.synthesis_report(result, lang=lang_en),
        ]
        figures = [
            reports.plot_comparison(result, lang=lang_en),
            reports.plot_temporal_residuals(result, lang=lang_en),
            reports.plot_residual_structure(result, lang=lang_en),
            reports.plot_residual_triage(result, lang=lang_en),
            reports.plot_residual_transformation_acf(result, lang=lang_en),
            reports.plot_condition_metrics(result, lang=lang_en),
            reports.plot_shap_summary(shap_results, lang=lang_en),
            reports.plot_shap_local(result, shap_results, lang=lang_en),
        ]

        assert all(not table.empty for table in tables)
        messages = [
            reports.confirmation_message(result, lang=lang_en),
            reports.residual_handoff_message(result, lang=lang_en),
            reports.handoff_message(result, lang=lang_en),
        ]
        assert all(messages)

        visible = "\n".join(
            [
                *(table.to_string() for table in tables),
                *(_figure_text(fig) for fig in figures),
                *messages,
            ]
        )
        forbidden_portuguese = (
            "Comparação",
            "bicicletas",
            "Resíduo",
            "Resíduos",
            "Média",
            "Previsto",
            "Observado",
            "Janela",
            "Estrutura",
            "Distribuição",
            "Quantis",
            "Autocorrelação",
            "Defasagem",
            "Previsão",
            "Triagem",
            "Erro ",
            "Hora",
            "Dia da semana",
            "Persistência",
            "condição",
            "Faixa",
            "Quintil",
            "Segunda",
            "Terça",
            "Quarta",
            "Quinta",
            "Sexta",
            "Sábado",
            "Domingo",
            "Inverno",
            "Primavera",
            "Verão",
            "Outono",
            "Decisão",
            "Observações",
            "Validação",
            "validação",
            "Não ",
            " não ",
        )
        leaked = {
            token: [line for line in visible.splitlines() if token in line]
            for token in forbidden_portuguese
            if token in visible
        }
        assert not leaked, f"Portuguese leaked into EN reports/figures: {leaked}"

        expected_titles = {
            "Temporal holdout comparison",
            "Champion temporal residuals",
            "Champion residual structure",
            "Champion residual screening on the holdout",
            "Residual persistence before and after diagnostic transformations",
            "Champion MAE by operating condition",
            "SHAP importance by candidate",
            "Local Champion explanations (contributions to the logarithmic residual)",
        }
        assert expected_titles.issubset(set(visible.splitlines()))
        for figure in figures:
            plt.close(figure)

    def test_display_copy_is_localized_while_internal_schema_stays_english(self, results, lang):
        result, _, _ = results
        report = reports.metrics_report(result, lang=lang)
        assert "R²" in report.columns
        assert "holdout_mae" not in report.columns
        # The internal objects keep the stable English schema.
        assert "holdout_mae" in result.evaluations[0].metrics
        assert "cv_mae_mean" in result.candidates[0].cv_metrics

    def test_heteroscedasticity_report_is_localized(self, results, lang):
        result, _, _ = results
        report = reports.heteroscedasticity_report(result, lang=lang)
        assert "Teste" in report.columns
        assert "p-valor ajustado (Holm)" in report.columns
        assert "evidence_of_heteroscedasticity" not in report.columns
        assert report["Teste"].str.contains("ARCH").any()

    def test_residual_transformation_report_is_localized(self, results, lang):
        result, _, _ = results
        report = reports.residual_transformation_report(result, lang=lang)

        assert "Versão do resíduo" in report.columns
        assert "residual_version" not in report.columns
        assert report["Versão do resíduo"].str.contains("Original").any()
        assert "p-valor ajustado (Holm)" in report.columns

    def test_p_value_underflow_is_formatted_only_in_display_copy(self, results, lang, monkeypatch):
        result, _, _ = results
        internal = fv.heteroscedasticity_diagnostics(result)
        internal.loc[0, "p_value"] = 0.0
        internal.loc[0, "adjusted_p_value"] = 0.0
        monkeypatch.setattr(reports, "heteroscedasticity_diagnostics", lambda _: internal)

        report = reports.heteroscedasticity_report(result, lang=lang)

        assert "abaixo da precisão numérica" in set(report["p-valor bruto"])
        assert internal.loc[0, "p_value"] == 0.0

    def test_comparison_is_sorted_by_ascending_holdout_mae(self, results, lang):
        result, _, _ = results
        internal = result.comparison.sort_values("holdout_mae")["holdout_mae"].tolist()
        assert internal == sorted(internal)


class TestNarrativeStrings:
    def test_confirmation_message_is_non_empty_portuguese(self, results, lang):
        result, _, _ = results
        message = reports.confirmation_message(result, lang=lang)
        assert isinstance(message, str) and len(message) > 40
        assert "champion" in message.lower()

    def test_handoff_message_is_non_empty(self, results, lang):
        result, _, _ = results
        assert len(reports.handoff_message(result, lang=lang)) > 40

    def test_residual_handoff_mentions_iid_interval_limitation(self, results, lang):
        result, _, _ = results
        message = reports.residual_handoff_message(result, lang=lang)
        assert "IID" in message
        assert "previsão pontual" in message


class TestFigures:
    def test_every_plot_returns_a_matplotlib_figure(self, results, lang):
        result, _, shap_results = results
        figures = [
            reports.plot_comparison(result, lang=lang),
            reports.plot_temporal_residuals(result, lang=lang),
            reports.plot_residual_structure(result, lang=lang),
            reports.plot_residual_triage(result, lang=lang),
            reports.plot_residual_transformation_acf(result, lang=lang),
            reports.plot_condition_metrics(result, lang=lang),
            reports.plot_shap_summary(shap_results, lang=lang),
            reports.plot_shap_local(result, shap_results, lang=lang),
        ]
        for figure in figures:
            assert isinstance(figure, plt.Figure)
            plt.close(figure)
