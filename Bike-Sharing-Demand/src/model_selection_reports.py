"""Reports and charts for the v4 model-selection notebook.

Every function here takes the objects produced by
:mod:`src.model_selection_workflow` and returns something displayable — a
DataFrame, a matplotlib figure, or an HTML diagram. The computation itself
already happened; this layer only shapes it for reading.

The project's localization boundary is respected throughout: the internal
column names stay in English and stable, and only the returned display copy is
translated, through :func:`src.i18n.localize_table`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.patches import Patch

from src.catboost_refinement import CatBoostAblationResults
from src.cv import expanding_meteorological_year_report
from src.environment import (
    TRACKED_PACKAGES,
    describe_environment,
    model_pip_requirements,
    version_drift,
)
from src.i18n import localize_table, resolve_lang
from src.model_selection_workflow import (
    DevelopmentData,
    ModelSelectionConfig,
    ModelSelectionResults,
)
from src.modeling_pipeline import (
    BOOSTING_BUDGET_STRATEGIES,
    encoder_space,
    estimator_family,
    modeler_space,
    selector_space,
)
from src.temporal_optimizer import CODE_VERSION, CV_STRATEGY_VERSION
from src.utils import public_path

_BASELINE_COLOR = "#9e9e9e"
_CANDIDATE_COLOR = "#4c78a8"
_CHAMPION_COLOR = "#f58518"


def _key_value_frame(lang, rows: List[tuple]) -> pd.DataFrame:
    """Build a two-column label/value report from already-canonical PT labels."""
    labels = lang({"item": "Item", "value": "Valor"})
    return pd.DataFrame([{labels["item"]: label, labels["value"]: value} for label, value in rows])


# ---------------------------------------------------------------------------
# Setup reports
# ---------------------------------------------------------------------------


def run_configuration_report(config: ModelSelectionConfig, lang=None) -> pd.DataFrame:
    """Summarize the declared run mode and the budget it resolves to."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "mode": "Modo de execução",
            "estimators": "Estimators candidatos",
            "trials": "Trials por estimator (total configurado)",
            "trial_timeout": "Timeout por trial (s)",
            "study_timeout": "Timeout por estudo (s)",
            "study_timeout_map": "Timeout específico por estimator (s)",
            "challengers": "Challengers congelados",
            "definitive": "Congela candidatos definitivos",
            "cv_version": "Versão da estratégia de CV",
            "code_version": "Versão do código",
            "window": "Janela de treino (anos)",
            "weights": "Pesos dos folds",
            "profile": "Perfil de busca",
            "target": "Estratégia do alvo",
            "metric": "Métrica de seleção",
            "seeded_trials": "Trials iniciais enfileiradas",
            "regime": "Política de regime",
            "excluded_interval": "Intervalo excluído do ajuste",
            "selection_years": "Anos meteorológicos de seleção",
            "stress_years": "Anos meteorológicos de estresse",
        }
    )
    return _key_value_frame(
        lang,
        [
            (labels["mode"], config.run_mode),
            (labels["estimators"], ", ".join(config.candidate_estimators)),
            (labels["trials"], config.resolved_trials),
            (labels["trial_timeout"], config.resolved_trial_timeout),
            (labels["study_timeout"], config.resolved_study_timeout),
            (labels["study_timeout_map"], dict(config.study_timeout_by_estimator)),
            (labels["challengers"], config.n_challengers),
            (labels["definitive"], config.freezes_definitive_candidates),
            (labels["cv_version"], CV_STRATEGY_VERSION),
            (labels["code_version"], CODE_VERSION),
            (
                labels["window"],
                config.train_window_years if config.train_window_years is not None else "expanding",
            ),
            (labels["weights"], config.fold_weights),
            (labels["profile"], config.search_profile),
            (labels["target"], config.target_strategy),
            (labels["metric"], config.selection_metric),
            (labels["regime"], config.regime_policy),
            (
                labels["excluded_interval"],
                f"{config.regime_exclusion_start} — {config.regime_exclusion_end}",
            ),
            (labels["selection_years"], tuple(config.selection_test_years)),
            (labels["stress_years"], tuple(config.stress_test_years)),
            (
                labels["seeded_trials"],
                {
                    estimator: len(trials)
                    for estimator, trials in config.enqueued_trials_by_estimator.items()
                },
            ),
        ],
    )


def environment_report(lang=None) -> pd.DataFrame:
    """State which interpreter and which library versions are executing this notebook.

    Exists to be *read*, not merely computed: the guard in
    :func:`src.environment.require_environment` already refuses a wrong
    environment, but a saved notebook is also an audit record, and the only way
    a reader can confirm which environment produced the outputs below is to see
    its name and executable without exposing the contributor's local path.
    """
    lang = resolve_lang(lang)
    labels = lang(
        {
            "environment": "Ambiente conda",
            "executable": "Interpretador (sys.executable)",
            "python": "Versão do Python",
            "fingerprint": "Fingerprint do ambiente",
            "drift": "Divergências em relação ao environment.yml",
            "requirements": "Requisitos declarados no modelo",
        }
    )
    description = describe_environment()
    drift = version_drift()
    none_label = lang({"none": "nenhuma"})["none"]
    rows = [
        (labels["environment"], description["environment_name"]),
        (
            labels["executable"],
            f"{description['environment_name']}/{Path(description['python_executable']).name}",
        ),
        (labels["python"], description["python_version"]),
        (labels["fingerprint"], description["environment_fingerprint"]),
        (
            labels["drift"],
            none_label
            if not drift
            else ", ".join(f"{name} {pin}->{got}" for name, (pin, got) in drift.items()),
        ),
        (labels["requirements"], ", ".join(model_pip_requirements())),
    ]
    rows.extend((name, description[f"version_{name}"]) for name in TRACKED_PACKAGES)
    return _key_value_frame(lang, rows)


def _window_label(start, end) -> str:
    """Render a closed date window, or a dash when the window is empty."""
    if start is None or end is None:
        return "—"
    return f"{start:%d/%m/%Y} — {end:%d/%m/%Y}"


def holdout_seal_report(development: DevelopmentData, lang=None) -> pd.DataFrame:
    """Report the development, holdout and discarded windows — dates and counts only.

    The post-holdout line is what makes the split auditable: the v4 source
    extends past the holdout, and those later rows belong to neither split.
    Reporting their count proves they were dropped rather than folded back
    into development.
    """
    lang = resolve_lang(lang)
    labels = lang(
        {
            "dev_window": "Período de desenvolvimento",
            "dev_rows": "Linhas de desenvolvimento",
            "holdout_window": "Período do holdout",
            "rows": "Linhas seladas no holdout",
            "sealed": "Selado",
            "post_window": "Período pós-holdout descartado",
            "post_rows": "Linhas pós-holdout excluídas",
            "folds": "Folds disponíveis",
            "fingerprint": "Fingerprint do dataset",
            "regime_fingerprint": "Fingerprint do regime",
            "eligible_rows": "Linhas elegíveis para ajuste final",
            "excluded_rows": "Linhas excluídas do ajuste",
        }
    )
    holdout = development.holdout
    return _key_value_frame(
        lang,
        [
            (labels["dev_window"], _window_label(holdout.dev_start, holdout.dev_end)),
            (labels["dev_rows"], f"{len(development.X_dev):,}"),
            (labels["holdout_window"], _window_label(holdout.start, holdout.end)),
            (labels["rows"], f"{holdout.n_rows:,}"),
            (labels["sealed"], holdout.sealed),
            (
                labels["post_window"],
                _window_label(holdout.post_holdout_start, holdout.post_holdout_end),
            ),
            (labels["post_rows"], f"{holdout.n_post_holdout_rows:,}"),
            (labels["folds"], development.n_folds),
            (labels["fingerprint"], development.fingerprint),
            (labels["regime_fingerprint"], development.regime_fingerprint),
            (
                labels["eligible_rows"],
                f"{int(development.train_eligible_mask.sum()):,}",
            ),
            (
                labels["excluded_rows"],
                f"{int((~development.train_eligible_mask).sum()):,}",
            ),
        ],
    )


def fold_audit_report(development: DevelopmentData, lang=None) -> pd.DataFrame:
    """Audit every expanding fold: windows, row counts, real gap, seasons, leakage flag."""
    lang = resolve_lang(lang)
    report = expanding_meteorological_year_report(
        development.X_dev, development.splitter, lang=lang
    )
    labels = lang(
        {
            "year": "Ano meteorológico",
            "role": "Papel no protocolo",
            "train_used": "Linhas de treino elegíveis",
            "train_excluded": "Linhas de treino excluídas",
            "score_rows": "Linhas usadas na seleção",
        }
    )
    roles = lang({"selection": "seleção", "stress": "estresse"})
    details = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        development.splitter.split(development.X_dev, development.y_dev)
    ):
        test_year = int(development.config.test_years[fold_idx])
        is_selection = test_year in set(development.config.selection_test_years)
        train_used = int(development.train_eligible_mask[train_idx].sum())
        score_rows = int(development.score_eligible_mask[test_idx].sum()) if is_selection else 0
        details.append(
            {
                labels["year"]: test_year,
                labels["role"]: roles["selection"] if is_selection else roles["stress"],
                labels["train_used"]: train_used,
                labels["train_excluded"]: int(len(train_idx) - train_used),
                labels["score_rows"]: score_rows,
            }
        )
    return pd.concat([report.reset_index(drop=True), pd.DataFrame(details)], axis=1)


def _timeline_segments(
    timestamps: pd.Series,
    mask,
    bridge_gap: pd.Timedelta = pd.Timedelta(days=45),
) -> List[tuple]:
    """Collapse one timeline class into calendar segments.

    The chart audits protocol geometry rather than source completeness, so
    isolated missing hourly stamps are bridged. A long exclusion such as the
    2020 regime remains a visible break.
    """
    selected = timestamps.loc[np.asarray(mask, dtype=bool)].sort_values().reset_index(drop=True)
    if selected.empty:
        return []
    groups = selected.diff().gt(bridge_gap).fillna(True).cumsum()
    return [
        (group.iloc[0], group.iloc[-1] + pd.Timedelta(hours=1))
        for _, group in selected.groupby(groups, sort=False)
    ]


def plot_fold_audit(development: DevelopmentData, lang=None):
    """Plot expanding-fold geometry, gap, stress fold and regime exclusions."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Geometria da validação cruzada por anos meteorológicos",
            "subtitle": ("Janela expansiva, gap temporal e política de operações normais"),
            "x": "Data",
            "fold": "Fold {fold} — {role} ({year})",
            "selection": "seleção",
            "stress": "estresse",
            "regime": "Regime operacional",
            "train": "Treino elegível",
            "train_excluded": "Treino excluído (2020)",
            "gap": "Gap temporal de 48 h",
            "test_selection": "Teste de seleção",
            "test_stress": "Teste de estresse",
            "test_excluded": "Teste excluído do score",
            "normal_regime": "Operações normais",
            "excluded_regime": "Regime excluído (2020)",
        }
    )
    colors = {
        "train": "#4C78A8",
        "train_excluded": "#A7A7A7",
        "gap": "#F2CF5B",
        "test_selection": "#F58518",
        "test_stress": "#E45756",
        "test_excluded": "#B279A2",
        "normal_regime": "#72B7B2",
        "excluded_regime": "#A7A7A7",
    }
    timestamps = (
        development.splitter._get_timestamps(development.X_dev)
        .reset_index(drop=True)
        .astype("datetime64[ns]")
    )
    train_eligible = np.asarray(development.train_eligible_mask, dtype=bool)
    score_eligible = np.asarray(development.score_eligible_mask, dtype=bool)
    selection_years = set(development.config.selection_test_years)
    splits = list(development.splitter.split(development.X_dev, development.y_dev))

    fig, ax = plt.subplots(figsize=(14, 0.85 * len(splits) + 3.4))
    bar_height = 0.62

    def draw_segments(mask, y, color, hatch=None):
        for start, end in _timeline_segments(timestamps, mask):
            left = mdates.date2num(start.to_pydatetime())
            width = mdates.date2num(end.to_pydatetime()) - left
            ax.barh(
                y,
                width,
                left=left,
                height=bar_height,
                color=color,
                edgecolor="white" if hatch is None else "#666666",
                linewidth=0.4,
                hatch=hatch,
            )

    tick_labels = []
    n_rows = len(timestamps)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        test_year = int(development.config.test_years[fold_idx])
        is_selection = test_year in selection_years
        role = labels["selection"] if is_selection else labels["stress"]
        tick_labels.append(labels["fold"].format(fold=fold_idx + 1, role=role, year=test_year))

        train_mask = np.zeros(n_rows, dtype=bool)
        train_mask[train_idx] = True
        test_mask = np.zeros(n_rows, dtype=bool)
        test_mask[test_idx] = True

        draw_segments(train_mask & train_eligible, fold_idx, colors["train"])
        draw_segments(
            train_mask & ~train_eligible,
            fold_idx,
            colors["train_excluded"],
            hatch="///",
        )

        gap_start = timestamps.iloc[train_idx].max() + pd.Timedelta(hours=1)
        gap_end = timestamps.iloc[test_idx].min()
        gap_left = mdates.date2num(gap_start.to_pydatetime())
        gap_width = mdates.date2num(gap_end.to_pydatetime()) - gap_left
        ax.barh(
            fold_idx,
            gap_width,
            left=gap_left,
            height=bar_height,
            color=colors["gap"],
            edgecolor="white",
            linewidth=0.4,
        )
        gap_center = gap_start + (gap_end - gap_start) / 2
        ax.vlines(
            mdates.date2num(gap_center.to_pydatetime()),
            fold_idx - bar_height / 2,
            fold_idx + bar_height / 2,
            color=colors["gap"],
            linewidth=3.0,
            zorder=5,
        )

        if is_selection:
            draw_segments(
                test_mask & score_eligible,
                fold_idx,
                colors["test_selection"],
            )
            draw_segments(
                test_mask & ~score_eligible,
                fold_idx,
                colors["test_excluded"],
                hatch="\\\\\\",
            )
        else:
            draw_segments(test_mask, fold_idx, colors["test_stress"])

    regime_row = len(splits)
    tick_labels.append(labels["regime"])
    draw_segments(train_eligible, regime_row, colors["normal_regime"])
    draw_segments(
        ~train_eligible,
        regime_row,
        colors["excluded_regime"],
        hatch="///",
    )

    ax.set_yticks(range(len(tick_labels)), labels=tick_labels)
    ax.invert_yaxis()
    ax.set_xlim(
        mdates.date2num((timestamps.min() - pd.Timedelta(days=30)).to_pydatetime()),
        mdates.date2num((timestamps.max() + pd.Timedelta(days=30)).to_pydatetime()),
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))
    ax.set_xlabel(labels["x"])
    ax.set_title(f"{labels['title']}\n{labels['subtitle']}")
    ax.grid(axis="x", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)

    legend = [
        Patch(facecolor=colors["train"], label=labels["train"]),
        Patch(
            facecolor=colors["train_excluded"],
            edgecolor="#666666",
            hatch="///",
            label=labels["train_excluded"],
        ),
        Patch(facecolor=colors["gap"], label=labels["gap"]),
        Patch(
            facecolor=colors["test_selection"],
            label=labels["test_selection"],
        ),
        Patch(facecolor=colors["test_stress"], label=labels["test_stress"]),
        Patch(
            facecolor=colors["test_excluded"],
            edgecolor="#666666",
            hatch="\\\\\\",
            label=labels["test_excluded"],
        ),
        Patch(facecolor=colors["normal_regime"], label=labels["normal_regime"]),
        Patch(
            facecolor=colors["excluded_regime"],
            edgecolor="#666666",
            hatch="///",
            label=labels["excluded_regime"],
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def feature_space_report(config: ModelSelectionConfig, lang=None) -> pd.DataFrame:
    """List the candidate features and the branch each one is routed through."""
    lang = resolve_lang(lang)
    labels = lang({"feature": "Feature candidata", "branch": "Ramo de pré-processamento"})
    numeric_label = lang({"numeric": "Numérico"})["numeric"]
    categorical_label = lang({"categorical": "Categórico"})["categorical"]
    rows = [
        {labels["feature"]: name, labels["branch"]: numeric_label}
        for name in config.numeric_features
    ]
    rows.extend(
        {labels["feature"]: name, labels["branch"]: categorical_label}
        for name in config.categorical_features
    )
    return pd.DataFrame(rows)


def search_space_report(config: ModelSelectionConfig, lang=None) -> pd.DataFrame:
    """Show the representation and selection spaces each estimator family receives.

    This is the table that makes the architecture visible: the ``modeler_name``
    candidates are not a single global list but a per-family menu, and the
    feature selectors follow the estimator's own capabilities.
    """
    lang = resolve_lang(lang)
    estimators = ["DummyRegressor"] + list(config.candidate_estimators)
    rows = [
        {
            "estimator": name,
            "family": estimator_family(name),
            "n_modelers": len(modeler_space(name, config.search_profile)),
            "modeler_space": ", ".join(modeler_space(name, config.search_profile)),
            "encoder_space": ", ".join(encoder_space(name, config.search_profile)),
            "selector_space": ", ".join(selector_space(name, config.search_profile)),
            "boosting_budget_space": (
                ", ".join(BOOSTING_BUDGET_STRATEGIES)
                if name in {"XGBRegressor", "LGBMRegressor", "CatBoostRegressor"}
                and config.search_profile == "refined"
                else "not applicable"
            ),
        }
        for name in estimators
    ]
    return localize_table(
        pd.DataFrame(rows),
        lang,
        columns={
            "estimator": "Estimator",
            "family": "Família",
            "n_modelers": "Estratégias disponíveis",
            "modeler_space": "Espaço de representação (modeler_name)",
            "encoder_space": "Espaço de encoders",
            "selector_space": "Espaço de seleção de features",
            "boosting_budget_space": "Espaço de orçamento do boosting",
        },
        value_columns=["boosting_budget_space"],
        value_labels={"not applicable": "não aplicável"},
    )


def dynamic_pipeline_diagram(pipeline: Any):
    """Render a pipeline as sklearn's interactive HTML diagram for the notebook."""
    from IPython.display import HTML
    from sklearn.utils import estimator_html_repr

    return HTML(estimator_html_repr(pipeline))


def pipeline_spec_report(spec, lang=None) -> pd.DataFrame:
    """Describe one sampled pipeline: representation, encoder, scaler, selector, target."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "estimator": "Estimator",
            "family": "Família",
            "modeler": "Representação (modeler_name)",
            "encoder": "Encoder categórico",
            "scaler": "Scaler numérico",
            "normalizer": "Normalizador",
            "selector": "Seletor de features",
            "target": "Transformação do alvo",
            "boosting_budget": "Estratégia do orçamento de boosting",
            "n_features": "Features entregues ao estimator",
        }
    )
    none_label = lang({"none": "não utilizado"})["none"]
    return _key_value_frame(
        lang,
        [
            (labels["estimator"], spec.estimator),
            (labels["family"], spec.family),
            (labels["modeler"], spec.modeler_name),
            (labels["encoder"], spec.encoder),
            (labels["scaler"], spec.scaler),
            (labels["normalizer"], spec.normalizer or none_label),
            (labels["selector"], spec.selector),
            (labels["target"], spec.target_transform),
            (
                labels["boosting_budget"],
                spec.boosting_budget_strategy or none_label,
            ),
            (labels["n_features"], spec.n_features_selected or none_label),
        ],
    )


# ---------------------------------------------------------------------------
# Result reports
# ---------------------------------------------------------------------------


def _comparison_frame(results: ModelSelectionResults) -> pd.DataFrame:
    """Internal-schema comparison table, sorted by ascending mean CV MAE."""
    champion_run_id = results.champion["run_id"]
    rows = [
        {
            "estimator": outcome.estimator,
            "role": (
                "champion"
                if outcome.run_id == champion_run_id
                else ("baseline" if outcome.is_baseline else "candidate")
            ),
            "modeler_name": outcome.spec.modeler_name,
            "encoder": outcome.spec.encoder,
            "scaler": outcome.spec.scaler,
            "selector": outcome.spec.selector,
            "target_transform": outcome.spec.target_transform,
            "boosting_budget_strategy": outcome.spec.boosting_budget_strategy,
            "n_features_selected": outcome.spec.n_features_selected,
            "cv_mae_mean": outcome.cv_mae_mean,
            "cv_mae_weighted": outcome.cv_mae_selection,
            "cv_rmse_mean": outcome.evaluation.cv_metrics.get("cv_rmse_mean"),
            "cv_r2_mean": outcome.evaluation.cv_metrics.get("cv_r2_mean"),
            "cv_r2_median": outcome.evaluation.cv_metrics.get("cv_r2_median"),
            "cv_r2_weighted": outcome.evaluation.cv_metrics.get("cv_r2_weighted"),
            "cv_wape_mean": outcome.evaluation.cv_metrics.get("cv_wape_mean"),
            "cv_mean_bias": outcome.evaluation.cv_metrics.get("cv_mean_bias"),
            "cv_mean_abs_fold_bias": outcome.evaluation.cv_metrics.get("cv_mean_abs_fold_bias"),
            "r2_gap": outcome.evaluation.cv_metrics.get("r2_gap"),
            "trials_planned": outcome.trials_planned,
            "trials_completed": outcome.evaluation.trials_completed,
            "termination_reason": outcome.termination_reason,
            "final_n_estimators": outcome.evaluation.final_n_estimators,
            "n_folds_cap_hit": outcome.evaluation.n_folds_cap_hit,
            "run_id": outcome.run_id,
        }
        for outcome in results.outcomes
    ]
    return pd.DataFrame(rows).sort_values("cv_mae_weighted").reset_index(drop=True)


_COMPARISON_COLUMNS = {
    "estimator": "Estimator",
    "role": "Papel",
    "modeler_name": "Representação",
    "encoder": "Encoder",
    "scaler": "Scaler",
    "selector": "Seletor",
    "target_transform": "Alvo",
    "boosting_budget_strategy": "Orçamento do boosting",
    "n_features_selected": "Features",
    "cv_mae_mean": "MAE médio normal (CV)",
    "cv_mae_weighted": "MAE normal ponderado (seleção)",
    "cv_rmse_mean": "RMSE médio normal (CV)",
    "cv_r2_mean": "R² médio normal (CV)",
    "cv_r2_median": "R² mediano normal (CV)",
    "cv_r2_weighted": "R² ponderado normal (CV)",
    "cv_wape_mean": "WAPE médio normal (CV)",
    "cv_mean_bias": "Viés médio normal (CV)",
    "cv_mean_abs_fold_bias": "Média do |viés| por fold",
    "r2_gap": "Gap de R² (treino-teste)",
    "trials_planned": "Trials planejadas",
    "trials_completed": "Trials concluídas",
    "termination_reason": "Motivo do encerramento",
    "final_n_estimators": "Iterações do refit final",
    "n_folds_cap_hit": "Folds no teto de boosting",
    "run_id": "Run ID",
}

_TERMINATION_LABELS = {
    "trial_limit": "limite de trials",
    "study_timeout": "timeout do estudo",
}

_ROLE_LABELS = {
    "champion": "champion",
    "challenger": "challenger",
    "candidate": "candidato",
    "baseline": "baseline",
}


def comparison_report(results: ModelSelectionResults, lang=None) -> pd.DataFrame:
    """Compare every estimator by cross-validation only, winning pipeline included.

    The pipeline columns are what distinguish this from a plain metric table:
    each row records which representation, encoder, scaler, selector and target
    transform the search actually chose for that estimator.
    """
    lang = resolve_lang(lang)
    return localize_table(
        _comparison_frame(results),
        lang,
        columns=_COMPARISON_COLUMNS,
        value_columns=["role", "termination_reason"],
        value_labels={**_ROLE_LABELS, **_TERMINATION_LABELS},
    )


_CATBOOST_ABLATION_VARIANTS = {
    "v4_raw_fixed_283": "CatBoost v4 — alvo bruto, 283 iterações fixas",
    "v4_raw_temporal_early_stopping": ("CatBoost v4 — alvo bruto, early stopping temporal"),
}


def catboost_ablation_report(
    results: CatBoostAblationResults,
    lang=None,
) -> pd.DataFrame:
    """Summarize both old-configuration replays under the current CV."""
    lang = resolve_lang(lang)
    return localize_table(
        results.summary,
        lang,
        columns={
            "variant": "Configuração avaliada",
            "early_stopping": "Early stopping temporal",
            "configured_iterations": "Teto/iterações configuradas",
            "final_n_estimators": "Iterações indicadas para refit",
            "cv_mae_mean": "MAE médio (CV)",
            "cv_mae_weighted": "MAE ponderado (CV)",
            "cv_rmse_mean": "RMSE médio (CV)",
            "cv_r2_mean": "R² médio (CV)",
            "cv_r2_median": "R² mediano (CV)",
            "cv_r2_weighted": "R² ponderado (CV)",
            "cv_wape_mean": "WAPE médio (CV)",
            "cv_mean_bias": "Viés médio (CV)",
            "cv_mean_abs_fold_bias": "Média do |viés| por fold",
            "cv_mae_std": "Desvio-padrão do MAE",
        },
        value_columns=["variant"],
        value_labels=_CATBOOST_ABLATION_VARIANTS,
    )


def catboost_ablation_fold_report(
    results: CatBoostAblationResults,
    lang=None,
) -> pd.DataFrame:
    """Expose fold-level errors so a favorable mean cannot hide instability."""
    lang = resolve_lang(lang)
    return localize_table(
        results.fold_metrics,
        lang,
        columns={
            "variant": "Configuração avaliada",
            "fold": "Fold",
            "n_train": "Observações de treino",
            "n_test": "Observações de teste",
            "best_iteration": "Melhor iteração",
            "iteration_ceiling": "Teto de iterações",
            "best_iteration_cap_hit": "Teto atingido",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R²",
            "wape": "WAPE",
            "mean_bias": "Viés médio",
            "train_r2": "R² de treino",
        },
        value_columns=["variant"],
        value_labels=_CATBOOST_ABLATION_VARIANTS,
    )


def plot_comparison(results: ModelSelectionResults, lang=None):
    """Horizontal bar chart of selection MAE, with baseline and champion highlighted."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "x": "MAE normal ponderado por recência (bicicletas/hora)",
            "title": (
                "Comparação de estimators no regime de operações normais "
                "pela validação cruzada temporal"
            ),
            "smoke": "modo smoke — infraestrutura validada, ranking não definitivo",
        }
    )
    frame = _comparison_frame(results)
    colors = [
        _CHAMPION_COLOR
        if role == "champion"
        else (_BASELINE_COLOR if role == "baseline" else _CANDIDATE_COLOR)
        for role in frame["role"]
    ]

    fig, ax = plt.subplots(figsize=(11, 0.9 * len(frame) + 2.2))
    ax.barh(frame["estimator"], frame["cv_mae_weighted"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(labels["x"])
    ax.set_title(
        labels["title"] if not results.is_provisional else f"{labels['title']}\n({labels['smoke']})"
    )
    for y, value in enumerate(frame["cv_mae_weighted"]):
        ax.text(value, y, f" {value:,.0f}", va="center", fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig


def fold_metrics_report(
    results: ModelSelectionResults, estimator: Optional[str] = None, lang=None
) -> pd.DataFrame:
    """Per-fold metrics of the champion (or of ``estimator`` when given).

    ``best_iteration`` records the boosting budget each fold discovered on its
    own training tail; it is empty for estimators that have no such budget. The
    ceiling and the cap flag sit next to it because a budget close to its
    ceiling was possibly ended by the limit rather than by the validation loss,
    and that distinction is invisible in the number alone. Optuna ranks the
    configured recency-weighted mean of the MAE column.
    """
    lang = resolve_lang(lang)
    outcome = results.outcomes_by_estimator[estimator] if estimator else results.best_outcome
    frame = outcome.evaluation.fold_metrics.copy()
    frame.insert(0, "estimator", outcome.estimator)
    return localize_table(
        frame,
        lang,
        columns={
            "estimator": "Estimator",
            "fold": "Fold",
            "test_year": "Ano meteorológico",
            "fold_role": "Papel",
            "n_train": "Linhas de treino",
            "n_train_excluded": "Treino excluído",
            "n_test": "Linhas de teste",
            "n_selection_test": "Teste usado na seleção",
            "selection_mae": "MAE normal",
            "selection_rmse": "RMSE normal",
            "selection_r2": "R² normal",
            "selection_wape": "WAPE normal",
            "selection_mean_bias": "Viés normal",
            "mae": "MAE diagnóstico completo",
            "rmse": "RMSE diagnóstico completo",
            "r2": "R² diagnóstico completo",
            "wape": "WAPE diagnóstico completo",
            "mean_bias": "Viés diagnóstico completo",
            "best_iteration": "Iterações (early stopping)",
            "iteration_ceiling": "Teto de iterações",
            "best_iteration_cap_hit": "Atingiu o teto",
        },
        value_columns=["fold_role"],
        value_labels={"selection": "seleção", "stress": "estresse"},
    )


def condition_metrics_report(
    results: ModelSelectionResults, estimator: Optional[str] = None, lang=None
) -> pd.DataFrame:
    """Champion performance by season and by temperature extreme, averaged over folds.

    Seasons come from the normal-regime rows of each selection fold; the
    temperature bands are the coldest and warmest five per cent of those rows.
    Both are aggregated across folds so a condition that only degrades in one
    year is still visible in the row count. Stress-only folds are intentionally
    kept out of this primary condition summary.
    """
    lang = resolve_lang(lang)
    outcome = results.outcomes_by_estimator[estimator] if estimator else results.best_outcome
    frames: List[pd.DataFrame] = []

    seasonal = outcome.evaluation.seasonal_metrics
    if not seasonal.empty:
        grouped = (
            seasonal.groupby("season", observed=False)
            .agg(n=("n", "sum"), mae=("mae", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"))
            .reset_index()
            .rename(columns={"season": "condition"})
        )
        grouped.insert(0, "scope", "season")
        frames.append(grouped)

    extreme = outcome.evaluation.extreme_metrics
    if not extreme.empty:
        grouped = (
            extreme.groupby("band", observed=False)
            .agg(n=("n", "sum"), mae=("mae", "mean"), rmse=("rmse", "mean"))
            .reset_index()
            .rename(columns={"band": "condition"})
        )
        grouped.insert(0, "scope", "temperature_band")
        frames.append(grouped)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "estimator", outcome.estimator)
    return localize_table(
        combined,
        lang,
        columns={
            "estimator": "Estimator",
            "scope": "Recorte",
            "condition": "Condição",
            "n": "Observações",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R²",
        },
        value_columns=["scope", "condition"],
        value_labels={
            "season": "estação",
            "temperature_band": "faixa de temperatura",
            "Winter": "Inverno",
            "Spring": "Primavera",
            "Summer": "Verão",
            "Autumn": "Outono",
            "cold_extreme": "extremo frio (5% mais frio)",
            "hot_extreme": "extremo quente (5% mais quente)",
        },
    )


def selection_report(results: ModelSelectionResults, lang=None) -> pd.DataFrame:
    """Champion and challengers with the full pipeline that produced each metric."""
    lang = resolve_lang(lang)
    rows: List[Dict[str, Any]] = []
    for role, candidate in [("champion", results.champion)] + [
        ("challenger", c) for c in results.challengers
    ]:
        rows.append(
            {
                "role": role,
                "estimator": candidate.get("estimator"),
                "cv_mae_mean": candidate.get("cv_mae_mean"),
                "modeler_name": candidate.get("modeler_name"),
                "encoder": candidate.get("encoder"),
                "scaler": candidate.get("scaler"),
                "selector": candidate.get("selector"),
                "target_transform": candidate.get("target_transform"),
                "boosting_budget_strategy": candidate.get("boosting_budget_strategy"),
                "final_n_estimators": candidate.get("final_n_estimators"),
                "n_folds_cap_hit": candidate.get("n_folds_cap_hit"),
                "trials_completed": candidate.get("trials_completed"),
                "termination_reason": candidate.get("termination_reason"),
                "run_mode": candidate.get("run_mode"),
                "run_id": candidate.get("run_id"),
            }
        )
    return localize_table(
        pd.DataFrame(rows),
        lang,
        columns={
            "role": "Papel",
            "estimator": "Estimator",
            "cv_mae_mean": "MAE médio normal (CV)",
            "modeler_name": "Representação",
            "encoder": "Encoder",
            "scaler": "Scaler",
            "selector": "Seletor",
            "target_transform": "Alvo",
            "boosting_budget_strategy": "Orçamento do boosting",
            "final_n_estimators": "Iterações do refit final",
            "n_folds_cap_hit": "Folds no teto de boosting",
            "trials_completed": "Trials concluídas",
            "termination_reason": "Motivo do encerramento",
            "run_mode": "Modo",
            "run_id": "Run ID",
        },
        value_columns=["role", "termination_reason"],
        value_labels={**_ROLE_LABELS, **_TERMINATION_LABELS},
    )


def handoff_report(results: ModelSelectionResults, lang=None) -> pd.DataFrame:
    """State what was written for notebook 05 and whether it is definitive."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "mode": "Modo de execução",
            "provisional": "Artefatos provisórios",
            "manifest": "Manifesto gravado em",
            "experiment": "Experimento MLflow",
            "champion_run": "Run ID do champion",
            "champion_uri": "URI do modelo do champion",
            "fingerprint": "Fingerprint do dataset",
            "regime": "Política de regime",
            "regime_fingerprint": "Fingerprint do regime",
            "environment": "Ambiente de execução",
            "verified": "Artefato do champion conferido",
        }
    )
    return _key_value_frame(
        lang,
        [
            (labels["mode"], results.config.run_mode),
            (labels["provisional"], results.is_provisional),
            (labels["manifest"], public_path(results.manifest_path)),
            (labels["experiment"], results.experiment_name),
            (labels["champion_run"], results.champion["run_id"]),
            (labels["champion_uri"], results.champion["model_uri"]),
            (labels["fingerprint"], results.development.fingerprint),
            (labels["regime"], results.selection.get("regime_policy")),
            (
                labels["regime_fingerprint"],
                results.selection.get("regime_fingerprint"),
            ),
            (
                labels["environment"],
                f"{results.selection.get('environment_name')} "
                f"({results.selection.get('environment_fingerprint')})",
            ),
            (labels["verified"], results.champion.get("model_artifact_verified")),
        ],
    )
