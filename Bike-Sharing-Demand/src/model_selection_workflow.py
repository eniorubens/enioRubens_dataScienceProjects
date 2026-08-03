"""Workflow layer for the v4 model-selection notebook.

Notebook 04 is a presentation surface: it declares a
:class:`ModelSelectionConfig`, calls the public functions here, and displays
what they return. Everything with a decision in it — loading and sealing the
data, resolving the smoke/full budget, running the baseline, looping over the
candidate estimators, logging to MLflow, assembling the comparison, choosing a
champion and freezing the hand-off artifacts — lives in this module, where it
can be tested without a kernel.

Nothing here ever materialises the final holdout: the only holdout object that
exists anywhere in the flow is the metadata-only
:class:`src.cv.HoldoutSummary` returned by :func:`src.cv.split_dev_holdout`.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd

from src.cv import ExpandingMeteorologicalYearSplit, HoldoutSummary, split_dev_holdout
from src.data import read_data
from src.environment import (
    describe_environment,
    require_clean_git_source,
    require_environment,
)
from src.mlflow_integration import ExperimentConfigV4, MLflowTracker
from src.modeling_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PipelineSpec,
    SEARCH_PROFILE_BROAD,
    TARGET_STRATEGY_DIRECT,
    build_dynamic_pipeline,
)
from src.normal_operations import (
    DEFAULT_EXCLUSION_END,
    DEFAULT_EXCLUSION_START,
    DEFAULT_SELECTION_FOLD_WEIGHTS,
    DEFAULT_SELECTION_TEST_YEARS,
    DEFAULT_STRESS_TEST_YEARS,
    REGIME_POLICY_NORMAL_OPERATIONS,
    normal_operations_mask,
    regime_fingerprint,
)
from src.temporal_optimizer import (
    CODE_VERSION,
    CV_STRATEGY_NAME,
    CV_STRATEGY_VERSION,
    DEFAULT_CANDIDATES_ROOT,
    DEFAULT_ESTIMATORS,
    DEFAULT_INVALID_CONFIGS_PATH,
    DEFAULT_STUDIES_DIR,
    MAX_SMOKE_TRIALS,
    RUN_MODE_FULL,
    RUN_MODE_SMOKE,
    RUN_MODES,
    FoldEvaluation,
    TemporalRegressionOptimizer,
    dataset_fingerprint,
    freeze_candidates,
    select_champion_and_challengers,
)
from src.tracking import log_temporal_model_run

logger = logging.getLogger(__name__)

BASELINE_ESTIMATOR = "DummyRegressor"

# Budget defaults per run mode: (trials per estimator, trial timeout, study timeout).
#
# A full study stops at whichever limit arrives first — 400 trials or four
# hours — and which one it was is recorded as ``termination_reason``, so a
# study that ran out of clock is never read as one that exhausted its budget.
#
# What makes a smoke run small is the trial *count*, not the clock: the two
# trials it draws are ordinary members of the search space and may well be
# expensive ones (a wrapper selector refitting a 500-tree booster five times
# per fold, say). Cutting the timeouts too finely would make smoke runs fail
# on cost rather than on correctness, and a cost failure is recorded in the
# invalid-configuration blocklist, where it would wrongly mark a perfectly
# valid pipeline shape as unusable.
FULL_TRIALS_PER_ESTIMATOR = 400
FULL_STUDY_TIMEOUT_SECONDS = 14_400.0
DEFAULT_TRIAL_TIMEOUT_SECONDS = 1_800.0

_MODE_BUDGETS: Dict[str, Tuple[int, float, float]] = {
    RUN_MODE_SMOKE: (MAX_SMOKE_TRIALS, DEFAULT_TRIAL_TIMEOUT_SECONDS, 7200.0),
    RUN_MODE_FULL: (
        FULL_TRIALS_PER_ESTIMATOR,
        DEFAULT_TRIAL_TIMEOUT_SECONDS,
        FULL_STUDY_TIMEOUT_SECONDS,
    ),
}


# ---------------------------------------------------------------------------
# Declarative configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelSelectionConfig:
    """Everything the notebook declares, and nothing it has to compute.

    Parameters
    ----------
    run_mode:
        ``"smoke"`` proves the infrastructure with at most
        :data:`src.temporal_optimizer.MAX_SMOKE_TRIALS` trials per estimator
        and produces only provisional artifacts; ``"full"`` runs the real
        budget and is the only mode allowed to freeze the definitive
        candidates for notebook 05.
    estimators:
        Candidate estimators, excluding the baseline (which always runs).
    include_random_forest, include_catboost:
        Opt-in extras appended to ``estimators``.
    trials_per_estimator, trial_timeout, study_timeout:
        Override the per-mode defaults. In smoke mode the trial count is
        capped regardless of what is requested here.
    n_challengers:
        How many runners-up are frozen alongside the champion.
    allow_boosting_truncation:
        Escape hatch for freezing a definitive champion whose folds
        systematically hit their boosting ceiling. Off by default: such a
        champion's tree count records the limit rather than the data, and the
        correct response is to raise the ceiling, not to accept the number.
    """

    run_mode: str = RUN_MODE_SMOKE
    estimators: Sequence[str] = tuple(e for e in DEFAULT_ESTIMATORS if e != BASELINE_ESTIMATOR)
    include_random_forest: bool = False
    include_catboost: bool = False
    trials_per_estimator: Optional[int] = None
    trial_timeout: Optional[float] = None
    study_timeout: Optional[float] = None
    study_timeout_by_estimator: Mapping[str, float] = field(default_factory=dict)
    enqueued_trials_by_estimator: Mapping[str, Sequence[Mapping[str, Any]]] = field(
        default_factory=dict
    )
    n_challengers: int = 2
    allow_boosting_truncation: bool = False
    seed: int = 42
    target: str = "Rented Bike Count"
    holdout_start: str = "2023-12-01"
    holdout_end: str = "2024-11-30"
    test_years: Sequence[int] = (2019, 2020, 2021, 2022, 2023)
    gap_hours: int = 48
    train_window_years: Optional[int] = None
    regime_policy: str = REGIME_POLICY_NORMAL_OPERATIONS
    regime_exclusion_start: str = DEFAULT_EXCLUSION_START
    regime_exclusion_end: str = DEFAULT_EXCLUSION_END
    selection_test_years: Sequence[int] = DEFAULT_SELECTION_TEST_YEARS
    stress_test_years: Sequence[int] = DEFAULT_STRESS_TEST_YEARS
    fold_weights: Optional[Sequence[float]] = DEFAULT_SELECTION_FOLD_WEIGHTS
    search_profile: str = SEARCH_PROFILE_BROAD
    target_strategy: str = TARGET_STRATEGY_DIRECT
    selection_metric: str = "cv_mae_weighted"
    numeric_features: Sequence[str] = tuple(NUMERICAL_FEATURES)
    categorical_features: Sequence[str] = tuple(CATEGORICAL_FEATURES)
    tracking_uri: Optional[str] = None
    studies_dir: Path = DEFAULT_STUDIES_DIR
    invalid_configs_path: Path = DEFAULT_INVALID_CONFIGS_PATH
    candidates_root: Path = DEFAULT_CANDIDATES_ROOT

    def __post_init__(self) -> None:
        if self.run_mode not in RUN_MODES:
            raise ValueError(f"run_mode must be one of {RUN_MODES}, got '{self.run_mode}'.")
        if self.n_challengers < 0:
            raise ValueError("n_challengers must be non-negative.")
        if any(value <= 0 for value in self.study_timeout_by_estimator.values()):
            raise ValueError("Estimator-specific study timeouts must be positive.")
        if self.regime_policy != REGIME_POLICY_NORMAL_OPERATIONS:
            raise ValueError(
                f"Unsupported regime_policy '{self.regime_policy}'. "
                f"Expected '{REGIME_POLICY_NORMAL_OPERATIONS}'."
            )
        if not set(self.selection_test_years).issubset(self.test_years):
            raise ValueError("selection_test_years must be a subset of test_years.")
        if not set(self.stress_test_years).issubset(self.test_years):
            raise ValueError("stress_test_years must be a subset of test_years.")
        if set(self.selection_test_years).intersection(self.stress_test_years):
            raise ValueError("Selection and stress-test years must be disjoint.")
        if set(self.selection_test_years).union(self.stress_test_years) != set(self.test_years):
            raise ValueError("Every test year must be declared as either selection or stress.")
        ordered_selection_years = tuple(
            year for year in self.test_years if year in set(self.selection_test_years)
        )
        if tuple(self.selection_test_years) != ordered_selection_years:
            raise ValueError(
                "selection_test_years must follow the same order as test_years "
                "because fold_weights are positional."
            )
        if self.fold_weights is not None and len(self.fold_weights) != len(
            self.selection_test_years
        ):
            raise ValueError(
                "fold_weights must contain one value per normal-regime selection fold."
            )
        unknown_seed_estimators = set(self.enqueued_trials_by_estimator).difference(
            self.candidate_estimators
        )
        if unknown_seed_estimators:
            raise ValueError(
                "Seeded trials were declared for estimators outside this run: "
                f"{sorted(unknown_seed_estimators)}."
            )

    @property
    def is_smoke(self) -> bool:
        """Whether this configuration only validates the infrastructure."""
        return self.run_mode == RUN_MODE_SMOKE

    @property
    def freezes_definitive_candidates(self) -> bool:
        """Whether this run may write the manifest notebook 05 consumes."""
        return self.run_mode == RUN_MODE_FULL

    @property
    def candidate_estimators(self) -> List[str]:
        """Candidate estimators after applying the opt-in extras."""
        estimators = [name for name in self.estimators if name != BASELINE_ESTIMATOR]
        if self.include_random_forest and "RandomForestRegressor" not in estimators:
            estimators.append("RandomForestRegressor")
        if self.include_catboost and "CatBoostRegressor" not in estimators:
            estimators.append("CatBoostRegressor")
        return estimators

    @property
    def resolved_trials(self) -> int:
        """Trial budget per estimator, capped in smoke mode."""
        default_trials = _MODE_BUDGETS[self.run_mode][0]
        trials = self.trials_per_estimator or default_trials
        return min(trials, MAX_SMOKE_TRIALS) if self.is_smoke else trials

    @property
    def resolved_trial_timeout(self) -> float:
        """Per-trial wall-clock guard, in seconds."""
        return self.trial_timeout or _MODE_BUDGETS[self.run_mode][1]

    @property
    def resolved_study_timeout(self) -> float:
        """Per-study wall-clock guard, in seconds."""
        return self.study_timeout or _MODE_BUDGETS[self.run_mode][2]

    def resolved_study_timeout_for(self, estimator: str) -> float:
        """Estimator-specific cumulative budget, falling back to the mode default."""

        return float(self.study_timeout_by_estimator.get(estimator, self.resolved_study_timeout))

    @property
    def baseline_trials(self) -> int:
        """Trial budget for the constant baseline, whose space is tiny."""
        return min(self.resolved_trials, 3)


# ---------------------------------------------------------------------------
# Development data
# ---------------------------------------------------------------------------


@dataclass
class DevelopmentData:
    """The sealed development split and the CV that will be run over it."""

    X_dev: pd.DataFrame
    y_dev: pd.Series
    holdout: HoldoutSummary
    splitter: ExpandingMeteorologicalYearSplit
    fingerprint: str
    config: ModelSelectionConfig
    train_eligible_mask: np.ndarray
    score_eligible_mask: np.ndarray
    regime_fingerprint: str
    environment: Dict[str, str] = field(default_factory=dict)

    @property
    def n_folds(self) -> int:
        """Number of expanding folds actually available in the development data."""
        return self.splitter.get_n_splits(self.X_dev)


def prepare_development_data(config: ModelSelectionConfig) -> DevelopmentData:
    """Load the raw data, seal the final holdout by date and build the CV.

    The holdout rows are sliced and discarded inside
    :func:`src.cv.split_dev_holdout`; what comes back is the development split
    plus a metadata-only summary, so no notebook variable can carry holdout
    observations into a modeling call.

    The interpreter is verified first, before a single row is read: everything
    downstream — the dataset fingerprint, the Optuna studies, the MLflow runs —
    inherits whichever environment this function ran under, and none of those
    artifacts records it on its own.
    """
    require_environment()
    raw = read_data()
    X_dev, y_dev, holdout = split_dev_holdout(
        raw,
        target=config.target,
        holdout_start=config.holdout_start,
        holdout_end=config.holdout_end,
    )
    splitter = ExpandingMeteorologicalYearSplit(
        test_years=tuple(config.test_years),
        gap=config.gap_hours,
        max_train_years=config.train_window_years,
    )
    train_eligible_mask = normal_operations_mask(
        X_dev,
        config.regime_exclusion_start,
        config.regime_exclusion_end,
    )
    score_eligible_mask = train_eligible_mask.copy()
    regime_fingerprint_value = regime_fingerprint(
        X_dev,
        policy=config.regime_policy,
        exclusion_start=config.regime_exclusion_start,
        exclusion_end=config.regime_exclusion_end,
        selection_test_years=config.selection_test_years,
        stress_test_years=config.stress_test_years,
    )
    return DevelopmentData(
        X_dev=X_dev,
        y_dev=y_dev,
        holdout=holdout,
        splitter=splitter,
        fingerprint=dataset_fingerprint(X_dev, y_dev),
        config=config,
        train_eligible_mask=train_eligible_mask,
        score_eligible_mask=score_eligible_mask,
        regime_fingerprint=regime_fingerprint_value,
        environment=describe_environment(),
    )


def sample_dynamic_pipeline(
    development: DevelopmentData,
    estimator: str = "Ridge",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, PipelineSpec]:
    """Build one illustrative pipeline with fixed choices, for inspection.

    Uses ``optuna.trial.FixedTrial`` so the sampled values are pinned rather
    than drawn, which makes the rendered diagram reproducible. Nothing is
    fitted here.
    """
    defaults: Dict[str, Any] = {
        "modeler_name": "Periodic_Spline",
        "encoder": "MeanEncoder",
        "standardizer": "StandardScaler",
        "normalizer": "QuantileUniform",
        "alpha": 1.0,
        "selector": "SelectKBest",
        "kbest_k": 12,
        "kbest_score_func": "f_regression",
        "target_transform": "log1p",
        # ``FixedTrial`` must contain every conditional parameter that may be
        # requested by the configured target strategy.  The illustrative
        # sample stays deterministic even when the search itself uses "auto".
        "target_strategy": TARGET_STRATEGY_DIRECT,
        "trend_extrapolation_damping": 0.5,
    }
    defaults.update(params or {})
    return build_dynamic_pipeline(
        optuna.trial.FixedTrial(defaults),
        estimator,
        numeric_features=development.config.numeric_features,
        categorical_features=development.config.categorical_features,
        search_profile=development.config.search_profile,
        target_strategy=development.config.target_strategy,
    )


# ---------------------------------------------------------------------------
# Per-estimator search
# ---------------------------------------------------------------------------


@dataclass
class EstimatorOutcome:
    """One estimator's study, winning pipeline and MLflow run."""

    estimator: str
    study: "optuna.Study"
    evaluation: FoldEvaluation
    run_id: str
    is_baseline: bool = False
    trials_planned: int = 0
    termination_reason: Optional[str] = None
    cv_fingerprint: Optional[str] = None

    @property
    def spec(self) -> PipelineSpec:
        """The dynamic-pipeline choices behind this estimator's winning trial."""
        return self.evaluation.spec

    @property
    def cv_mae_mean(self) -> float:
        """Unweighted mean MAE across the development folds."""
        return float(self.evaluation.cv_metrics["cv_mae_mean"])

    @property
    def cv_mae_selection(self) -> float:
        """Recency-weighted MAE used by Optuna and champion selection."""
        return float(self.study.best_value)


def _log_outcome(
    development: DevelopmentData,
    optimizer: TemporalRegressionOptimizer,
    study: "optuna.Study",
    evaluation: FoldEvaluation,
) -> str:
    """Write one MLflow run: params, CV metrics, diagnostics, pipeline, provenance."""
    config = development.config
    with tempfile.TemporaryDirectory(prefix="bsd_v4_mlflow_") as staging:
        pipeline_tags = evaluation.spec.as_tags()
        pipeline_tags["cv_fingerprint"] = optimizer.cv_fingerprint
        pipeline_tags["selection_metric"] = config.selection_metric
        pipeline_tags["regime_policy"] = config.regime_policy
        pipeline_tags["regime_fingerprint"] = development.regime_fingerprint
        return log_temporal_model_run(
            estimator_name=optimizer.estimator,
            params=evaluation.best_params,
            cv_metrics=evaluation.cv_metrics,
            fold_metrics=evaluation.fold_metrics,
            seasonal_metrics=evaluation.seasonal_metrics,
            extreme_metrics=evaluation.extreme_metrics,
            trials_dataframe=study.trials_dataframe(),
            feature_manifest={
                "numeric_features": list(config.numeric_features),
                "categorical_features": list(config.categorical_features),
                "excluded": {
                    "Year": "collinear with the expanding window's growth curve",
                    "is_anomalous_2020": "retrospective demand-derived regime label",
                    "operational_disruption_interval": (
                        f"{config.regime_exclusion_start} — " f"{config.regime_exclusion_end}"
                    ),
                },
                "modeler_space": optimizer.modeler_space,
                "selector_space": optimizer.selector_space,
            },
            cv_manifest={
                "cv_strategy": CV_STRATEGY_NAME,
                "cv_strategy_version": CV_STRATEGY_VERSION,
                "test_years": list(config.test_years),
                "gap_hours": config.gap_hours,
                "train_window_years": config.train_window_years,
                "fold_weights": list(optimizer.fold_weights),
                "selection_test_years": list(config.selection_test_years),
                "stress_test_years": list(config.stress_test_years),
                "regime_policy": config.regime_policy,
                "regime_exclusion_start": config.regime_exclusion_start,
                "regime_exclusion_end": config.regime_exclusion_end,
                "regime_fingerprint": development.regime_fingerprint,
                "n_training_rows_eligible": int(development.train_eligible_mask.sum()),
                "n_training_rows_excluded": int((~development.train_eligible_mask).sum()),
                "selection_metric": config.selection_metric,
                "search_profile": config.search_profile,
                "target_strategy": (
                    TARGET_STRATEGY_DIRECT
                    if optimizer.estimator == BASELINE_ESTIMATOR
                    else config.target_strategy
                ),
                "n_folds": development.n_folds,
                "development": {
                    "start": str(development.holdout.dev_start),
                    "end": str(development.holdout.dev_end),
                    "n_rows": development.holdout.n_dev_rows,
                },
                "holdout": {
                    "start": str(development.holdout.start.date()),
                    "end": str(development.holdout.end.date()),
                    "n_rows": development.holdout.n_rows,
                    "sealed": development.holdout.sealed,
                },
                "post_holdout_discarded": {
                    "start": str(development.holdout.post_holdout_start),
                    "end": str(development.holdout.post_holdout_end),
                    "n_rows": development.holdout.n_post_holdout_rows,
                },
            },
            model_object=evaluation.fitted_pipeline,
            input_example=development.X_dev.iloc[development.train_eligible_mask].head(5),
            dataset_fingerprint=development.fingerprint,
            cv_strategy=CV_STRATEGY_NAME,
            cv_strategy_version=CV_STRATEGY_VERSION,
            code_version=CODE_VERSION,
            run_mode=config.run_mode,
            pipeline_spec=pipeline_tags,
            n_features_selected=evaluation.spec.n_features_selected,
            trials_planned=optimizer.trials,
            trials_completed=evaluation.trials_completed,
            termination_reason=optimizer.termination_reason,
            best_iterations_by_fold=evaluation.best_iterations_by_fold,
            final_n_estimators=evaluation.final_n_estimators,
            iteration_aggregation=evaluation.iteration_aggregation,
            cap_hits_by_fold=evaluation.cap_hits_by_fold,
            n_folds_cap_hit=evaluation.n_folds_cap_hit,
            n_folds_with_budget=evaluation.n_folds_with_budget,
            iteration_ceiling=evaluation.iteration_ceiling,
            systematic_truncation=evaluation.systematic_truncation,
            run_name=f"{optimizer.estimator} [{config.run_mode}]",
            artifacts_dir=Path(staging),
        )


def optimize_estimator(
    development: DevelopmentData,
    estimator: str,
    is_baseline: bool = False,
) -> EstimatorOutcome:
    """Run (or resume) one estimator's study, evaluate the winner and log it.

    The optimizer receives only ``X_dev``/``y_dev`` and the splitter; there is
    no argument through which the holdout could travel.
    """
    config = development.config
    optimizer = TemporalRegressionOptimizer(
        estimator,
        development.X_dev,
        development.y_dev,
        development.splitter,
        run_mode=config.run_mode,
        trials=config.baseline_trials if is_baseline else config.resolved_trials,
        trial_timeout=config.resolved_trial_timeout,
        study_timeout=config.resolved_study_timeout_for(estimator),
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
        studies_dir=config.studies_dir,
        invalid_configs_path=config.invalid_configs_path,
        seed=config.seed,
        fold_weights=config.fold_weights,
        search_profile=(SEARCH_PROFILE_BROAD if is_baseline else config.search_profile),
        target_strategy=(TARGET_STRATEGY_DIRECT if is_baseline else config.target_strategy),
        enqueued_trials=(
            None if is_baseline else config.enqueued_trials_by_estimator.get(estimator)
        ),
        train_eligible_mask=development.train_eligible_mask,
        score_eligible_mask=development.score_eligible_mask,
        selection_test_years=config.selection_test_years,
        regime_policy=config.regime_policy,
        regime_fingerprint_value=development.regime_fingerprint,
    )
    study = optimizer.optimize()
    evaluation = optimizer.evaluate_best(study)
    run_id = _log_outcome(development, optimizer, study, evaluation)
    logger.info(
        "[%s] cv_mae_weighted=%.2f modeler=%s encoder=%s selector=%s run_id=%s",
        estimator,
        study.best_value,
        evaluation.spec.modeler_name,
        evaluation.spec.encoder,
        evaluation.spec.selector,
        run_id,
    )
    return EstimatorOutcome(
        estimator=estimator,
        study=study,
        evaluation=evaluation,
        run_id=run_id,
        is_baseline=is_baseline,
        trials_planned=optimizer.trials,
        termination_reason=optimizer.termination_reason,
        cv_fingerprint=optimizer.cv_fingerprint,
    )


# ---------------------------------------------------------------------------
# End-to-end selection
# ---------------------------------------------------------------------------


@dataclass
class ModelSelectionResults:
    """Everything the notebook needs to display, already computed."""

    config: ModelSelectionConfig
    development: DevelopmentData
    outcomes: List[EstimatorOutcome]
    selection: Dict[str, Any]
    manifest_path: Path
    experiment_name: str
    experiment_id: str
    tracker: MLflowTracker = field(repr=False, default=None)

    @property
    def outcomes_by_estimator(self) -> Dict[str, EstimatorOutcome]:
        """Outcomes keyed by estimator name."""
        return {outcome.estimator: outcome for outcome in self.outcomes}

    @property
    def outcomes_by_run_id(self) -> Dict[str, EstimatorOutcome]:
        """Outcomes keyed by MLflow run id — the key freezing is validated against."""
        return {outcome.run_id: outcome for outcome in self.outcomes}

    @property
    def champion(self) -> Dict[str, Any]:
        """The selected champion candidate, as recorded in the manifest."""
        return self.selection["champion"]

    @property
    def challengers(self) -> List[Dict[str, Any]]:
        """The selected challenger candidates."""
        return self.selection.get("challengers", [])

    @property
    def best_outcome(self) -> EstimatorOutcome:
        """The outcome whose MLflow run was selected as champion."""
        return self.outcomes_by_run_id[self.champion["run_id"]]

    @property
    def is_provisional(self) -> bool:
        """Whether these results are a smoke validation rather than a selection."""
        return self.config.is_smoke


def run_model_selection(
    config: ModelSelectionConfig,
    development: Optional[DevelopmentData] = None,
) -> ModelSelectionResults:
    """Run the whole selection: baseline, candidates, comparison, champion, freeze.

    The baseline always runs first so that every candidate is read against a
    constant predictor evaluated through the identical temporal CV. Champion
    selection queries MLflow rather than the in-memory outcomes, and matches
    exactly on dataset fingerprint, CV strategy version, code version and run
    mode, so a stale or smoke run can never be returned. Freezing is keyed on
    the selected ``run_id``.
    """
    require_environment()
    if config.freezes_definitive_candidates:
        # This is intentionally checked before Optuna or MLflow is opened. A
        # four-hour search must never be attributed to a HEAD that does not
        # contain the source that actually produced it.
        require_clean_git_source()
    if development is None:
        development = prepare_development_data(config)

    tracker = MLflowTracker(ExperimentConfigV4(tracking_uri=config.tracking_uri))
    experiment_id = tracker.setup_experiment()

    outcomes = [optimize_estimator(development, BASELINE_ESTIMATOR, is_baseline=True)]
    outcomes.extend(
        optimize_estimator(development, estimator) for estimator in config.candidate_estimators
    )

    selection = select_champion_and_challengers(
        tracker,
        development.fingerprint,
        n_challengers=config.n_challengers,
        metric_name=config.selection_metric,
        run_mode=config.run_mode,
        cv_fingerprint_value=outcomes[0].cv_fingerprint,
        regime_policy=config.regime_policy,
        regime_fingerprint_value=development.regime_fingerprint,
    )
    pipelines_by_run_id = {
        outcome.run_id: outcome.evaluation.fitted_pipeline for outcome in outcomes
    }
    manifest_path = freeze_candidates(
        selection,
        pipelines_by_run_id,
        run_mode=config.run_mode,
        candidates_root=config.candidates_root,
        allow_boosting_truncation=config.allow_boosting_truncation,
    )

    return ModelSelectionResults(
        config=config,
        development=development,
        outcomes=outcomes,
        selection=selection,
        manifest_path=manifest_path,
        experiment_name=tracker.config.experiment_name,
        experiment_id=experiment_id,
        tracker=tracker,
    )
