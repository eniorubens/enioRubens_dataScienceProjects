"""One-shot final-holdout validation for the v4 model-selection candidates.

Notebook 05 is a confirmatory step, not a second model search. Three candidates
were frozen by notebook 04 under an expanding meteorological-year cross
validation restricted to the normal-operations regime; here each is measured
exactly once on the sealed temporal holdout (Dec/2023 through Nov/2024), and the
pre-registered champion is either confirmed or not. Nothing is refit, no
hyperparameter is touched, and the holdout is opened by a single audited
function so that no notebook variable can carry it into a modeling call.

Every entry point is fail-closed. The candidate manifest, the pipelines' own
provenance stamps and the original MLflow selection runs are all confronted
against the declared contract before a single holdout row is read; any
divergence raises. The holdout materialisation verifies the interpreter, the
dataset fingerprint and the regime fingerprint, and refuses to proceed unless
the holdout is exactly the 8,784 hours of the authorised window. The 744 hours
of December/2024 that exist in the source but fall after the holdout are counted
and discarded, never returned.

The module owns the numbers; ``src.final_validation_reports`` owns their
presentation. The public surface is deliberately small —
:func:`prepare_final_validation`, :func:`run_final_validation` and
:func:`run_shap_validation` — so the notebook stays a thin declarative layer.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer
from statsmodels.stats.diagnostic import (
    het_arch,
    het_breuschpagan,
    het_goldfeldquandt,
    het_white,
)

from src.cv import _resolve_timestamps, split_dev_holdout
from src.data import read_data
from src.environment import (
    ENVIRONMENT_NAME,
    TRACKED_PACKAGES,
    describe_environment,
    describe_git_source_state,
    environment_fingerprint,
    package_versions,
    require_environment,
)
from src.normal_operations import (
    DEFAULT_EXCLUSION_END,
    DEFAULT_EXCLUSION_START,
    DEFAULT_SELECTION_TEST_YEARS,
    DEFAULT_STRESS_TEST_YEARS,
    REGIME_POLICY_NORMAL_OPERATIONS,
    regime_fingerprint,
)
from src.periodic_features import CosTransformer, PeriodicSplineTransformer, SinTransformer
from src.temporal_optimizer import (
    CODE_VERSION,
    CV_STRATEGY_NAME,
    CV_STRATEGY_VERSION,
    dataset_fingerprint,
)
from src.tracking import pipeline_provenance
from src.trend import RobustTrendResidualRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

FINAL_VALIDATION_CODE_VERSION = "final_validation_v1"
FINAL_EXPERIMENT_NAME = "bike_sharing_demand_v4_final_validation"

HOLDOUT_START = "2023-12-01"
HOLDOUT_END = "2024-11-30"
EXPECTED_HOLDOUT_ROWS = 8_784
EXPECTED_POST_HOLDOUT_ROWS = 744

# Data-derived fingerprints the notebook-04 selection stamped on its manifest.
# They are recomputed from the freshly sealed holdout and confronted, so a
# changed dataset or a changed regime definition stops this workflow instead of
# validating the frozen models against data they were never selected on.
EXPECTED_DATASET_FINGERPRINT = "24f188a95febd94d"
EXPECTED_ENVIRONMENT_FINGERPRINT = "8f80d9216d47a792"
EXPECTED_REGIME_FINGERPRINT = "1043b392ad5e2806"

# The pre-registered champion. The holdout is not a fresh search: CatBoost is
# named the champion before the holdout is opened, and the rule below can only
# confirm or fail to confirm it.
PREREGISTERED_CHAMPION_RUN_ID = "2106a5bf79e945f7a4a9d161e2ee01d3"
PREREGISTERED_CHAMPION_ESTIMATOR = "CatBoostRegressor"

# Confirmation thresholds, fixed before the holdout is read (see the module
# docstring and :func:`decide_confirmation`).
CONFIRM_MAE_RATIO = 1.05
CONFIRM_R2_MARGIN = 0.02

CONFIRMED = "champion_confirmed"
NOT_CONFIRMED = "champion_not_confirmed"

# SHAP explains the residual model on the log-residual scale; the sample is
# capped and drawn deterministically so the three candidates share the exact
# same rows.
SHAP_MAX_SAMPLE = 500
SHAP_RANDOM_STATE = 42
SHAP_RTOL = 1e-6
SHAP_ATOL = 1e-6

HETEROSCEDASTICITY_ALPHA = 0.05
HETEROSCEDASTICITY_ARCH_LAGS = (24, 168)
HETEROSCEDASTICITY_GQ_DROP_FRACTION = 0.2
RESIDUAL_DIAGNOSTIC_DECILES = 10
RESIDUAL_ROLLING_WINDOW = 168
RESIDUAL_ROLLING_MIN_PERIODS = 24
RESIDUAL_SCALE_MIN_OBSERVATIONS = 24
RESIDUAL_SCALE_FLOOR = 1e-6
RESIDUAL_ACF_LAGS = (1, 24, 168)

ROLE_CHAMPION = "champion"
ROLE_CHALLENGER = "challenger"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES_ROOT = _PROJECT_ROOT / "dataset" / "normal_operations" / "candidates_v4"
DEFAULT_MANIFEST_PATH = DEFAULT_CANDIDATES_ROOT / "candidates_manifest.json"
DEFAULT_RUNTIME_ROOT = _PROJECT_ROOT / "dataset" / "normal_operations" / "final_validation_v4"
DEFAULT_TRACKING_URI = f"file:{_PROJECT_ROOT / 'mlruns'}"

FINAL_MANIFEST_NAME = "final_validation_manifest.json"

# The manifest keys whose values are a fixed contract for a definitive
# notebook-05 run. Data-derived fingerprints are confronted separately, against
# values recomputed from the data, not against these literals.
_MANIFEST_CONTRACT: Dict[str, Any] = {
    "run_mode": "full",
    "provisional": False,
    "selection_metric": "cv_mae_weighted",
    "cv_strategy_version": CV_STRATEGY_VERSION,
    "code_version": CODE_VERSION,
    "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
    "environment_name": ENVIRONMENT_NAME,
    "environment_fingerprint": EXPECTED_ENVIRONMENT_FINGERPRINT,
    "regime_policy": REGIME_POLICY_NORMAL_OPERATIONS,
    "regime_fingerprint": EXPECTED_REGIME_FINGERPRINT,
}

# What every original selection run must still assert. These are read, never
# written: the selection experiment is immutable and no holdout metric is ever
# added to it.
_REQUIRED_RUN_TAGS: Tuple[str, ...] = (
    "estimator",
    "dataset_fingerprint",
    "cv_strategy_version",
    "code_version",
    "run_mode",
    "environment_name",
    "environment_fingerprint",
    "regime_policy",
    "regime_fingerprint",
    "model_logged",
    "model_artifact_verified",
    "git_source_dirty",
    "params_hash",
    "pipeline_spec_hash",
)

# The provenance stamp carried by a frozen pipeline, and the manifest key each
# element must reproduce.
_PROVENANCE_TO_MANIFEST: Tuple[Tuple[str, str], ...] = (
    ("source_run_id", "run_id"),
    ("best_params_hash", "params_hash"),
    ("pipeline_spec_hash", "pipeline_spec_hash"),
    ("code_version", "code_version"),
    ("dataset_fingerprint", "dataset_fingerprint"),
)

_CV_METRIC_KEYS: Tuple[str, ...] = (
    "cv_mae_mean",
    "cv_rmse_mean",
    "cv_r2_mean",
    "cv_wape_mean",
    "cv_mean_bias",
    "cv_mae_weighted",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FinalValidationConfig:
    """Everything the notebook declares for the confirmatory holdout run.

    Parameters
    ----------
    manifest_path:
        The definitive candidate manifest written by a full notebook-04 run.
    runtime_root:
        Directory the final artifacts are persisted under. The complete
        manifest written there is what makes a re-run idempotent.
    holdout_start, holdout_end:
        The authorised holdout window. Any timestamp outside it aborts the
        materialisation.
    error_quantiles:
        Quantiles of the absolute error reported per candidate.
    temperature_cold_c, temperature_hot_c:
        Thresholds for the extreme-temperature segments whose previous
        fragility motivates their explicit reporting.
    log_to_mlflow:
        Whether the run is recorded in the dedicated final-validation
        experiment. Turned off in tests.
    """

    manifest_path: Path = DEFAULT_MANIFEST_PATH
    runtime_root: Path = DEFAULT_RUNTIME_ROOT
    holdout_start: str = HOLDOUT_START
    holdout_end: str = HOLDOUT_END
    target: str = "Rented Bike Count"
    experiment_name: str = FINAL_EXPERIMENT_NAME
    tracking_uri: Optional[str] = DEFAULT_TRACKING_URI
    shap_max_sample: int = SHAP_MAX_SAMPLE
    shap_random_state: int = SHAP_RANDOM_STATE
    shap_rtol: float = SHAP_RTOL
    shap_atol: float = SHAP_ATOL
    heteroscedasticity_alpha: float = HETEROSCEDASTICITY_ALPHA
    heteroscedasticity_arch_lags: Sequence[int] = HETEROSCEDASTICITY_ARCH_LAGS
    heteroscedasticity_gq_drop_fraction: float = HETEROSCEDASTICITY_GQ_DROP_FRACTION
    confirm_mae_ratio: float = CONFIRM_MAE_RATIO
    confirm_r2_margin: float = CONFIRM_R2_MARGIN
    error_quantiles: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99)
    temperature_cold_c: float = 0.0
    temperature_hot_c: float = 30.0
    log_to_mlflow: bool = True

    @property
    def final_manifest_path(self) -> Path:
        """Location of the complete final manifest, written last on success."""
        return Path(self.runtime_root) / FINAL_MANIFEST_NAME


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FrozenCandidate:
    """A loaded, provenance-verified frozen pipeline and its manifest record."""

    role: str
    run_id: str
    estimator: str
    pipeline: Any
    artifact_path: Path
    artifact_sha256: str
    provenance: Dict[str, str]
    manifest_entry: Dict[str, Any]
    cv_metrics: Dict[str, float]

    @property
    def is_champion(self) -> bool:
        """Whether this candidate is the pre-registered champion."""
        return self.role == ROLE_CHAMPION


@dataclass
class FinalEvaluationData:
    """The sealed holdout and the provenance recomputed while materialising it.

    Carries the holdout observations because measuring the frozen models is the
    whole point of this workflow — but only this object, produced by the single
    audited materialisation function, ever holds them.
    """

    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    timestamps: pd.Series
    dataset_fingerprint: str
    holdout_fingerprint: str
    regime_fingerprint: str
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp
    n_dev_rows: int
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    n_holdout_rows: int
    n_post_holdout_rows: int
    post_holdout_start: Optional[pd.Timestamp]
    post_holdout_end: Optional[pd.Timestamp]
    environment: Dict[str, str]


@dataclass
class CandidateHoldoutEvaluation:
    """One candidate's single-shot holdout predictions, metrics and residuals.

    Convention, documented so no interpretation is inverted downstream:
    ``bias`` is ``mean(y_pred - y_true)`` (positive means over-estimation), and
    the per-row ``residual`` reported for the diagnostics is ``y_true - y_pred``
    (positive means under-estimation).
    """

    role: str
    run_id: str
    estimator: str
    predictions: np.ndarray
    residuals: pd.Series
    metrics: Dict[str, float]
    cv_metrics: Dict[str, float]

    @property
    def is_champion(self) -> bool:
        """Whether this evaluation belongs to the pre-registered champion."""
        return self.role == ROLE_CHAMPION


@dataclass
class ShapCandidateExplanation:
    """SHAP decomposition of one candidate's residual model on the shared sample.

    The values explain the log-residual prediction of the core estimator, which
    is not additive on the bicycles-per-hour scale because of the ``expm1``
    reconstruction. Both additivity identities are checked at construction and
    their worst error is recorded.
    """

    role: str
    run_id: str
    estimator: str
    sample_positions: np.ndarray
    feature_names: List[str]
    feature_sources: List[str]
    shap_values: np.ndarray
    expected_value: float
    matrix: np.ndarray
    detailed_importance: pd.DataFrame
    grouped_importance: pd.DataFrame
    additivity_max_error: float
    reconstruction_max_error: float
    local_examples: Dict[str, int] = field(default_factory=dict)

    @property
    def is_champion(self) -> bool:
        """Whether this explanation belongs to the pre-registered champion."""
        return self.role == ROLE_CHAMPION


@dataclass
class FinalValidationResults:
    """Everything the notebook needs to display, already computed."""

    config: FinalValidationConfig
    manifest: Dict[str, Any]
    candidates: List[FrozenCandidate]
    data: Optional[FinalEvaluationData]
    evaluations: List[CandidateHoldoutEvaluation]
    comparison: pd.DataFrame
    confirmation: Dict[str, Any]
    segmented: Dict[str, pd.DataFrame]
    predictions: pd.DataFrame
    manifest_fingerprint: str
    final_manifest_path: Path
    parent_run_id: Optional[str] = None
    child_run_ids: Dict[str, str] = field(default_factory=dict)
    loaded_from_cache: bool = False
    shap: Optional[List[ShapCandidateExplanation]] = None

    @property
    def champion_evaluation(self) -> CandidateHoldoutEvaluation:
        """The pre-registered champion's holdout evaluation."""
        return next(item for item in self.evaluations if item.is_champion)

    @property
    def decision(self) -> str:
        """The confirmation outcome — confirmed or not confirmed."""
        return self.confirmation["decision"]


# ---------------------------------------------------------------------------
# Manifest audit
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    """Read the candidate manifest JSON, raising a clear error if it is absent."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate manifest not found at {path}. Notebook 05 validates the "
            "definitive candidates frozen by a full notebook-04 run; without that "
            "manifest there is nothing to validate."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """A stable 16-hex digest of the manifest content, for cross-referencing."""
    payload = json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def audit_manifest(manifest: Mapping[str, Any]) -> None:
    """Confront the manifest against the fixed notebook-05 contract, fail-closed.

    The check is deliberately literal: a provisional manifest, a smoke run, a
    different code or CV-strategy version, or a different dataset/regime/environment
    fingerprint each aborts, because validating the frozen models would then mean
    validating them against a protocol they were never selected under.
    """
    problems: List[str] = []
    for key, expected in _MANIFEST_CONTRACT.items():
        actual = manifest.get(key)
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected {expected!r})")

    champion = manifest.get("champion") or {}
    if champion.get("run_id") != PREREGISTERED_CHAMPION_RUN_ID:
        problems.append(
            f"champion.run_id={champion.get('run_id')!r} "
            f"(expected {PREREGISTERED_CHAMPION_RUN_ID!r})"
        )
    if champion.get("estimator") != PREREGISTERED_CHAMPION_ESTIMATOR:
        problems.append(
            f"champion.estimator={champion.get('estimator')!r} "
            f"(expected {PREREGISTERED_CHAMPION_ESTIMATOR!r})"
        )
    challengers = manifest.get("challengers") or []
    if len(challengers) != 2:
        problems.append(f"expected exactly 2 challengers, found {len(challengers)}")

    if problems:
        raise ValueError(
            "The candidate manifest does not match the notebook-05 contract:\n  - "
            + "\n  - ".join(problems)
            + "\nThe holdout is not opened until the manifest is exactly the one a "
            "definitive notebook-04 run produced."
        )


def manifest_entries(manifest: Mapping[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return ``(role, entry)`` pairs, champion first, then the challengers."""
    entries: List[Tuple[str, Dict[str, Any]]] = [(ROLE_CHAMPION, dict(manifest["champion"]))]
    entries.extend((ROLE_CHALLENGER, dict(entry)) for entry in manifest.get("challengers", []))
    return entries


# ---------------------------------------------------------------------------
# Source-run audit (MLflow, read-only)
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """A read-only snapshot of one original selection run."""

    run_id: str
    status: str
    tags: Dict[str, str]


def _fetch_run_record(run_id: str, tracking_uri: Optional[str] = None) -> RunRecord:
    """Fetch one run's status and tags from MLflow without modifying it."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    client = MlflowClient()
    run = client.get_run(run_id)
    return RunRecord(
        run_id=run_id,
        status=run.info.status,
        tags=dict(run.data.tags),
    )


def verify_run_record(entry: Mapping[str, Any], record: RunRecord) -> List[str]:
    """Return the list of problems with an original selection run, empty if sound.

    The run must be ``FINISHED``; its ``run_id`` must be the one the manifest
    names; every required provenance tag must be present and, where the manifest
    also records it, must agree; the model must be logged and verified; and the
    source that produced it must have been committed (``git_source_dirty=false``).
    """
    problems: List[str] = []
    if record.status != "FINISHED":
        problems.append(f"status is {record.status!r}, expected 'FINISHED'")
    if record.run_id != entry.get("run_id"):
        problems.append(
            f"run_id {record.run_id!r} does not match the manifest entry "
            f"{entry.get('run_id')!r}"
        )

    for tag in _REQUIRED_RUN_TAGS:
        if tag not in record.tags:
            problems.append(f"missing provenance tag {tag!r}")

    if record.tags.get("model_logged") != "true":
        problems.append("model_logged is not 'true'")
    if record.tags.get("model_artifact_verified") != "true":
        problems.append("model_artifact_verified is not 'true'")
    if record.tags.get("git_source_dirty") != "false":
        problems.append("git_source_dirty is not 'false'")

    # Tags whose value the manifest also records must agree exactly.
    for tag in (
        "estimator",
        "dataset_fingerprint",
        "cv_strategy_version",
        "code_version",
        "run_mode",
        "environment_name",
        "environment_fingerprint",
        "regime_policy",
        "regime_fingerprint",
        "params_hash",
        "pipeline_spec_hash",
    ):
        expected = entry.get(tag)
        if expected is not None and record.tags.get(tag) != str(expected):
            problems.append(
                f"tag {tag!r}={record.tags.get(tag)!r} does not match manifest {expected!r}"
            )
    return problems


def audit_source_runs(
    manifest: Mapping[str, Any],
    fetch: Optional[Callable[[str], RunRecord]] = None,
    tracking_uri: Optional[str] = None,
) -> Dict[str, RunRecord]:
    """Confront every candidate's original MLflow run against the manifest.

    ``fetch`` is injectable so the audit can be unit-tested with synthetic run
    records; by default it reads the real runs read-only. The selection runs are
    never modified — no holdout metric is written to them.
    """
    effective_tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
    fetcher = fetch or (lambda run_id: _fetch_run_record(run_id, effective_tracking_uri))
    records: Dict[str, RunRecord] = {}
    problems: List[str] = []
    for role, entry in manifest_entries(manifest):
        run_id = entry["run_id"]
        record = fetcher(run_id)
        records[run_id] = record
        for problem in verify_run_record(entry, record):
            problems.append(f"{role} {entry.get('estimator')} (run {run_id}): {problem}")
    if problems:
        raise ValueError(
            "The original selection runs do not match the manifest:\n  - " + "\n  - ".join(problems)
        )
    return records


# ---------------------------------------------------------------------------
# Frozen-candidate loading and provenance confrontation
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Full SHA-256 of a file's bytes, recorded so the artifact is auditable."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _assert_fitted_frozen_pipeline(pipeline: Any, run_id: str) -> Any:
    """Return the fitted core pipeline, refusing anything that is not fitted.

    Never calls ``fit``: a frozen candidate is loaded already fitted, and the
    whole point of this workflow is that it is only ever used through
    ``predict``. An unfitted object here means the wrong artifact was handed in.
    """
    if not isinstance(pipeline, RobustTrendResidualRegressor):
        raise ValueError(
            f"Frozen candidate for run {run_id} is a {type(pipeline).__name__}, not the "
            "RobustTrendResidualRegressor every v4 candidate is wrapped in."
        )
    core = getattr(pipeline, "estimator_", None)
    if core is None:
        raise ValueError(
            f"Frozen candidate for run {run_id} is not fitted: its dynamic estimator_ "
            "is missing. Notebook 05 never fits a candidate, so an unfitted artifact "
            "cannot be used."
        )
    return core


def _confront_pipeline_provenance(entry: Mapping[str, Any], pipeline: Any) -> Dict[str, str]:
    """Assert the pipeline's own stamp reproduces the manifest entry's identity."""
    run_id = entry["run_id"]
    core = _assert_fitted_frozen_pipeline(pipeline, run_id)

    try:
        regressor = core.named_steps["regressor"].regressor_
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"Frozen candidate for run {run_id} has no fitted 'regressor' step and "
            "cannot be a v4 selection artifact."
        ) from exc
    actual_estimator = type(regressor).__name__
    if actual_estimator != entry.get("estimator"):
        raise ValueError(
            f"Frozen candidate for run {run_id} wraps a {actual_estimator} but the "
            f"manifest declares estimator={entry.get('estimator')!r}."
        )

    provenance = pipeline_provenance(pipeline)
    if provenance is None:
        raise ValueError(
            f"Frozen candidate for run {run_id} carries no provenance stamp; only a "
            "pipeline stamped at selection time can be proven to belong to its run."
        )
    mismatches = [
        f"{stamp_key}={provenance.get(stamp_key)!r} != {entry.get(manifest_key)!r}"
        for stamp_key, manifest_key in _PROVENANCE_TO_MANIFEST
        if entry.get(manifest_key) is not None
        and str(provenance.get(stamp_key)) != str(entry.get(manifest_key))
    ]
    if mismatches:
        raise ValueError(
            f"Provenance mismatch for run {run_id}: " + "; ".join(mismatches) + ". The "
            "frozen artifact was produced by a different run, parameter set or code version."
        )
    return dict(provenance)


def load_frozen_candidate(role: str, entry: Mapping[str, Any]) -> FrozenCandidate:
    """Load one frozen pipeline, verify it against the manifest and describe it.

    The pipeline is unpickled (never fitted), its file hashed for the audit
    trail, its provenance stamp confronted with the manifest entry, and its
    regressor class checked against the declared estimator.
    """
    artifact_path = Path(entry["artifact_path"])
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Frozen pipeline for run {entry.get('run_id')} is missing at {artifact_path}."
        )
    with gzip.open(artifact_path, "rb") as handle:
        pipeline = pickle.load(handle)
    provenance = _confront_pipeline_provenance(entry, pipeline)
    cv_metrics = {key: float(entry[key]) for key in _CV_METRIC_KEYS if entry.get(key) is not None}
    return FrozenCandidate(
        role=role,
        run_id=entry["run_id"],
        estimator=entry["estimator"],
        pipeline=pipeline,
        artifact_path=artifact_path,
        artifact_sha256=_sha256_file(artifact_path),
        provenance=provenance,
        manifest_entry=dict(entry),
        cv_metrics=cv_metrics,
    )


def load_frozen_candidates(manifest: Mapping[str, Any]) -> List[FrozenCandidate]:
    """Load and verify all frozen candidates, champion first."""
    return [load_frozen_candidate(role, entry) for role, entry in manifest_entries(manifest)]


# ---------------------------------------------------------------------------
# Holdout materialisation — the single authorised opening of the holdout
# ---------------------------------------------------------------------------


@dataclass
class SealedHoldout:
    """The holdout slice and the split metadata, with no fingerprints yet."""

    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    timestamps: pd.Series
    dev_start: Optional[pd.Timestamp]
    dev_end: Optional[pd.Timestamp]
    n_dev_rows: int
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    n_holdout_rows: int
    n_post_holdout_rows: int
    post_holdout_start: Optional[pd.Timestamp]
    post_holdout_end: Optional[pd.Timestamp]


def seal_holdout(
    raw: pd.DataFrame,
    target: str,
    holdout_start: str,
    holdout_end: str,
    expected_rows: int = EXPECTED_HOLDOUT_ROWS,
) -> SealedHoldout:
    """Slice the holdout by date, discard the post-holdout tail and assert the window.

    Development is everything strictly before the holdout start; the holdout is
    the inclusive ``[start, end]`` window; anything after the window — the 744
    hours of December/2024 in the real data — is counted and dropped, never
    returned. The function aborts unless the holdout is exactly ``expected_rows``
    observations and every returned timestamp is inside the authorised window,
    and it guarantees features, target and timestamps stay aligned. It is pure in
    ``raw``, so it can be exercised on a synthetic frame of the same shape.
    """
    dates = _resolve_timestamps(raw)
    start = pd.Timestamp(holdout_start)
    end_inclusive = pd.Timestamp(holdout_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    dev_mask = (dates < start).to_numpy()
    holdout_mask = ((dates >= start) & (dates <= end_inclusive)).to_numpy()
    post_mask = (dates > end_inclusive).to_numpy()

    holdout_df = raw.loc[holdout_mask].reset_index(drop=True)
    timestamps = dates[holdout_mask].reset_index(drop=True)

    if len(holdout_df) != expected_rows:
        raise ValueError(
            f"The holdout has {len(holdout_df)} rows, expected {expected_rows}. "
            "The authorised window is Dec/2023 through Nov/2024 inclusive."
        )
    if bool((timestamps < start).any()) or bool((timestamps > end_inclusive).any()):
        raise ValueError("A holdout timestamp falls outside the authorised window.")

    y_holdout = holdout_df[target].reset_index(drop=True)
    X_holdout = holdout_df.drop(columns=[target])
    if not (len(X_holdout) == len(y_holdout) == len(timestamps)):
        raise ValueError("Holdout features, target and timestamps are not aligned.")

    dev_dates = dates[dev_mask]
    post_dates = dates[post_mask]
    return SealedHoldout(
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        timestamps=timestamps,
        dev_start=dev_dates.min() if len(dev_dates) else None,
        dev_end=dev_dates.max() if len(dev_dates) else None,
        n_dev_rows=int(dev_mask.sum()),
        holdout_start=start,
        holdout_end=pd.Timestamp(holdout_end),
        n_holdout_rows=len(X_holdout),
        n_post_holdout_rows=int(post_mask.sum()),
        post_holdout_start=post_dates.min() if len(post_dates) else None,
        post_holdout_end=post_dates.max() if len(post_dates) else None,
    )


def materialize_final_holdout(config: FinalValidationConfig) -> FinalEvaluationData:
    """Open the sealed holdout once, after verifying every provenance invariant.

    The interpreter is checked before a row is read. The raw dataset is loaded
    through :func:`src.data.read_data`; the development split is recomputed
    exactly as notebook 04 did (via :func:`src.cv.split_dev_holdout`) so its
    fingerprint can be confronted with the manifest's; the regime fingerprint is
    recomputed and confronted; and the holdout is sliced by :func:`seal_holdout`.
    The 744 hours of December/2024 that follow the holdout are counted and
    dropped, never returned. The function aborts unless the holdout is exactly the
    authorised 8,784-hour window.
    """
    require_environment()
    raw = read_data()

    X_dev, y_dev, _ = split_dev_holdout(
        raw,
        target=config.target,
        holdout_start=config.holdout_start,
        holdout_end=config.holdout_end,
    )
    dev_fingerprint = dataset_fingerprint(X_dev, y_dev)
    if dev_fingerprint != EXPECTED_DATASET_FINGERPRINT:
        raise ValueError(
            f"Development dataset fingerprint {dev_fingerprint} does not match the "
            f"selection fingerprint {EXPECTED_DATASET_FINGERPRINT}. The frozen models were "
            "selected on different data and must not be validated here."
        )

    regime = regime_fingerprint(
        X_dev,
        policy=REGIME_POLICY_NORMAL_OPERATIONS,
        exclusion_start=DEFAULT_EXCLUSION_START,
        exclusion_end=DEFAULT_EXCLUSION_END,
        selection_test_years=DEFAULT_SELECTION_TEST_YEARS,
        stress_test_years=DEFAULT_STRESS_TEST_YEARS,
    )
    if regime != EXPECTED_REGIME_FINGERPRINT:
        raise ValueError(
            f"Regime fingerprint {regime} does not match the selection regime "
            f"{EXPECTED_REGIME_FINGERPRINT}."
        )

    sealed = seal_holdout(raw, config.target, config.holdout_start, config.holdout_end)
    holdout_fingerprint = dataset_fingerprint(sealed.X_holdout, sealed.y_holdout)

    return FinalEvaluationData(
        X_holdout=sealed.X_holdout,
        y_holdout=sealed.y_holdout,
        timestamps=sealed.timestamps,
        dataset_fingerprint=dev_fingerprint,
        holdout_fingerprint=holdout_fingerprint,
        regime_fingerprint=regime,
        dev_start=sealed.dev_start,
        dev_end=sealed.dev_end,
        n_dev_rows=sealed.n_dev_rows,
        holdout_start=sealed.holdout_start,
        holdout_end=sealed.holdout_end,
        n_holdout_rows=sealed.n_holdout_rows,
        n_post_holdout_rows=sealed.n_post_holdout_rows,
        post_holdout_start=sealed.post_holdout_start,
        post_holdout_end=sealed.post_holdout_end,
        environment=describe_environment(),
    )


# ---------------------------------------------------------------------------
# Prediction, metrics and residuals
# ---------------------------------------------------------------------------


def predict_holdout(candidate: FrozenCandidate, X_holdout: pd.DataFrame) -> np.ndarray:
    """Predict demand on the holdout, asserting the output contract.

    The frozen pipeline is used through ``predict`` only. The result must be
    finite, non-negative and one value per holdout row, in holdout order.
    """
    predictions = np.asarray(candidate.pipeline.predict(X_holdout), dtype=float)
    if predictions.shape != (len(X_holdout),):
        raise ValueError(
            f"{candidate.estimator} produced {predictions.shape} predictions for "
            f"{len(X_holdout)} holdout rows."
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"{candidate.estimator} produced non-finite predictions.")
    if np.any(predictions < 0):
        raise ValueError(f"{candidate.estimator} produced negative predictions.")
    return predictions


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE = sum|y-yhat| / sum|y|, defined when the denominator is zero.

    A conventional MAPE is avoided because demand can be zero. When every true
    value is zero the ratio is defined as ``0`` if the errors are also zero and
    ``inf`` otherwise, rather than raising.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    numer = float(np.sum(np.abs(y_true - y_pred)))
    if denom == 0.0:
        return 0.0 if numer == 0.0 else float("inf")
    return numer / denom


def holdout_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cv_metrics: Optional[Mapping[str, float]] = None,
    quantiles: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
) -> Dict[str, float]:
    """Compute the single-shot holdout metrics for one candidate.

    ``bias`` follows the notebook-04 convention ``mean(y_pred - y_true)``, so a
    positive bias is systematic over-estimation. The absolute-error quantiles
    and the differences against the aggregated CV metrics are included so the
    holdout can be read against what the selection expected.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics: Dict[str, float] = {
        "holdout_mae": mae,
        "holdout_rmse": rmse,
        "holdout_r2": float(r2_score(y_true, y_pred)),
        "holdout_wape": weighted_absolute_percentage_error(y_true, y_pred),
        "holdout_median_abs_error": float(np.median(abs_errors)),
        "holdout_mean_bias": float(np.mean(errors)),
        "holdout_mean_abs_residual": float(np.mean(abs_errors)),
        "holdout_n": float(len(y_true)),
    }
    for quantile in quantiles:
        label = int(round(quantile * 100))
        metrics[f"holdout_abs_error_q{label:02d}"] = float(np.quantile(abs_errors, quantile))

    if cv_metrics:
        pairs = [
            ("holdout_mae", "cv_mae_mean", "mae"),
            ("holdout_rmse", "cv_rmse_mean", "rmse"),
            ("holdout_r2", "cv_r2_mean", "r2"),
            ("holdout_wape", "cv_wape_mean", "wape"),
        ]
        for holdout_key, cv_key, short in pairs:
            if cv_key in cv_metrics:
                metrics[f"{short}_holdout_minus_cv"] = float(
                    metrics[holdout_key] - float(cv_metrics[cv_key])
                )
    return metrics


def evaluate_candidate(
    candidate: FrozenCandidate,
    data: FinalEvaluationData,
    quantiles: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
) -> CandidateHoldoutEvaluation:
    """Predict once and assemble the metrics and residuals for one candidate."""
    predictions = predict_holdout(candidate, data.X_holdout)
    y_true = data.y_holdout.to_numpy(dtype=float)
    residuals = pd.Series(y_true - predictions, index=data.timestamps.to_numpy(), name="residual")
    metrics = holdout_metrics(y_true, predictions, candidate.cv_metrics, quantiles)
    return CandidateHoldoutEvaluation(
        role=candidate.role,
        run_id=candidate.run_id,
        estimator=candidate.estimator,
        predictions=predictions,
        residuals=residuals,
        metrics=metrics,
        cv_metrics=dict(candidate.cv_metrics),
    )


def comparison_frame(evaluations: Sequence[CandidateHoldoutEvaluation]) -> pd.DataFrame:
    """One row per candidate with the headline holdout and CV metrics."""
    rows = []
    for evaluation in evaluations:
        row = {
            "role": evaluation.role,
            "estimator": evaluation.estimator,
            "run_id": evaluation.run_id,
            "holdout_mae": evaluation.metrics["holdout_mae"],
            "holdout_rmse": evaluation.metrics["holdout_rmse"],
            "holdout_r2": evaluation.metrics["holdout_r2"],
            "holdout_wape": evaluation.metrics["holdout_wape"],
            "holdout_median_abs_error": evaluation.metrics["holdout_median_abs_error"],
            "holdout_mean_bias": evaluation.metrics["holdout_mean_bias"],
            "cv_mae_mean": evaluation.cv_metrics.get("cv_mae_mean", float("nan")),
            "cv_rmse_mean": evaluation.cv_metrics.get("cv_rmse_mean", float("nan")),
            "cv_r2_mean": evaluation.cv_metrics.get("cv_r2_mean", float("nan")),
            "mae_holdout_minus_cv": evaluation.metrics.get("mae_holdout_minus_cv", float("nan")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pre-registered confirmation rule
# ---------------------------------------------------------------------------


def decide_confirmation(
    evaluations: Sequence[CandidateHoldoutEvaluation],
    mae_ratio: float = CONFIRM_MAE_RATIO,
    r2_margin: float = CONFIRM_R2_MARGIN,
) -> Dict[str, Any]:
    """Apply the pre-registered rule confirming (or not) the CatBoost champion.

    The champion is confirmed only if its holdout MAE is within ``mae_ratio`` of
    the best holdout MAE and its holdout R² is within ``r2_margin`` below the
    best holdout R². When it is not confirmed the best holdout candidate is named
    descriptively, but the manifest is not touched, the search is not reopened,
    and no model is retrained — a definitive switch would require a fresh,
    independent temporal window.
    """
    by_role = {evaluation.role: evaluation for evaluation in evaluations}
    champion = next(item for item in evaluations if item.is_champion)
    champion_mae = champion.metrics["holdout_mae"]
    champion_r2 = champion.metrics["holdout_r2"]

    maes = {item.estimator: item.metrics["holdout_mae"] for item in evaluations}
    r2s = {item.estimator: item.metrics["holdout_r2"] for item in evaluations}
    best_mae = min(maes.values())
    best_r2 = max(r2s.values())

    mae_threshold = mae_ratio * best_mae
    r2_threshold = best_r2 - r2_margin
    mae_ok = champion_mae <= mae_threshold
    r2_ok = champion_r2 >= r2_threshold
    decision = CONFIRMED if (mae_ok and r2_ok) else NOT_CONFIRMED

    best_holdout = min(evaluations, key=lambda item: item.metrics["holdout_mae"])
    return {
        "decision": decision,
        "champion_run_id": champion.run_id,
        "champion_estimator": champion.estimator,
        "champion_holdout_mae": champion_mae,
        "champion_holdout_r2": champion_r2,
        "best_holdout_mae": best_mae,
        "best_holdout_r2": best_r2,
        "mae_ratio": mae_ratio,
        "r2_margin": r2_margin,
        "mae_threshold": mae_threshold,
        "r2_threshold": r2_threshold,
        "mae_condition_met": bool(mae_ok),
        "r2_condition_met": bool(r2_ok),
        "best_holdout_estimator": best_holdout.estimator,
        "best_holdout_run_id": best_holdout.run_id,
        "champion_mae_gap_to_best": champion_mae - best_mae,
        "champion_r2_gap_to_best": champion_r2 - best_r2,
        "roles_present": sorted(by_role),
        "independent_window_required_for_switch": decision == NOT_CONFIRMED,
    }


# ---------------------------------------------------------------------------
# Engineered frame and operational-condition segmentation
# ---------------------------------------------------------------------------


def engineered_holdout_frame(candidate: FrozenCandidate, X_holdout: pd.DataFrame) -> pd.DataFrame:
    """Return the post-feature-engineering frame the models actually see.

    Uses the candidate's already-fitted feature-engineering and elapsed-hours
    steps through ``transform`` only. Feature engineering is target-free and was
    fitted on the identical development split for every candidate, so the frame
    is representative of what all three receive; it supplies the calendar and
    weather columns the segmentation reads.
    """
    core = candidate.pipeline.estimator_
    frame = core[:2].transform(X_holdout)
    return frame.reset_index(drop=True)


def segmentation_labels(
    engineered: pd.DataFrame,
    y_true: np.ndarray,
    config: FinalValidationConfig,
) -> Dict[str, pd.Series]:
    """Build the operational-condition segment labels, one Series per view.

    Every label Series is positionally aligned to the holdout, so the segmented
    metrics preserve the full row count: no observation is dropped and none is
    counted twice.
    """
    engineered = engineered.reset_index(drop=True)
    y_true = np.asarray(y_true, dtype=float)

    temperature = engineered["Temperature(C)"].astype(float)
    temperature_band = pd.cut(
        temperature,
        bins=[-np.inf, config.temperature_cold_c, 10.0, 20.0, config.temperature_hot_c, np.inf],
        labels=["<=0C", "0-10C", "10-20C", "20-30C", ">30C"],
    ).astype("object")
    extreme_cold = np.where(temperature <= config.temperature_cold_c, "extreme_cold", "other")
    extreme_hot = np.where(temperature >= config.temperature_hot_c, "extreme_hot", "other")

    demand_quintile = pd.qcut(
        pd.Series(y_true), q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop"
    ).astype("object")

    labels: Dict[str, pd.Series] = {
        "season": engineered["Seasons"].astype("object"),
        "month": engineered["Month"].astype(int).astype("object"),
        "hour": engineered["Hour"].astype(int).astype("object"),
        "week_status": engineered["WeekStatus"].astype("object"),
        "holiday": engineered["Holiday"].astype("object"),
        "functioning_day": engineered["Functioning Day"].astype("object"),
        "temperature_band": pd.Series(temperature_band, name="temperature_band"),
        "extreme_cold": pd.Series(extreme_cold, name="extreme_cold"),
        "extreme_hot": pd.Series(extreme_hot, name="extreme_hot"),
        "demand_quintile": pd.Series(demand_quintile, name="demand_quintile"),
    }
    if "Rush_Period" in engineered.columns:
        labels["rush_period"] = engineered["Rush_Period"].astype("object")

    return {key: value.reset_index(drop=True).rename(key) for key, value in labels.items()}


def segment_metrics(
    evaluations: Sequence[CandidateHoldoutEvaluation],
    labels: pd.Series,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Per-segment holdout metrics for every candidate under one segmentation.

    The returned frame has one row per ``(segment, candidate)``; the ``n``
    column sums back to the holdout size within each candidate, which is what
    the count-preservation test asserts.
    """
    y_true = np.asarray(y_true, dtype=float)
    labels = pd.Series(labels).reset_index(drop=True)
    rows = []
    for evaluation in evaluations:
        y_pred = np.asarray(evaluation.predictions, dtype=float)
        for segment in pd.unique(labels[labels.notna()]):
            mask = (labels == segment).to_numpy()
            count = int(mask.sum())
            if count == 0:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            rows.append(
                {
                    "segment": segment,
                    "estimator": evaluation.estimator,
                    "role": evaluation.role,
                    "n": count,
                    "mae": float(mean_absolute_error(yt, yp)),
                    "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                    "wape": weighted_absolute_percentage_error(yt, yp),
                    "mean_bias": float(np.mean(yp - yt)),
                }
            )
    return pd.DataFrame(rows)


def segmented_metrics(
    evaluations: Sequence[CandidateHoldoutEvaluation],
    engineered: pd.DataFrame,
    y_true: np.ndarray,
    config: FinalValidationConfig,
) -> Dict[str, pd.DataFrame]:
    """Every operational-condition segmentation, keyed by view name."""
    labels = segmentation_labels(engineered, y_true, config)
    return {name: segment_metrics(evaluations, series, y_true) for name, series in labels.items()}


# ---------------------------------------------------------------------------
# Residual diagnostics (scalars; the reports layer draws the plots)
# ---------------------------------------------------------------------------


def durbin_watson(residuals: np.ndarray) -> float:
    """Durbin-Watson statistic of a residual series (2 means no autocorrelation)."""
    residuals = np.asarray(residuals, dtype=float)
    diff = np.diff(residuals)
    denom = float(np.sum(residuals**2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(diff**2) / denom)


def autocorrelation(residuals: np.ndarray, lag: int) -> float:
    """Sample autocorrelation of ``residuals`` at a single ``lag``."""
    residuals = np.asarray(residuals, dtype=float)
    if lag <= 0 or lag >= len(residuals):
        return float("nan")
    centered = residuals - residuals.mean()
    denom = float(np.sum(centered**2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(centered[lag:] * centered[:-lag]) / denom)


def autocorrelation_function(residuals: np.ndarray, n_lags: int) -> np.ndarray:
    """Autocorrelation for lags ``0..n_lags`` (lag 0 is 1 by construction)."""
    return np.array([1.0] + [autocorrelation(residuals, lag) for lag in range(1, n_lags + 1)])


def partial_autocorrelation_function(residuals: np.ndarray, n_lags: int) -> np.ndarray:
    """Partial autocorrelation via the Levinson-Durbin recursion, lags ``0..n_lags``."""
    acf = autocorrelation_function(residuals, n_lags)
    pacf = np.ones(n_lags + 1, dtype=float)
    if n_lags >= 1:
        phi = np.zeros((n_lags + 1, n_lags + 1), dtype=float)
        phi[1, 1] = acf[1]
        pacf[1] = acf[1]
        for k in range(2, n_lags + 1):
            numerator = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
            denominator = 1.0 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
            phi_kk = numerator / denominator if denominator != 0 else 0.0
            phi[k, k] = phi_kk
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi_kk * phi[k - 1, k - j]
            pacf[k] = phi_kk
    return pacf


def residual_diagnostics(residuals: np.ndarray) -> Dict[str, float]:
    """A compact scalar summary of a residual series, treated as diagnostics.

    Normality is deliberately not reduced to a pass/fail test: with thousands of
    observations a normality test rejects trivial departures, so skewness,
    kurtosis, the Durbin-Watson statistic and the autocorrelation at the daily
    and weekly lags are reported as diagnostics instead.
    """
    residuals = np.asarray(residuals, dtype=float)
    series = pd.Series(residuals)
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "skew": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
        "durbin_watson": durbin_watson(residuals),
        "autocorr_lag_1": autocorrelation(residuals, 1),
        "autocorr_lag_24": autocorrelation(residuals, 24),
        "autocorr_lag_168": autocorrelation(residuals, 168),
    }


_HETEROSCEDASTICITY_COLUMNS: Tuple[str, ...] = (
    "role",
    "estimator",
    "test",
    "null_hypothesis",
    "statistic",
    "p_value",
    "adjusted_p_value",
    "alpha",
    "evidence_of_heteroscedasticity",
    "n_observations",
    "diagnostic_specification",
    "limitations",
    "status",
    "reason",
)


def _standardized_prediction(predictions: np.ndarray) -> np.ndarray:
    """Return a deterministic, numerically stable fitted-value covariate."""
    centered = predictions - float(np.mean(predictions))
    scale = float(np.std(centered, ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("predictions have no finite variation")
    return centered / scale


def _auxiliary_matrix(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Build an auxiliary matrix with an intercept and verify basic rank."""
    matrix = np.column_stack([np.ones(len(columns[0]), dtype=float), *columns])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("auxiliary matrix contains non-finite values")
    if matrix.shape[0] <= matrix.shape[1] + 5:
        raise ValueError("insufficient observations for the auxiliary regression")
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError("auxiliary matrix does not have sufficient rank")
    return matrix


def _validate_residual_inputs(
    evaluation: CandidateHoldoutEvaluation,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned finite residuals and predictions for diagnostics."""
    residuals = evaluation.residuals.to_numpy(dtype=float)
    predictions = np.asarray(evaluation.predictions, dtype=float)
    if residuals.shape[0] != predictions.shape[0]:
        raise ValueError("residuals and predictions have different lengths")
    if residuals.shape[0] < 40:
        raise ValueError("insufficient observations for heteroscedasticity diagnostics")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals contain non-finite values")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("predictions contain non-finite values")
    if float(np.std(residuals, ddof=0)) <= 0.0:
        raise ValueError("residuals have no variation")
    return residuals, predictions


def _heteroscedasticity_row(
    evaluation: CandidateHoldoutEvaluation,
    test: str,
    null_hypothesis: str,
    statistic: float,
    p_value: float,
    alpha: float,
    n_observations: int,
    diagnostic_specification: str,
    limitations: str,
    status: str = "ok",
    reason: str = "",
) -> Dict[str, Any]:
    """Build one stable-schema heteroscedasticity diagnostic row."""
    return {
        "role": evaluation.role,
        "estimator": evaluation.estimator,
        "test": test,
        "null_hypothesis": null_hypothesis,
        "statistic": statistic,
        "p_value": p_value,
        "adjusted_p_value": np.nan,
        "alpha": alpha,
        "evidence_of_heteroscedasticity": pd.NA,
        "n_observations": n_observations,
        "diagnostic_specification": diagnostic_specification,
        "limitations": limitations,
        "status": status,
        "reason": reason,
    }


def _not_applicable_heteroscedasticity_row(
    evaluation: CandidateHoldoutEvaluation,
    test: str,
    null_hypothesis: str,
    alpha: float,
    n_observations: int,
    diagnostic_specification: str,
    limitations: str,
    reason: str,
) -> Dict[str, Any]:
    """Build an explicit not-applicable row instead of silently emitting NaNs."""
    return _heteroscedasticity_row(
        evaluation=evaluation,
        test=test,
        null_hypothesis=null_hypothesis,
        statistic=np.nan,
        p_value=np.nan,
        alpha=alpha,
        n_observations=n_observations,
        diagnostic_specification=diagnostic_specification,
        limitations=limitations,
        status="not_applicable",
        reason=reason,
    )


def _holm_adjust(candidate_rows: List[Dict[str, Any]]) -> None:
    """Apply Holm correction in-place within one candidate's diagnostics."""
    indexed = [
        (index, float(row["p_value"]))
        for index, row in enumerate(candidate_rows)
        if row["status"] == "ok" and np.isfinite(float(row["p_value"]))
    ]
    if not indexed:
        return
    ordered = sorted(indexed, key=lambda item: item[1])
    m = len(ordered)
    previous = 0.0
    for rank, (index, p_value) in enumerate(ordered, start=1):
        adjusted = min((m - rank + 1) * p_value, 1.0)
        adjusted = max(adjusted, previous)
        previous = adjusted
        candidate_rows[index]["adjusted_p_value"] = float(adjusted)
    for row in candidate_rows:
        adjusted = row["adjusted_p_value"]
        if row["status"] == "ok" and np.isfinite(float(adjusted)):
            row["evidence_of_heteroscedasticity"] = bool(adjusted < row["alpha"])


def _run_diagnostic(
    evaluation: CandidateHoldoutEvaluation,
    test: str,
    null_hypothesis: str,
    alpha: float,
    n_observations: int,
    diagnostic_specification: str,
    limitations: str,
    runner: Callable[[], Tuple[float, float]],
) -> Dict[str, Any]:
    """Run one statsmodels diagnostic and convert failures into explicit rows."""
    try:
        statistic, p_value = runner()
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            raise ValueError("test returned a non-finite statistic or p-value")
    except Exception as exc:  # noqa: BLE001 - diagnostics must report applicability failures.
        return _not_applicable_heteroscedasticity_row(
            evaluation,
            test,
            null_hypothesis,
            alpha,
            n_observations,
            diagnostic_specification,
            limitations,
            str(exc),
        )
    return _heteroscedasticity_row(
        evaluation,
        test,
        null_hypothesis,
        float(statistic),
        float(p_value),
        alpha,
        n_observations,
        diagnostic_specification,
        limitations,
    )


def _heteroscedasticity_rows_for_evaluation(
    evaluation: CandidateHoldoutEvaluation,
    alpha: float,
    arch_lags: Sequence[int],
    gq_drop_fraction: float,
) -> List[Dict[str, Any]]:
    """Compute all heteroscedasticity diagnostics for one candidate."""
    common_limitations = (
        "Diagnostic only; BP, White and Goldfeld-Quandt assume independent errors "
        "and can be affected by temporal dependence."
    )
    arch_limitations = (
        "Diagnostic only; preserves the residual time order and targets conditional "
        "heteroscedasticity, not champion selection."
    )
    try:
        residuals, predictions = _validate_residual_inputs(evaluation)
        fitted_z = _standardized_prediction(predictions)
        n_observations = len(residuals)
        temporal_reason = ""
        if not evaluation.residuals.index.is_monotonic_increasing:
            temporal_reason = "residual index is not monotonic increasing"
    except Exception as exc:  # noqa: BLE001 - return every requested diagnostic explicitly.
        n_observations = len(evaluation.residuals)
        rows = [
            _not_applicable_heteroscedasticity_row(
                evaluation,
                "Breusch-Pagan (Koenker)",
                "Residual variance is constant against fitted-value auxiliaries.",
                alpha,
                n_observations,
                "Koenker robust LM test; auxiliary variables: intercept, "
                "standardized prediction and squared standardized prediction.",
                common_limitations,
                str(exc),
            ),
            _not_applicable_heteroscedasticity_row(
                evaluation,
                "White",
                "Residual variance is constant against fitted-value auxiliaries.",
                alpha,
                n_observations,
                "White LM test with low-dimensional exog: intercept and standardized "
                "prediction; statsmodels adds the square term.",
                common_limitations,
                str(exc),
            ),
            _not_applicable_heteroscedasticity_row(
                evaluation,
                "Goldfeld-Quandt",
                "The two ordered residual subsamples have equal variance.",
                alpha,
                n_observations,
                f"Two-sided test sorted by prediction; central {gq_drop_fraction:.0%} removed.",
                common_limitations,
                str(exc),
            ),
        ]
        for lag in arch_lags:
            rows.append(
                _not_applicable_heteroscedasticity_row(
                    evaluation,
                    f"Engle ARCH (lag {lag})",
                    "There is no ARCH effect up to the specified lag.",
                    alpha,
                    n_observations,
                    f"LM test on squared residual lags 1..{lag}, using temporal order.",
                    arch_limitations,
                    str(exc),
                )
            )
        return rows

    bp_spec = (
        "Koenker robust LM test; auxiliary variables: intercept, standardized "
        "prediction and squared standardized prediction."
    )
    white_spec = (
        "White LM test with low-dimensional exog: intercept and standardized "
        "prediction; statsmodels adds the square term."
    )
    gq_spec = f"Two-sided test sorted by prediction; central {gq_drop_fraction:.0%} removed."
    rows = [
        _run_diagnostic(
            evaluation,
            "Breusch-Pagan (Koenker)",
            "Residual variance is constant against fitted-value auxiliaries.",
            alpha,
            n_observations,
            bp_spec,
            common_limitations,
            lambda: (lambda result: (result[0], result[1]))(
                het_breuschpagan(residuals, _auxiliary_matrix([fitted_z, fitted_z**2]))
            ),
        ),
        _run_diagnostic(
            evaluation,
            "White",
            "Residual variance is constant against fitted-value auxiliaries.",
            alpha,
            n_observations,
            white_spec,
            common_limitations,
            lambda: (lambda result: (result[0], result[1]))(
                het_white(residuals, _auxiliary_matrix([fitted_z]))
            ),
        ),
        _run_diagnostic(
            evaluation,
            "Goldfeld-Quandt",
            "The two ordered residual subsamples have equal variance.",
            alpha,
            n_observations,
            gq_spec,
            common_limitations,
            lambda: (lambda result: (result[0], result[1]))(
                het_goldfeldquandt(
                    residuals,
                    _auxiliary_matrix([fitted_z]),
                    idx=1,
                    drop=gq_drop_fraction,
                    alternative="two-sided",
                )
            ),
        ),
    ]
    for lag in arch_lags:
        lag = int(lag)
        spec = f"LM test on squared residual lags 1..{lag}, using temporal order."
        if temporal_reason:
            rows.append(
                _not_applicable_heteroscedasticity_row(
                    evaluation,
                    f"Engle ARCH (lag {lag})",
                    "There is no ARCH effect up to the specified lag.",
                    alpha,
                    n_observations,
                    spec,
                    arch_limitations,
                    temporal_reason,
                )
            )
            continue
        if lag <= 0 or n_observations < max(40, 3 * lag):
            rows.append(
                _not_applicable_heteroscedasticity_row(
                    evaluation,
                    f"Engle ARCH (lag {lag})",
                    "There is no ARCH effect up to the specified lag.",
                    alpha,
                    n_observations,
                    spec,
                    arch_limitations,
                    f"insufficient observations for ARCH lag {lag}",
                )
            )
            continue
        rows.append(
            _run_diagnostic(
                evaluation,
                f"Engle ARCH (lag {lag})",
                "There is no ARCH effect up to the specified lag.",
                alpha,
                n_observations,
                spec,
                arch_limitations,
                lambda lag=lag: (lambda result: (result[0], result[1]))(
                    het_arch(residuals, nlags=lag, ddof=0)
                ),
            )
        )
    return rows


def heteroscedasticity_diagnostics(
    results: FinalValidationResults,
    config: Optional[FinalValidationConfig] = None,
) -> pd.DataFrame:
    """Formal residual heteroscedasticity diagnostics for frozen candidates.

    The diagnostics reuse only predictions and residuals already present in
    ``FinalValidationResults``. They do not materialize the holdout, refit a
    model, call ``predict`` or change the pre-registered champion decision.
    Holm correction is applied within each candidate over all applicable tests.
    """
    config = config or results.config
    alpha = float(getattr(config, "heteroscedasticity_alpha", HETEROSCEDASTICITY_ALPHA))
    arch_lags = tuple(getattr(config, "heteroscedasticity_arch_lags", HETEROSCEDASTICITY_ARCH_LAGS))
    gq_drop = float(
        getattr(
            config,
            "heteroscedasticity_gq_drop_fraction",
            HETEROSCEDASTICITY_GQ_DROP_FRACTION,
        )
    )
    rows: List[Dict[str, Any]] = []
    for evaluation in results.evaluations:
        candidate_rows = _heteroscedasticity_rows_for_evaluation(
            evaluation=evaluation,
            alpha=alpha,
            arch_lags=arch_lags,
            gq_drop_fraction=gq_drop,
        )
        _holm_adjust(candidate_rows)
        rows.extend(candidate_rows)
    return pd.DataFrame(rows, columns=_HETEROSCEDASTICITY_COLUMNS)


# ---------------------------------------------------------------------------
# Post-holdout residual triage for the confirmed champion
# ---------------------------------------------------------------------------


_RESIDUAL_PROFILE_VIEWS: Tuple[str, ...] = (
    "month",
    "hour",
    "weekday",
    "season",
    "predicted_demand_decile",
    "observed_demand_decile",
)

_RESIDUAL_VERSIONS: Tuple[str, ...] = (
    "raw",
    "globally_debiased",
    "calendar_demeaned",
    "level_standardized",
)


def _season_from_month(month: int) -> str:
    """Meteorological season used by the Seoul bike notebooks."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _deterministic_deciles(
    values: Sequence[float],
    n_deciles: int = RESIDUAL_DIAGNOSTIC_DECILES,
) -> pd.Series:
    """Return deterministic qcut labels, with explicit fallbacks for ties."""
    series = pd.Series(np.asarray(values, dtype=float))
    if series.empty:
        return pd.Series([], dtype="object")
    if not np.all(np.isfinite(series)):
        raise ValueError("decile input contains non-finite values")
    if series.nunique(dropna=False) < 2:
        return pd.Series(["D01"] * len(series), dtype="object")

    q = max(1, min(int(n_deciles), len(series)))
    try:
        codes = pd.qcut(series, q=q, labels=False, duplicates="drop")
    except ValueError:
        codes = pd.Series([np.nan] * len(series))
    if codes.isna().all():
        ranked = series.rank(method="first")
        codes = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    codes = pd.Series(codes).astype(int)
    return codes.map(lambda value: f"D{value + 1:02d}").astype("object")


def _validate_against_persisted_predictions(
    results: FinalValidationResults,
    evaluation: CandidateHoldoutEvaluation,
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: np.ndarray,
    residuals: np.ndarray,
) -> None:
    """Audit the diagnostic vectors against the persisted prediction artifact."""
    stored = results.predictions
    if stored is None or stored.empty:
        return
    required = {
        "timestamp",
        "y_true",
        f"pred_{evaluation.estimator}",
        f"residual_{evaluation.estimator}",
    }
    missing = sorted(required.difference(stored.columns))
    if missing:
        raise ValueError(
            "stored predictions are missing the columns required for residual diagnostics: "
            + ", ".join(missing)
        )

    stored_timestamps = pd.DatetimeIndex(pd.to_datetime(stored["timestamp"]))
    if not stored_timestamps.equals(timestamps):
        raise ValueError("stored prediction timestamps do not match champion residual timestamps")
    np.testing.assert_allclose(stored["y_true"].to_numpy(dtype=float), y_true)
    np.testing.assert_allclose(
        stored[f"pred_{evaluation.estimator}"].to_numpy(dtype=float),
        predictions,
    )
    np.testing.assert_allclose(
        stored[f"residual_{evaluation.estimator}"].to_numpy(dtype=float),
        residuals,
    )


def champion_residual_diagnostic_frame(
    results: FinalValidationResults,
    n_deciles: int = RESIDUAL_DIAGNOSTIC_DECILES,
) -> pd.DataFrame:
    """Timestamp-aligned champion residual frame built from stored results only.

    ``bias`` follows the holdout-metrics convention ``y_pred - y_true``. The row
    residual follows the plotting convention ``y_true - y_pred``. The function
    audits against ``results.predictions`` when that persisted artifact is
    available and does not call, refit or materialize any model or dataset.
    """
    evaluation = results.champion_evaluation
    residuals_before = evaluation.residuals.copy(deep=True)
    predictions_before = np.asarray(evaluation.predictions, dtype=float).copy()

    timestamps = pd.DatetimeIndex(pd.to_datetime(evaluation.residuals.index))
    residuals = evaluation.residuals.to_numpy(dtype=float, copy=True)
    predictions = np.asarray(evaluation.predictions, dtype=float).copy()
    if len(timestamps) != len(residuals) or len(residuals) != len(predictions):
        raise ValueError("timestamps, residuals and predictions have different lengths")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("champion residual timestamps are not monotonic increasing")
    if not timestamps.is_unique:
        raise ValueError("champion residual timestamps are not unique")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("champion residuals contain non-finite values")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("champion predictions contain non-finite values")

    y_true = predictions + residuals
    if not np.all(np.isfinite(y_true)):
        raise ValueError("reconstructed target contains non-finite values")

    _validate_against_persisted_predictions(
        results=results,
        evaluation=evaluation,
        timestamps=timestamps,
        y_true=y_true,
        predictions=predictions,
        residuals=residuals,
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y_true,
            "y_pred": predictions,
            "residual": residuals,
            "bias": predictions - y_true,
        }
    )
    frame["absolute_error"] = frame["residual"].abs()
    frame["squared_error"] = frame["residual"] ** 2
    frame["month"] = frame["timestamp"].dt.month.astype(int)
    frame["hour"] = frame["timestamp"].dt.hour.astype(int)
    frame["weekday"] = frame["timestamp"].dt.weekday.astype(int)
    frame["hour_of_week"] = (frame["weekday"] * 24 + frame["hour"]).astype(int)
    frame["season"] = frame["month"].map(_season_from_month)
    frame["predicted_demand_decile"] = _deterministic_deciles(frame["y_pred"], n_deciles)
    frame["observed_demand_decile"] = _deterministic_deciles(frame["y_true"], n_deciles)

    if not evaluation.residuals.equals(residuals_before):
        raise RuntimeError("residual diagnostics mutated the champion residual series")
    np.testing.assert_array_equal(
        np.asarray(evaluation.predictions, dtype=float), predictions_before
    )
    return frame


def _profile_one_dimension(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    """Aggregate champion error metrics for one segmentation view."""
    rows: List[Dict[str, Any]] = []
    for segment, group in frame.groupby(view, observed=True, sort=True):
        n = int(len(group))
        squared = group["squared_error"].to_numpy(dtype=float)
        bias = group["bias"].to_numpy(dtype=float)
        residual = group["residual"].to_numpy(dtype=float)
        rows.append(
            {
                "view": view,
                "segment": segment,
                "n": n,
                "observed_mean": float(group["y_true"].mean()),
                "predicted_mean": float(group["y_pred"].mean()),
                "bias_mean": float(np.mean(bias)),
                "residual_mean": float(np.mean(residual)),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.mean(squared))),
                "residual_std": float(np.std(residual, ddof=1)) if n > 1 else float("nan"),
                "overestimation_share": float(np.mean(bias > 0.0)),
                "underestimation_share": float(np.mean(residual > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def champion_error_profiles(
    results: FinalValidationResults,
    n_deciles: int = RESIDUAL_DIAGNOSTIC_DECILES,
) -> Dict[str, pd.DataFrame]:
    """Champion error profiles by calendar and demand-level dimensions."""
    frame = champion_residual_diagnostic_frame(results, n_deciles=n_deciles)
    return {view: _profile_one_dimension(frame, view) for view in _RESIDUAL_PROFILE_VIEWS}


def champion_rolling_residual_diagnostics(
    results: FinalValidationResults,
    window: int = RESIDUAL_ROLLING_WINDOW,
    min_periods: int = RESIDUAL_ROLLING_MIN_PERIODS,
) -> pd.DataFrame:
    """Weekly default rolling bias, MAE, RMSE, residual mean and residual scale."""
    if window <= 0:
        raise ValueError("rolling window must be positive")
    if min_periods <= 0 or min_periods > window:
        raise ValueError("rolling min_periods must be in 1..window")
    frame = champion_residual_diagnostic_frame(results).set_index("timestamp")
    rolling = pd.DataFrame(index=frame.index)
    rolling["bias_rolling"] = frame["bias"].rolling(window, min_periods=min_periods).mean()
    rolling["mae_rolling"] = frame["absolute_error"].rolling(window, min_periods=min_periods).mean()
    rolling["rmse_rolling"] = np.sqrt(
        frame["squared_error"].rolling(window, min_periods=min_periods).mean()
    )
    rolling["residual_mean_rolling"] = (
        frame["residual"].rolling(window, min_periods=min_periods).mean()
    )
    rolling["residual_std_rolling"] = (
        frame["residual"].rolling(window, min_periods=min_periods).std()
    )
    return rolling.reset_index()


def _rms(values: pd.Series) -> float:
    """Root mean square with finite-value validation."""
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(array**2)))


def _relative_magnitude_reduction(raw_value: float, transformed_value: float) -> float:
    """Relative reduction in absolute magnitude versus the raw diagnostic."""
    if not np.isfinite(raw_value) or abs(raw_value) <= 0.0 or not np.isfinite(transformed_value):
        return float("nan")
    return float(1.0 - abs(transformed_value) / abs(raw_value))


def champion_residual_transformation_frame(
    results: FinalValidationResults,
    scale_min_observations: int = RESIDUAL_SCALE_MIN_OBSERVATIONS,
    scale_floor: float = RESIDUAL_SCALE_FLOOR,
) -> pd.DataFrame:
    """Return raw, debiased, calendar-demeaned and level-standardized residuals.

    The demeaning and standardization are descriptive transformations estimated
    inside the already-opened holdout. They are diagnostics only, not a validated
    calibrator that may be attached to the current champion.
    """
    if scale_min_observations <= 0:
        raise ValueError("scale_min_observations must be positive")
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")

    base = champion_residual_diagnostic_frame(results)
    raw = base["residual"].astype(float)
    globally_debiased = raw - float(raw.mean())
    calendar_mean = raw.groupby(base["hour_of_week"], observed=True).transform("mean")
    calendar_demeaned = raw - calendar_mean
    global_scale = max(_rms(calendar_demeaned), scale_floor)
    group_key = [base["season"], base["predicted_demand_decile"]]

    def scale_or_nan(values: pd.Series) -> float:
        if len(values) < scale_min_observations:
            return float("nan")
        scale = _rms(values)
        if not np.isfinite(scale) or scale < scale_floor:
            return float("nan")
        return max(scale, scale_floor)

    grouped_scale = calendar_demeaned.groupby(group_key, observed=True).transform(scale_or_nan)
    scale = grouped_scale.fillna(global_scale).clip(lower=scale_floor)
    level_standardized = calendar_demeaned / scale
    versions = {
        "raw": raw,
        "globally_debiased": globally_debiased,
        "calendar_demeaned": calendar_demeaned,
        "level_standardized": level_standardized,
    }
    rows = []
    base_columns = [
        "timestamp",
        "month",
        "hour",
        "weekday",
        "hour_of_week",
        "season",
        "predicted_demand_decile",
        "observed_demand_decile",
    ]
    for version, values in versions.items():
        version_frame = base[base_columns].copy()
        version_frame["residual_version"] = version
        version_frame["diagnostic_residual"] = values.to_numpy(dtype=float)
        version_frame["diagnostic_scale"] = (
            scale.to_numpy(dtype=float) if version == "level_standardized" else np.nan
        )
        rows.append(version_frame)
    return pd.concat(rows, ignore_index=True)


_RESIDUAL_TRANSFORMATION_DIAGNOSTIC_COLUMNS: Tuple[str, ...] = (
    "residual_version",
    "arch_lag",
    "n_observations",
    "n_effective",
    "mean",
    "std",
    "durbin_watson",
    "autocorr_lag_1",
    "autocorr_lag_24",
    "autocorr_lag_168",
    "squared_autocorr_lag_1",
    "squared_autocorr_lag_24",
    "squared_autocorr_lag_168",
    "arch_statistic",
    "arch_statistic_per_observation",
    "p_value",
    "adjusted_p_value",
    "alpha",
    "evidence_of_arch",
    "arch_statistic_reduction_vs_raw",
    "autocorr_lag_1_reduction_vs_raw",
    "autocorr_lag_24_reduction_vs_raw",
    "autocorr_lag_168_reduction_vs_raw",
    "squared_autocorr_lag_1_reduction_vs_raw",
    "squared_autocorr_lag_24_reduction_vs_raw",
    "squared_autocorr_lag_168_reduction_vs_raw",
    "status",
    "reason",
    "diagnostic_note",
)


def _arch_row_for_residual_version(
    version: str,
    residuals: np.ndarray,
    arch_lag: int,
    alpha: float,
) -> Dict[str, Any]:
    """One row of ARCH and ACF diagnostics for one residual transformation."""
    squared = residuals**2
    row: Dict[str, Any] = {
        "residual_version": version,
        "arch_lag": int(arch_lag),
        "n_observations": int(len(residuals)),
        "n_effective": int(max(len(residuals) - arch_lag, 0)),
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan"),
        "durbin_watson": durbin_watson(residuals),
        "alpha": float(alpha),
        "diagnostic_note": (
            "Holdout-only descriptive transformation; not a validated calibration "
            "or a champion-selection criterion."
        ),
        "status": "ok",
        "reason": "",
    }
    for lag in RESIDUAL_ACF_LAGS:
        row[f"autocorr_lag_{lag}"] = autocorrelation(residuals, lag)
        row[f"squared_autocorr_lag_{lag}"] = autocorrelation(squared, lag)

    if not np.all(np.isfinite(residuals)):
        row.update(
            {
                "arch_statistic": np.nan,
                "arch_statistic_per_observation": np.nan,
                "p_value": np.nan,
                "adjusted_p_value": np.nan,
                "evidence_of_arch": pd.NA,
                "status": "not_applicable",
                "reason": "residuals contain non-finite values",
            }
        )
        return row
    if len(residuals) < max(40, 3 * arch_lag):
        row.update(
            {
                "arch_statistic": np.nan,
                "arch_statistic_per_observation": np.nan,
                "p_value": np.nan,
                "adjusted_p_value": np.nan,
                "evidence_of_arch": pd.NA,
                "status": "not_applicable",
                "reason": f"insufficient observations for ARCH lag {arch_lag}",
            }
        )
        return row
    try:
        statistic, p_value = (lambda result: (result[0], result[1]))(
            het_arch(residuals, nlags=int(arch_lag), ddof=0)
        )
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            raise ValueError("test returned a non-finite statistic or p-value")
        row["arch_statistic"] = float(statistic)
        row["arch_statistic_per_observation"] = float(statistic) / row["n_effective"]
        row["p_value"] = float(p_value)
        row["adjusted_p_value"] = np.nan
        row["evidence_of_arch"] = pd.NA
    except Exception as exc:  # noqa: BLE001 - diagnostics report applicability failures.
        row.update(
            {
                "arch_statistic": np.nan,
                "arch_statistic_per_observation": np.nan,
                "p_value": np.nan,
                "adjusted_p_value": np.nan,
                "evidence_of_arch": pd.NA,
                "status": "not_applicable",
                "reason": str(exc),
            }
        )
    return row


def _holm_adjust_arch_rows(rows: List[Dict[str, Any]]) -> None:
    """Apply Holm correction inside each residual transformation."""
    for version in _RESIDUAL_VERSIONS:
        indices = [
            index
            for index, row in enumerate(rows)
            if row["residual_version"] == version
            and row["status"] == "ok"
            and np.isfinite(float(row["p_value"]))
        ]
        ordered = sorted(indices, key=lambda index: float(rows[index]["p_value"]))
        previous = 0.0
        m = len(ordered)
        for rank, index in enumerate(ordered, start=1):
            adjusted = min((m - rank + 1) * float(rows[index]["p_value"]), 1.0)
            adjusted = max(adjusted, previous)
            previous = adjusted
            rows[index]["adjusted_p_value"] = float(adjusted)
        for index in indices:
            rows[index]["evidence_of_arch"] = bool(
                rows[index]["adjusted_p_value"] < rows[index]["alpha"]
            )


def _add_reductions_vs_raw(rows: List[Dict[str, Any]]) -> None:
    """Add relative reductions in magnitude versus the raw residual row."""
    raw_by_lag = {
        row["arch_lag"]: row
        for row in rows
        if row["residual_version"] == "raw" and row["status"] == "ok"
    }
    for row in rows:
        raw = raw_by_lag.get(row["arch_lag"])
        if raw is None or row["status"] != "ok":
            row["arch_statistic_reduction_vs_raw"] = np.nan
        else:
            row["arch_statistic_reduction_vs_raw"] = _relative_magnitude_reduction(
                raw["arch_statistic"], row["arch_statistic"]
            )
        for lag in RESIDUAL_ACF_LAGS:
            acf_key = f"autocorr_lag_{lag}"
            square_key = f"squared_autocorr_lag_{lag}"
            if raw is None:
                row[f"{acf_key}_reduction_vs_raw"] = np.nan
                row[f"{square_key}_reduction_vs_raw"] = np.nan
                continue
            row[f"{acf_key}_reduction_vs_raw"] = _relative_magnitude_reduction(
                raw[acf_key], row[acf_key]
            )
            row[f"{square_key}_reduction_vs_raw"] = _relative_magnitude_reduction(
                raw[square_key], row[square_key]
            )


def champion_residual_transformation_diagnostics(
    results: FinalValidationResults,
    config: Optional[FinalValidationConfig] = None,
) -> pd.DataFrame:
    """ARCH, Durbin-Watson and ACF diagnostics after residual transformations."""
    config = config or results.config
    alpha = float(getattr(config, "heteroscedasticity_alpha", HETEROSCEDASTICITY_ALPHA))
    arch_lags = tuple(getattr(config, "heteroscedasticity_arch_lags", HETEROSCEDASTICITY_ARCH_LAGS))
    transformed = champion_residual_transformation_frame(results)
    rows: List[Dict[str, Any]] = []
    for version in _RESIDUAL_VERSIONS:
        values = (
            transformed.loc[transformed["residual_version"] == version, "diagnostic_residual"]
            .to_numpy(dtype=float)
            .copy()
        )
        for lag in arch_lags:
            rows.append(_arch_row_for_residual_version(version, values, int(lag), alpha))
    _holm_adjust_arch_rows(rows)
    _add_reductions_vs_raw(rows)
    return pd.DataFrame(rows, columns=_RESIDUAL_TRANSFORMATION_DIAGNOSTIC_COLUMNS)


def champion_residual_triage(
    results: FinalValidationResults,
    config: Optional[FinalValidationConfig] = None,
) -> pd.DataFrame:
    """One-row-per-ARCH-lag triage of bias, autocorrelation and residual scale."""
    config = config or results.config
    base = champion_residual_diagnostic_frame(results)
    profiles = champion_error_profiles(results)
    by_level = profiles["predicted_demand_decile"]
    residual_std = by_level["residual_std"].replace(0.0, np.nan)
    level_std_ratio = float(residual_std.max() / residual_std.min())
    mae_by_level = by_level["mae"].replace(0.0, np.nan)
    level_mae_ratio = float(mae_by_level.max() / mae_by_level.min())
    diagnostics = champion_residual_transformation_diagnostics(results, config=config)
    raw = diagnostics[diagnostics["residual_version"] == "raw"].set_index("arch_lag")
    standardized = diagnostics[diagnostics["residual_version"] == "level_standardized"].set_index(
        "arch_lag"
    )
    rows = []
    residuals = base["residual"].to_numpy(dtype=float)
    bias = base["bias"].to_numpy(dtype=float)
    mae = float(base["absolute_error"].mean())
    for arch_lag in tuple(
        getattr(config, "heteroscedasticity_arch_lags", HETEROSCEDASTICITY_ARCH_LAGS)
    ):
        raw_row = raw.loc[int(arch_lag)] if int(arch_lag) in raw.index else pd.Series(dtype=float)
        standardized_row = (
            standardized.loc[int(arch_lag)]
            if int(arch_lag) in standardized.index
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "estimator": results.champion_evaluation.estimator,
                "arch_lag": int(arch_lag),
                "mae": mae,
                "bias_mean": float(np.mean(bias)),
                "bias_abs_to_mae_ratio": float(abs(np.mean(bias)) / mae) if mae else np.nan,
                "durbin_watson": durbin_watson(residuals),
                "autocorr_lag_1": autocorrelation(residuals, 1),
                "autocorr_lag_24": autocorrelation(residuals, 24),
                "autocorr_lag_168": autocorrelation(residuals, 168),
                "predicted_level_residual_std_ratio": level_std_ratio,
                "predicted_level_mae_ratio": level_mae_ratio,
                "raw_arch_statistic": raw_row.get("arch_statistic", np.nan),
                "standardized_arch_statistic": standardized_row.get("arch_statistic", np.nan),
                "arch_statistic_reduction_after_standardization": standardized_row.get(
                    "arch_statistic_reduction_vs_raw", np.nan
                ),
                "raw_arch_statistic_per_observation": raw_row.get(
                    "arch_statistic_per_observation", np.nan
                ),
                "standardized_arch_statistic_per_observation": standardized_row.get(
                    "arch_statistic_per_observation", np.nan
                ),
                "evidence_of_arch_after_standardization": standardized_row.get(
                    "evidence_of_arch", pd.NA
                ),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SHAP: decomposition, deterministic names, additivity
# ---------------------------------------------------------------------------


def _core_estimator(pipeline: RobustTrendResidualRegressor):
    """Return the fitted tree estimator and its transformed-feature transformer."""
    core = pipeline.estimator_
    ttr = core.named_steps["regressor"]
    if not isinstance(ttr, TransformedTargetRegressor):
        raise ValueError("The 'regressor' step is not a TransformedTargetRegressor.")
    transformer = ttr.transformer_
    return core, ttr.regressor_, transformer


def assert_identity_target_transformer(transformer: Any) -> None:
    """Refuse to proceed unless the target transformer is a verified identity.

    The three current candidates wrap their tree in a ``TransformedTargetRegressor``
    whose transformer is a ``FunctionTransformer`` with neither ``func`` nor
    ``inverse_func``. That the residual model therefore predicts the log-residual
    directly is checked, never assumed: a non-identity transformer would break the
    SHAP additivity identity silently.
    """
    if not isinstance(transformer, FunctionTransformer):
        raise ValueError(
            f"Expected an identity FunctionTransformer as the target transformer, got "
            f"{type(transformer).__name__}."
        )
    if transformer.func is not None or transformer.inverse_func is not None:
        raise ValueError(
            "The target transformer is not an identity (func/inverse_func are set); the "
            "SHAP decomposition would no longer be additive on the residual scale."
        )


def transformed_feature_matrix(
    pipeline: RobustTrendResidualRegressor, X: pd.DataFrame
) -> np.ndarray:
    """The numeric matrix handed to the core estimator — the SHAP input space."""
    core = pipeline.estimator_
    return np.asarray(core[:-1].transform(X), dtype=float)


def _branch_is_periodic(branch: Any) -> Optional[str]:
    """Return the periodic kind of a branch (``spline``/``sin``/``cos``) or ``None``."""
    steps = getattr(branch, "steps", None)
    if steps is None:
        return None
    for _, step in steps:
        if isinstance(step, PeriodicSplineTransformer):
            return "spline"
        if isinstance(step, SinTransformer):
            return "sin"
        if isinstance(step, CosTransformer):
            return "cos"
    return None


def resolve_transformed_feature_names(
    modeling: ColumnTransformer, engineered: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """Resolve stable names — and their conceptual source — for the SHAP columns.

    ``get_feature_names_out`` on the full pipeline fails for the periodic
    branches, whose inner ``FunctionTransformer``/spline steps expose no names.
    Instead each fitted branch is transformed on ``engineered`` to learn its
    exact output width, and names are generated deterministically: the plain
    numeric and categorical branches keep their one-to-one column names, while a
    cyclic branch produces ``<Column>_spline_00``, ``_sin`` or ``_cos``. Every
    generated name is also mapped to the conceptual feature it expands, so the
    grouped importance can sum an expansion back to its origin. A branch whose
    width does not match a resolvable naming is refused rather than labelled with
    opaque indices.
    """
    if not isinstance(modeling, ColumnTransformer):
        raise ValueError(
            f"SHAP name resolution expects a ColumnTransformer modeling step, got "
            f"{type(modeling).__name__}."
        )
    names: List[str] = []
    sources: List[str] = []
    for branch_name, transformer, columns in modeling.transformers_:
        if branch_name == "remainder" or transformer in ("drop", "passthrough"):
            continue
        column_list = list(columns)
        width = np.asarray(transformer.transform(engineered[column_list])).shape[1]

        periodic = _branch_is_periodic(transformer)
        if periodic is not None:
            if len(column_list) != 1:
                raise ValueError(
                    f"Periodic branch {branch_name!r} spans {len(column_list)} columns; only "
                    "single-column cyclic encodings can be named unambiguously."
                )
            source = column_list[0]
            if periodic == "spline":
                names.extend(f"{source}_spline_{index:02d}" for index in range(width))
            elif periodic == "sin":
                names.extend(f"{source}_sin_{index:02d}" for index in range(width))
            else:
                names.extend(f"{source}_cos_{index:02d}" for index in range(width))
            sources.extend([source] * width)
            continue

        if width != len(column_list):
            raise ValueError(
                f"Branch {branch_name!r} produced {width} columns from {len(column_list)} "
                "inputs; the transformed names cannot be resolved unambiguously."
            )
        names.extend(column_list)
        sources.extend(column_list)

    return names, sources


def select_shap_sample(
    timestamps: pd.Series,
    engineered: pd.DataFrame,
    max_sample: int = SHAP_MAX_SAMPLE,
    random_state: int = SHAP_RANDOM_STATE,
) -> np.ndarray:
    """Deterministically choose the holdout rows explained by SHAP.

    The same rows are used for all three candidates. Selection is stratified by
    season and, inside each season, drawn without replacement by a seeded
    generator, so the sample spans the whole holdout and its four seasons
    without ever consulting the target.
    """
    n = len(timestamps)
    if n <= max_sample:
        return np.arange(n)
    seasons = engineered["Seasons"].astype("object").reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    chosen: List[int] = []
    unique_seasons = sorted(pd.unique(seasons))
    for season in unique_seasons:
        positions = np.where((seasons == season).to_numpy())[0]
        quota = int(round(max_sample * len(positions) / n))
        quota = min(max(quota, 1), len(positions))
        chosen.extend(rng.choice(positions, size=quota, replace=False).tolist())
    chosen = sorted(set(chosen))
    if len(chosen) > max_sample:
        rng2 = np.random.default_rng(random_state + 1)
        chosen = sorted(rng2.choice(chosen, size=max_sample, replace=False).tolist())
    return np.array(chosen, dtype=int)


def _grouped_importance(shap_values: np.ndarray, sources: Sequence[str]) -> pd.DataFrame:
    """Sum SHAP within a conceptual feature per row, then mean the absolute value.

    Summing per observation before taking the absolute value preserves both the
    direction and the additivity inside a group, so a cyclic expansion's several
    columns contribute a single, correctly-signed conceptual attribution.
    """
    sources = list(sources)
    grouped_per_row: Dict[str, np.ndarray] = {}
    for index, source in enumerate(sources):
        column = shap_values[:, index]
        grouped_per_row[source] = grouped_per_row.get(source, 0.0) + column
    rows = [
        {"feature": source, "mean_abs_shap": float(np.mean(np.abs(values)))}
        for source, values in grouped_per_row.items()
    ]
    frame = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    return frame.reset_index(drop=True)


def _detailed_importance(shap_values: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Per transformed-feature mean absolute SHAP value, ranked."""
    frame = pd.DataFrame(
        {
            "feature": list(names),
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return frame.reset_index(drop=True)


def explain_candidate(
    candidate: FrozenCandidate,
    data: FinalEvaluationData,
    engineered: pd.DataFrame,
    sample_positions: np.ndarray,
    config: FinalValidationConfig,
    local_examples: Optional[Mapping[str, int]] = None,
) -> ShapCandidateExplanation:
    """Compute the SHAP decomposition of one candidate on the shared sample.

    The tree-path-dependent :class:`shap.TreeExplainer` is applied to the fitted
    core estimator with ``model_output="raw"``, which is additive to numerical
    precision on the log-residual scale. Two identities are asserted at
    ``rtol/atol`` = 1e-6: the SHAP values plus the expected value reconstruct the
    core estimator's residual prediction, and the trend baseline plus that
    residual prediction reconstruct the pipeline's demand prediction. Any failure
    raises.
    """
    import shap

    pipeline = candidate.pipeline
    core, tree, transformer = _core_estimator(pipeline)
    assert_identity_target_transformer(transformer)

    names, sources = resolve_transformed_feature_names(core.named_steps["modeling"], engineered)

    X_sample = data.X_holdout.iloc[sample_positions]
    matrix = transformed_feature_matrix(pipeline, X_sample)
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"Resolved {len(names)} feature names but the transformed matrix has "
            f"{matrix.shape[1]} columns for {candidate.estimator}."
        )

    explainer = shap.TreeExplainer(tree, model_output="raw")
    shap_values = np.asarray(explainer.shap_values(matrix), dtype=float)
    if shap_values.ndim != 2 or shap_values.shape != matrix.shape:
        raise ValueError(
            f"SHAP values shape {shap_values.shape} does not match the matrix "
            f"{matrix.shape} for {candidate.estimator}."
        )
    expected_value = float(np.asarray(explainer.expected_value, dtype=float).reshape(-1)[0])

    residual_prediction = np.asarray(tree.predict(matrix), dtype=float)
    reconstruction = expected_value + shap_values.sum(axis=1)
    additivity_error = float(np.max(np.abs(reconstruction - residual_prediction)))
    if not np.allclose(
        reconstruction, residual_prediction, rtol=config.shap_rtol, atol=config.shap_atol
    ):
        raise ValueError(
            f"SHAP additivity failed for {candidate.estimator}: max error {additivity_error}."
        )

    log_baseline = np.asarray(pipeline.predict_log_baseline(X_sample), dtype=float)
    demand_reconstruction = np.clip(np.expm1(log_baseline + residual_prediction), 0.0, None)
    pipeline_prediction = np.asarray(pipeline.predict(X_sample), dtype=float)
    reconstruction_error = float(np.max(np.abs(demand_reconstruction - pipeline_prediction)))
    if not np.allclose(
        demand_reconstruction, pipeline_prediction, rtol=config.shap_rtol, atol=config.shap_atol
    ):
        raise ValueError(
            f"Trend+residual reconstruction failed for {candidate.estimator}: "
            f"max error {reconstruction_error}."
        )

    return ShapCandidateExplanation(
        role=candidate.role,
        run_id=candidate.run_id,
        estimator=candidate.estimator,
        sample_positions=np.asarray(sample_positions, dtype=int),
        feature_names=list(names),
        feature_sources=list(sources),
        shap_values=shap_values,
        expected_value=expected_value,
        matrix=matrix,
        detailed_importance=_detailed_importance(shap_values, names),
        grouped_importance=_grouped_importance(shap_values, sources),
        additivity_max_error=additivity_error,
        reconstruction_max_error=reconstruction_error,
        local_examples=dict(local_examples or {}),
    )


def local_example_positions(
    evaluation: CandidateHoldoutEvaluation,
    sample_positions: np.ndarray,
) -> Dict[str, int]:
    """Positions (within the SHAP sample) of a median, a maximum under- and over-estimation.

    Residual convention ``y_true - y_pred``: the largest positive residual is the
    worst under-estimation and the most negative is the worst over-estimation.
    Only rows inside the shared SHAP sample are eligible, so the local
    explanations reuse the already-computed SHAP values. Generic in the
    evaluation, so it serves both the champion and — when the champion is not
    confirmed — the best holdout candidate.
    """
    residuals = evaluation.residuals.to_numpy()
    sample_positions = np.asarray(sample_positions, dtype=int)
    sample_residuals = residuals[sample_positions]
    abs_errors = np.abs(sample_residuals)
    median_local = int(np.argmin(np.abs(abs_errors - np.median(abs_errors))))
    return {
        "median_abs_error": median_local,
        "largest_underestimation": int(np.argmax(sample_residuals)),
        "largest_overestimation": int(np.argmin(sample_residuals)),
    }


# ---------------------------------------------------------------------------
# Persistence and idempotency
# ---------------------------------------------------------------------------


def _tracked_versions() -> Dict[str, str]:
    """Installed versions of the numerically relevant libraries, for the manifest."""
    return dict(package_versions(TRACKED_PACKAGES))


def _predictions_frame(
    data: FinalEvaluationData, evaluations: Sequence[CandidateHoldoutEvaluation]
) -> pd.DataFrame:
    """Timestamp-aligned holdout target, per-candidate predictions and residuals."""
    frame = pd.DataFrame(
        {"timestamp": data.timestamps.to_numpy(), "y_true": data.y_holdout.to_numpy()}
    )
    for evaluation in evaluations:
        frame[f"pred_{evaluation.estimator}"] = evaluation.predictions
        frame[f"residual_{evaluation.estimator}"] = evaluation.residuals.to_numpy()
    return frame


def _build_final_manifest(
    config: FinalValidationConfig,
    manifest: Mapping[str, Any],
    candidates: Sequence[FrozenCandidate],
    data: FinalEvaluationData,
    evaluations: Sequence[CandidateHoldoutEvaluation],
    confirmation: Mapping[str, Any],
    parent_run_id: Optional[str],
    child_run_ids: Mapping[str, str],
) -> Dict[str, Any]:
    """Assemble the complete final-validation manifest written last on success."""
    git_state = describe_git_source_state()
    return {
        "final_validation_code_version": FINAL_VALIDATION_CODE_VERSION,
        "candidate_manifest_fingerprint": manifest_fingerprint(manifest),
        "dataset_fingerprint": data.dataset_fingerprint,
        "holdout_fingerprint": data.holdout_fingerprint,
        "regime_fingerprint": data.regime_fingerprint,
        "environment_name": ENVIRONMENT_NAME,
        "environment_fingerprint": environment_fingerprint(),
        "cv_strategy": CV_STRATEGY_NAME,
        "cv_strategy_version": CV_STRATEGY_VERSION,
        "selection_code_version": CODE_VERSION,
        "regime_policy": REGIME_POLICY_NORMAL_OPERATIONS,
        "git_commit": git_state["git_commit"],
        "git_source_fingerprint": git_state["git_source_fingerprint"],
        "git_source_dirty": git_state["git_source_dirty"],
        "evaluated_at": pd.Timestamp.utcnow().isoformat(),
        "holdout": {
            "start": str(data.holdout_start.date()),
            "end": str(data.holdout_end.date()),
            "n_rows": data.n_holdout_rows,
        },
        "development": {
            "start": str(data.dev_start),
            "end": str(data.dev_end),
            "n_rows": data.n_dev_rows,
        },
        "post_holdout_discarded": {
            "start": str(data.post_holdout_start),
            "end": str(data.post_holdout_end),
            "n_rows": data.n_post_holdout_rows,
        },
        "confirmation_rule": {
            "mae_ratio": config.confirm_mae_ratio,
            "r2_margin": config.confirm_r2_margin,
        },
        "confirmation": dict(confirmation),
        "decision": confirmation["decision"],
        "shap": {
            "max_sample": config.shap_max_sample,
            "random_state": config.shap_random_state,
            "rtol": config.shap_rtol,
            "atol": config.shap_atol,
        },
        "parent_run_id": parent_run_id,
        "child_run_ids": dict(child_run_ids),
        "library_versions": _tracked_versions(),
        "candidates": [
            {
                "role": candidate.role,
                "run_id": candidate.run_id,
                "estimator": candidate.estimator,
                "artifact_path": str(candidate.artifact_path),
                "artifact_sha256": candidate.artifact_sha256,
                "provenance": candidate.provenance,
                "cv_metrics": candidate.cv_metrics,
                "holdout_metrics": next(
                    item.metrics for item in evaluations if item.run_id == candidate.run_id
                ),
            }
            for candidate in candidates
        ],
    }


def _persist_results(
    config: FinalValidationConfig,
    final_manifest: Mapping[str, Any],
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    segmented: Mapping[str, pd.DataFrame],
) -> None:
    """Write every artifact, then the complete manifest last.

    Partial or incompatible pre-existing artifacts are never overwritten
    silently: :func:`_check_reusable` decides earlier whether a complete run may
    be reused, and a mismatch there aborts before this function is reached.
    """
    root = Path(config.runtime_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "segmented_metrics").mkdir(parents=True, exist_ok=True)

    predictions.to_csv(root / "holdout_predictions.csv", index=False)
    comparison.to_csv(root / "holdout_comparison.csv", index=False)
    (root / "general_metrics.json").write_text(
        json.dumps(
            {row["estimator"]: row for row in comparison.to_dict("records")},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    for name, frame in segmented.items():
        frame.to_csv(root / "segmented_metrics" / f"{name}.csv", index=False)

    # Written last, so its presence proves every other artifact is complete.
    Path(config.final_manifest_path).write_text(
        json.dumps(final_manifest, indent=2, default=str), encoding="utf-8"
    )


def _check_reusable(
    config: FinalValidationConfig, manifest: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Return a complete, compatible cached manifest to reuse, or ``None``.

    A previous run is reusable only when its final manifest exists, was produced
    by the same final-validation code version, and matches the candidate
    manifest, dataset, regime and environment fingerprints of the current
    request. An incompatible cached manifest raises rather than being reused or
    overwritten silently.
    """
    path = Path(config.final_manifest_path)
    if not path.exists():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "final_validation_code_version": FINAL_VALIDATION_CODE_VERSION,
        "candidate_manifest_fingerprint": manifest_fingerprint(manifest),
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "regime_fingerprint": EXPECTED_REGIME_FINGERPRINT,
        "environment_fingerprint": environment_fingerprint(),
    }
    mismatches = [
        f"{key}={cached.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if cached.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "A final-validation manifest already exists at "
            f"{path} but is incompatible with the current request:\n  - "
            + "\n  - ".join(mismatches)
            + "\nMove or delete the incompatible artifacts before re-running; they are not "
            "overwritten silently."
        )
    return cached


def _guard_partial_artifacts(config: FinalValidationConfig) -> None:
    """Refuse to run over a partial prior run rather than overwriting it silently.

    The complete manifest is written last, so its absence beside other artifacts
    means a previous run was interrupted. Those leftovers are not overwritten:
    the run aborts and asks for them to be moved or removed, mirroring the
    incompatible-cache refusal in :func:`_check_reusable`.
    """
    root = Path(config.runtime_root)
    if Path(config.final_manifest_path).exists():
        return
    leftovers = [
        name
        for name in ("holdout_predictions.csv", "holdout_comparison.csv", "general_metrics.json")
        if (root / name).exists()
    ]
    if leftovers:
        raise ValueError(
            f"Partial final-validation artifacts were found under {root} without a complete "
            f"manifest ({', '.join(leftovers)}). A previous run did not finish; move or delete "
            "these files before re-running — they are not overwritten silently."
        )


def _results_from_cache(
    config: FinalValidationConfig,
    manifest: Dict[str, Any],
    candidates: List[FrozenCandidate],
    cached: Dict[str, Any],
) -> FinalValidationResults:
    """Rebuild a results object from persisted artifacts, without opening the holdout."""
    root = Path(config.runtime_root)
    predictions = pd.read_csv(root / "holdout_predictions.csv")
    comparison = pd.read_csv(root / "holdout_comparison.csv")
    segmented = {
        path.stem: pd.read_csv(path) for path in sorted((root / "segmented_metrics").glob("*.csv"))
    }
    evaluations: List[CandidateHoldoutEvaluation] = []
    timestamps = pd.to_datetime(predictions["timestamp"])
    y_true = predictions["y_true"].to_numpy(dtype=float)
    for candidate in candidates:
        pred = predictions[f"pred_{candidate.estimator}"].to_numpy(dtype=float)
        residuals = pd.Series(
            predictions[f"residual_{candidate.estimator}"].to_numpy(dtype=float),
            index=timestamps.to_numpy(),
            name="residual",
        )
        evaluations.append(
            CandidateHoldoutEvaluation(
                role=candidate.role,
                run_id=candidate.run_id,
                estimator=candidate.estimator,
                predictions=pred,
                residuals=residuals,
                metrics=holdout_metrics(y_true, pred, candidate.cv_metrics, config.error_quantiles),
                cv_metrics=dict(candidate.cv_metrics),
            )
        )
    return FinalValidationResults(
        config=config,
        manifest=manifest,
        candidates=candidates,
        data=None,
        evaluations=evaluations,
        comparison=comparison,
        confirmation=dict(cached["confirmation"]),
        segmented=segmented,
        predictions=predictions,
        manifest_fingerprint=cached["candidate_manifest_fingerprint"],
        final_manifest_path=Path(config.final_manifest_path),
        parent_run_id=cached.get("parent_run_id"),
        child_run_ids=dict(cached.get("child_run_ids", {})),
        loaded_from_cache=True,
    )


# ---------------------------------------------------------------------------
# MLflow logging (dedicated final-validation experiment)
# ---------------------------------------------------------------------------


def _log_final_validation_to_mlflow(
    config: FinalValidationConfig,
    data: FinalEvaluationData,
    candidates: Sequence[FrozenCandidate],
    evaluations: Sequence[CandidateHoldoutEvaluation],
    confirmation: Mapping[str, Any],
    manifest_fp: str,
) -> Tuple[str, Dict[str, str]]:
    """Record the run in the dedicated final-validation experiment.

    A parent run holds the shared provenance and the decision; one child run per
    candidate carries its source identity and holdout metrics. No model artifact
    is logged — the frozen pipelines already exist and are referenced by run id,
    path and hash. The selection experiment is never touched.
    """
    import mlflow

    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    shared_tags = {
        "final_validation_code_version": FINAL_VALIDATION_CODE_VERSION,
        "candidate_manifest_fingerprint": manifest_fp,
        "dataset_fingerprint": data.dataset_fingerprint,
        "holdout_fingerprint": data.holdout_fingerprint,
        "regime_policy": REGIME_POLICY_NORMAL_OPERATIONS,
        "regime_fingerprint": data.regime_fingerprint,
        "confirmation_decision": confirmation["decision"],
        **describe_environment(),
        **describe_git_source_state(),
    }

    child_run_ids: Dict[str, str] = {}
    with mlflow.start_run(run_name="final_validation") as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tags(shared_tags)
        mlflow.log_metric("champion_holdout_mae", confirmation["champion_holdout_mae"])
        mlflow.log_metric("champion_holdout_r2", confirmation["champion_holdout_r2"])
        mlflow.log_metric("best_holdout_mae", confirmation["best_holdout_mae"])

        evaluation_by_run = {item.run_id: item for item in evaluations}
        for candidate in candidates:
            evaluation = evaluation_by_run[candidate.run_id]
            with mlflow.start_run(
                run_name=f"{candidate.estimator} [{candidate.role}]", nested=True
            ):
                child_run_ids[candidate.run_id] = mlflow.active_run().info.run_id
                mlflow.set_tags(
                    {
                        "source_run_id": candidate.run_id,
                        "source_estimator": candidate.estimator,
                        "candidate_role": candidate.role,
                        "holdout_fingerprint": data.holdout_fingerprint,
                        "candidate_manifest_fingerprint": manifest_fp,
                        "regime_policy": REGIME_POLICY_NORMAL_OPERATIONS,
                        "regime_fingerprint": data.regime_fingerprint,
                        "final_validation_code_version": FINAL_VALIDATION_CODE_VERSION,
                        "confirmation_decision": confirmation["decision"],
                        "source_artifact_sha256": candidate.artifact_sha256,
                        **describe_environment(),
                    }
                )
                for key, value in evaluation.metrics.items():
                    if np.isfinite(value):
                        mlflow.log_metric(key, float(value))
    return parent_run_id, child_run_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class FinalValidationPlan:
    """The pre-registered protocol, assembled before the holdout is opened."""

    config: FinalValidationConfig
    manifest: Dict[str, Any]
    candidates: List[FrozenCandidate]
    manifest_fingerprint: str
    run_records: Dict[str, RunRecord] = field(default_factory=dict)

    @property
    def champion(self) -> FrozenCandidate:
        """The pre-registered champion candidate."""
        return next(candidate for candidate in self.candidates if candidate.is_champion)


def prepare_final_validation(
    config: Optional[FinalValidationConfig] = None,
    audit_runs: bool = True,
    run_fetch: Optional[Callable[[str], RunRecord]] = None,
) -> FinalValidationPlan:
    """Audit the manifest, the original runs and the frozen pipelines — no holdout.

    This is the pre-registration step: it proves the three candidates are exactly
    the ones notebook 04 froze, that their pipelines carry the matching
    provenance stamps, and that their original selection runs are finished and
    verified — all before the holdout is materialised. ``audit_runs`` may be
    disabled where MLflow is not reachable; ``run_fetch`` is injectable for
    testing.
    """
    config = config or FinalValidationConfig()
    require_environment()
    manifest = load_manifest(config.manifest_path)
    audit_manifest(manifest)
    candidates = load_frozen_candidates(manifest)
    run_records: Dict[str, RunRecord] = {}
    if audit_runs:
        run_records = audit_source_runs(manifest, fetch=run_fetch, tracking_uri=config.tracking_uri)
    return FinalValidationPlan(
        config=config,
        manifest=manifest,
        candidates=candidates,
        manifest_fingerprint=manifest_fingerprint(manifest),
        run_records=run_records,
    )


def run_final_validation(
    config: Optional[FinalValidationConfig] = None,
    plan: Optional[FinalValidationPlan] = None,
) -> FinalValidationResults:
    """Materialise the holdout once, evaluate the three candidates and decide.

    Idempotent: when a complete, compatible final manifest already exists the
    persisted results are loaded and the holdout is not reopened, no MLflow run
    is created, and nothing is overwritten. An incompatible cached manifest
    aborts. Otherwise the holdout is opened once, each candidate is predicted a
    single time, the metrics, residuals, segmentation and pre-registered
    confirmation are computed, every artifact is persisted (the complete manifest
    last), and the run is recorded in the dedicated final-validation experiment.
    """
    config = config or FinalValidationConfig()
    if plan is None:
        plan = prepare_final_validation(config, audit_runs=False)
    manifest = plan.manifest
    candidates = plan.candidates

    cached = _check_reusable(config, manifest)
    if cached is not None:
        return _results_from_cache(config, manifest, candidates, cached)
    _guard_partial_artifacts(config)

    data = materialize_final_holdout(config)
    evaluations = [evaluate_candidate(c, data, config.error_quantiles) for c in candidates]
    comparison = comparison_frame(evaluations)
    confirmation = decide_confirmation(
        evaluations, config.confirm_mae_ratio, config.confirm_r2_margin
    )
    engineered = engineered_holdout_frame(plan.champion, data.X_holdout)
    segmented = segmented_metrics(evaluations, engineered, data.y_holdout.to_numpy(), config)
    predictions = _predictions_frame(data, evaluations)

    parent_run_id: Optional[str] = None
    child_run_ids: Dict[str, str] = {}
    if config.log_to_mlflow:
        parent_run_id, child_run_ids = _log_final_validation_to_mlflow(
            config, data, candidates, evaluations, confirmation, plan.manifest_fingerprint
        )

    final_manifest = _build_final_manifest(
        config, manifest, candidates, data, evaluations, confirmation, parent_run_id, child_run_ids
    )
    _persist_results(config, final_manifest, predictions, comparison, segmented)

    return FinalValidationResults(
        config=config,
        manifest=manifest,
        candidates=candidates,
        data=data,
        evaluations=evaluations,
        comparison=comparison,
        confirmation=confirmation,
        segmented=segmented,
        predictions=predictions,
        manifest_fingerprint=plan.manifest_fingerprint,
        final_manifest_path=Path(config.final_manifest_path),
        parent_run_id=parent_run_id,
        child_run_ids=child_run_ids,
        loaded_from_cache=False,
    )


def run_shap_validation(
    results: FinalValidationResults,
    config: Optional[FinalValidationConfig] = None,
) -> List[ShapCandidateExplanation]:
    """Explain the three residual models on one shared, deterministic holdout sample.

    The holdout data already opened by :func:`run_final_validation` is reused; if
    the results were loaded from cache the holdout is materialised once here. The
    same sample rows explain all three candidates, and the additivity and
    trend+residual reconstruction identities are asserted for each. Local
    explanations of a median, a worst under- and a worst over-estimation are
    attached to the champion; when the champion is not confirmed they are also
    attached to the candidate that performed best on the holdout, so a
    comparative local explanation is available for it too.
    """
    config = config or results.config
    data = results.data or materialize_final_holdout(config)
    engineered = engineered_holdout_frame(
        next(c for c in results.candidates if c.is_champion), data.X_holdout
    )
    sample_positions = select_shap_sample(
        data.timestamps, engineered, config.shap_max_sample, config.shap_random_state
    )

    evaluation_by_run = {item.run_id: item for item in results.evaluations}
    local_by_run: Dict[str, Dict[str, int]] = {
        results.champion_evaluation.run_id: local_example_positions(
            results.champion_evaluation, sample_positions
        )
    }
    if results.confirmation.get("decision") == NOT_CONFIRMED:
        best_run = results.confirmation["best_holdout_run_id"]
        local_by_run[best_run] = local_example_positions(
            evaluation_by_run[best_run], sample_positions
        )

    explanations: List[ShapCandidateExplanation] = [
        explain_candidate(
            candidate,
            data,
            engineered,
            sample_positions,
            config,
            local_examples=local_by_run.get(candidate.run_id),
        )
        for candidate in results.candidates
    ]
    results.shap = explanations
    return explanations
