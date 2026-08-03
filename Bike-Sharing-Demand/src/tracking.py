"""MLflow tracking utilities for Bike Sharing Demand v3.

§7 spec:
  - URI: file:./mlruns
  - experiment: bike_sharing_demand_v3
  - 1 run per estimator (XGB, HistGB, LGBM)
  - Log params, per-fold metrics, LOSO metrics, tags (measurement_era, git_commit)
  - Artifacts: metric_dataframe.csv (complement, do not replace)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import mlflow
import pandas as pd
import yaml

from src.environment import (
    describe_environment,
    describe_git_source_state,
    model_code_paths,
    model_pip_requirements,
    require_environment,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "bike_sharing_demand_v3"
EXPERIMENT_NAME_V4 = "bike_sharing_demand_v4_model_selection"
MLRUNS_DIR = Path(__file__).resolve().parent.parent / "mlruns"
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "dataset"


def params_hash(params: Mapping[str, Any]) -> str:
    """Provenance hash of a parameter set, stable across an MLflow round-trip.

    Values are stringified before hashing because MLflow stores every param as
    a string: hashing the typed dict at write time and the retrieved dict at
    read time would otherwise never agree, and this hash exists precisely so
    that a frozen pipeline can be proven to belong to the run it was selected
    from.
    """
    payload = {str(key): str(value) for key, value in dict(params).items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def spec_hash(spec: Optional[Mapping[str, Any]]) -> str:
    """Provenance hash of a pipeline spec, stable across an MLflow round-trip.

    Same stringify-then-hash contract as :func:`params_hash`, applied to the
    dynamic-pipeline choices (representation, encoder, scaler, normalizer,
    selector, target transform). Two pipelines with identical hyperparameters
    but different representations hash differently, which is what makes the
    pair (params, spec) a usable identity for an artifact.
    """
    return params_hash(spec or {})


# Attribute the provenance stamp is written under. The trailing underscore
# follows the sklearn convention for state that only exists after fitting.
PROVENANCE_ATTRIBUTE = "v4_provenance_"


def stamp_pipeline_provenance(
    pipeline: Any,
    source_run_id: str,
    best_params: Mapping[str, Any],
    pipeline_spec: Optional[Mapping[str, Any]],
    code_version: str,
    dataset_fingerprint: str,
) -> Dict[str, str]:
    """Write the run's identity onto the fitted pipeline object itself.

    Freezing a candidate has to prove that the object being pickled is the
    object the selected run measured. Recomputing a hash of the run's own
    metadata cannot prove that — it only shows the metadata is self-consistent,
    and would accept any pipeline handed over under that ``run_id``. Carrying
    the identity *on the artifact* closes the gap: the stamp travels with the
    object through pickling and through MLflow, and an unstamped or
    differently-stamped pipeline is rejected.

    The stamp is applied inside the MLflow run, before the model is logged, so
    the copy stored as an artifact carries it too.
    """
    provenance = {
        "source_run_id": str(source_run_id),
        "best_params_hash": params_hash(best_params),
        "pipeline_spec_hash": spec_hash(pipeline_spec),
        "code_version": str(code_version),
        "dataset_fingerprint": str(dataset_fingerprint),
    }
    try:
        setattr(pipeline, PROVENANCE_ATTRIBUTE, provenance)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"Cannot stamp provenance on a {type(pipeline).__name__}: it rejects "
            "attribute assignment, so its identity could never be verified at "
            "freeze time."
        ) from exc
    return provenance


def pipeline_provenance(pipeline: Any) -> Optional[Dict[str, str]]:
    """Read back the provenance stamp of ``pipeline``, or ``None`` if unstamped."""
    provenance = getattr(pipeline, PROVENANCE_ATTRIBUTE, None)
    return dict(provenance) if isinstance(provenance, Mapping) else None


# Distributions that must never appear in a logged model's requirements. The
# name is not hypothetical: an execution under an environment holding another
# project installed in editable mode — one that also declares a top-level
# package called ``src`` — had MLflow's dependency inference attribute this
# project's own modules to it, so the artifact demanded a package that exists
# nowhere and could not be reinstalled anywhere.
FORBIDDEN_MODEL_REQUIREMENTS: tuple = ("customer-segmentation-nba",)


def verify_model_artifact(
    run_id: str,
    artifact_path: str = "model",
    forbidden: tuple = FORBIDDEN_MODEL_REQUIREMENTS,
) -> Dict[str, Any]:
    """Check that a logged model is self-contained and free of foreign requirements.

    Four properties are asserted against the artifact as it was actually
    written, not against the arguments that were passed:

    * ``requirements.txt`` and ``conda.yaml`` name no forbidden distribution,
      which is what proves the dependency list was declared rather than
      inferred from the environment;
    * ``MLmodel`` declares a code path instead of ``code: null``, so a consumer
      is told where the project's own modules live;
    * ``code/src`` is present, so those modules actually travel with the model.

    Returns a report rather than raising, because how a failure is handled
    depends on the run mode: fatal for a definitive candidate, recorded for a
    smoke run.
    """
    local_dir = Path(
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    )
    problems: List[str] = []

    for name in ("requirements.txt", "conda.yaml"):
        path = local_dir / name
        if not path.exists():
            problems.append(f"{name} is missing from the logged model")
            continue
        text = path.read_text(encoding="utf-8").lower()
        problems.extend(
            f"{name} declares the foreign requirement '{token}'"
            for token in forbidden
            if token.lower() in text
        )

    mlmodel_path = local_dir / "MLmodel"
    if not mlmodel_path.exists():
        problems.append("MLmodel is missing from the logged model")
    else:
        document = yaml.safe_load(mlmodel_path.read_text(encoding="utf-8")) or {}
        flavor = (document.get("flavors") or {}).get("sklearn") or {}
        if not flavor.get("code"):
            problems.append("MLmodel carries no code path (flavors.sklearn.code is null)")

    if not (local_dir / "code" / "src").is_dir():
        problems.append("code/src is not present in the artifact")

    return {"artifact_dir": str(local_dir), "problems": problems, "verified": not problems}


def _git_commit() -> str:
    """Return the current HEAD commit hash (short), or 'unknown'."""
    try:
        return describe_git_source_state()["git_commit"]
    except Exception:
        return "unknown"


def setup_mlflow(
    tracking_uri: Optional[str] = None, experiment_name: str = EXPERIMENT_NAME
) -> None:
    """Configure MLflow URI and activate (creating if needed) ``experiment_name``.

    ``experiment_name`` used to be ignored here: the module-level
    ``EXPERIMENT_NAME`` (v3) was always activated, so a caller configuring a
    custom experiment (e.g. ``MLflowTracker.setup_experiment()``) had its runs
    silently land in the v3 experiment instead. Passing the name through and
    actually activating it fixes that.
    """
    uri = tracking_uri or f"file:{MLRUNS_DIR}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking: %s  experiment: %s", uri, experiment_name)


def log_estimator_run(
    estimator_name: str,
    params: Dict[str, Any],
    inner_cv_metrics: Dict[str, float],
    holdout_metrics: Dict[str, float],
    loso_df: Optional[pd.DataFrame] = None,
    artifacts_dir: str = "./dataset",
    measurement_era: str = "v3",
    run_name: Optional[str] = None,
) -> str:
    """Log one estimator run to MLflow.

    Parameters
    ----------
    estimator_name:
        e.g. 'XGBRegressor'. Used for the estimator tag; always preserved.
    params:
        Optuna best params dict.
    inner_cv_metrics:
        Dict with keys like 'test_r2_mean', 'test_rmse_mean' etc.
    holdout_metrics:
        Dict with keys like 'holdout_r2', 'holdout_rmse' etc.
    loso_df:
        Optional DataFrame from evaluate_loso_generalization (cols: Season, MAE, RMSE, R2).
    artifacts_dir:
        Directory containing metric_dataframe.csv.
    measurement_era:
        Free-form tag, e.g. 'v3'.
    run_name:
        Display name in the MLflow UI. Defaults to estimator_name when None.
        Use to label experiment batches (e.g. '1st run', '2nd run post-tuning').

    Returns
    -------
    str
        The MLflow run_id for this run.
    """
    with mlflow.start_run(run_name=run_name or estimator_name) as run:
        mlflow.set_tags(
            {
                "estimator": estimator_name,
                "measurement_era": measurement_era,
                "git_commit": _git_commit(),
            }
        )

        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception as exc:
                logger.warning("Could not log param %s: %s", k, exc)

        for k, v in inner_cv_metrics.items():
            try:
                mlflow.log_metric(f"cv_{k}", float(v))
            except Exception as exc:
                logger.warning("Could not log metric %s: %s", k, exc)

        for k, v in holdout_metrics.items():
            try:
                mlflow.log_metric(f"holdout_{k}", float(v))
            except Exception as exc:
                logger.warning("Could not log metric %s: %s", k, exc)

        if loso_df is not None and not loso_df.empty:
            for _, row in loso_df.iterrows():
                season = str(row.get("Season", "unknown")).lower()
                for metric in ["MAE", "RMSE", "R2"]:
                    if metric in row:
                        try:
                            mlflow.log_metric(f"loso_{season}_{metric.lower()}", float(row[metric]))
                        except Exception as exc:
                            logger.warning(
                                "Could not log LOSO metric %s/%s: %s", season, metric, exc
                            )

        csv_path = Path(artifacts_dir) / "metric_dataframe.csv"
        if csv_path.exists():
            mlflow.log_artifact(str(csv_path), artifact_path="metrics")

        run_id = run.info.run_id

    logger.info("MLflow run logged: %s  run_id=%s", estimator_name, run_id)
    return run_id


def log_loso_run(
    estimator_name: str,
    loso_df: pd.DataFrame,
    measurement_era: str = "v3",
) -> str:
    """Log a dedicated LOSO-only MLflow run for an estimator.

    Complements log_estimator_run; does not replace it.
    """
    run_name = f"{estimator_name}_LOSO"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "estimator": estimator_name,
                "evaluation": "LOSO",
                "measurement_era": measurement_era,
                "git_commit": _git_commit(),
            }
        )

        summary = loso_df.groupby("Model", observed=False)[["MAE", "RMSE", "R2"]].mean()
        for model_label, row in summary.iterrows():
            for metric in ["MAE", "RMSE", "R2"]:
                mlflow.log_metric(f"loso_mean_{metric.lower()}", float(row[metric]))

        for _, row in loso_df.iterrows():
            season = str(row.get("Season", "unknown")).lower()
            for metric in ["MAE", "RMSE", "R2"]:
                if metric in row:
                    mlflow.log_metric(f"loso_{season}_{metric.lower()}", float(row[metric]))

        run_id = run.info.run_id

    logger.info("MLflow LOSO run logged: %s  run_id=%s", run_name, run_id)
    return run_id


# ---------------------------------------------------------------------------
# v5 temporal model-selection logging (ForwardMeteorologicalYearSplit)
# ---------------------------------------------------------------------------


def log_temporal_model_run(
    estimator_name: str,
    params: Dict[str, Any],
    cv_metrics: Dict[str, float],
    fold_metrics: Optional[pd.DataFrame] = None,
    seasonal_metrics: Optional[pd.DataFrame] = None,
    extreme_metrics: Optional[pd.DataFrame] = None,
    trials_dataframe: Optional[pd.DataFrame] = None,
    feature_manifest: Optional[Dict[str, Any]] = None,
    cv_manifest: Optional[Dict[str, Any]] = None,
    model_object: Any = None,
    input_example: Optional[pd.DataFrame] = None,
    dataset_fingerprint: str = "unknown",
    cv_strategy: str = "ForwardMeteorologicalYearSplit",
    cv_strategy_version: str = "ForwardMeteorologicalYearSplit_v3_normal_operations",
    code_version: str = "unknown",
    run_mode: str = "full",
    pipeline_spec: Optional[Dict[str, Any]] = None,
    n_features_selected: Optional[int] = None,
    trials_planned: Optional[int] = None,
    trials_completed: Optional[int] = None,
    termination_reason: Optional[str] = None,
    best_iterations_by_fold: Optional[Any] = None,
    final_n_estimators: Optional[int] = None,
    iteration_aggregation: Optional[str] = None,
    cap_hits_by_fold: Optional[Any] = None,
    n_folds_cap_hit: Optional[int] = None,
    n_folds_with_budget: Optional[int] = None,
    iteration_ceiling: Optional[int] = None,
    systematic_truncation: Optional[bool] = None,
    measurement_era: str = "v4",
    run_name: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
) -> str:
    """Log one v4 model-selection run — CV-only, never holdout metrics.

    ``cv_metrics`` keys are stored as-is (e.g. ``cv_mae_mean`` -> metric
    ``cv_mae_mean``, matching the champion-selection query in
    ``src.temporal_optimizer.select_champion_and_challengers``). Diagnostics
    (fold/seasonal/extreme CSVs, the Optuna trials export, feature/CV
    manifests, and — when ``model_object`` is given — the fitted pipeline via
    ``mlflow.sklearn.log_model`` with a signature/input example and an HTML
    repr) are all logged as artifacts. No holdout metric is ever written by
    this function.

    Parameters
    ----------
    pipeline_spec:
        The dynamic-pipeline choices behind this run (``modeler_name``,
        encoder, scaler, normalizer, selector, target transform). Written as
        tags so a run can be filtered by representation strategy, and
        recorded again in the hand-off manifest.
    run_mode, cv_strategy_version, code_version, dataset_fingerprint:
        The four provenance tags champion selection matches on exactly. A run
        missing any of them can never be selected — see
        ``select_champion_and_challengers``.
    trials_planned, trials_completed, termination_reason:
        Study budget, what the study actually holds, and which limit ended it
        (``"trial_limit"`` or ``"study_timeout"``) — so a search cut short by
        the clock is visible from the run alone.
    best_iterations_by_fold, final_n_estimators, iteration_aggregation:
        For the boosting estimators, the budget each fold discovered, the
        single budget the persisted artifact was refit with, and the rule that
        collapsed one into the other.
    cap_hits_by_fold, n_folds_cap_hit, n_folds_with_budget, iteration_ceiling,
    systematic_truncation:
        Whether early stopping was ended by the data or by its own ceiling,
        per fold and in aggregate — so a budget that merely records where the
        limit was cannot be mistaken for a converged one.
    """
    from sklearn.utils import estimator_html_repr

    # Nothing persistent has been created yet: an MLflow run opened under the
    # wrong interpreter would record that interpreter's package versions as if
    # they were the project's.
    environment = require_environment()

    if artifacts_dir is None:
        artifacts_dir = DEFAULT_ARTIFACTS_DIR

    with mlflow.start_run(run_name=run_name or estimator_name) as run:
        run_id = run.info.run_id
        git_state = describe_git_source_state()
        tags = {
            "estimator": estimator_name,
            "measurement_era": measurement_era,
            "cv_strategy": cv_strategy,
            "cv_strategy_version": cv_strategy_version,
            "code_version": code_version,
            "run_mode": run_mode,
            "dataset_fingerprint": dataset_fingerprint,
            "params_hash": params_hash(params),
            "pipeline_spec_hash": spec_hash(pipeline_spec),
            "final_holdout": "2023-12-01/2024-11-30",
            **git_state,
            # Flipped to "true" only once the model really reaches the artifact
            # store. Selection reads it, so a run without a retrievable model
            # can never become a definitive champion.
            "model_logged": "false",
            # Flipped to "true" only once the stored artifact has been read back
            # and found self-contained — see verify_model_artifact.
            "model_artifact_verified": "false",
        }
        # Which interpreter and which library versions produced these numbers.
        # Two of these tags are part of the fail-closed selection filter, so a
        # run made under a different stack cannot win a champion query.
        tags.update(environment)
        if termination_reason is not None:
            tags["termination_reason"] = str(termination_reason)
        if best_iterations_by_fold is not None:
            tags["best_iterations_by_fold"] = json.dumps(list(best_iterations_by_fold))
        if final_n_estimators is not None:
            tags["final_n_estimators"] = str(int(final_n_estimators))
        if iteration_aggregation is not None:
            tags["iteration_aggregation"] = str(iteration_aggregation)
        if cap_hits_by_fold is not None:
            tags["boosting_cap_hits_by_fold"] = json.dumps(
                [bool(value) for value in cap_hits_by_fold]
            )
        if iteration_ceiling is not None:
            tags["iteration_ceiling"] = str(int(iteration_ceiling))
        if systematic_truncation is not None:
            tags["systematic_truncation"] = str(bool(systematic_truncation)).lower()
        if pipeline_spec:
            tags.update({str(k): str(v) for k, v in pipeline_spec.items() if v is not None})
        mlflow.set_tags(tags)

        if systematic_truncation:
            logger.warning(
                "[%s] run %s records systematic boosting truncation (%s of %s folds at the "
                "%s-iteration ceiling); it cannot be frozen as a definitive candidate.",
                estimator_name,
                run_id,
                n_folds_cap_hit,
                n_folds_with_budget,
                iteration_ceiling,
            )

        for name, value in [
            ("trials_planned", trials_planned),
            ("trials_completed", trials_completed),
            ("n_features_selected", n_features_selected),
            ("final_n_estimators", final_n_estimators),
            ("n_folds_cap_hit", n_folds_cap_hit),
            ("n_folds_with_budget", n_folds_with_budget),
        ]:
            if value is not None:
                mlflow.log_metric(name, float(value))

        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception as exc:
                logger.warning("Could not log param %s: %s", k, exc)

        for k, v in cv_metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception as exc:
                logger.warning("Could not log metric %s: %s", k, exc)

        tmp_dir = Path(artifacts_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in [
            ("fold_metrics", fold_metrics),
            ("seasonal_metrics", seasonal_metrics),
            ("extreme_metrics", extreme_metrics),
            ("optuna_trials", trials_dataframe),
        ]:
            if frame is not None and not frame.empty:
                path = tmp_dir / f"{estimator_name}_{name}.csv"
                frame.to_csv(path, index=False)
                mlflow.log_artifact(str(path), artifact_path="diagnostics")

        if feature_manifest is not None:
            path = tmp_dir / f"{estimator_name}_feature_manifest.json"
            path.write_text(json.dumps(feature_manifest, indent=2, default=str), encoding="utf-8")
            mlflow.log_artifact(str(path), artifact_path="manifests")

        if cv_manifest is not None:
            path = tmp_dir / f"{estimator_name}_cv_manifest.json"
            path.write_text(json.dumps(cv_manifest, indent=2, default=str), encoding="utf-8")
            mlflow.log_artifact(str(path), artifact_path="manifests")

        if model_object is not None:
            stamp_pipeline_provenance(
                model_object,
                source_run_id=run_id,
                best_params=params,
                pipeline_spec=pipeline_spec,
                code_version=code_version,
                dataset_fingerprint=dataset_fingerprint,
            )
            try:
                signature = None
                if input_example is not None:
                    signature = mlflow.models.infer_signature(
                        input_example, model_object.predict(input_example)
                    )
                mlflow.sklearn.log_model(
                    model_object,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    # The pipeline's steps come from this project's own modules,
                    # so the artifact has to carry them; without this MLflow
                    # stores a pickle nothing outside the repository can open.
                    code_paths=model_code_paths(),
                    # Declared, never inferred. Inference walks the installed
                    # distributions and attributes imported top-level modules to
                    # them, which is how an unrelated editable install exposing
                    # a package named ``src`` ends up as a requirement of this
                    # model.
                    pip_requirements=model_pip_requirements(),
                )
                mlflow.set_tag("model_logged", "true")
            except Exception as exc:
                mlflow.set_tag("model_logged", "false")
                # A run whose model never reached the artifact store cannot be
                # frozen as a definitive candidate: notebook 05 would be handed
                # a manifest pointing at a model URI that does not resolve. In
                # full mode the failure is therefore raised, which marks the
                # run FAILED and — since selection only considers FINISHED runs
                # — removes it from contention. A smoke run only proves the
                # machinery and produces provisional artifacts, so there the
                # failure is recorded and the run continues.
                if run_mode == "full":
                    raise RuntimeError(
                        f"mlflow.sklearn.log_model failed for {estimator_name} "
                        f"(run {run_id}): {exc}. A full run whose model is not stored "
                        "cannot be frozen as a definitive candidate."
                    ) from exc
                logger.warning("Could not log model artifact for %s: %s", estimator_name, exc)
            else:
                verification = verify_model_artifact(run_id)
                mlflow.set_tag("model_artifact_verified", str(verification["verified"]).lower())
                if verification["problems"]:
                    message = (
                        f"The model logged for {estimator_name} (run {run_id}) is not "
                        "self-contained: " + "; ".join(verification["problems"]) + "."
                    )
                    if run_mode == "full":
                        raise RuntimeError(
                            message + " A definitive candidate must be loadable outside "
                            "this repository, so the run is failed rather than kept."
                        )
                    logger.warning(message)

            try:
                # estimator_html_repr() embeds Unicode collapse/expand arrows
                # for nested pipelines — Path.write_text()'s default encoding
                # is the OS locale (cp1252 on Windows), which cannot encode
                # them; without encoding="utf-8" this raised inside the same
                # try/except as log_model() above, so a failure here used to
                # be misreported as "could not log model artifact" even
                # though the model itself had already been saved successfully.
                repr_path = tmp_dir / f"{estimator_name}_pipeline_repr.html"
                repr_path.write_text(estimator_html_repr(model_object), encoding="utf-8")
                mlflow.log_artifact(str(repr_path), artifact_path="manifests")
            except Exception as exc:
                logger.warning("Could not log pipeline HTML for %s: %s", estimator_name, exc)

    logger.info("MLflow v4 run logged: %s  run_id=%s", estimator_name, run_id)
    return run_id
