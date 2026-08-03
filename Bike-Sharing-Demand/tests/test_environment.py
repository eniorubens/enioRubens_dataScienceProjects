"""Interpreter guard, environment provenance and model portability.

Three separate guarantees are asserted here, in increasing order of what they
cost to get wrong:

* the project refuses to run under any interpreter other than the
  ``Bike-Sharing`` environment, and refuses *before* touching anything
  persistent;
* every run records which interpreter and which library versions produced it,
  and that record participates in fail-closed selection;
* a logged model is genuinely self-contained — it declares pinned requirements
  rather than inferred ones, it carries this project's own modules, and it
  loads and predicts from outside the repository.

The last one is proved by actually doing it, in a subprocess started outside
the project directory with the project absent from ``PYTHONPATH``. Asserting
on the arguments passed to ``log_model`` would only restate the call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import src.model_selection_workflow as workflow
import src.environment as environment
import src.temporal_optimizer as topt
import src.tracking as tracking
from src.environment import (
    ENVIRONMENT_NAME,
    MODEL_REQUIREMENT_PACKAGES,
    TRACKED_PACKAGES,
    check_environment,
    describe_git_source_state,
    describe_environment,
    environment_fingerprint,
    model_code_paths,
    model_pip_requirements,
    package_versions,
    pinned_versions,
    require_clean_git_source,
    require_environment,
    source_tree_fingerprint,
    version_drift,
)
from src.mlflow_integration import ExperimentConfigV4, MLflowTracker
from src.periodic_features import SinTransformer
from src.tracking import log_temporal_model_run, verify_model_artifact

FOREIGN_EXECUTABLE = r"C:\Users\someone\miniforge3\envs\Churn-ML\python.exe"
FOREIGN_PREFIX = r"C:\Users\someone\miniforge3\envs\Churn-ML"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


class TestTheRunningInterpreterIsTheProjectEnvironment:
    def test_the_suite_itself_runs_under_bike_sharing(self):
        assert check_environment() == []
        assert Path(sys.prefix).name == ENVIRONMENT_NAME
        assert "envs" in Path(sys.executable).parts

    def test_require_environment_returns_the_description_it_validated(self):
        description = require_environment()
        assert description["environment_name"] == ENVIRONMENT_NAME
        assert description["python_executable"] == sys.executable


class TestAForeignInterpreterIsRefused:
    def test_both_properties_are_reported_as_problems(self):
        problems = check_environment(executable=FOREIGN_EXECUTABLE, prefix=FOREIGN_PREFIX)
        assert len(problems) == 2
        assert any("sys.prefix" in problem for problem in problems)
        assert any("sys.executable" in problem for problem in problems)

    def test_require_environment_raises(self):
        with pytest.raises(RuntimeError, match="Wrong Python environment"):
            require_environment(executable=FOREIGN_EXECUTABLE, prefix=FOREIGN_PREFIX)

    def test_the_right_prefix_with_a_foreign_executable_still_fails(self):
        """Each property catches something the other cannot; neither alone is
        sufficient."""
        problems = check_environment(executable=FOREIGN_EXECUTABLE, prefix=sys.prefix)
        assert len(problems) == 1
        assert "sys.executable" in problems[0]

    def test_a_directory_merely_named_bike_sharing_is_not_the_environment(self):
        """Matching the name anywhere in the path would accept a checkout or a
        scratch folder; the ``envs/<name>`` pair is what identifies a conda
        environment."""
        problems = check_environment(
            executable=r"D:\Projects\Bike-Sharing\python.exe",
            prefix=r"D:\Projects\Bike-Sharing",
        )
        assert any("sys.executable" in problem for problem in problems)


class TestTheGuardDoesNotReadCondaDefaultEnv:
    """``CONDA_DEFAULT_ENV`` is set by ``conda activate`` and is simply absent
    when the interpreter is launched by absolute path, from ``multiprocessing``
    or from nbconvert — which is exactly when a wrong environment is least
    visible."""

    def test_the_check_passes_with_the_variable_removed(self, monkeypatch):
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
        assert check_environment() == []

    def test_the_check_fails_with_the_variable_claiming_the_right_environment(self, monkeypatch):
        monkeypatch.setenv("CONDA_DEFAULT_ENV", ENVIRONMENT_NAME)
        assert check_environment(executable=FOREIGN_EXECUTABLE, prefix=FOREIGN_PREFIX)


class TestTheGuardRunsBeforeAnythingPersistent:
    """Ordering is the whole point: a wrong environment must cost seconds, not
    a dataset fingerprint, an Optuna study or an MLflow run."""

    @staticmethod
    def _refuse(*args, **kwargs):
        raise RuntimeError("Wrong Python environment (simulated)")

    def test_no_data_is_read_before_the_check(self, monkeypatch):
        def explode():
            raise AssertionError("read_data() ran before the environment was verified")

        monkeypatch.setattr(workflow, "require_environment", self._refuse)
        monkeypatch.setattr(workflow, "read_data", explode)
        with pytest.raises(RuntimeError, match="simulated"):
            workflow.prepare_development_data(workflow.ModelSelectionConfig())

    def test_no_optuna_study_is_created_before_the_check(self, monkeypatch):
        monkeypatch.setattr(topt, "require_environment", self._refuse)
        with pytest.raises(RuntimeError, match="simulated"):
            topt.TemporalRegressionOptimizer(
                "Ridge", pd.DataFrame({"x": [1.0]}), pd.Series([1.0]), cv=None
            )

    def test_no_mlflow_run_is_opened_before_the_check(self, monkeypatch):
        monkeypatch.setattr(tracking, "require_environment", self._refuse)
        with pytest.raises(RuntimeError, match="simulated"):
            log_temporal_model_run("Ridge", params={}, cv_metrics={"cv_mae_mean": 1.0})
        assert mlflow.active_run() is None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestEnvironmentProvenance:
    def test_every_package_the_task_names_is_recorded(self):
        description = describe_environment()
        for name in (
            "scikit-learn",
            "pandas",
            "numpy",
            "mlflow",
            "optuna",
            "xgboost",
            "lightgbm",
            "catboost",
            "category-encoders",
            "feature-engine",
        ):
            assert name in TRACKED_PACKAGES
            assert description[f"version_{name}"] not in ("", "not-installed")

    def test_the_description_names_the_interpreter_and_its_version(self):
        description = describe_environment()
        assert description["python_executable"] == sys.executable
        assert description["python_version"].startswith("3.")
        assert len(description["environment_fingerprint"]) == 16

    def test_the_fingerprint_is_stable_within_one_environment(self):
        assert environment_fingerprint() == environment_fingerprint()

    def test_the_fingerprint_changes_with_the_package_set(self):
        """Two environments differing in one library must not share a
        fingerprint, or a study built under one could absorb trials from the
        other."""
        assert environment_fingerprint(("mlflow",)) != environment_fingerprint(("numpy",))
        assert environment_fingerprint() != environment_fingerprint(
            TRACKED_PACKAGES, environment_name="Churn-ML"
        )


class TestGitSourceProvenance:
    @staticmethod
    def _fake_git(status):
        def run(arguments, project_root=PROJECT_ROOT):
            if arguments[0] == "rev-parse":
                return "abc1234"
            return status

        return run

    def test_a_clean_source_tree_is_described_without_a_diff_hash(self, monkeypatch):
        monkeypatch.setattr(environment, "_run_git", self._fake_git(""))
        state = describe_git_source_state()
        assert state == {
            "git_commit": "abc1234",
            "git_source_dirty": "false",
            "git_source_status_hash": "clean",
            "git_source_fingerprint": source_tree_fingerprint(),
        }

    def test_an_uncommitted_source_file_is_detected_and_refused(self, monkeypatch):
        monkeypatch.setattr(
            environment,
            "_run_git",
            self._fake_git(" M src/temporal_optimizer.py"),
        )
        state = describe_git_source_state()
        assert state["git_source_dirty"] == "true"
        assert len(state["git_source_status_hash"]) == 16
        with pytest.raises(RuntimeError, match="requires committed"):
            require_clean_git_source()

    def test_the_source_fingerprint_changes_with_file_content(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        module = src / "model.py"
        module.write_text("VALUE = 1\n", encoding="utf-8")
        first = source_tree_fingerprint(tmp_path, ("src",))
        module.write_text("VALUE = 2\n", encoding="utf-8")
        second = source_tree_fingerprint(tmp_path, ("src",))
        assert first != second

    def test_the_environment_matches_its_own_declaration(self):
        assert version_drift() == {}, "environment.yml and the installed packages disagree"

    def test_the_declaration_pins_every_tracked_package(self):
        pins = pinned_versions()
        installed = package_versions()
        for name in TRACKED_PACKAGES:
            assert pins.get(name) == installed[name]


class TestModelRequirements:
    def test_every_requirement_is_pinned_to_an_exact_version(self):
        for requirement in model_pip_requirements():
            assert "==" in requirement
            assert requirement.split("==")[1].strip()

    def test_the_minimum_load_and_predict_set_is_covered(self):
        declared = {line.split("==")[0] for line in model_pip_requirements()}
        assert declared == set(MODEL_REQUIREMENT_PACKAGES)
        for name in ("mlflow", "cloudpickle", "numpy", "pandas", "scipy", "scikit-learn"):
            assert name in declared

    def test_the_versions_come_from_the_environment_file(self):
        pins = pinned_versions()
        for requirement in model_pip_requirements():
            name, _, version = requirement.partition("==")
            if name in pins:
                assert version == pins[name]

    def test_the_code_path_is_the_projects_own_src_directory(self):
        paths = [Path(path) for path in model_code_paths()]
        assert paths == [PROJECT_ROOT / "src"]
        assert (paths[0] / "modeling_pipeline.py").exists()


# ---------------------------------------------------------------------------
# What a logged model actually contains
# ---------------------------------------------------------------------------


def _portable_pipeline() -> Pipeline:
    """A fitted pipeline whose unpickling genuinely requires ``src``.

    ``SinTransformer`` lives in :mod:`src.periodic_features`, so a consumer
    without the project's modules cannot open this pickle — which is what makes
    the code-path assertions meaningful rather than decorative.
    """
    pipeline = Pipeline(
        [
            ("hour_sin", SinTransformer(24)),
            ("regressor", TransformedTargetRegressor(regressor=Ridge())),
        ]
    )
    return pipeline.fit(_example_frame(), pd.Series([10.0, 20.0, 30.0, 40.0]))


def _example_frame() -> pd.DataFrame:
    return pd.DataFrame({"Hour": [0.0, 6.0, 12.0, 18.0]})


@pytest.fixture
def logged_model(tmp_path):
    """Log one v4 run with a real model and return ``(tracking_uri, run_id)``."""
    tracking_uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
    MLflowTracker(ExperimentConfigV4(tracking_uri=tracking_uri)).setup_experiment()
    run_id = log_temporal_model_run(
        estimator_name="Ridge",
        params={"alpha": 1.0},
        cv_metrics={"cv_mae_mean": 1.0},
        model_object=_portable_pipeline(),
        input_example=_example_frame(),
        dataset_fingerprint="fp_portability",
        run_mode="smoke",
        artifacts_dir=tmp_path / "artifacts",
    )
    return tracking_uri, run_id


class TestTheLoggedArtifactIsSelfContained:
    """The four properties the packaging fix has to produce, read back from the
    artifact as written rather than from the arguments that were passed."""

    def test_requirements_txt_declares_no_foreign_distribution(self, logged_model):
        _, run_id = logged_model
        artifact = Path(verify_model_artifact(run_id)["artifact_dir"])
        requirements = (artifact / "requirements.txt").read_text(encoding="utf-8")
        assert "customer-segmentation-nba" not in requirements
        for name in MODEL_REQUIREMENT_PACKAGES:
            assert f"{name}==" in requirements

    def test_conda_yaml_declares_no_foreign_distribution(self, logged_model):
        _, run_id = logged_model
        artifact = Path(verify_model_artifact(run_id)["artifact_dir"])
        assert "customer-segmentation-nba" not in (artifact / "conda.yaml").read_text("utf-8")

    def test_mlmodel_carries_a_code_path_rather_than_null(self, logged_model):
        _, run_id = logged_model
        artifact = Path(verify_model_artifact(run_id)["artifact_dir"])
        mlmodel = (artifact / "MLmodel").read_text(encoding="utf-8")
        assert "code: code" in mlmodel
        assert "code: null" not in mlmodel

    def test_the_projects_modules_travel_with_the_model(self, logged_model):
        _, run_id = logged_model
        artifact = Path(verify_model_artifact(run_id)["artifact_dir"])
        assert (artifact / "code" / "src" / "periodic_features.py").exists()
        assert (artifact / "code" / "src" / "modeling_pipeline.py").exists()

    def test_the_run_records_that_the_model_was_logged_and_verified(self, logged_model):
        _, run_id = logged_model
        tags = mlflow.get_run(run_id).data.tags
        assert tags["model_logged"] == "true"
        assert tags["model_artifact_verified"] == "true"

    def test_the_run_records_the_environment_that_produced_it(self, logged_model):
        _, run_id = logged_model
        tags = mlflow.get_run(run_id).data.tags
        assert tags["environment_name"] == ENVIRONMENT_NAME
        assert tags["environment_fingerprint"] == environment_fingerprint()
        assert tags["python_executable"] == sys.executable
        assert tags["version_scikit-learn"] == package_versions()["scikit-learn"]

    def test_verification_reports_a_foreign_requirement(self, logged_model):
        _, run_id = logged_model
        artifact = Path(verify_model_artifact(run_id)["artifact_dir"])
        (artifact / "requirements.txt").write_text("customer-segmentation-nba==2.0.0\n", "utf-8")
        report = verify_model_artifact(run_id)
        assert report["verified"] is False
        assert any("customer-segmentation-nba" in problem for problem in report["problems"])


class TestArtifactVerificationIsFailClosedInFullMode:
    @staticmethod
    def _broken_artifact(*args, **kwargs):
        return {
            "artifact_dir": "",
            "problems": ["MLmodel carries no code path (flavors.sklearn.code is null)"],
            "verified": False,
        }

    def test_a_full_run_fails_when_the_artifact_is_not_self_contained(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tracking, "verify_model_artifact", self._broken_artifact)
        MLflowTracker(
            ExperimentConfigV4(tracking_uri=f"file:{(tmp_path / 'mlruns').as_posix()}")
        ).setup_experiment()
        with pytest.raises(RuntimeError, match="not self-contained"):
            log_temporal_model_run(
                estimator_name="Ridge",
                params={"alpha": 1.0},
                cv_metrics={"cv_mae_mean": 1.0},
                model_object=_portable_pipeline(),
                input_example=_example_frame(),
                run_mode="full",
                artifacts_dir=tmp_path / "artifacts",
            )

    def test_a_smoke_run_records_the_problem_and_continues(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tracking, "verify_model_artifact", self._broken_artifact)
        MLflowTracker(
            ExperimentConfigV4(tracking_uri=f"file:{(tmp_path / 'mlruns').as_posix()}")
        ).setup_experiment()
        run_id = log_temporal_model_run(
            estimator_name="Ridge",
            params={"alpha": 1.0},
            cv_metrics={"cv_mae_mean": 1.0},
            model_object=_portable_pipeline(),
            input_example=_example_frame(),
            run_mode="smoke",
            artifacts_dir=tmp_path / "artifacts",
        )
        tags = mlflow.get_run(run_id).data.tags
        assert tags["model_logged"] == "true"
        assert tags["model_artifact_verified"] == "false"


# ---------------------------------------------------------------------------
# Portability, proved by doing it
# ---------------------------------------------------------------------------


PORTABILITY_SCRIPT = textwrap.dedent(
    """
    import importlib
    import json
    import sys

    import numpy as np

    tracking_uri, model_uri = sys.argv[1], sys.argv[2]
    result = {}

    # 1. Before loading, the project's package must be unreachable: this
    #    subprocess stands outside the repository and PYTHONPATH is empty, so
    #    anything that works afterwards works because of the artifact.
    try:
        importlib.import_module("src")
        result["src_importable_before"] = True
    except ImportError:
        result["src_importable_before"] = False

    import mlflow
    from mlflow.models import Model

    mlflow.set_tracking_uri(tracking_uri)
    local_dir = mlflow.artifacts.download_artifacts(model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    # 2. The code path shipped inside the model is what makes it importable.
    src_module = importlib.import_module("src.periodic_features")
    result["src_importable_after"] = True
    result["src_came_from_the_artifact"] = str(local_dir) in str(src_module.__file__)

    example = Model.load(local_dir).load_input_example(local_dir)
    result["example_rows"] = int(len(example))

    predictions = np.asarray(model.predict(example))
    result["prediction_shape"] = list(predictions.shape)
    result["all_finite"] = bool(np.isfinite(predictions).all())

    print(json.dumps(result))
    """
)


class TestTheModelLoadsAndPredictsOutsideTheRepository:
    """The real portability check: a fresh interpreter, started outside the
    project directory, with ``PYTHONPATH`` cleared, loading the model purely
    from its MLflow URI."""

    @pytest.fixture(scope="class")
    def portability(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("portability")
        tracking_uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
        MLflowTracker(ExperimentConfigV4(tracking_uri=tracking_uri)).setup_experiment()
        run_id = log_temporal_model_run(
            estimator_name="Ridge",
            params={"alpha": 1.0},
            cv_metrics={"cv_mae_mean": 1.0},
            model_object=_portable_pipeline(),
            input_example=_example_frame(),
            dataset_fingerprint="fp_portability",
            run_mode="smoke",
            artifacts_dir=tmp_path / "artifacts",
        )

        workdir = tmp_path / "elsewhere"
        workdir.mkdir()
        script = workdir / "load_and_predict.py"
        script.write_text(PORTABILITY_SCRIPT, encoding="utf-8")

        environ = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        completed = subprocess.run(
            [sys.executable, str(script), tracking_uri, f"runs:/{run_id}/model"],
            cwd=str(workdir),
            env=environ,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def test_the_subprocess_ran_under_the_project_environment(self):
        assert Path(sys.executable).parent.name == ENVIRONMENT_NAME

    def test_src_is_unreachable_before_the_model_is_loaded(self, portability):
        assert portability["src_importable_before"] is False

    def test_loading_the_model_makes_src_importable_from_the_artifact(self, portability):
        assert portability["src_importable_after"] is True
        assert portability["src_came_from_the_artifact"] is True

    def test_the_input_example_travels_with_the_model(self, portability):
        assert portability["example_rows"] == len(_example_frame())

    def test_prediction_has_the_expected_shape_and_is_finite(self, portability):
        assert portability["prediction_shape"] == [len(_example_frame())]
        assert portability["all_finite"] is True


def test_no_source_file_references_the_wrong_environment():
    """The project must not name another environment anywhere in its code,
    which is how a hard-coded interpreter path outlives the decision to stop
    using it."""
    offenders = []
    for path in sorted((PROJECT_ROOT / "src").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "Churn-ML" in text or "churn-ml" in text:
            offenders.append(path.name)
    assert offenders == []
