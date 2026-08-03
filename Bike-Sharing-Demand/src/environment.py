"""Interpreter guard and environment provenance for Bike-Sharing-Demand v4.

Every execution of this project — tests, notebooks, Optuna studies, MLflow
runs — must happen under the ``Bike-Sharing`` conda environment. The guard is
not a style preference: running under a different environment silently changes
what a run *means*. The concrete failure that motivated this module was an
execution under an unrelated environment that had another project installed in
editable mode declaring the same top-level package name ``src``; MLflow's
dependency inference then wrote that foreign project into the logged model's
``requirements.txt``, producing an artifact that could never be reinstalled.

The check reads the interpreter itself, never ``CONDA_DEFAULT_ENV``. That
variable is set by ``conda activate`` and is simply absent when the interpreter
is invoked by absolute path, when a subprocess is spawned by ``multiprocessing``
or ``nbconvert``, or when an IDE launches a kernel directly — exactly the
situations where a wrong environment is most likely and least visible.
``sys.prefix`` and ``sys.executable`` are properties of the running process and
cannot be out of sync with it.

The module also owns the environment's *provenance*: the versions of the
packages that decide numerical results, the fingerprint that summarises them,
and the pinned requirement list a logged model must declare.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

# The one environment this project may run under.
ENVIRONMENT_NAME = "Bike-Sharing"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
ENVIRONMENT_FILE = _PROJECT_ROOT / "environment.yml"

# Distributions whose version can change a fitted model's numbers. They are the
# fingerprint's ingredients and are logged with every run. Names are *pip
# distribution* names (``scikit-learn``, not ``sklearn``), because that is what
# both ``importlib.metadata`` and a requirements file speak.
TRACKED_PACKAGES: Tuple[str, ...] = (
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
)

# The minimal set needed to *load and predict* with a persisted pipeline. It is
# deliberately narrower than the environment: a model consumer needs the
# serialization stack, the array/frame layer, and the estimator libraries the
# pipeline's steps come from — not Optuna, not matplotlib, not the notebook
# toolchain. Declaring it explicitly is what stops MLflow from inferring the
# whole environment, which is how a foreign project ends up in the artifact.
MODEL_REQUIREMENT_PACKAGES: Tuple[str, ...] = (
    "mlflow",
    "cloudpickle",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "category-encoders",
    "feature-engine",
)

# Marker for a distribution that is not installed. Kept as a value rather than
# raising, so a fingerprint can still be computed and the gap is visible in the
# logged tags instead of aborting a run at its very last step.
MISSING_VERSION = "not-installed"

# Only files that can change the fitted model belong to the source-state gate.
# Notebook outputs, tests and runtime artifacts may legitimately change during
# an execution; the implementation and its declared environment may not.
GIT_SOURCE_PATHS: Tuple[str, ...] = ("src", "environment.yml", ".flake8")


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def _executable_is_inside_environment(
    executable: str, environment_name: str = ENVIRONMENT_NAME
) -> bool:
    """Whether ``executable`` lives under an ``envs/<environment_name>`` directory.

    Matching the pair ``envs``/``<name>`` rather than just the name anywhere in
    the path avoids accepting an unrelated directory that happens to be called
    ``Bike-Sharing`` — a checkout, a scratch folder — as if it were the conda
    environment. Comparison is case-insensitive because Windows paths are.
    """
    parts = [part.lower() for part in Path(executable).parts]
    target = environment_name.lower()
    return any(
        part == "envs" and parts[index + 1] == target for index, part in enumerate(parts[:-1])
    )


def check_environment(
    executable: Optional[str] = None,
    prefix: Optional[str] = None,
    environment_name: str = ENVIRONMENT_NAME,
) -> List[str]:
    """Return the list of environment problems — empty when the interpreter is right.

    Both properties are checked because each catches something the other does
    not: ``sys.prefix`` identifies the environment the interpreter belongs to,
    while ``sys.executable`` proves it was launched from that environment's own
    directory rather than being, say, a venv layered on top of it.

    ``executable`` and ``prefix`` are injectable so the failure path can be
    tested without a second interpreter.
    """
    executable = sys.executable if executable is None else executable
    prefix = sys.prefix if prefix is None else prefix

    problems: List[str] = []
    actual_prefix_name = Path(prefix).name
    if actual_prefix_name != environment_name:
        problems.append(
            f"sys.prefix is '{prefix}' (environment '{actual_prefix_name}'), "
            f"expected an environment named '{environment_name}'"
        )
    if not _executable_is_inside_environment(executable, environment_name):
        problems.append(
            f"sys.executable is '{executable}', which is not inside an "
            f"'envs/{environment_name}' directory"
        )
    return problems


def require_environment(
    executable: Optional[str] = None,
    prefix: Optional[str] = None,
    environment_name: str = ENVIRONMENT_NAME,
) -> Dict[str, str]:
    """Raise unless the running interpreter is the project's environment.

    Called at the head of every entry point that would otherwise commit the
    wrong environment to something persistent — loading the dataset, creating
    or resuming an Optuna study, opening an MLflow run. Failing there costs
    seconds; failing after four hours of search costs the search.

    Returns
    -------
    dict
        The environment description, so a caller can log what it validated.
    """
    problems = check_environment(executable, prefix, environment_name)
    if problems:
        raise RuntimeError(
            f"Wrong Python environment for Bike-Sharing-Demand v4. Expected the "
            f"'{environment_name}' conda environment; found:\n  - "
            + "\n  - ".join(problems)
            + "\nRun everything — tests, notebooks, Optuna, MLflow — with "
            f"the '{environment_name}' interpreter, by absolute path if necessary."
        )
    return describe_environment()


# ---------------------------------------------------------------------------
# Git source provenance
# ---------------------------------------------------------------------------


def _run_git(arguments: Sequence[str], project_root: Path = _PROJECT_ROOT) -> str:
    """Run a read-only Git query rooted at this project and return stdout."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Git provenance query failed: {detail or arguments}")
    return completed.stdout.strip()


def source_tree_fingerprint(
    project_root: Path = _PROJECT_ROOT,
    source_paths: Sequence[str] = GIT_SOURCE_PATHS,
) -> str:
    """Hash the bytes and relative names of every model-producing source file."""
    root = Path(project_root)
    files: List[Path] = []
    for relative in source_paths:
        path = root / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                child
                for child in path.rglob("*")
                if child.is_file()
                and "__pycache__" not in child.parts
                and child.suffix not in {".pyc", ".pyo"}
            )

    hasher = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def describe_git_source_state(
    project_root: Path = _PROJECT_ROOT,
    source_paths: Sequence[str] = GIT_SOURCE_PATHS,
) -> Dict[str, str]:
    """Describe the committed revision and the model-producing source state.

    The repository that contains this project is broader than the project
    directory, so status is deliberately restricted to ``source_paths``.
    Runtime artifacts and unrelated sibling projects cannot make a clean
    modeling checkout look dirty, while any modified or untracked file under
    ``src`` is detected.
    """
    commit = _run_git(("rev-parse", "--short", "HEAD"), project_root)
    status = _run_git(
        ("status", "--porcelain", "--untracked-files=all", "--", *source_paths),
        project_root,
    )
    dirty = bool(status)
    status_hash = hashlib.sha256(status.encode("utf-8")).hexdigest()[:16] if dirty else "clean"
    return {
        "git_commit": commit or "unknown",
        "git_source_dirty": str(dirty).lower(),
        "git_source_status_hash": status_hash,
        "git_source_fingerprint": source_tree_fingerprint(project_root, source_paths),
    }


def require_clean_git_source(
    project_root: Path = _PROJECT_ROOT,
    source_paths: Sequence[str] = GIT_SOURCE_PATHS,
) -> Dict[str, str]:
    """Refuse a definitive run whose model-producing source is not committed."""
    state = describe_git_source_state(project_root, source_paths)
    if state["git_source_dirty"] == "true":
        status = _run_git(
            ("status", "--short", "--untracked-files=all", "--", *source_paths),
            project_root,
        )
        raise RuntimeError(
            "A full model-selection run requires committed model-producing source. "
            "The following paths differ from HEAD:\n"
            f"{status}\nCommit the intended src/environment configuration before "
            "starting the definitive search."
        )
    return state


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def package_version(distribution: str) -> str:
    """Installed version of ``distribution``, or :data:`MISSING_VERSION`."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return MISSING_VERSION


def package_versions(distributions: Sequence[str] = TRACKED_PACKAGES) -> Dict[str, str]:
    """Installed versions of ``distributions``, in the order given."""
    return {name: package_version(name) for name in distributions}


def python_version() -> str:
    """The interpreter's version, without the build/compiler suffix."""
    return ".".join(str(part) for part in sys.version_info[:3])


def environment_fingerprint(
    distributions: Sequence[str] = TRACKED_PACKAGES,
    environment_name: str = ENVIRONMENT_NAME,
) -> str:
    """Hash the environment's identity: its name, its Python and its libraries.

    Two runs sharing a fingerprint were produced by the same numerical stack.
    The fingerprint is part of the Optuna study name and of the fail-closed
    MLflow selection filter, so a run produced under a different stack can
    neither contribute trials to a study nor win a champion query — which is
    precisely what must not happen after an execution under the wrong
    environment.
    """
    payload = {
        "environment_name": environment_name,
        "python_version": python_version(),
        "packages": package_versions(distributions),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def describe_environment(
    distributions: Sequence[str] = TRACKED_PACKAGES,
    environment_name: str = ENVIRONMENT_NAME,
) -> Dict[str, str]:
    """Flat, string-valued description of the environment, ready for MLflow tags."""
    description = {
        "environment_name": environment_name,
        "python_executable": sys.executable,
        "python_version": python_version(),
        "environment_fingerprint": environment_fingerprint(distributions, environment_name),
    }
    description.update(
        {f"version_{name}": version for name, version in package_versions(distributions).items()}
    )
    return description


# ---------------------------------------------------------------------------
# Pinned requirements for a logged model
# ---------------------------------------------------------------------------


def pinned_versions(environment_file: Path = ENVIRONMENT_FILE) -> Dict[str, str]:
    """Read the ``pip:`` pins declared in ``environment.yml``.

    The file is the project's declaration of what the environment *is*; the
    installed metadata only reports what it currently happens to be. Reading
    the pins from the declaration means a logged model advertises the versions
    the project stands behind, and any drift between the two is a detectable
    condition rather than a silent one.
    """
    if not Path(environment_file).exists():
        return {}
    document = yaml.safe_load(Path(environment_file).read_text(encoding="utf-8")) or {}
    pins: Dict[str, str] = {}
    for dependency in document.get("dependencies", []) or []:
        if not isinstance(dependency, dict):
            continue
        for entry in dependency.get("pip", []) or []:
            if not isinstance(entry, str) or "==" not in entry:
                continue
            name, _, version = entry.partition("==")
            pins[name.strip()] = version.strip()
    return pins


def version_drift(
    distributions: Sequence[str] = TRACKED_PACKAGES,
    environment_file: Path = ENVIRONMENT_FILE,
) -> Dict[str, Tuple[str, str]]:
    """Packages whose installed version differs from the ``environment.yml`` pin.

    Returns a mapping ``name -> (pinned, installed)``. An empty mapping means
    the environment matches its own declaration.
    """
    pins = pinned_versions(environment_file)
    drift: Dict[str, Tuple[str, str]] = {}
    for name in distributions:
        pinned = pins.get(name)
        installed = package_version(name)
        if pinned is not None and pinned != installed:
            drift[name] = (pinned, installed)
    return drift


def model_pip_requirements(
    distributions: Sequence[str] = MODEL_REQUIREMENT_PACKAGES,
    environment_file: Path = ENVIRONMENT_FILE,
) -> List[str]:
    """Explicit, fully pinned ``pip_requirements`` for ``mlflow.sklearn.log_model``.

    Versions come from ``environment.yml`` where the project pins them and from
    the installed metadata otherwise (``cloudpickle`` is pulled in by MLflow and
    is not pinned by the project, but must still be pinned in the artifact,
    since it is the serialization format the pickle was written with).

    Passing this list is what stops MLflow from inferring requirements by
    scanning the environment. Inference is not merely noisy: it walks the
    installed distributions and attributes imported top-level modules to them,
    so any other project installed in editable mode that exposes a package
    named ``src`` is attributed the project's own modules and lands in the
    artifact as a requirement that cannot be resolved anywhere.
    """
    pins = pinned_versions(environment_file)
    requirements: List[str] = []
    for name in distributions:
        version = pins.get(name) or package_version(name)
        if version == MISSING_VERSION:
            raise RuntimeError(
                f"Cannot pin '{name}' for the model requirements: it is neither "
                f"pinned in {Path(environment_file).name} nor installed in the "
                "running environment."
            )
        requirements.append(f"{name}=={version}")
    return requirements


def model_code_paths() -> List[str]:
    """Local source directories a logged model must carry to be loadable.

    The persisted pipeline holds references to ``src.feature_engineering``,
    ``src.periodic_features`` and ``src.modeling_pipeline``, so unpickling it
    anywhere requires the package itself, not just its third-party
    dependencies. MLflow copies these paths under the model's ``code/``
    directory and prepends that directory to ``sys.path`` at load time, which
    makes ``import src`` resolve from the artifact rather than from wherever
    the consumer happens to be standing.
    """
    return [str(SRC_DIR)]
