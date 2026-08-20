"""Optional experiment tracking with a dependency-free no-op default."""

from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    """Define the small tracking surface used by orchestration code."""

    def log_params(self, params: dict[str, Any]) -> None:
        """Record experiment parameters."""

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Record experiment metrics."""

    def log_artifact(self, path: str | Path) -> None:
        """Record one artifact path."""

    def close(self) -> None:
        """Close the tracking run."""


class NullTracker:
    """Implement the tracker contract without external dependencies."""

    def log_params(self, params: dict[str, Any]) -> None:
        """Ignore experiment parameters by design."""

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Ignore experiment metrics by design."""

    def log_artifact(self, path: str | Path) -> None:
        """Ignore an artifact path by design."""

    def close(self) -> None:
        """Close a no-op run."""


class MlflowTracker:
    """Adapt MLflow lazily without making it a package dependency.

    Args:
        experiment_name: MLflow experiment name.

    Raises:
        RuntimeError: If MLflow is not installed or the run cannot start.
    """

    def __init__(self, experiment_name: str) -> None:
        """Start an MLflow run only when this adapter is explicitly created.

        Args:
            experiment_name: MLflow experiment name.

        Raises:
            RuntimeError: If MLflow cannot be imported or started.
        """

        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "MLflow is optional; install it before selecting this tracker"
            ) from error
        try:
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self._run = mlflow.start_run()
        except Exception as error:
            raise RuntimeError("Could not start the MLflow run") from error

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameter values in the active MLflow run."""

        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log numeric metrics in the active MLflow run."""

        self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: str | Path) -> None:
        """Log one local artifact in the active MLflow run."""

        self._mlflow.log_artifact(str(path))

    def close(self) -> None:
        """End the active MLflow run."""

        self._mlflow.end_run()


def create_tracker(kind: str = "noop", **kwargs: Any) -> Tracker:
    """Create a no-op or explicitly requested optional tracker.

    Args:
        kind: ``noop`` or ``mlflow``.
        **kwargs: Adapter-specific arguments, such as ``experiment_name``.

    Returns:
        Tracker implementation selected by ``kind``.

    Raises:
        ValueError: If ``kind`` is unsupported.
        RuntimeError: If the MLflow adapter is selected but unavailable.
    """

    normalized = kind.strip().lower()
    if normalized == "noop":
        return NullTracker()
    if normalized == "mlflow":
        return MlflowTracker(**kwargs)
    raise ValueError(f"Unsupported tracker kind: {kind}")
