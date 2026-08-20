"""Framework-neutral prediction and artifact contracts."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Prediction:
    """Represent one model prediction in an API-friendly shape.

    Args:
        label: Predicted class label.
        score: Optional numeric model score whose meaning is in ``metadata``.
        model_version: Optional version exposed by the artifact registry.
        metadata: JSON-friendly auxiliary values.
    """

    label: str
    score: float | None = None
    model_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the prediction into a JSON-friendly dictionary.

        Returns:
            Dictionary suitable for a Flask, CLI, or batch adapter.
        """

        return {
            "label": self.label,
            "score": self.score,
            "model_version": self.model_version,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictionBatch:
    """Represent an ordered batch of predictions."""

    predictions: tuple[Prediction, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the batch into a JSON-friendly dictionary.

        Returns:
            Dictionary with prediction records and their count.
        """

        return {
            "count": len(self.predictions),
            "predictions": [item.to_dict() for item in self.predictions],
        }


class Predictor(Protocol):
    """Define the model boundary consumed by future applications."""

    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        """Predict labels for an ordered sequence of texts.

        Args:
            texts: Input narratives in request order.

        Returns:
            Predictions aligned with ``texts``.

        Raises:
            ValueError: If input validation fails.
        """

        ...


@dataclass(frozen=True)
class ArtifactManifest:
    """Describe a persisted estimator without owning its serving framework.

    Args:
        artifact_id: Stable identifier for the persisted artifact.
        task: Logical prediction task.
        framework: Training framework, for example ``scikit-learn``.
        model_path: Project-relative or externally managed model path.
        schema_version: Contract version for request and response fields.
        created_at: ISO-8601 creation timestamp.
        metadata: JSON-friendly training and provenance metadata.
    """

    artifact_id: str
    task: str
    framework: str = "scikit-learn"
    model_path: str = ""
    schema_version: str = "1"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest into a JSON-friendly dictionary.

        Returns:
            Dictionary suitable for artifact metadata or registry storage.
        """

        return {
            "artifact_id": self.artifact_id,
            "task": self.task,
            "framework": self.framework,
            "model_path": self.model_path,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }
