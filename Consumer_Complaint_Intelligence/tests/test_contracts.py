"""Tests for framework-neutral prediction and tracking contracts."""

import unittest

from consumer_complaint_intelligence.contracts import (
    ArtifactManifest,
    Prediction,
    PredictionBatch,
)
from consumer_complaint_intelligence.service import PredictionService
from consumer_complaint_intelligence.tracking import (
    NullTracker,
    create_tracker,
)


class _FakePredictor:
    """Provide deterministic predictions for service tests."""

    def predict(self, texts: tuple[str, ...]) -> PredictionBatch:
        """Return one fixed label per input text."""

        return PredictionBatch(
            tuple(Prediction(label="test", score=1.0) for _ in texts)
        )


class ContractTests(unittest.TestCase):
    """Verify serializable contracts and application independence."""

    def test_prediction_and_manifest_are_serializable(self) -> None:
        """Return stable dictionaries without Flask or MLflow objects."""

        prediction = Prediction(label="Debt collection", score=0.8)
        batch = PredictionBatch((prediction,))
        manifest = ArtifactManifest(artifact_id="demo", task="routing")
        self.assertEqual(batch.to_dict()["count"], 1)
        self.assertEqual(manifest.to_dict()["artifact_id"], "demo")

    def test_service_delegates_and_validates(self) -> None:
        """Preserve order and reject invalid request payloads."""

        service = PredictionService(_FakePredictor())
        result = service.predict(["one", "two"])
        self.assertEqual(result.to_dict()["count"], 2)
        with self.assertRaises(ValueError):
            service.predict([])
        with self.assertRaises(ValueError):
            service.predict(["one", 2])  # type: ignore[list-item]

    def test_noop_tracker_is_default_and_callable(self) -> None:
        """Keep optional tracking absent from the default dependency graph."""

        tracker = create_tracker()
        self.assertIsInstance(tracker, NullTracker)
        tracker.log_params({"sample_rows": 10})
        tracker.log_metrics({"f1": 0.5})
        tracker.close()

    def test_unknown_tracker_is_rejected(self) -> None:
        """Reject accidental silent fallback for unsupported tracker names."""

        with self.assertRaises(ValueError):
            create_tracker("unknown")
