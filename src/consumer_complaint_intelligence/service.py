"""Application-neutral service wrapper around the predictor contract."""

from typing import Sequence

from .contracts import PredictionBatch, Predictor


class PredictionService:
    """Validate request shape and delegate inference to a predictor.

    The class is deliberately unaware of Flask, HTTP, serialization libraries,
    and model tracking. An API adapter can translate its result later.

    Args:
        predictor: Object implementing the framework-neutral ``Predictor``
            protocol.
    """

    def __init__(self, predictor: Predictor) -> None:
        """Initialize the service with a predictor implementation.

        Args:
            predictor: Predictor used for inference.

        Raises:
            TypeError: If ``predictor`` does not expose a callable ``predict``.
        """

        if not callable(getattr(predictor, "predict", None)):
            raise TypeError("predictor must expose a callable predict method")
        self._predictor = predictor

    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        """Return predictions while preserving input order.

        Args:
            texts: Ordered sequence of complaint narratives.

        Returns:
            Prediction batch returned by the configured predictor.

        Raises:
            ValueError: If ``texts`` is empty or contains non-string values.
        """

        values = tuple(texts)
        if not values:
            raise ValueError("texts must contain at least one item")
        if not all(isinstance(text, str) for text in values):
            raise ValueError("texts must contain only strings")
        return self._predictor.predict(values)
