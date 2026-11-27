"""Optional bridge to the external model-optimization toolkit.

The churn project can be imported and tested without the sibling toolkit. The
notebooks can also be opened without it; backend operations receive an
actionable error if the optional dependency is absent.
"""

from __future__ import annotations

from typing import Any

_BACKEND_NAMES = {
    "train_all_models",
    "run_baseline_estimator",
    "run_baseline_logistic_regression",
    "Optimizer",
    "CsvModelStore",
    "MlflowModelStore",
    "apply_threshold_decision",
    "get_positive_scores",
    "predict_with_threshold",
    "compute_business_score",
    "build_business_weight_config",
    "get_classification_scoring",
}

try:  # Optional by design: CI for this project must not need the toolkit.
    from opt_binary_clf_pipe import (  # type: ignore[import-not-found]
        CsvModelStore,
        MlflowModelStore,
        Optimizer,
        apply_threshold_decision,
        build_business_weight_config,
        compute_business_score,
        get_classification_scoring,
        get_positive_scores,
        predict_with_threshold,
        run_baseline_estimator,
        run_baseline_logistic_regression,
        train_all_models,
    )
except ImportError as exc:  # pragma: no cover - exercised by dependency checks
    _BACKEND_ERROR = exc
else:
    _BACKEND_ERROR = None


def require_backend() -> None:
    """Raise a clear installation error when the optional backend is missing."""
    if _BACKEND_ERROR is not None:
        raise ModuleNotFoundError(
            "The notebook modeling backend is optional and is not installed. "
            "Install the sibling ds_toolkit/opt_binary_clf_pipe package or "
            "run the project tests, which do not require it."
        ) from _BACKEND_ERROR


def _missing_backend(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    require_backend()
    raise AssertionError("unreachable")


if _BACKEND_ERROR is not None:
    for _name in _BACKEND_NAMES:
        globals()[_name] = _missing_backend


def __getattr__(name: str) -> Any:
    if name in _BACKEND_NAMES:
        return _missing_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_BACKEND_NAMES | {"require_backend"})
