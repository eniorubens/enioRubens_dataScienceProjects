"""
opt_binary_clf_pipe
==============
Reusable modeling library for binary classification projects.

Public API
----------
.. code-block:: python

    from opt_binary_clf_pipe import (
        train_all_models,       # full experiment loop
        run_baseline_estimator, # single baseline run
        Optimizer,              # Optuna pipeline optimizer
        CsvModelStore,          # CSV + gzip-pickle backend
        MlflowModelStore,       # MLflow experiment backend
        ModelStore,             # abstract interface (for custom backends)
    )
"""
from __future__ import annotations

__version__ = "1.0.0"
__author__ = "opt_binary_clf_pipe contributors"

from .callbacks import EarlyStoppingCallback
from .estimators import define_estimators, get_estimator
from .optimizer import Optimizer, apply_threshold_decision
from .persistence import CsvModelStore, MlflowModelStore, ModelStore
from .pipeline_builder import (
    build_baseline_preprocessor,
    build_full_pipeline,
    build_num_transformer,
    build_preprocessor,
)
from .scoring import (
    build_business_weight_config,
    compute_business_score,
    get_classification_scoring,
    get_objective_scoring,
    get_positive_scores,
    predict_with_threshold,
)
from .trainer import run_baseline_estimator, run_baseline_logistic_regression, train_all_models

__all__ = [
    # Core workflow
    "train_all_models",
    "run_baseline_estimator",
    "run_baseline_logistic_regression",
    # Optimizer
    "Optimizer",
    "apply_threshold_decision",
    # Storage
    "ModelStore",
    "CsvModelStore",
    "MlflowModelStore",
    # Pipeline construction
    "build_baseline_preprocessor",
    "build_full_pipeline",
    "build_num_transformer",
    "build_preprocessor",
    # Scoring & metrics
    "get_classification_scoring",
    "get_objective_scoring",
    "compute_business_score",
    "build_business_weight_config",
    "get_positive_scores",
    "predict_with_threshold",
    # Estimator registry
    "define_estimators",
    "get_estimator",
    # Callbacks
    "EarlyStoppingCallback",
]
