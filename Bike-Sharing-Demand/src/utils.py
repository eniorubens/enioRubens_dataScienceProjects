"""Logging setup, seeds, paths, and shared configuration utilities."""

import logging
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MULTILANG_CACHE_DIR = PROJECT_ROOT / ".multilang_cache"

RANDOM_STATE: int = 42

LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"


def setup_logging(
    level: int = logging.INFO,
    fmt: str = LOG_FORMAT,
) -> logging.Logger:
    """Configure root logger and return the project-level logger."""
    logging.basicConfig(level=level, format=fmt)
    logger = logging.getLogger("bike_sharing_v3")
    logger.setLevel(level)
    return logger


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Set NumPy random seed for reproducibility."""
    np.random.seed(seed)


def public_path(value: object) -> object:
    """Return a portable path for user-facing reports.

    Paths inside the project are rendered relative to its root. Absolute paths
    outside it are reduced to their final component so saved notebook outputs
    do not expose a contributor's local directory layout.
    """
    if value is None:
        return None
    text = str(value)
    if not text or text.startswith(("runs:/", "models:/")):
        return value
    path = Path(text)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


logger = setup_logging()
