"""Data loading and I/O utilities — v4 (Seoul public bicycle, Sep 2015 – Dec 2024).

The v4 dataset (``Seoul_public_bicycle.csv``, Kaggle ``lnoahl/seoul-bike-sharing-dataset``,
KOGL Type 1 — see ``dataset/DATASET_README.md`` for attribution) covers ~9.3 years of
hourly Ttareungyi rentals with KMA/ASOS weather. ``read_data`` adapts its schema to the
v3 column names so the whole inherited toolkit — ``build_preprocessing_pipeline``,
``RegressionOptimizer._NUMERICAL_FEATURES``/``_CATEGORICAL_FEATURES``, the tests'
synthetic fixtures — keeps working unchanged.
"""

import gzip
import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "dataset" / "Seoul_public_bicycle.csv"
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# v4 → v3 schema adapter. Unit conversions:
#  - visibility arrives in meters; v3's "Visibility (10m)" is in tens of meters (÷10).
#  - all other mapped columns share units with v3 (README vs. v3 dataset docs).
_COLUMN_MAP = {
    "count": "Rented Bike Count",
    "temperature": "Temperature(C)",
    "humidity": "Humidity(%)",
    "windspeed": "Wind speed (m/s)",
    "dew_point": "Dew point temperature(C)",
    "solar_radiation": "Solar Radiation (MJ/m2)",
    "precipitation": "Rainfall(mm)",
    "snowfall": "Snowfall (cm)",
    # New-in-v4 weather signals, named in the same style; not yet in the
    # optimizer's _NUMERICAL_FEATURES — promoting them is a modeling decision.
    "sunshine": "Sunshine (hr)",
    "cloud_cover": "Cloud Cover (oktas)",
    "ground_temp": "Ground Temp(C)",
}

# KMA/ASOS convention: precipitation-type gauges are left blank when there is
# nothing to measure, so NaN in these columns means 0, not "unknown".
_ZERO_WHEN_MISSING = ["Rainfall(mm)", "Snowfall (cm)", "Sunshine (hr)", "Solar Radiation (MJ/m2)"]

_SEASON_BY_MONTH = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


def read_data(path: str | Path = _DEFAULT_DATASET) -> pd.DataFrame:
    """Read the 2015-2024 Seoul public bicycle dataset in v3-compatible shape.

    Adapter steps (see ``_COLUMN_MAP``):

    - ``datetime`` is split into ``DateTime`` (midnight-normalised date) +
      ``Hour``, matching the raw v3 layout that ``DateHourJoiner`` re-joins.
    - ``holiday`` (1/0) becomes ``Holiday`` ("Holiday"/"No Holiday").
    - ``Seasons`` is derived from the calendar month (meteorological, matching
      the v3 dataset's Winter/Spring/Summer/Autumn labels).
    - ``Functioning Day`` is constant "Yes": the v4 source has no non-service
      flag, so v3's deterministic zero-demand rule is inert here.
    - ``visibility`` (m) → ``Visibility (10m)`` (÷10).
    - NaN → 0 for ``_ZERO_WHEN_MISSING`` (KMA leaves zero readings blank).
    - ``ID`` and the CSV's ``weekday`` are dropped (the pipeline re-derives
      weekday features from the date).

    Rows are sorted by timestamp. The ~390 missing hourly stamps (mostly
    2015-2017) are NOT reindexed/filled here — whether absence means zero
    demand or no service is an EDA decision, not a loader default.
    """
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime", ignore_index=True)

    out = pd.DataFrame()
    out["DateTime"] = df["datetime"].dt.normalize()
    out["Rented Bike Count"] = df["count"]
    out["Hour"] = df["datetime"].dt.hour.astype(np.int64)

    for src, dst in _COLUMN_MAP.items():
        if dst == "Rented Bike Count":
            continue
        out[dst] = df[src]
    out["Visibility (10m)"] = df["visibility"] / 10.0

    for col in _ZERO_WHEN_MISSING:
        out[col] = out[col].fillna(0.0)

    out["Seasons"] = df["datetime"].dt.month.map(_SEASON_BY_MONTH)
    out["Holiday"] = np.where(df["holiday"].astype(int) == 1, "Holiday", "No Holiday")
    out["Functioning Day"] = "Yes"

    """logger.info(
        "Loaded v4 dataset: %s rows × %s cols (%s → %s) from %s",
        len(out), out.shape[1], out["DateTime"].min().date(), out["DateTime"].max().date(), path,
    )"""
    return out


def read_model(
    description: str,
    models_dir: str | Path = _DEFAULT_MODELS_DIR,
) -> Tuple[object, object, object]:
    """Load a saved estimator, pipeline, and transformer from compressed pickles.

    Port of notebook cell [106] (exec_count=106).

    Parameters
    ----------
    description:
        Model name used as the filename prefix (e.g., ``"XGBRegressor"``).
    models_dir:
        Directory containing ``.pkl.gz`` artefacts.

    Returns
    -------
    Tuple[model, pipeline, transformer]
    """
    models_dir = Path(models_dir)

    with gzip.open(models_dir / f"{description}_estimator.pkl.gz", "rb") as f:
        model = pickle.load(f)

    with gzip.open(models_dir / f"{description}_pipeline.pkl.gz", "rb") as f:
        pipeline = pickle.load(f)

    with gzip.open(models_dir / f"{description}_transformer.pkl.gz", "rb") as f:
        transformer = pickle.load(f)

    logger.info("Loaded model artefacts for '%s'", description)
    return model, pipeline, transformer
