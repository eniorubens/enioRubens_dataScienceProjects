"""Reproducible operational replay for the public-facing notebook 08.

The demonstration consumes only frozen, selection-fold OOF predictions from
notebook 07.  It never opens the final holdout, refits an estimator, or turns a
historical replay into a claim about live production inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from src.data import read_data

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_ROOT = _PROJECT_ROOT / "dataset" / "normal_operations" / "conformal_uncertainty_v1"
DEFAULT_MANIFEST_PATH = DEFAULT_RUNTIME_ROOT / "conformal_uncertainty_manifest.json"
DEFAULT_PREDICTIONS_PATH = DEFAULT_RUNTIME_ROOT / "conformal_oof_predictions.csv.gz"
DEFAULT_RAW_DATA_PATH = _PROJECT_ROOT / "dataset" / "Seoul_public_bicycle.csv"

DEMO_CODE_VERSION = "operational_demo_v1_historical_replay"

PREDICTION_COLUMNS = (
    "candidate_id",
    "method_id",
    "coverage",
    "timestamp",
    "fold",
    "test_year",
    "fold_role",
    "weekday",
    "hour",
    "Rush_Period",
    "Seasons",
    "Rainfall Cat",
    "y_true",
    "y_pred",
    "lower",
    "upper",
    "interval_available",
    "status",
    "calibration_size",
    "alpha_used",
    "covered",
    "width",
)

PROFILE_COLUMNS = (
    "Temperature(C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Holiday",
    "Functioning Day",
)

WEEKDAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


@dataclass(frozen=True)
class OperationalDemoConfig:
    """Configuration for one deterministic historical forecast replay."""

    manifest_path: Path = DEFAULT_MANIFEST_PATH
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH
    raw_data_path: Path = DEFAULT_RAW_DATA_PATH
    candidate_id: str = "U4b_g0p01"
    coverage: float = 0.90
    random_state: int = 2026
    planned_capacity: float = 4000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "predictions_path", Path(self.predictions_path))
        object.__setattr__(self, "raw_data_path", Path(self.raw_data_path))
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty.")
        if not 0.0 < float(self.coverage) < 1.0:
            raise ValueError("coverage must lie in (0, 1).")
        if int(self.random_state) < 0:
            raise ValueError("random_state must be non-negative.")
        if not np.isfinite(self.planned_capacity) or self.planned_capacity < 0:
            raise ValueError("planned_capacity must be finite and non-negative.")


@dataclass
class OperationalReplayResult:
    """All information required to present and audit one replay."""

    config: OperationalDemoConfig
    manifest: Mapping[str, Any]
    profile: Dict[str, Any]
    timestamp: pd.Timestamp
    fold: int
    test_year: int
    eligible_rows: int
    point_prediction: float
    lower: float
    upper: float
    actual_demand: float
    absolute_error: float
    covered: bool
    interval_width: float
    calibration_size: int
    alpha_used: float
    decision_code: str
    additional_capacity_to_point: int
    additional_capacity_to_upper: int
    source_audit: pd.DataFrame = field(repr=False)

    @property
    def planned_capacity(self) -> float:
        return float(self.config.planned_capacity)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_full_manifest(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Conformal manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("run_mode") != "full":
        raise ValueError("Operational replay requires notebook 07 full artifacts.")
    return manifest


def _load_eligible_predictions(config: OperationalDemoConfig) -> pd.DataFrame:
    if not config.predictions_path.exists():
        raise FileNotFoundError(
            f"Conformal prediction artifact not found: {config.predictions_path}"
        )
    frame = pd.read_csv(
        config.predictions_path,
        usecols=list(PREDICTION_COLUMNS),
        parse_dates=["timestamp"],
        low_memory=False,
    )
    mask = (
        frame["candidate_id"].eq(config.candidate_id)
        & np.isclose(frame["coverage"].astype(float), float(config.coverage))
        & frame["fold_role"].eq("selection")
        & frame["interval_available"].fillna(False).astype(bool)
        & frame["status"].eq("ok")
    )
    eligible = frame.loc[mask].copy()
    if eligible.empty:
        raise ValueError(
            "No selection-fold scored interval matches the requested candidate and coverage."
        )
    if not eligible["fold_role"].eq("selection").all():
        raise AssertionError("Replay eligibility leaked a non-selection fold.")
    if eligible[["lower", "y_pred", "upper", "y_true"]].isna().any().any():
        raise ValueError("Eligible replay rows contain missing predictions or targets.")
    return eligible.sort_values(["timestamp", "fold"], kind="stable").reset_index(drop=True)


def _prepare_context(context_frame: pd.DataFrame) -> pd.DataFrame:
    context = context_frame.copy(deep=True)
    if "timestamp" not in context.columns:
        required = {"DateTime", "Hour"}
        missing = required.difference(context.columns)
        if missing:
            raise ValueError(f"Context frame is missing temporal columns: {sorted(missing)}")
        context["timestamp"] = pd.to_datetime(context["DateTime"]) + pd.to_timedelta(
            context["Hour"].astype(int), unit="h"
        )
    else:
        context["timestamp"] = pd.to_datetime(context["timestamp"])
    if context["timestamp"].duplicated().any():
        raise ValueError("Context frame must contain one row per timestamp.")
    return context


def _capacity_decision(
    planned_capacity: float,
    lower: float,
    point_prediction: float,
    upper: float,
) -> str:
    if planned_capacity < lower:
        return "critical_shortage"
    if planned_capacity < point_prediction:
        return "reinforcement_recommended"
    if planned_capacity < upper:
        return "attention_zone"
    return "capacity_compatible"


def _source_audit(config: OperationalDemoConfig, context_is_memory: bool) -> pd.DataFrame:
    rows = []
    for label, path in (
        ("conformal_manifest", config.manifest_path),
        ("conformal_predictions", config.predictions_path),
    ):
        rows.append(
            {
                "artifact": label,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if context_is_memory:
        rows.append(
            {
                "artifact": "context_frame",
                "path": "in_memory",
                "bytes": np.nan,
                "sha256": "not_applicable",
            }
        )
    else:
        rows.append(
            {
                "artifact": "raw_context",
                "path": str(config.raw_data_path.resolve()),
                "bytes": config.raw_data_path.stat().st_size,
                "sha256": _sha256(config.raw_data_path),
            }
        )
    return pd.DataFrame(rows)


def build_operational_replay(
    config: Optional[OperationalDemoConfig] = None,
    context_frame: Optional[pd.DataFrame] = None,
) -> OperationalReplayResult:
    """Build one deterministic replay without refitting or holdout access."""

    config = config or OperationalDemoConfig()
    manifest = _read_full_manifest(config.manifest_path)
    eligible = _load_eligible_predictions(config)
    selected = eligible.sample(n=1, random_state=config.random_state).iloc[0]

    if context_frame is None:
        context_frame = read_data(config.raw_data_path)
        context_is_memory = False
    else:
        context_is_memory = True
    context = _prepare_context(context_frame)
    matches = context.loc[context["timestamp"].eq(selected["timestamp"])]
    if len(matches) != 1:
        raise ValueError("The selected replay timestamp has no unique context row.")
    raw = matches.iloc[0]

    if "Rented Bike Count" in raw.index and not np.isclose(
        float(raw["Rented Bike Count"]), float(selected["y_true"])
    ):
        raise ValueError("Raw context target disagrees with the frozen OOF target.")

    weekday = int(selected["weekday"])
    profile: Dict[str, Any] = {
        "timestamp": pd.Timestamp(selected["timestamp"]),
        "weekday": WEEKDAY_NAMES.get(weekday, str(weekday)),
        "hour": int(selected["hour"]),
        "Rush_Period": selected["Rush_Period"],
        "Seasons": selected["Seasons"],
        "Rainfall Cat": selected["Rainfall Cat"],
    }
    for column in PROFILE_COLUMNS:
        profile[column] = raw[column] if column in raw.index else np.nan

    point_prediction = float(selected["y_pred"])
    lower = float(selected["lower"])
    upper = float(selected["upper"])
    actual = float(selected["y_true"])
    planned = float(config.planned_capacity)

    return OperationalReplayResult(
        config=config,
        manifest=manifest,
        profile=profile,
        timestamp=pd.Timestamp(selected["timestamp"]),
        fold=int(selected["fold"]),
        test_year=int(selected["test_year"]),
        eligible_rows=len(eligible),
        point_prediction=point_prediction,
        lower=lower,
        upper=upper,
        actual_demand=actual,
        absolute_error=abs(actual - point_prediction),
        covered=bool(selected["covered"]),
        interval_width=float(selected["width"]),
        calibration_size=int(selected["calibration_size"]),
        alpha_used=float(selected["alpha_used"]),
        decision_code=_capacity_decision(planned, lower, point_prediction, upper),
        additional_capacity_to_point=max(0, math.ceil(point_prediction - planned)),
        additional_capacity_to_upper=max(0, math.ceil(upper - planned)),
        source_audit=_source_audit(config, context_is_memory),
    )
