import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src import operational_demo_reports as reports
from src.operational_demo import (
    OperationalDemoConfig,
    build_operational_replay,
)


def _prediction_row(**overrides):
    row = {
        "candidate_id": "U4b_g0p01",
        "method_id": "U4b",
        "coverage": 0.90,
        "timestamp": "2022-10-22 07:00:00",
        "fold": 4,
        "test_year": 2022,
        "fold_role": "selection",
        "weekday": 5,
        "hour": 7,
        "Rush_Period": "Non-Rush",
        "Seasons": "Autumn",
        "Rainfall Cat": "No Rain",
        "y_true": 110.0,
        "y_pred": 100.0,
        "lower": 50.0,
        "upper": 150.0,
        "interval_available": True,
        "status": "ok",
        "calibration_size": 2160,
        "alpha_used": 0.08,
        "covered": True,
        "width": 100.0,
    }
    row.update(overrides)
    return row


def _context_frame(target=110.0):
    return pd.DataFrame(
        {
            "DateTime": [pd.Timestamp("2022-10-22")],
            "Hour": [7],
            "Temperature(C)": [10.6],
            "Humidity(%)": [86.0],
            "Wind speed (m/s)": [1.8],
            "Visibility (10m)": [79.5],
            "Solar Radiation (MJ/m2)": [0.0],
            "Rainfall(mm)": [0.0],
            "Snowfall (cm)": [0.0],
            "Holiday": ["No Holiday"],
            "Functioning Day": ["Yes"],
            "Rented Bike Count": [target],
        }
    )


def _config(tmp_path, rows=None, **overrides):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"run_mode": "full"}), encoding="utf-8")
    predictions_path = tmp_path / "predictions.csv.gz"
    pd.DataFrame(rows or [_prediction_row()]).to_csv(
        predictions_path, index=False, compression="gzip"
    )
    return OperationalDemoConfig(
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        raw_data_path=tmp_path / "unused.csv",
        **overrides,
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"candidate_id": ""}, "candidate_id"),
        ({"coverage": 0.0}, "coverage"),
        ({"coverage": 1.0}, "coverage"),
        ({"random_state": -1}, "random_state"),
        ({"planned_capacity": -1}, "planned_capacity"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OperationalDemoConfig(**kwargs)


def test_replay_requires_full_manifest(tmp_path):
    config = _config(tmp_path)
    config.manifest_path.write_text(json.dumps({"run_mode": "smoke"}), encoding="utf-8")
    with pytest.raises(ValueError, match="full artifacts"):
        build_operational_replay(config, _context_frame())


def test_replay_filters_warmup_stress_and_other_candidates(tmp_path):
    rows = [
        _prediction_row(),
        _prediction_row(fold_role="stress"),
        _prediction_row(interval_available=False, status="warmup"),
        _prediction_row(candidate_id="U1"),
        _prediction_row(coverage=0.80),
    ]
    result = build_operational_replay(_config(tmp_path, rows), _context_frame())
    assert result.eligible_rows == 1
    assert result.fold == 4
    assert result.test_year == 2022


def test_replay_is_deterministic_and_does_not_mutate_context(tmp_path):
    rows = [
        _prediction_row(timestamp="2022-10-22 07:00:00"),
        _prediction_row(
            timestamp="2022-10-22 08:00:00",
            hour=8,
            y_true=120.0,
        ),
    ]
    context = pd.concat(
        [
            _context_frame(),
            _context_frame(target=120.0).assign(Hour=8),
        ],
        ignore_index=True,
    )
    before = context.copy(deep=True)
    config = _config(tmp_path, rows, random_state=17)
    first = build_operational_replay(config, context)
    second = build_operational_replay(config, context)
    assert first.timestamp == second.timestamp
    pd.testing.assert_frame_equal(context, before)


def test_profile_never_exposes_target(tmp_path):
    result = build_operational_replay(_config(tmp_path), _context_frame())
    assert "Rented Bike Count" not in result.profile
    assert result.profile["weekday"] == "Saturday"
    assert result.actual_demand == 110.0


def test_replay_rejects_target_disagreement(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        build_operational_replay(_config(tmp_path), _context_frame(target=999.0))


def test_replay_rejects_missing_context_timestamp(tmp_path):
    context = _context_frame().assign(DateTime=pd.Timestamp("2022-10-23"))
    with pytest.raises(ValueError, match="no unique context row"):
        build_operational_replay(_config(tmp_path), context)


@pytest.mark.parametrize(
    "capacity, expected, point_gap, upper_gap",
    [
        (40.0, "critical_shortage", 60, 110),
        (75.0, "reinforcement_recommended", 25, 75),
        (125.0, "attention_zone", 0, 25),
        (175.0, "capacity_compatible", 0, 0),
    ],
)
def test_capacity_decision_contract(tmp_path, capacity, expected, point_gap, upper_gap):
    result = build_operational_replay(
        _config(tmp_path, planned_capacity=capacity), _context_frame()
    )
    assert result.decision_code == expected
    assert result.additional_capacity_to_point == point_gap
    assert result.additional_capacity_to_upper == upper_gap


def test_source_audit_covers_frozen_inputs_and_memory_context(tmp_path):
    result = build_operational_replay(_config(tmp_path), _context_frame())
    assert set(result.source_audit["artifact"]) == {
        "conformal_manifest",
        "conformal_predictions",
        "context_frame",
    }
    assert result.source_audit["sha256"].notna().all()


def test_reports_separate_forecast_from_later_audit(tmp_path):
    result = build_operational_replay(_config(tmp_path), _context_frame())
    forecast = reports.forecast_report(result)
    audit = reports.audit_report(result)
    assert (
        not forecast.astype(str)
        .apply(lambda column: column.str.contains("observada", case=False).any())
        .any()
    )
    assert (
        audit.astype(str)
        .apply(lambda column: column.str.contains("observada", case=False).any())
        .any()
    )


def test_reports_use_pt_br_number_formatting(tmp_path):
    result = build_operational_replay(_config(tmp_path, planned_capacity=4000.0), _context_frame())
    protocol = reports.protocol_report(result.config)
    rendered = " ".join(protocol.astype(str).to_numpy().ravel())
    assert "4.000" in rendered


def test_plot_and_synthesis_are_available(tmp_path):
    result = build_operational_replay(_config(tmp_path), _context_frame())
    figure = reports.plot_operational_forecast(result)
    try:
        assert len(figure.axes) == 1
        assert "U4b" in reports.synthesis_report(result)
    finally:
        plt.close(figure)
