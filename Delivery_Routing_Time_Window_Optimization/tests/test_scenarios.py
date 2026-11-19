import math

import numpy as np
import pandas as pd
import pytest

from delivery_routing_vrptw import (
    SolverConfig,
    exact,
    family_of,
    solve,
    validate_instance,
    verify,
    widen_windows,
    window_scenario_summary,
)


def test_factor_one_is_identity(inst):
    out = widen_windows(inst, 1.0)
    assert out.name == inst.name
    for field in ("x", "y", "demand", "ready", "due", "service"):
        assert np.array_equal(getattr(out, field), getattr(inst, field))
    assert out.ready is not inst.ready


@pytest.mark.parametrize("mode", ["symmetric", "later_due"])
@pytest.mark.parametrize("factor", [1.25, 1.5, 2.0, 5.0])
def test_windows_contain_original_and_integral(inst, mode, factor):
    out = widen_windows(inst, factor, mode)
    validate_instance(out)
    assert np.all(out.ready <= inst.ready) and np.all(out.due >= inst.due)
    assert np.issubdtype(out.ready.dtype, np.integer) and np.issubdtype(out.due.dtype, np.integer)
    assert out.ready[0] == inst.ready[0] and out.due[0] == inst.due[0]
    for field in ("x", "y", "demand", "service"):
        assert np.array_equal(getattr(out, field), getattr(inst, field))
    assert out.capacity == inst.capacity and out.vehicle_limit == inst.vehicle_limit


def test_symmetric_and_later_due_values(inst):
    # customer 3: window [20, 70], width 50 -> extra 25
    sym = widen_windows(inst, 1.5, "symmetric")
    assert sym.ready[3] == 7 and sym.due[3] == 83  # 7.5 down, 82.5 up
    assert sym.ready[2] == 0  # customer 2 window [0, 60] is clipped at 0
    late = widen_windows(inst, 1.5, "later_due")
    assert late.ready[3] == 20 and late.due[3] == 93  # 95 clipped to 100 - 2 - d(3,0)=5


def test_clipping_to_return_feasibility(inst):
    out = widen_windows(inst, 10.0, "later_due")
    back = exact(inst)[:, 0]
    assert np.all(out.due[1:] + inst.service[1:] + back[1:] <= inst.horizon + 1e-9)
    assert np.all(out.ready >= 0)
    sym = widen_windows(inst, 10.0, "symmetric")
    assert np.all(sym.ready >= 0)


def test_keep_original_due_beyond_latest(inst):
    tight = type(inst)(**{
        **inst.__dict__, "due": np.array([100, 50, 60, 70, 80, 90]), "service": np.array([0, 2, 2, 2, 2, 20])
    })
    out = widen_windows(tight, 3.0, "later_due")
    assert out.due[5] == 90  # latest = 100 - 20 - 5 = 75 < 90, so the original due is kept


def test_name_and_family(inst):
    out = widen_windows(inst, 1.25)
    assert out.name == "R101@w1.25"
    assert out.family == "R1"
    assert widen_windows(inst, 2.0).name == "R101@w2"
    assert family_of("rc104@w1.5") == "RC1"
    with pytest.raises(ValueError):
        family_of("X101@w1")


def test_invalid_arguments(inst):
    with pytest.raises(ValueError):
        widen_windows(inst, 0.9)
    with pytest.raises(ValueError):
        widen_windows(inst, 1.5, "nope")


def test_original_solution_stays_feasible(inst):
    routes = solve(inst, SolverConfig(time_limit_s=1)).routes
    assert verify(inst, routes).feasible
    for mode in ("symmetric", "later_due"):
        for factor in (1.25, 2.0):
            assert verify(widen_windows(inst, factor, mode), routes).feasible


def test_summary_without_sintef_columns():
    rows = pd.DataFrame({
        "instance": ["A", "A", "A", "B", "B"],
        "mode": ["symmetric"] * 5,
        "factor": [1.0, 1.5, 2.0, 1.0, 1.5],
        "feasible": [True, True, False, True, True],
        "vehicles": [10, 9, 0, 5, 5],
        "distance": [1000.0, 950.0, 0.0, 200.0, 210.0],
    })
    s = window_scenario_summary(rows)
    assert not any("ref" in c or "gap" in c or "sintef" in c.lower() for c in s.columns)
    a15 = s[(s.instance == "A") & (s.factor == 1.5)].iloc[0]
    assert a15.vehicles_saved == 1 and math.isclose(a15.distance_change_pct, -5.0)
    assert np.isnan(s[(s.instance == "A") & (s.factor == 2.0)].iloc[0].vehicles_saved)
    b15 = s[(s.instance == "B") & (s.factor == 1.5)].iloc[0]
    assert b15.vehicles_saved == 0 and math.isclose(b15.distance_change_pct, 5.0)
    base = s[s.factor == 1.0]
    assert (base.vehicles_saved == 0).all() and (base.distance_change_pct == 0).all()
