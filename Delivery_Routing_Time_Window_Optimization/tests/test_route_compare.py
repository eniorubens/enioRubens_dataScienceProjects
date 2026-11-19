import json
import math

import numpy as np
import pandas as pd
import pytest

from delivery_routing_vrptw import (
    allow_early,
    early_delivery_summary,
    exact,
    family_of,
    reallocation,
    reallocation_summary,
    reallocation_table,
    schedule_metrics,
    tolerance_label,
    validate_instance,
    verify,
)

ROUTES = [[1, 2], [3, 5], [4]]


def test_schedule_metrics_matches_verify(inst):
    m = schedule_metrics(inst, ROUTES, exact(inst))
    rep = verify(inst, ROUTES, exact(inst))
    assert m["total_duration"] == pytest.approx(rep.duration)
    assert m["total_waiting"] == pytest.approx(rep.waiting)
    assert m["distance"] == pytest.approx(rep.distance) and m["vehicles"] == rep.vehicles == 3
    assert m["total_service"] == pytest.approx(2 + 2 + 2 + 2 + 0)
    assert m["total_duration"] == pytest.approx(m["total_travel"] + m["total_waiting"] + m["total_service"])
    assert m["total_travel"] == pytest.approx(m["distance"])
    assert m["mean_route_duration"] == pytest.approx(m["total_duration"] / 3)
    assert m["makespan"] <= m["total_duration"]


def test_schedule_metrics_waiting_and_empty(inst):
    m = schedule_metrics(inst, [[3], [], [1, 2], [4, 5]])
    assert m["vehicles"] == 3 and m["total_waiting"] > 0  # customers 3 and 5 open after arrival
    empty = schedule_metrics(inst, [])
    assert empty["makespan"] == 0 and empty["vehicles"] == 0 and empty["mean_route_duration"] == 0


def test_allow_early_zero_is_identity(inst):
    out = allow_early(inst, 0.0)
    assert out.name == inst.name and out.ready is not inst.ready
    assert np.array_equal(out.ready, inst.ready) and np.array_equal(out.due, inst.due)


def test_allow_early_values(inst):
    out = allow_early(inst, 0.5)
    # customer 3: ready 20, width 50 -> floor(20 - 25) clipped to 0; customer 5: ready 30, width 60 -> 0
    assert list(out.ready) == [0, 0, 0, 0, 0, 0] and np.array_equal(out.due, inst.due)
    assert out.name == "R101@e0.5"
    small = allow_early(inst, 0.2)
    assert small.ready[3] == 10 and small.ready[5] == 18  # 20-10, 30-12
    assert small.name == "R101@e0.2"


@pytest.mark.parametrize("tol", [0.1, 0.25, 1.0, 3.0, math.inf])
def test_allow_early_contains_original(inst, tol):
    out = allow_early(inst, tol)
    validate_instance(out)
    assert np.all(out.ready <= inst.ready) and np.all(out.ready >= 0) and np.array_equal(out.due, inst.due)
    assert out.ready[0] == inst.ready[0]
    assert np.issubdtype(out.ready.dtype, np.integer)
    if math.isinf(tol):
        assert np.all(out.ready == 0) and out.name == "R101@efull"


def test_allow_early_invalid(inst):
    with pytest.raises(ValueError):
        allow_early(inst, -0.1)
    with pytest.raises(ValueError):
        allow_early(inst, math.nan)


def test_labels_and_family():
    assert tolerance_label(0.0) == "0" and tolerance_label(0.25) == "0.25" and tolerance_label(math.inf) == "full"
    assert family_of("R101@e0.5") == "R1" and family_of("RC205@efull") == "RC2"


def test_allow_early_never_worse_for_same_plan(inst):
    early = allow_early(inst, math.inf)
    base = schedule_metrics(inst, ROUTES)
    new = schedule_metrics(early, ROUTES)
    assert verify(early, ROUTES).feasible
    assert new["total_waiting"] <= base["total_waiting"] and new["total_duration"] <= base["total_duration"]


def test_reallocation_identical():
    plan = [[1, 2], [3, 4], [5]]
    r = reallocation(plan, [list(x) for x in plan[::-1]])  # same routes, different route order
    assert r["customers_reassigned"] == 0 and r["customers_reassigned_share"] == 0
    assert r["pairs_kept"] == 1.0 and r["pair_agreement"] == 1.0 and r["arcs_kept"] == 1.0


def test_reallocation_swap_one_customer():
    a, b = [[1, 2, 3], [4, 5, 6]], [[1, 2, 6], [4, 5, 3]]
    r = reallocation(a, b)
    assert r["customers_reassigned"] == 2 and r["customers_reassigned_share"] == pytest.approx(2 / 6)
    assert r["pairs_kept"] == pytest.approx(2 / 6)  # (1,2) and (4,5) kept of 6 co-routed pairs
    assert r["pair_agreement"] == pytest.approx(1 - 8 / 15)  # 8 of the 15 pairs change togetherness
    assert r["vehicles_a"] == r["vehicles_b"] == 2


def test_reallocation_route_split():
    r = reallocation([[1, 2, 3, 4]], [[1, 2], [3, 4]])
    assert r["customers_reassigned"] == 2 and r["vehicles_b"] == 2
    assert r["pairs_kept"] == pytest.approx(2 / 6)
    assert r["pair_agreement"] == pytest.approx(2 / 6)
    assert r["arcs_kept"] == pytest.approx(4 / 5)  # only 2-3 is lost of 0-1, 1-2, 2-3, 3-4, 4-0


def test_reallocation_tie_break_and_errors():
    r = reallocation([[1, 2], [3, 4]], [[1, 3], [2, 4]])  # equal overlaps: A0-B0, A1-B1
    assert r["customers_reassigned"] == 2
    with pytest.raises(ValueError):
        reallocation([[1, 2]], [[1, 3]])
    with pytest.raises(ValueError):
        reallocation([[1, 1]], [[1]])


def _rows():
    plans = {
        ("A", "0", 0): [[1, 2], [3, 4]],
        ("A", "0", 1): [[1, 2], [3, 4]],
        ("A", "0.5", 0): [[1, 2], [3, 4]],
        ("A", "0.5", 1): [[1, 2, 3, 4]],
        ("A", "full", 0): [[1, 3], [2, 4]],
        ("A", "full", 1): [[1, 3], [2, 4]],
    }
    out = []
    for (inst, lab, rep), routes in plans.items():
        base = {"0": 100.0, "0.5": 90.0, "full": 80.0}[lab]
        out.append(
            {
                "instance": inst, "tolerance_label": lab, "replicate": rep, "feasible": True,
                "vehicles": len(routes), "distance": base + rep, "total_duration": base * 2 + rep,
                "total_waiting": base / 2, "makespan": base,
                "total_duration_min": base * 2 - 10, "total_waiting_min": base / 4, "routes": json.dumps(routes),
            }
        )
    return pd.DataFrame(out)


def test_summary():
    s = early_delivery_summary(_rows())
    assert list(s["tolerance_label"]) == ["0", "0.5", "full"]
    full = s[s["tolerance_label"] == "full"].iloc[0]
    # two replicates: position ceil(2/2) = 1st of the lexicographic order, i.e. replicate 0 (distance 80)
    assert full["rep_replicate"] == 0 and full["distance_rep"] == 80
    assert full["distance_min"] == 80 and full["distance_max"] == 81
    base = s.iloc[0]
    assert base["distance_pct"] == 0 and base["vehicles_saved"] == 0
    assert full["distance_pct"] == pytest.approx(100 * (80 - 100) / 100)
    assert full["waiting_pct"] == pytest.approx(-20.0) and full["tolerance"] == math.inf
    half = s[s["tolerance_label"] == "0.5"].iloc[0]
    assert full["duration_min_pct"] == pytest.approx(100 * (150 - 190) / 190)
    assert full["waiting_min_pct"] == pytest.approx(-20.0)
    # replicate 1 of "0.5" has one vehicle and sorts first: the representative plan is that real plan
    assert half["rep_replicate"] == 1 and half["vehicles_rep"] == 1 and half["distance_rep"] == 91
    assert half["vehicles_saved"] == 1 and half["vehicles_min"] == 1 and half["vehicles_max"] == 2


def test_summary_is_one_real_plan_not_a_column_wise_median():
    rows = pd.DataFrame([
        {"instance": "A", "tolerance_label": "0", "replicate": r, "feasible": True, "vehicles": v,
         "distance": d, "total_duration": d + 10, "total_waiting": 10.0, "makespan": d,
         "total_duration_min": d + 5, "total_waiting_min": 5.0}
        for r, (v, d) in enumerate([(3, 50.0), (4, 40.0), (5, 30.0)])
    ])
    base = early_delivery_summary(rows).iloc[0]
    # column-wise medians would give (4, 40) only by luck; with these data it must be the plan (4, 40)
    assert (base["vehicles_rep"], base["distance_rep"]) == (4, 40.0) and base["rep_replicate"] == 1
    shuffled = early_delivery_summary(rows.iloc[::-1].reset_index(drop=True)).iloc[0]
    assert (shuffled["vehicles_rep"], shuffled["distance_rep"]) == (4, 40.0)
    rows2 = rows.assign(vehicles=[3, 3, 5], distance=[50.0, 60.0, 30.0])
    plan = early_delivery_summary(rows2).iloc[0]
    # sorted: (3, 50), (3, 60), (5, 30) -> median is (3, 60); a column-wise median would be (3, 50)
    assert (plan["vehicles_rep"], plan["distance_rep"]) == (3, 60.0)


def test_summary_infeasible_excluded():
    rows = _rows()
    rows.loc[1, "feasible"] = False
    s = early_delivery_summary(rows.astype({"feasible": str}))
    base = s.iloc[0]
    assert base["n_feasible"] == 1 and base["distance_rep"] == 100


def test_reallocation_table_and_noise():
    t = reallocation_table(_rows())
    assert set(t["tolerance_label"]) == {"0.5", "full", "noise"}
    noise = t[t["tolerance_label"] == "noise"]
    assert len(noise) == 2 and (noise["customers_reassigned"] == 0).all()
    full = t[t["tolerance_label"] == "full"]
    assert (full["customers_reassigned"] == 2).all()
    half = t[t["tolerance_label"] == "0.5"].sort_values("replicate")
    assert list(half["customers_reassigned"]) == [0, 2]
    agg = reallocation_summary(t)
    assert list(agg["tolerance_label"]) == ["0.5", "full", "noise"]
    assert agg.set_index("tolerance_label").loc["noise", "customers_reassigned_share"] == 0
    assert reallocation_summary(pd.DataFrame()).empty


def test_reallocation_table_single_replicate_has_no_noise():
    rows = _rows()
    rows = rows[rows["replicate"] == 0]
    assert "noise" not in set(reallocation_table(rows)["tolerance_label"])


def _make(rows):
    from conftest import make_text

    from delivery_routing_vrptw import parse_instance

    return parse_instance(make_text(rows=rows))


def test_optimal_departure_first_customer_opens_late(inst):
    # customer 3 opens at 20 and is 5 away from the depot: wait 15, no later waits
    m = schedule_metrics(inst, [[3]])
    assert m["total_delay"] == pytest.approx(15) and m["total_waiting"] == pytest.approx(15)
    assert m["total_waiting_min"] == pytest.approx(0)
    assert m["total_duration_min"] == pytest.approx(m["total_duration"] - 15)
    assert m["makespan"] == pytest.approx(m["total_duration"])


def test_optimal_departure_due_binds():
    rows = [
        (0, 0, 0, 0, 0, 100, 0),
        (1, 3, 4, 1, 0, 50, 0),  # 5 from the depot, no wait
        (2, 6, 8, 1, 0, 20, 0),  # reached at 10, due 20: slack 10
        (3, 9, 12, 1, 40, 90, 0),  # reached at 15, opens at 40: wait 25
    ]
    m = schedule_metrics(_make(rows), [[1, 2, 3]])
    assert m["total_waiting"] == pytest.approx(25)
    assert m["total_delay"] == pytest.approx(10)  # limited by the due of customer 2, not by the wait
    assert m["total_waiting_min"] == pytest.approx(15)
    assert m["total_duration_min"] == pytest.approx(m["total_duration"] - 10)


def test_optimal_departure_no_waits(inst):
    m = schedule_metrics(inst, [[1], [2]])
    assert m["total_waiting"] == 0 and m["total_delay"] == 0
    assert m["total_duration_min"] == m["total_duration"]
    assert m["mean_route_duration_min"] == m["mean_route_duration"]


def _simulate(instance, route, delta):
    matrix = exact(instance)
    t, prev = float(delta), 0
    for c in route:
        t = max(t + matrix[prev, c], instance.ready[c])
        if t > instance.due[c] + 1e-9:
            return None
        t += instance.service[c]
        prev = c
    return t + matrix[prev, 0]


@pytest.mark.parametrize("route", [[3], [1, 2], [3, 5], [4, 3], [5, 3, 1], [1, 3, 2, 4], [5]])
def test_optimal_departure_brute_force(inst, route):
    base = _simulate(inst, route, 0)
    assert base is not None
    delta = schedule_metrics(inst, [route])["total_delay"]
    for d in np.arange(0, delta + 1e-9, 0.25):
        assert _simulate(inst, route, d) == pytest.approx(base)  # feasible, same end time
    assert _simulate(inst, route, delta) == pytest.approx(base)
    after = _simulate(inst, route, delta + 0.01)
    assert after is None or after > base + 1e-9  # infeasible, or the end time grows
