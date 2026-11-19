import numpy as np
import pytest

from delivery_routing_vrptw import (
    CustomerMap,
    exact,
    insert_customers,
    limit_fleet,
    measure_lateness,
    remove_customers,
    restrict_routes,
    scale_capacity,
    verify,
)

ROUTES = [[1, 2], [3, 5], [4]]


def test_scale_capacity(inst):
    same = scale_capacity(inst, 1.0)
    assert same.capacity == 10 and same.name == inst.name and same is not inst
    small = scale_capacity(inst, 0.85)  # floor(8.5) = 8 >= max demand 6
    assert small.capacity == 8 and small.name == "R101@q0.85"
    assert np.array_equal(small.demand, inst.demand) and inst.capacity == 10
    assert scale_capacity(inst, 0.9).capacity == 9
    with pytest.raises(ValueError, match="below the largest demand"):
        scale_capacity(inst, 0.5)  # 5 < 6
    with pytest.raises(ValueError):
        scale_capacity(inst, 0)


def test_limit_fleet(inst):
    two = limit_fleet(inst, 2)
    assert two.vehicle_limit == 2 and inst.vehicle_limit == 3 and two.name == "R101@k2"
    assert not verify(two, ROUTES).feasible and verify(two, ROUTES).too_many_vehicles
    with pytest.raises(ValueError):
        limit_fleet(inst, 0)


def test_remove_customers_and_mapping(inst):
    reduced, cmap = remove_customers(inst, [2, 4])
    assert isinstance(cmap, CustomerMap) and reduced.n_customers == 3
    assert cmap.new_to_old == (0, 1, 3, 5) and cmap.old_to_new == {1: 1, 3: 2, 5: 3}
    assert np.array_equal(reduced.x, inst.x[[0, 1, 3, 5]]) and np.array_equal(reduced.due, inst.due[[0, 1, 3, 5]])
    assert reduced.capacity == inst.capacity and reduced.vehicle_limit == inst.vehicle_limit
    assert reduced.name == "R101@rm2" and inst.n_customers == 5
    # round trip of plans, original ids <-> reduced ids
    assert cmap.to_reduced(ROUTES) == [[1], [2, 3]]
    assert cmap.to_original([[1], [2, 3]]) == [[1], [3, 5]]
    assert cmap.to_original(cmap.to_reduced(ROUTES)) == restrict_routes(ROUTES, [1, 3, 5])
    # distances are those of the original customers
    assert exact(reduced)[1, 2] == pytest.approx(exact(inst)[1, 3])
    assert verify(reduced, cmap.to_reduced(ROUTES)).feasible


def test_remove_customers_validation(inst):
    with pytest.raises(ValueError):
        remove_customers(inst, [0])
    with pytest.raises(ValueError):
        remove_customers(inst, [6])
    same, cmap = remove_customers(inst, [])
    assert same.n_customers == 5 and cmap.new_to_old == tuple(range(6))
    assert cmap.to_reduced([[], [1]]) == [[1]]


def test_restrict_routes():
    assert restrict_routes([[1, 2, 3], [4], [5, 6]], [1, 3, 5]) == [[1, 3], [5]]
    assert restrict_routes([[1]], []) == []


def test_insert_customers_builds_feasible_plan(inst):
    base = [[1, 2], [3]]
    plan = insert_customers(inst, base, [4, 5], scale=100)
    assert plan is not None and base == [[1, 2], [3]]
    assert sorted(c for r in plan for c in r) == [1, 2, 3, 4, 5]
    assert verify(inst, plan).feasible
    assert insert_customers(inst, ROUTES, []) == ROUTES


def test_insert_customers_respects_fleet_limit(inst):
    assert insert_customers(limit_fleet(inst, 1), [[1, 2]], [3, 4, 5]) is None


def test_measure_lateness_does_not_fail_on_late_service(inst):
    from dataclasses import replace

    late = replace(inst, due=np.array([100, 4, 60, 70, 80, 90]))  # customer 1 would start at 5 > 4
    assert not verify(late, ROUTES).feasible
    rep = measure_lateness(late, ROUTES)
    assert rep.hard_feasible and rep.n_late == 1 and rep.late[0][0] == 1
    assert rep.total_lateness == pytest.approx(1.0) and rep.max_lateness == pytest.approx(1.0)
    assert rep.vehicles == 3 and rep.distance == pytest.approx(verify(late, ROUTES).distance)
    ok = measure_lateness(inst, ROUTES)
    assert ok.hard_feasible and ok.n_late == 0 and ok.total_lateness == 0.0


def test_measure_lateness_flags_hard_violations(inst):
    rep = measure_lateness(inst, [[1, 2, 3], [4]])  # overload (16 > 10) and customer 5 unserved
    assert not rep.hard_feasible and rep.capacity_violations and rep.unserved == [5]
    assert measure_lateness(inst, [[1], [2], [3], [4], [5]]).too_many_vehicles
