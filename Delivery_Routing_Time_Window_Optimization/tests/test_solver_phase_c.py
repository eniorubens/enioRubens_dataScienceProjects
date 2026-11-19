import pytest
from conftest import make_text

from delivery_routing_vrptw import (
    SolverConfig,
    limit_fleet,
    measure_lateness,
    parse_instance,
    remove_customers,
    solve,
    verify,
)

# two customers 5 units either side of the depot, both due at 10: serving both with one vehicle needs
# 5 (go) + 2 (service) + 10 (cross) = 17 > 10 for the second one, so only a soft due date makes it feasible.
TIGHT_ROWS = [
    (0, 0, 0, 0, 0, 100, 0),
    (1, 5, 0, 1, 0, 10, 2),
    (2, -5, 0, 1, 0, 10, 2),
]


@pytest.fixture
def tight():
    return parse_instance(make_text(rows=TIGHT_ROWS, vehicles=2, capacity=100))


def test_defaults_keep_v1_behaviour(inst):
    sol = solve(inst, SolverConfig(time_limit_s=0.5))
    assert sol.meta["soft_due_cost_per_unit"] is None and sol.meta["dropped"] == []
    assert verify(inst, sol.routes).feasible


def test_per_customer_penalties_protect_priorities(inst):
    cfg = SolverConfig(time_limit_s=1, num_vehicles=2)
    sol = solve(inst, cfg, drop_penalty=10**9, drop_penalties={3: 10**10})
    dropped = sol.meta["dropped"]
    assert dropped and 3 not in dropped
    sub, cmap = remove_customers(limit_fleet(inst, 2), dropped)
    assert verify(sub, cmap.to_reduced(sol.routes)).feasible


def test_drop_penalties_alone_make_only_those_droppable(inst):
    sol = solve(inst, SolverConfig(time_limit_s=0.5), drop_penalties={1: 10**9})
    assert sol.meta["dropped"] == [] and verify(inst, sol.routes).feasible


def test_drop_penalties_validation(inst):
    with pytest.raises(ValueError):
        solve(inst, SolverConfig(time_limit_s=0.5), drop_penalties={9: 100})
    with pytest.raises(ValueError):
        solve(inst, SolverConfig(time_limit_s=0.5), drop_penalties={1: 0})


def test_soft_due_dates_allow_late_service(tight):
    one = limit_fleet(tight, 1)
    hard = solve(one, SolverConfig(time_limit_s=0.5, num_vehicles=1))
    assert hard.routes == []  # a hard window cannot be met with one vehicle
    soft = solve(one, SolverConfig(time_limit_s=0.5, num_vehicles=1), soft_due_cost_per_unit=1)
    assert len(soft.routes) == 1 and sorted(soft.routes[0]) == [1, 2]
    assert not verify(one, soft.routes).feasible  # the rigid verifier still rejects it
    rep = measure_lateness(one, soft.routes)
    assert rep.hard_feasible and rep.n_late == 1 and rep.total_lateness == pytest.approx(7.0)
    assert soft.meta["soft_due_cost_per_unit"] == 1


def test_soft_cost_validation(tight):
    with pytest.raises(ValueError):
        solve(tight, SolverConfig(time_limit_s=0.5), soft_due_cost_per_unit=0)


def test_soft_windows_no_effect_when_hard_plan_exists(inst):
    soft = solve(inst, SolverConfig(time_limit_s=0.5), soft_due_cost_per_unit=10)
    rep = measure_lateness(inst, soft.routes)
    assert rep.hard_feasible and rep.n_late == 0


def test_soft_depot_return_stays_hard(tight):
    short = parse_instance(make_text(rows=[(0, 0, 0, 0, 0, 14, 0), *TIGHT_ROWS[1:]], vehicles=1, capacity=100))
    sol = solve(short, SolverConfig(time_limit_s=0.5, num_vehicles=1), soft_due_cost_per_unit=1)
    assert sol.routes == []  # 5 + 2 + 10 + 2 + 5 = 24 > horizon 14, however cheap lateness is
