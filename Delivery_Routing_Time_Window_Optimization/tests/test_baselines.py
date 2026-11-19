from dataclasses import replace

import pytest

from delivery_routing_vrptw import greedy_deadline, nearest_neighbor_tw, verify


@pytest.mark.parametrize("fn", [greedy_deadline, nearest_neighbor_tw])
def test_feasible_and_deterministic(inst, fn):
    a, b = fn(inst), fn(inst)
    rep = verify(inst, a.routes)
    assert rep.feasible and not rep.unserved
    assert a.routes == b.routes
    assert a.method == fn.__name__
    assert a.num_vehicles == 3  # total demand 25 with capacity 10


def test_nn_weights(inst):
    sol = nearest_neighbor_tw(inst, weights=(1.0, 0.0, 0.0))
    assert verify(inst, sol.routes).feasible


def test_unservable_customer_raises(inst):
    due = inst.due.copy()
    due[1] = 1
    bad = replace(inst, due=due)
    with pytest.raises(ValueError, match="cannot be served"):
        greedy_deadline(bad)
    with pytest.raises(ValueError, match="cannot be served"):
        nearest_neighbor_tw(bad)
