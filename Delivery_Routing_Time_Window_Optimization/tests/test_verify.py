from dataclasses import replace

import numpy as np
import pytest

from delivery_routing_vrptw import report_to_dict, verify

GOOD = [[1, 2], [3, 5], [4]]


def test_feasible_hand_computed(inst):
    rep = verify(inst, GOOD)
    assert rep.feasible
    assert rep.vehicles == 3
    assert rep.distance == pytest.approx(20 + 10 + np.sqrt(90) + 20)
    assert rep.waiting == pytest.approx(15.0)
    assert rep.duration == pytest.approx(24 + (27 + np.sqrt(90)) + 22)
    assert not (rep.unserved or rep.duplicates or rep.unknown)
    assert report_to_dict(rep)["vehicles"] == 3


def test_empty_routes_ignored(inst):
    assert verify(inst, [[], *GOOD, []]).vehicles == 3


def test_custom_matrix(inst):
    rep = verify(inst, GOOD, matrix=np.zeros((6, 6)))
    assert rep.distance == 0.0


def test_unserved(inst):
    rep = verify(inst, [[1, 2], [3, 5]])
    assert not rep.feasible and rep.unserved == [4]


def test_duplicate(inst):
    rep = verify(inst, [[1, 2], [2, 3, 5], [4]])
    assert not rep.feasible and rep.duplicates == [2]


def test_unknown(inst):
    rep = verify(inst, [[1, 2], [3, 5], [4, 9]])
    assert not rep.feasible and rep.unknown == [9]


def test_capacity(inst):
    rep = verify(inst, [[1, 2, 3], [5], [4]])
    assert not rep.feasible and rep.capacity_violations == [(0, 16)]


def test_window(inst):
    tight = inst.due.copy()
    tight[1] = 4
    rep = verify(replace(inst, due=tight), GOOD)
    assert not rep.feasible
    assert rep.window_violations == [(1, 5.0, 4.0)]


def test_depot_return(inst):
    short = inst.due.copy()
    short[0] = 20
    rep = verify(replace(inst, due=short), GOOD)
    assert not rep.feasible and 0 in rep.depot_violations


def test_too_many_vehicles(inst):
    rep = verify(replace(inst, vehicle_limit=2), GOOD)
    assert not rep.feasible and rep.too_many_vehicles
