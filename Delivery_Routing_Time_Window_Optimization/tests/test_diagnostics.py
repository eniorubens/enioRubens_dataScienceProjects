from dataclasses import replace

import pytest
from conftest import make_text

from delivery_routing_vrptw import diagnostics, nearest_neighbor_tw, parse_instance, verify
from delivery_routing_vrptw.diagnostics import VARIANTS

# Five customers on a cross with tight windows; capacity 10, every demand 5.
TIGHT_ROWS = [
    (0, 0, 0, 0, 0, 100, 0),
    (1, 10, 0, 5, 0, 12, 0),
    (2, 20, 0, 5, 0, 22, 0),
    (3, -10, 0, 5, 0, 12, 0),
    (4, -20, 0, 5, 0, 22, 0),
    (5, 0, 10, 5, 0, 12, 0),
]


@pytest.fixture
def tight():
    return parse_instance(make_text("R101", TIGHT_ROWS, vehicles=5, capacity=10))


def test_demo_violations_each_variant_flags_its_violation(tight):
    inst = tight
    base = [[1, 2], [3, 4], [5]]
    assert verify(inst, base).feasible
    variants = diagnostics.demo_violations(inst, base)
    assert tuple(variants) == VARIANTS
    assert base == [[1, 2], [3, 4], [5]]  # input untouched
    reports = {k: verify(inst, r) for k, r in variants.items()}
    assert reports["original"].feasible
    assert reports["drop_customer"].unserved == [1]
    assert reports["duplicate_customer"].duplicates == [1]
    assert reports["break_window"].window_violations
    assert reports["break_capacity"].capacity_violations


def test_demo_violations_needs_two_routes(tight):
    inst = tight
    with pytest.raises(ValueError, match="two"):
        diagnostics.demo_violations(inst, [[1, 2]])


def test_demo_violations_capacity_impossible(tight):
    inst = tight
    with pytest.raises(ValueError, match="capacity"):
        diagnostics.demo_violations(inst, [[1], [2], [5]])  # 5 + 5 <= 10: merging cannot overload


def test_break_window_impossible(inst):
    wide = replace(inst, due=inst.due * 0 + 100)  # every window as wide as the horizon
    with pytest.raises(ValueError, match="window"):
        diagnostics._break_window(wide, [[1, 2], [3, 4], [5]], diagnostics.exact(wide))


def test_violation_table(tight):
    inst = tight
    routes = nearest_neighbor_tw(inst).routes
    variants = diagnostics.demo_violations(inst, routes)
    table = diagnostics.violation_table(inst, variants, "pt")
    assert len(table) == 5
    assert table["feasible"].tolist() == [True, False, False, False, False]
    assert table["variant"].iloc[1] == "Um cliente removido"
    assert table["n_unserved"].iloc[1] == 1 and table["n_duplicates"].iloc[2] == 1
    assert table["n_capacity_violations"].iloc[4] >= 1
