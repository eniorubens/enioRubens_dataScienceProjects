import math

import pytest

from delivery_routing_vrptw import (
    Reference,
    Report,
    compare,
    family_of,
    make_split,
    read_split,
    write_split,
)
from delivery_routing_vrptw.benchmark import DEFAULT_DEV_COUNTS

REF = Reference("C101", 10, 828.94, "RT")
SIZES = {"C1": 9, "C2": 8, "R1": 12, "R2": 11, "RC1": 8, "RC2": 8}


def rep(vehicles, distance, feasible=True):
    return Report(feasible=feasible, vehicles=vehicles, distance=distance, duration=0.0, waiting=0.0)


def fake_names():
    return [f"{fam[:-1]}{fam[-1]}{i:02d}" for fam, n in SIZES.items() for i in range(1, n + 1)]


def test_compare_equal_vehicles():
    out = compare(rep(10, 830.0), REF)
    assert out["distance_comparable"] and out["delta_vehicles"] == 0
    assert out["gap_distance_pct"] == pytest.approx(100 * (830 - 828.94) / 828.94)
    assert not out["beats_reference"]
    assert compare(rep(10, 828.94), REF)["beats_reference"] is False
    assert compare(rep(10, 828.0), REF)["beats_reference"] is True


def test_compare_fewer_vehicles():
    out = compare(rep(9, 900.0), REF)
    assert out["beats_reference"] and not out["distance_comparable"] and out["delta_vehicles"] == -1


def test_compare_more_vehicles():
    out = compare(rep(11, 700.0), REF)
    assert not out["beats_reference"] and not out["distance_comparable"]
    assert out["gap_distance_pct"] < 0  # always computed, but flagged as not comparable


def test_compare_infeasible():
    out = compare(rep(10, 830.0, feasible=False), REF)
    assert out["feasible"] is False and math.isnan(out["gap_distance_pct"])
    assert out["ref_vehicles"] == 10 and not out["beats_reference"]


def test_make_split_counts_and_reproducible():
    ns = fake_names()
    a = make_split(ns)
    assert a == make_split(list(reversed(ns)))
    assert len(a) == 56 and sum(v == "dev" for v in a.values()) == 14
    for fam, k in DEFAULT_DEV_COUNTS.items():
        assert sum(1 for n, v in a.items() if v == "dev" and family_of(n) == fam) == k
    assert make_split(ns, seed=1) != a


def test_make_split_too_many():
    with pytest.raises(ValueError):
        make_split(["C101", "C102"], dev_counts={"C1": 3})


def test_split_roundtrip(tmp_path):
    split = make_split(fake_names())
    path = tmp_path / "sub" / "split.csv"
    write_split(split, path)
    assert read_split(path) == split
    assert path.read_text().splitlines()[0] == "instance,family,set"


def test_rank_configurations():
    import pandas as pd

    from delivery_routing_vrptw import rank_configurations

    rows = []
    # config A: always best; B: ties A on i1 (same vehicles+distance), worse on i2; C: infeasible on i2
    spec = {
        "i1": {"A": (True, 10, 100.0), "B": (True, 10, 100.0), "C": (True, 11, 90.0)},
        "i2": {"A": (True, 10, 100.0), "B": (True, 10, 105.0), "C": (False, float("nan"), float("nan"))},
    }
    for inst, cfgs in spec.items():
        for c, (f, v, d) in cfgs.items():
            rows.append({
                "instance": inst, "fleet_strategy": "fixed_cost", "first_solution": c,
                "metaheuristic": "GUIDED_LOCAL_SEARCH", "feasible": f, "vehicles": v, "distance": d,
                "delta_vehicles": v - 10 if f else float("nan"),
                "gap_distance_pct": (d - 100) if f else float("nan"),
                "distance_comparable": bool(f and v == 10), "runtime_s": 1.0,
            })
    out = rank_configurations(pd.DataFrame(rows))
    assert list(out["first_solution"]) == ["A", "B", "C"]
    assert out.loc[0, "mean_rank"] == pytest.approx(1.25)  # (1.5 + 1) / 2
    assert out.loc[1, "mean_rank"] == pytest.approx(1.75)
    assert out.loc[2, "mean_rank"] == pytest.approx(3.0)
