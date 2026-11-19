import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import make_text

from delivery_routing_vrptw import SolverConfig, fixed_cost_for, parse_instance, phase_c, verify
from delivery_routing_vrptw.config import ExperimentConfig, PhaseCSection, load_config

ROUTES = [[1, 2], [3, 5], [4]]
PC = PhaseCSection()
CFG = SolverConfig(time_limit_s=1.0, scale=100, fixed_cost_k=10)
TIGHT_ROWS = [(0, 0, 0, 0, 0, 100, 0), (1, 5, 0, 1, 0, 10, 2), (2, -5, 0, 1, 0, 10, 2)]


# --------------------------------------------------------------------------- draws, penalties, labels
def test_draw_customers_is_reproducible_and_sized():
    a = phase_c.draw_customers(100, 0.2, 20261118, "R101", "priority")
    assert a == phase_c.draw_customers(100, 0.2, 20261118, "R101", "priority")
    assert len(a) == 20 and a == sorted(set(a)) and 1 <= a[0] and a[-1] <= 100
    assert a != phase_c.draw_customers(100, 0.2, 20261118, "R102", "priority")
    assert a != phase_c.draw_customers(100, 0.2, 20261118, "R101", "reopt")
    assert a != phase_c.draw_customers(100, 0.2, 1, "R101", "priority")
    assert len(phase_c.draw_customers(5, 0.1, 1, "x", "s")) == 1  # at least one customer


def test_base_drop_penalty_dominates_fixed_cost_and_detours(inst):
    cfg = SolverConfig(scale=100, fixed_cost_k=10)
    d_ub = sum(2 * math.hypot(x, y) for x, y in [(3, 4), (6, 8), (-3, 4), (-6, 8), (0, -5)])
    base = phase_c.base_drop_penalty(inst, cfg)
    assert abs(base - (fixed_cost_for(inst, cfg) + 100 * d_ub + 1)) < 1
    assert base > fixed_cost_for(inst, cfg) + 100 * 2 * math.hypot(6, 8)  # more than any single round trip


def test_fleet_class_and_labels():
    assert [phase_c.fleet_class(v, 5) for v in (4, 5, 6)] == ["reduced", "equal", "increased"]
    assert phase_c.level_label(1.0) == "1" and phase_c.level_label(0.9) == "0.9"


def test_solver_config_is_v2():
    cfg = phase_c.phase_c_solver_config(load_config(), 3.0, num_vehicles=4)
    assert (cfg.first_solution, cfg.metaheuristic, cfg.scale, cfg.fixed_cost_k) == (
        "SAVINGS", "GUIDED_LOCAL_SEARCH", 10000, 10
    )
    assert cfg.time_limit_s == 3.0 and cfg.num_vehicles == 4


# --------------------------------------------------------------------------- cells
def test_solver_cells_counts_and_reference_requirement():
    names = ["A", "B"]
    cap = phase_c.solver_cells("capacity", names, PC, 3)
    assert len(cap) == 2 * 3 * 3 and {c["level"] for c in cap} == {"1", "0.9", "0.8"}
    with pytest.raises(ValueError, match="reference"):
        phase_c.solver_cells("fleet_priority", names, PC, 3)
    ref = {"A": 5, "B": 6}
    fp = phase_c.solver_cells("fleet_priority", names, PC, 3, ref)
    assert len(fp) == 2 * 3 * 2 * 2 and {c["variant"] for c in fp} == {"uniform", "priority"}
    sw = phase_c.solver_cells("soft_windows", names, PC, 3, ref)
    assert len(sw) == 2 * 3 * 2 and {c["level"] for c in sw} == {"c1", "c10"}
    assert len({phase_c.cell_key(c) for c in cap + fp + sw}) == len(cap + fp + sw)
    with pytest.raises(ValueError):
        phase_c.solver_cells("nope", names, PC, 1)


def test_reopt_cells():
    assert len(phase_c.reopt_cells("removal", ["A", "B"], 3)) == 2 * 6
    ins = phase_c.reopt_cells("insertion", ["A", "B"], 3)
    assert len(ins) == 2 * 7 and sum(c["variant"] == "plan90" for c in ins) == 2
    assert {c["variant"] for c in phase_c.reopt_cells("insertion", ["A"], 3, "plan")} == {"plan90"}
    assert "plan90" not in {c["variant"] for c in phase_c.reopt_cells("insertion", ["A"], 3, "reopt")}
    with pytest.raises(ValueError):
        phase_c.reopt_cells("removal", ["A"], 3, "bad")


def test_capacity_cell():
    inst = parse_instance(make_text(vehicles=6))  # room for the extra vehicles that a smaller Q needs
    row = phase_c.solve_capacity_cell(inst, 0.9, 0, CFG)
    assert row["feasible"] and row["capacity"] == 9 and row["level"] == "0.9" and row["variant"] == ""
    assert row["instance"] == "R101" and row["vehicles"] == 4
    assert not phase_c.solve_capacity_cell(parse_instance(make_text(vehicles=3)), 0.9, 0, CFG)["feasible"]
    skipped = phase_c.solve_capacity_cell(inst, 0.5, 0, CFG)
    assert skipped["status"].startswith("skipped") and not skipped["feasible"]
    assert json.loads(phase_c.as_frame(skipped)["routes"].iloc[0]) == []


def test_fleet_priority_cell(inst):
    pri = phase_c.draw_customers(5, PC.priority_share, PC.seed, inst.name, "priority")
    easy = phase_c.solve_fleet_priority_cell(inst, 4, 1, "uniform", 0, CFG, PC)
    assert easy["K"] == 3 and easy["n_dropped"] == 0 and easy["feasible"] and easy["vehicles"] == 3
    hard_u = phase_c.solve_fleet_priority_cell(inst, 4, 2, "uniform", 0, CFG, PC)
    hard_p = phase_c.solve_fleet_priority_cell(inst, 4, 2, "priority", 0, CFG, PC)
    for row in (hard_u, hard_p):
        assert row["K"] == 2 and row["feasible"] and row["vehicles"] <= 2 and row["n_dropped"] >= 1
        assert json.loads(row["priority_ids"]) == pri
        dropped = json.loads(row["dropped_ids"])
        assert row["dropped_demand"] == sum(int(inst.demand[c]) for c in dropped)
        served = {c for r in json.loads(row["routes"]) for c in r}
        assert served.isdisjoint(dropped) and served | set(dropped) == {1, 2, 3, 4, 5}
    assert hard_p["n_dropped_priority"] == 0  # another customer can always be dropped instead
    assert hard_p["base_penalty"] == hard_u["base_penalty"] > 0
    assert phase_c.solve_fleet_priority_cell(inst, 1, 1, "uniform", 0, CFG, PC)["status"].startswith("skipped")
    with pytest.raises(ValueError):
        phase_c.solve_fleet_priority_cell(inst, 4, 1, "other", 0, CFG, PC)


def test_soft_cell():
    tight = parse_instance(make_text(rows=TIGHT_ROWS, vehicles=2, capacity=100))
    row = phase_c.solve_soft_cell(tight, 2, 1, 0, CFG)
    assert row["K"] == 1 and row["feasible"] and row["vehicles"] == 1 and row["n_late"] == 1
    assert row["total_lateness"] == pytest.approx(7.0) and row["max_lateness"] == pytest.approx(7.0)
    assert row["late_share"] == pytest.approx(0.5) and row["level"] == "c1"
    assert phase_c.solve_soft_cell(tight, 1, 10, 0, CFG)["status"].startswith("skipped")


def test_run_solver_cell_dispatch(inst):
    cell = {"block": "capacity", "instance": "R101", "level": "1", "variant": "", "replicate": 0, "factor": 1.0}
    assert phase_c.run_solver_cell(inst, cell, CFG, PC)["feasible"]
    with pytest.raises(ValueError):
        phase_c.run_solver_cell(inst, {**cell, "block": "x"}, CFG, PC)


def test_reopt_cells_removal_and_insertion(inst):
    sample = phase_c.draw_customers(5, PC.reopt_share, PC.seed, inst.name, "reopt")
    common = [c for c in range(1, 6) if c not in sample]
    warm = phase_c.solve_reopt_cell(inst, "removal", "warm", 0, CFG, PC, ROUTES)
    cold = phase_c.solve_reopt_cell(inst, "removal", "cold", 0, CFG, PC, ROUTES)
    for row in (warm, cold):
        assert row["feasible"] and json.loads(row["sample"]) == sample and row["n_common"] == 4
        assert sorted(c for r in json.loads(row["routes"]) for c in r) == common  # original ids, sample gone
        assert 0.0 <= row["realloc_customers_reassigned_share"] <= 1.0 and row["orig_vehicles"] >= 1
    assert warm["warm_start"] == "used" and cold["warm_start"] == "none"
    with pytest.raises(ValueError):
        phase_c.solve_reopt_cell(inst, "removal", "plan90", 0, CFG, PC, ROUTES)

    p90 = phase_c.solve_reopt_cell(inst, "insertion", "plan90", 0, CFG, PC, [])
    plan90 = json.loads(p90["routes"])
    assert p90["feasible"] and sorted(c for r in plan90 for c in r) == common
    assert math.isnan(p90["realloc_arcs_kept"]) and math.isnan(p90["delta_vehicles"])
    w = phase_c.solve_reopt_cell(inst, "insertion", "warm", 0, CFG, PC, plan90)
    c = phase_c.solve_reopt_cell(inst, "insertion", "cold", 0, CFG, PC, plan90)
    for row in (w, c):
        assert row["feasible"] and sorted(x for r in json.loads(row["routes"]) for x in r) == [1, 2, 3, 4, 5]
        assert verify(inst, json.loads(row["routes"])).feasible
        assert not math.isnan(row["realloc_pair_agreement"])
    assert w["warm_start_input"] == "built" and w["warm_start"] == "used" and c["warm_start"] == "none"


def test_as_frame_fixed_columns_and_unknown_fields(inst):
    row = phase_c.solve_capacity_cell(inst, 1.0, 0, CFG)
    frame = phase_c.as_frame(row)
    assert list(frame.columns) == phase_c.result_columns("capacity") and frame.columns[-1] == "routes"
    with pytest.raises(ValueError, match="schema"):
        phase_c.as_frame({**row, "surprise": 1})
    with pytest.raises(ValueError):
        phase_c.result_columns("nope")


# --------------------------------------------------------------------------- summaries
def capacity_rows():
    rows = []
    # instance A: reference vehicles over replicates 5, 4, 6 -> sorted 4, 5, 6 -> representative 5
    for rep, (veh, dist) in enumerate([(5, 100.0), (4, 110.0), (6, 90.0)]):
        rows.append(dict(instance="A", family="C1", level="1", variant="", replicate=rep, feasible=True, vehicles=veh,
                         distance=dist, routes=json.dumps([[1]]), capacity=10))
    for rep, veh in enumerate([6, 5, 7]):  # 0.9: representative 6 (one extra), distance 120
        rows.append(dict(instance="A", family="C1", level="0.9", variant="", replicate=rep, feasible=True, vehicles=veh,
                         distance=120.0 + rep, routes="[]", capacity=9))
    for rep in range(3):
        rows.append(dict(instance="A", family="C1", level="0.8", variant="", replicate=rep, feasible=False,
                         vehicles=np.nan, distance=np.nan, routes="[]", capacity=8))
        rows.append(dict(instance="B", family="R1", level="1", variant="", replicate=rep, feasible=True, vehicles=3,
                         distance=50.0, routes=json.dumps([[7]]), capacity=10))
        rows.append(dict(instance="B", family="R1", level="0.9", variant="", replicate=rep, feasible=True, vehicles=3,
                         distance=55.0, routes="[]", capacity=9))
        rows.append(dict(instance="B", family="R1", level="0.8", variant="", replicate=rep, feasible=True, vehicles=4,
                         distance=70.0, routes="[]", capacity=8))
    return pd.DataFrame(rows)


def test_reference_and_capacity_summary():
    rows = capacity_rows()
    ref = phase_c.reference_table(rows)
    a = ref[ref["instance"] == "A"].iloc[0]
    assert (a["ref_vehicles"], a["ref_distance"], a["ref_replicate"]) == (5, 100.0, 0) and a["ref_routes"] == [[1]]
    summary = phase_c.capacity_summary(rows, ref)
    sa = summary[(summary["instance"] == "A") & (summary["level"] == "0.9")].iloc[0]
    assert sa["extra_vehicles"] == 1 and sa["fleet_class"] == "increased"
    assert sa["distance_change_pct"] == pytest.approx(20.0)
    infeasible = summary[(summary["instance"] == "A") & (summary["level"] == "0.8")].iloc[0]
    assert not infeasible["feasible"] and math.isnan(infeasible["distance_change_pct"])
    agg = phase_c.capacity_aggregate(summary)
    row = agg[(agg["level"] == "0.9") & (agg["fleet_class"] == "increased")].iloc[0]
    assert row["n_instances"] == 1 and row["distance_change_mean"] == pytest.approx(20.0)
    allrow = agg[(agg["level"] == "0.8") & (agg["fleet_class"] == "all")].iloc[0]
    assert allrow["n_infeasible"] == 1
    # the distance of different fleet classes is never pooled: no row mixes classes
    assert set(agg["fleet_class"]) <= {"all", "reduced", "equal", "increased"}


def test_fleet_priority_summary_and_aggregate():
    ref = pd.DataFrame({"instance": ["A"], "ref_vehicles": [5], "ref_distance": [100.0]})
    rows = []
    for variant, dropped, pri_dropped, demand in (("uniform", 4, 2, 40), ("priority", 3, 0, 30)):
        for rep in range(3):
            rows.append(dict(instance="A", family="C1", level="K-1", variant=variant, replicate=rep, feasible=True,
                             vehicles=4, distance=90.0 + rep, K=4, n_customers=100, n_priority=20,
                             n_dropped=dropped, n_dropped_priority=pri_dropped, dropped_demand=demand))
    df = pd.DataFrame(rows)
    summary = phase_c.fleet_priority_summary(df, ref)
    assert len(summary) == 2 and set(summary["fleet_class"]) == {"reduced"}
    agg = phase_c.fleet_priority_aggregate(summary)
    pri = agg[agg["variant"] == "priority"].iloc[0]
    assert pri["dropped_priority_total"] == 0 and pri["priority_dropped_share"] == 0.0
    assert pri["common_dropped_share"] == pytest.approx(3 / 80)
    uni = agg[agg["variant"] == "uniform"].iloc[0]
    assert uni["priority_dropped_share"] == pytest.approx(2 / 20)
    diff = agg[agg["variant"] == "priority - uniform"].iloc[0]
    assert diff["dropped_priority_total"] == -2 and diff["dropped_total"] == -1 and diff["dropped_demand_mean"] == -10


def test_soft_summary_and_aggregate():
    ref = pd.DataFrame({"instance": ["A"], "ref_vehicles": [5], "ref_distance": [100.0]})
    rows = [dict(instance="A", family="C1", level=lvl, replicate=rep, feasible=True, vehicles=4, distance=95.0,
                 K=4, multiplier=m, n_late=n, late_share=n / 100, total_lateness=10.0 * n, max_lateness=7.0)
            for lvl, m, n in (("c1", 1, 5), ("c10", 10, 1)) for rep in range(3)]
    df = pd.DataFrame(rows).assign(variant="")
    summary = phase_c.soft_summary(df, ref)
    assert set(summary["level"]) == {"c1", "c10"}
    agg = phase_c.soft_aggregate(summary)
    low = agg[agg["level"] == "c1"].iloc[0]
    assert low["n_late_mean"] == 5 and low["fleet_class"] == "reduced"
    assert low["distance_change_mean"] == pytest.approx(-5.0)


def test_noise_table_and_reopt_aggregate():
    sample = [9]
    plans = {0: [[1, 2], [3, 9]], 1: [[1, 2], [3]], 2: [[1, 3], [2, 9]]}
    rows = []
    for rep, routes in plans.items():
        rows.append(dict(instance="A", variant="cold", replicate=rep, feasible=True, sample=json.dumps(sample),
                         routes=json.dumps(routes), vehicles=2, distance=50.0, orig_vehicles=2, orig_distance=49.0, delta_vehicles=0.0, delta_distance_pct=1.0,
                         runtime_s=3.0, **{f"realloc_{f}": 0.1 for f in phase_c.REALLOC_FIELDS}))
        rows.append(dict(instance="A", variant="warm", replicate=rep, feasible=True, sample=json.dumps(sample),
                         routes="[]", vehicles=2, distance=48.0, orig_vehicles=3, orig_distance=49.0, delta_vehicles=-1.0, delta_distance_pct=-2.0, runtime_s=3.0,
                         **{f"realloc_{f}": 0.05 for f in phase_c.REALLOC_FIELDS}))
    df = pd.DataFrame(rows)
    noise = phase_c.noise_table(df)
    assert len(noise) == 3 and (noise["variant"] == "noise").all()
    first = noise[noise["replicate"] == 0].iloc[0]  # plans 0 and 1 are equal once the sample is removed
    assert first["realloc_customers_reassigned"] == 0 and first["realloc_arcs_kept"] == 1.0
    last = noise[noise["replicate"] == 2].iloc[0]  # plan 2 against plan 0: genuinely different
    assert last["realloc_customers_reassigned"] > 0
    agg = phase_c.reopt_aggregate(df, noise)
    assert list(agg["variant"]) == ["warm", "cold", "noise"]
    warm = agg[agg["variant"] == "warm"].iloc[0]
    assert warm["n"] == 3 and warm["realloc_customers_reassigned_share"] == pytest.approx(0.05)
    assert warm["delta_vehicles_mean"] == -1.0
    summary = phase_c.reopt_summary(df)
    assert set(summary["variant"]) == {"warm", "cold"} and (summary["n_feasible"] == 3).all()


def test_read_results_roundtrip(tmp_path):
    inst = parse_instance(make_text(vehicles=6))
    row = phase_c.solve_capacity_cell(inst, 0.9, 0, CFG)
    skipped = phase_c.solve_capacity_cell(inst, 0.5, 1, CFG)
    path = tmp_path / "r.csv"
    for k, r in enumerate((row, skipped)):
        phase_c.as_frame(r).to_csv(path, mode="a", header=k == 0, index=False)
    df = phase_c.read_results(path)
    assert df["feasible"].tolist() == [True, False] and df["level"].tolist() == ["0.9", "0.5"]
    assert {phase_c.cell_key(r) for r in df.to_dict(orient="records")} == {
        ("R101", "0.9", "", 0), ("R101", "0.5", "", 1)
    }


# --------------------------------------------------------------------------- config
def test_phase_c_section_matches_protocol():
    pc = load_config().phase_c
    assert (pc.seed, pc.scale, pc.fixed_cost_k, pc.time_limit_s, pc.replicates) == (20261118, 10000, 10, 30, 3)
    assert pc.capacity_factors == (1.0, 0.9, 0.8) and pc.fleet_reductions == (1, 2)
    assert (pc.priority_share, pc.priority_penalty_multiplier) == (0.2, 10)
    assert pc.soft_cost_multipliers == (1, 10) and pc.soft_fleet_reduction == 1 and pc.reopt_share == 0.1
    assert pc.c1_draws == 1000 and pc.c1_travel_a == (0.05, 0.10, 0.20) and pc.c1_service_b == 0.20
    assert ExperimentConfig().phase_c == PhaseCSection()
    assert json.dumps(load_config().to_dict()["phase_c"])


@pytest.mark.parametrize("bad", [
    {"capacity_factors": (0.9, 0.8)}, {"capacity_factors": (1.0, 1.2)}, {"fleet_reductions": (0,)},
    {"priority_share": 1.0}, {"reopt_share": 0.0}, {"soft_cost_multipliers": ()}, {"time_limit_s": 0},
    {"c1_draws": 0}, {"c1_travel_a": (0.0,)}, {"priority_penalty_multiplier": 0}, {"scale": 0},
])
def test_phase_c_validation(bad):
    with pytest.raises(ValueError):
        PhaseCSection(**bad)


def test_phase_c_toml_unknown_key(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[phase_c]\nnope = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)
    path.write_text("[phase_c]\nreplicates = 2\n", encoding="utf-8")
    assert load_config(path).phase_c.replicates == 2


# --------------------------------------------------------------------------- block-specific representative
def test_orderings_registry():
    assert phase_c.ordering("fleet_priority") == [
        "n_dropped_priority", "n_dropped", "dropped_demand", "vehicles", "distance"
    ]
    assert phase_c.ordering("soft_windows") == ["n_late", "total_lateness", "vehicles", "distance"]
    for block in ("capacity", "removal", "insertion"):
        assert phase_c.ordering(block) == ["vehicles", "distance"]


def test_representative_ranks_more_drops_worse_despite_fewer_vehicles():
    # sorted by the lexicographic default (vehicles, distance) the middle one would be the 3-vehicle plan that
    # drops 9 customers; with the block ordering that plan is the worst and the 4-vehicle plan is ranked second.
    rows = pd.DataFrame([
        dict(instance="A", level="K-1", variant="u", replicate=0, feasible=True, vehicles=4, distance=100.0,
             n_dropped=2, n_dropped_priority=0, dropped_demand=20),
        dict(instance="A", level="K-1", variant="u", replicate=1, feasible=True, vehicles=3, distance=80.0,
             n_dropped=9, n_dropped_priority=0, dropped_demand=90),
        dict(instance="A", level="K-1", variant="u", replicate=2, feasible=True, vehicles=4, distance=105.0,
             n_dropped=1, n_dropped_priority=0, dropped_demand=10),
    ])
    keys = ["instance", "level", "variant"]
    assert phase_c.representative(rows, keys).iloc[0]["replicate"] == 0  # default: (3,80), (4,100), (4,105)
    pick = phase_c.representative(rows, keys, phase_c.ordering("fleet_priority")).iloc[0]
    assert pick["replicate"] == 0 and pick["n_dropped"] == 2  # order: 1 drop, 2 drops, 9 drops
    worst = rows.copy()
    worst.loc[0, "n_dropped"] = 12  # drops now 12, 9, 1 (replicates 0, 1, 2)
    pick = phase_c.representative(worst, keys, phase_c.ordering("fleet_priority")).iloc[0]
    assert pick["replicate"] == 1  # drops 1, 9, 12 -> middle is the 9-drop plan, not the 3-vehicle default
    # priority drops outrank everything
    pri = rows.copy()
    pri.loc[2, "n_dropped_priority"] = 1
    pri.loc[1, "n_dropped_priority"] = 1
    order = phase_c.ordering("fleet_priority")
    assert phase_c.representative(pri, keys, order).iloc[0]["replicate"] == 2  # 0 priority drops first; then 1(9), 2(1)


def test_representative_soft_ordering_ranks_lateness_worse_and_infeasible_last():
    rows = pd.DataFrame([
        dict(instance="A", level="c1", replicate=0, feasible=True, vehicles=3, distance=70.0, n_late=8,
             total_lateness=80.0),
        dict(instance="A", level="c1", replicate=1, feasible=True, vehicles=5, distance=120.0, n_late=1,
             total_lateness=3.0),
        dict(instance="A", level="c1", replicate=2, feasible=False, vehicles=np.nan, distance=np.nan, n_late=0,
             total_lateness=0.0),
    ])
    pick = phase_c.representative(rows, ["instance", "level"], phase_c.ordering("soft_windows")).iloc[0]
    assert pick["replicate"] == 0  # order: (1 late), (8 late), infeasible -> position 2 of 3 is the 8-late plan
    two = rows.iloc[:2]
    assert phase_c.representative(two, ["instance", "level"], phase_c.ordering("soft_windows")).iloc[0][
        "replicate"] == 1  # position ceil(2/2) = 1st: the plan with 1 late customer, despite 5 vehicles


def test_summaries_record_ordering_and_use_it():
    ref = pd.DataFrame({"instance": ["A"], "ref_vehicles": [5], "ref_distance": [100.0]})
    rows = pd.DataFrame([
        dict(instance="A", family="C1", level="K-1", variant="uniform", replicate=r, feasible=True, vehicles=v,
             distance=d, K=4, n_customers=100, n_priority=20, n_dropped=n, n_dropped_priority=0,
             dropped_demand=10 * n, capacity_infeasible=False)
        for r, (v, d, n) in enumerate([(4, 100.0, 2), (3, 80.0, 9), (4, 105.0, 1)])
    ])
    summary = phase_c.fleet_priority_summary(rows, ref)
    assert summary.iloc[0]["n_dropped"] == 2 and "n_dropped_priority" in summary.iloc[0]["representative_order"]


def test_capacity_infeasible_flag_and_breakdown():
    tight = parse_instance(make_text(rows=TIGHT_ROWS, vehicles=2, capacity=100))
    assert phase_c.capacity_lower_bound(tight) == 1
    wide = parse_instance(make_text(vehicles=6))
    assert phase_c.capacity_lower_bound(wide) == 3  # 25 units / Q = 10
    # soft cell: reference 3 vehicles, K = 2 < 3 -> structural infeasibility, not a search failure
    row = phase_c.solve_soft_cell(wide, 3, 1, 0, CFG)
    assert row["K"] == 2 and row["capacity_infeasible"] is True and not row["feasible"]
    ok = phase_c.solve_soft_cell(wide, 4, 1, 0, CFG)
    assert ok["K"] == 3 and ok["capacity_infeasible"] is False
    fp = phase_c.solve_fleet_priority_cell(wide, 4, 2, "uniform", 0, CFG, PC)
    assert fp["capacity_infeasible"] is True and fp["feasible"]  # dropping makes it feasible
    cap = phase_c.solve_capacity_cell(wide, 1.0, 0, CFG)
    assert cap["capacity_infeasible"] is False
    summary = pd.DataFrame([
        dict(level="c1", feasible=False, capacity_infeasible=True),
        dict(level="c1", feasible=False, capacity_infeasible=False),
        dict(level="c1", feasible=True, capacity_infeasible=False),
        dict(level="c10", feasible=True, capacity_infeasible=False),
    ])
    out = phase_c.infeasible_breakdown(summary).set_index("level")
    assert out.loc["c1", "n_capacity_infeasible"] == 1 and out.loc["c1", "n_search_failed"] == 1
    assert out.loc["c1", "n_feasible"] == 1 and out.loc["c10", "n_search_failed"] == 0
    assert "capacity_infeasible" in phase_c.result_columns("soft_windows")
    assert "capacity_infeasible" not in phase_c.result_columns("removal")

