import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delivery_routing_vrptw import phase_c, phase_c_view, viz_robustness

METHODS = ("ortools", "nearest_neighbor_tw", "greedy_deadline")
SCENARIOS = ("travel_0.05", "travel_0.10", "travel_0.20", "service", "travel_0.10+service")


def capacity_cells():
    rows = []
    for rep, (veh, dist) in enumerate([(5, 100.0), (4, 110.0), (6, 90.0)]):
        rows.append(dict(instance="A", family="C1", level="1", variant="", replicate=rep, feasible=True, vehicles=veh,
                         distance=dist, routes=json.dumps([[1]]), capacity=10, capacity_infeasible=False))
    for rep, veh in enumerate([6, 5, 7]):
        rows.append(dict(instance="A", family="C1", level="0.9", variant="", replicate=rep, feasible=True, vehicles=veh,
                         distance=120.0 + rep, routes="[]", capacity=9, capacity_infeasible=False))
    for rep in range(3):
        rows.append(dict(instance="A", family="C1", level="0.8", variant="", replicate=rep, feasible=False,
                         vehicles=np.nan, distance=np.nan, routes="[]", capacity=8, capacity_infeasible=False))
        rows.append(dict(instance="B", family="R1", level="1", variant="", replicate=rep, feasible=True, vehicles=3,
                         distance=50.0, routes=json.dumps([[7]]), capacity=10, capacity_infeasible=False))
        rows.append(dict(instance="B", family="R1", level="0.9", variant="", replicate=rep, feasible=True, vehicles=3,
                         distance=55.0, routes="[]", capacity=9, capacity_infeasible=False))
        rows.append(dict(instance="B", family="R1", level="0.8", variant="", replicate=rep, feasible=True, vehicles=4,
                         distance=70.0, routes="[]", capacity=8, capacity_infeasible=False))
    return pd.DataFrame(rows)


def robustness_summary():
    rows = []
    for scope in ("overall", "C1", "R1"):
        for method in METHODS:
            for k, scenario in enumerate(SCENARIOS):
                rows.append({
                    "scope": scope, "method": method, "scenario": scenario, "n_plans": 3,
                    "late_customer_share": 0.05 * (k + 1), "routes_with_late_share": 0.5, "draws_with_late_share": 1.0,
                    "mean_lateness_late": 2.0, "p95_lateness_late": 4.0, "mean_lateness_all": 0.1,
                    "late_return_share": 0.1, "nominal_slack_mean": 100.0, "nominal_slack_min": 1.0,
                })
    return pd.DataFrame(rows)


def test_robustness_tables():
    summary = robustness_summary()
    table = phase_c_view.robustness_table(summary, "overall", "en")
    assert len(table) == 15 and table.iloc[0]["scenario"] == "Travel +0–5%"
    assert table.iloc[0]["late_customer_pct"] == 5.0 and table.iloc[-1]["late_customer_pct"] == 25.0
    by_family = phase_c_view.robustness_by_family(summary, "travel_0.10", "pt")
    assert by_family["family"].tolist() == ["C1", "R1"] and by_family["OR-Tools"].tolist() == [10.0, 10.0]
    cells = pd.DataFrame({
        "instance": ["A", "A", "B"], "method": ["ortools", "ortools", "greedy_deadline"],
        "scenario": ["travel_0.05", "service", "travel_0.05"], "plan_feasible": [True, True, False],
        "n_draws": [1000, 1000, np.nan],
    })
    integrity = phase_c_view.robustness_integrity(cells, "en").set_index("method")
    assert integrity.loc["OR-Tools", "n_scenarios"] == 2 and integrity.loc["OR-Tools", "n_draws"] == 1000
    assert integrity.loc["Deadline-first greedy", "n_plans_stressed"] == 0


def test_slack_tables():
    corr = pd.DataFrame([{"scenario": s, "method": m, "spearman_slack_late_rate": -0.5}
                         for s in SCENARIOS for m in METHODS])
    out = phase_c_view.slack_correlation_table(corr, "en")
    assert len(out) == 5 and (out["OR-Tools"] == -0.5).all()
    bins = pd.DataFrame([{"scenario": "travel_0.10", "method": m, "slack_bin": b, "n_customers": 10, "late_rate": 0.25}
                         for m in METHODS for b in phase_c_view.SLACK_ORDER])
    table = phase_c_view.slack_bins_table(bins, "travel_0.10", "en")
    assert table["slack_bin"].tolist() == list(phase_c_view.SLACK_ORDER) and (table["OR-Tools"] == 25.0).all()


def test_capacity_tables():
    cells = capacity_cells()
    ref = phase_c_view.reference_plans(cells)
    by_level = phase_c_view.capacity_by_level(cells, ref, "en")
    assert set(by_level["level"]) == {"80%", "90%"}  # the reference level is not shown
    row = by_level[(by_level["level"] == "90%") & (by_level["fleet_class"] == "larger fleet")].iloc[0]
    assert row["n_instances"] == 1 and row["distance_change_mean"] == pytest.approx(20.0)
    assert by_level[(by_level["level"] == "80%") & (by_level["fleet_class"] == "all")]["n_infeasible"].iloc[0] == 1
    inst = phase_c_view.capacity_instances(cells, ref, "en")
    assert set(zip(inst["instance"], inst["level"], strict=True)) == {("A", "90%"), ("B", "80%")}


def priority_cells(dropped):
    rows = []
    for variant, (n_dropped, n_pri) in dropped.items():
        for rep in range(3):
            rows.append(dict(instance="A", family="C1", level="K-1", variant=variant, replicate=rep, feasible=True,
                             vehicles=4, distance=90.0 + rep, K=4, n_customers=100, n_priority=20,
                             n_dropped=n_dropped, n_dropped_priority=n_pri, dropped_demand=10 * n_dropped,
                             capacity_infeasible=False))
    return pd.DataFrame(rows)


def test_fleet_priority_tables_and_plot():
    ref = pd.DataFrame({"instance": ["A"], "ref_vehicles": [5], "ref_distance": [100.0]})
    cells = priority_cells({"uniform": (4, 2), "priority": (3, 0)}).assign(level="K-1")
    cells = pd.concat([cells, cells.assign(level="K-2")], ignore_index=True)
    table = phase_c_view.fleet_priority_by_level(cells, ref, "en")
    pri = table[(table["level"] == "K − 1") & (table["variant"] == "With priority (10×)")].iloc[0]
    assert pri["dropped_priority_total"] == 0 and pri["common_dropped_pct"] == pytest.approx(3.8, abs=0.05)
    diff = table[table["variant"] == "Difference (priority − uniform)"].iloc[0]
    assert diff["dropped_priority_total"] == -2
    inst = phase_c_view.fleet_priority_instances(cells, ref, "en")
    assert inst["n_prio_dropped_uniform"].tolist() == [2, 2] and inst["n_prio_dropped_priority_variant"].tolist() == [0, 0]
    fig = viz_robustness.plot_fleet_priority(cells, ref, "pt")
    assert len(fig.axes) == 2
    plt.close(fig)


def test_soft_tables():
    ref = pd.DataFrame({"instance": ["A", "B", "C"], "ref_vehicles": [5, 5, 5], "ref_distance": [100.0] * 3})
    rows = []
    for inst, ok, cap in (("A", True, False), ("B", False, True), ("C", False, False)):
        for lvl, m in (("c1", 1), ("c10", 10)):
            for rep in range(3):
                rows.append(dict(instance=inst, family="R1", level=lvl, variant="", replicate=rep, feasible=ok,
                                 vehicles=4 if ok else np.nan, distance=95.0 if ok else np.nan, K=4, multiplier=m,
                                 n_late=3 if ok else np.nan, late_share=0.03, total_lateness=12.0, max_lateness=7.0,
                                 runtime_s=30.0 if ok else 0.5, capacity_infeasible=cap))
    cells = pd.DataFrame(rows)
    breakdown = phase_c_view.soft_breakdown(cells, ref, "en").iloc[0]
    assert (breakdown["n_feasible"], breakdown["n_capacity_infeasible"], breakdown["n_search_failed"]) == (1, 1, 1)
    status = phase_c_view.soft_status_by_instance(cells, ref, "en").set_index("instance")
    assert status.loc["A", "outcome"] == "solved" and status.loc["B", "outcome"] == "capacity-infeasible"
    assert status.loc["C", "outcome"] == "initial construction failed" and status.loc["C", "runtime_max"] == 0.5
    by_level = phase_c_view.soft_by_level(cells, ref, "en")
    assert by_level["n_instances"].tolist() == [1, 1] and by_level["fleet_class"].iloc[0] == "reduced fleet"


def reopt_cells(share):
    rows = []
    for rep, routes in {0: [[1, 2], [3, 9]], 1: [[1, 2], [3]], 2: [[1, 3], [2, 9]]}.items():
        for variant, value in (("cold", share * 2), ("warm", share)):
            rows.append(dict(instance="A", variant=variant, replicate=rep, feasible=True, sample=json.dumps([9]),
                             routes=json.dumps(routes) if variant == "cold" else "[]", vehicles=2, distance=50.0,
                             orig_vehicles=2, orig_distance=49.0, delta_vehicles=-1.0, delta_distance_pct=1.0,
                             runtime_s=3.0, **{f"realloc_{f}": value for f in phase_c.REALLOC_FIELDS}))
    return pd.DataFrame(rows)


def test_reopt_table_and_plot():
    removal, insertion = reopt_cells(0.1), reopt_cells(0.2)
    table = phase_c_view.reopt_table(removal, insertion, "en")
    assert table["variant"].tolist()[:3] == ["Warm start", "Cold start", "Noise floor"]
    warm = table[(table["event"] == "Remove 10%") & (table["variant"] == "Warm start")].iloc[0]
    assert warm["reassigned_pct"] == 10.0 and warm["delta_vehicles_mean"] == -1.0
    noise = table[(table["event"] == "Insert 10%") & (table["variant"] == "Noise floor")].iloc[0]
    assert pd.isna(noise["delta_vehicles_mean"])
    fig = viz_robustness.plot_reoptimisation(removal, insertion, "en")
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_robustness():
    summary = robustness_summary()
    bins = pd.DataFrame([{"scenario": "travel_0.10", "method": m, "slack_bin": b, "n_customers": 10, "late_rate": 0.1}
                         for m in METHODS for b in phase_c_view.SLACK_ORDER])
    fig = viz_robustness.plot_robustness(summary, bins, "pt")
    assert len(fig.axes) == 2 and fig.axes[0].get_title().startswith("(a)")
    plt.close(fig)


def test_config_table_and_block_overview():
    from delivery_routing_vrptw.config import ExperimentConfig

    table = phase_c_view.config_table(ExperimentConfig(), "en")
    values = dict(zip(table["metric"], table["value"], strict=True))
    assert values["Integer scale and relative fixed cost"] == "10000, k = 10"
    assert values["Budget per run (s) · replicates · processes"] == "30 · 3 · 8"
    assert values["C2.2: limited fleet · priority share · priority penalty"] == "K − 1, K − 2 · 20% · 10×"
    assert values["C2.1: capacity (fraction of Q)"] == "100%, 90%, 80%"
    blocks = {"capacity": capacity_cells().assign(runtime_s=2.0), "removal": reopt_cells(0.1).assign(runtime_s=4.0)}
    out = phase_c_view.block_overview(blocks, "pt").set_index("block")
    assert out.loc["C2.1 · capacidade reduzida", "n_cells"] == 18 and out.loc["C2.1 · capacidade reduzida", "n_feasible"] == 15
    assert out.loc["C3 · remoção de 10%", "mean_runtime"] == 4.0
