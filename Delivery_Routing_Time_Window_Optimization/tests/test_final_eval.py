import math

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delivery_routing_vrptw import final_eval as fe
from delivery_routing_vrptw import viz_final
from delivery_routing_vrptw.config import ExperimentConfig
from delivery_routing_vrptw.evaluation import instance_table

SPLIT = {"C101": "eval", "C102": "eval", "R101": "eval", "RC101": "eval", "R201": "dev"}
FAMILY = {"C101": "C1", "C102": "C1", "R101": "R1", "RC101": "RC1"}
REF = {"C101": (10, 100.0), "C102": (10, 100.0), "R101": (10, 190.0), "RC101": (10, 280.0)}


def make_row(instance, method, replicate, vehicles, distance):
    ref_v, ref_d = REF[instance]
    delta = vehicles - ref_v
    return {
        "instance": instance, "family": FAMILY[instance], "method": method, "replicate": replicate,
        "feasible": True, "vehicles": vehicles, "distance": distance, "runtime_s": 120.0 + replicate / 100,
        "status": "ROUTING_SUCCESS" if method == "ortools" else None, "ref_vehicles": ref_v,
        "ref_distance": ref_d, "delta_vehicles": delta, "gap_distance_pct": 100 * (distance - ref_d) / ref_d,
        "distance_comparable": delta == 0, "beats_reference": delta < 0,
        "routes": [[1, 2], [3], [4, 5]],
    }


def make_rows():
    reps = {
        "C101": [(10, 100.0)] * 5,
        "C102": [(10, 110.0)] * 5,
        "R101": [(11, 200.0), (11, 200.0), (12, 210.0), (11, 205.0), (11, 200.0)],
        "RC101": [(12, 300.0), (12, 300.0), (12, 300.0), (12, 310.0), (12, 300.0)],
    }
    rows = []
    for inst, items in reps.items():
        rows += [make_row(inst, "ortools", i + 1, v, d) for i, (v, d) in enumerate(items)]
        rows.append(make_row(inst, "nearest_neighbor_tw", 1, 13, 400.0))
        rows.append(make_row(inst, "greedy_deadline", 1, 15, 500.0))
    return pd.DataFrame(rows)


@pytest.fixture
def rows():
    return make_rows()


@pytest.fixture
def table(rows):
    return instance_table(rows)


def test_reproduces_reference(table):
    flags = dict(zip(table["instance"], fe.reproduces_reference(table), strict=True))
    assert flags == {"C101": True, "C102": False, "R101": False, "RC101": False}


def test_protocol_table():
    selected = {"config": {"fleet_strategy": "fixed_cost", "first_solution": "SAVINGS",
                           "metaheuristic": "GUIDED_LOCAL_SEARCH"}}
    out = fe.protocol_table(ExperimentConfig(), selected, "pt")
    values = dict(zip(out["metric"], out["value"], strict=True))
    assert values["Réplicas por instância"] == "5"
    assert values["Orçamento da avaliação (s por execução)"] == "120"
    assert "posição 3 de 5" in values["Resultado principal"]


def test_integrity_table(rows):
    split = {**SPLIT, "R999": "eval"}
    out = fe.integrity_table(rows, split, "en")
    values = dict(zip(out["metric"], out["value"], strict=True))
    assert values["Cells in the file (total)"] == "28"
    assert values["OR-Tools cells (instances × replicates)"] == "20"
    assert values["Baseline cells"] == "8"
    assert values["Evaluation instances in the split"] == "5"
    assert values["Development instances in the results"] == "0"
    assert values["Cells feasible in the verifier"] == "28 / 28"
    assert values["OR-Tools replicates per instance"] == "5"


def test_main_table(rows):
    main = fe.main_table(rows, "en")
    assert list(main["scope"]) == ["Overall", "C1", "R1", "RC1"]
    overall = main.iloc[0]
    assert overall["n_instances"] == 4 and overall["mean_delta_vehicles"] == pytest.approx(0.75)
    assert overall["n_fleet_equal_ref"] == 2 and overall["n_beats_reference"] == 0
    assert overall["median_gap_equal_fleet"] == pytest.approx(5.0)
    assert main[main["scope"] == "RC1"]["median_gap_equal_fleet"].iloc[0] == "-"
    best = fe.main_table(rows, "pt", best=True)
    assert best["result_kind"].iloc[0] == "Melhor de 5 (à parte)"
    assert best["mean_delta_vehicles"].iloc[0] == pytest.approx(0.75)


def test_delta_distribution_and_instances(table):
    dist = fe.delta_distribution(table, "en")
    first = dist.iloc[0]
    assert (first["n_delta0"], first["n_delta1"], first["n_delta2"], first["n_below"]) == (2, 1, 1, 0)
    counts = fe.delta_counts(table)
    assert list(counts.index) == ["C1", "R1", "RC1"] and counts.loc["C1", "0"] == 2
    listed = fe.delta_instances(table, 2)
    assert list(listed["instance"]) == ["RC101"] and "delta_vs_ref" in listed.columns


def test_fleet_equal_tables(table):
    eq = fe.fleet_equal_table(table, "en")
    assert list(eq["instance"]) == ["C101", "C102"]
    assert list(eq["reproduces"]) == ["Yes", "No"]
    assert eq["gap_distance_pct"].tolist() == [0.0, 10.0]
    summ = fe.reproduction_summary(table, "en")
    overall = summ.iloc[0]
    assert (overall["n_fleet_equal_ref"], overall["n_reproduced"], overall["n_above"]) == (2, 1, 1)
    assert overall["max_gap"] == pytest.approx(10.0)
    assert summ[summ["scope"] == "R1"]["max_gap"].iloc[0] == "-"


def test_variability(table):
    var = fe.variability_table(table, "en")
    overall = var.iloc[0]
    assert overall["n_vehicles_vary"] == 1 and overall["max_vehicles_range"] == 1
    assert overall["n_distance_vary"] == 2
    listed = fe.varying_instances(table)
    assert list(listed["instance"]) == ["R101", "RC101"]
    assert listed["distance_range_pct"].iloc[0] == pytest.approx(5.0)


def test_method_comparison(rows):
    comp = fe.method_comparison(rows, "en")
    assert list(comp["method"])[0] == "OR-Tools"
    assert list(comp["method"])[1:] == ["Nearest neighbour with time windows", "Deadline-first greedy"]
    assert comp["mean_delta_vehicles"].tolist() == [0.75, 3.0, 5.0]
    wide = fe.method_by_family(rows, "pt")
    assert list(wide.columns) == ["method", "Geral", "C1", "R1", "RC1"]
    assert wide["Geral"].tolist() == [0.75, 3.0, 5.0]


def test_dev_vs_eval(table):
    time_df = pd.DataFrame({
        "instance": ["R201", "R201"], "family": ["R2", "R2"], "budget_s": [30.0, 120.0],
        "delta_vehicles": [1, 0], "distance_comparable": [False, True], "gap_distance_pct": [math.nan, 2.0],
    })
    out = fe.dev_vs_eval(time_df, table, 120.0, "en")
    assert list(out["set"]) == ["Development", "Evaluation"]
    assert list(out["fleet_equal_share"]) == ["1 / 1", "2 / 4"]
    assert list(out["n_runs"]) == [1, 5]
    fam = fe.dev_vs_eval_by_family(time_df, table, 120.0, "en")
    assert set(fam["family"]) == {"C1", "R1", "R2", "RC1"}
    assert fam[fam["family"] == "R2"].iloc[0]["Development · Mean Δ vehicles"] == 0.0


def test_map_picks_and_panels(rows, table, inst):
    assert fe.pick_map_instances(table) == ["C101", "RC101"]
    panels = fe.median_panels(["C101"], {"C101": inst}, rows, table, "en")
    instance, label, routes = panels[0]
    assert instance is inst and routes == [[1, 2], [3], [4, 5]] and "SINTEF: 10" in label
    with pytest.raises(ValueError):
        fe.pick_map_instances(table[table["family"] == "R1"])


def test_plots(table):
    fig = viz_final.plot_delta_distribution(table, "pt")
    assert fig.axes[0].get_title().startswith("Δ")
    fig = viz_final.plot_gap_fleet_equal(table, "en")
    assert len(fig.axes[0].patches) == 2
    plt.close("all")


# ------------------------------------------------------------------ time metrics derived from frozen routes
def _time_rows(inst):
    from delivery_routing_vrptw import verify
    plans = {
        ("ortools", 1): [[1, 2], [3], [4, 5]],
        ("ortools", 2): [[1], [2], [3], [4], [5]],
        ("ortools", 3): [[1, 2], [3, 4], [5]],
        ("greedy_deadline", 1): [[1], [2], [3], [4], [5]],
    }
    out = []
    for (method, rep), routes in plans.items():
        r = verify(inst, routes)
        out.append({"instance": "R101", "family": "R1", "method": method, "replicate": rep,
                    "feasible": r.feasible, "vehicles": r.vehicles, "distance": r.distance, "routes": routes})
    return pd.DataFrame(out)


def test_time_metrics_table_matches_schedule_metrics(inst):
    from delivery_routing_vrptw.route_compare import schedule_metrics
    rows = _time_rows(inst)
    tm = fe.time_metrics_table(rows, {"R101": inst})
    assert len(tm) == 4 and (tm["distance_diff"] == 0).all()
    first = tm.iloc[0]
    expected = schedule_metrics(inst, rows.iloc[0]["routes"])
    for key in fe.TIME_METRICS:
        assert first[key] == pytest.approx(expected[key])
    assert (tm["total_duration_min"] <= tm["total_duration"] + 1e-9).all()
    assert (tm["total_waiting_min"] <= tm["total_waiting"] + 1e-9).all()
    # duration = distance + total service + waiting (travel time equals distance in Solomon instances)
    service = float(inst.service.sum())
    assert (tm["total_duration"] - tm["distance"] - tm["total_waiting"]).round(6).eq(service).all()


def test_time_representatives_and_aggregate(inst):
    tm = fe.time_metrics_table(_time_rows(inst), {"R101": inst})
    rep = fe.time_representatives(tm)
    assert sorted(rep["method"]) == ["greedy_deadline", "ortools"]
    sel = rep[rep["method"] == "ortools"].iloc[0]
    # same rule as the evaluation: infeasible replicates last, ceil(3/2) = the 2nd replicate of the order
    from delivery_routing_vrptw.evaluation import sort_replicates
    order = sort_replicates(tm[tm["method"] == "ortools"])["replicate"].tolist()
    assert sel["replicate"] == order[1]
    agg = fe.time_aggregate(rep)
    assert set(agg["scope"]) == {"overall", "R1"}
    base = agg[(agg["scope"] == "overall") & (agg["method"] == "ortools")].iloc[0]
    assert base["ratio_distance_pct"] == 100.0 and base["ratio_duration_min_pct"] == 100.0
    other = agg[(agg["scope"] == "overall") & (agg["method"] == "greedy_deadline")].iloc[0]
    assert other["ratio_distance_pct"] == pytest.approx(100 * other["dist_median"] / base["dist_median"])
    for lang in ("pt", "en"):
        assert len(fe.time_method_table(agg, lang)) == 2
        assert fe.time_family_table(agg)["family"].tolist() == ["R1"]
        assert fe.time_ratio_table(agg, lang)["method"].nunique() == 1
        fig = viz_final.plot_time_metrics(agg, lang)
        assert len(fig.axes) == 2
    plt.close("all")


def test_load_eval_time_metrics(tmp_path, inst):
    from delivery_routing_vrptw import results
    tm = fe.time_metrics_table(_time_rows(inst), {"R101": inst})
    path = tmp_path / "time_metrics.csv"
    tm.to_csv(path, index=False)
    out = results.load_eval_time_metrics(path, {"R101": "eval"})
    assert out["feasible"].dtype == bool and len(out) == 4
    with pytest.raises(ValueError):
        results.load_eval_time_metrics(path, {"R101": "dev"})


# --------------------------------------------------------------------------- v1 against v2 (post-hoc)
def make_v2_rows():
    """Like ``make_rows`` but R101 gains a vehicle-free change (-1 vehicle) and C102 / RC101 move in distance."""
    reps = {
        "C101": [(10, 100.0)] * 5,
        "C102": [(10, 100.0)] * 5,                   # -9.1% against v1 (110.0): beyond the noise threshold
        "R101": [(10, 190.0)] * 5,                   # one vehicle fewer: distance not comparable
        "RC101": [(12, 300.2)] * 5,                  # +0.07%: within the noise
    }
    rows = []
    for inst, items in reps.items():
        rows += [make_row(inst, "ortools", i + 1, v, d) for i, (v, d) in enumerate(items)]
    return pd.DataFrame(rows)


def test_version_tables(rows):
    from delivery_routing_vrptw.evaluation import compare_versions

    by_instance, aggregate = compare_versions(rows, make_v2_rows())
    table = fe.version_aggregate_table(aggregate, "en")
    overall = table.iloc[0]
    assert overall["scope"] == "Overall" and overall["n_instances"] == 4
    assert overall["v1_n_fleet_equal"] == 2 and overall["v2_n_fleet_equal"] == 3
    assert table[table["scope"] == "R1"].iloc[0]["v2_mean_delta_vehicles"] == 0.0
    summary = fe.version_distance_summary(by_instance, "en").set_index("item")
    assert summary.loc["Smaller fleet in v2", "count"] == 1 and summary.loc["Smaller fleet in v2", "detail"] == "R101"
    assert summary.loc["Larger fleet in v2", "count"] == 0
    assert summary.loc["Equal fleet in both versions", "count"] == 3
    assert summary.loc["Distance within noise (equal fleet)", "count"] == 2
    assert summary.loc["Shorter distance in v2, beyond noise", "count"] == 1
    assert summary.loc["Longer distance in v2, beyond noise", "count"] == 0
    changes = fe.version_changes(by_instance, "en")
    assert list(changes["instance"]) == ["R101", "C102"]  # fleet changes first, then by distance difference
    assert changes.iloc[0]["version_distance_diff_pct"] == "-"
    fig = viz_final.plot_version_comparison(by_instance, "pt")
    assert fig.axes[0].get_ylabel().startswith("Distância v2")
    plt.close(fig)


def test_fixed_cost_ratio_table_and_summary(inst):
    from delivery_routing_vrptw import solver

    d_ub = solver.distance_upper_bound(inst)
    table = fe.fixed_cost_ratio_table({"C101": inst, "R101": inst})
    assert list(table["instance"]) == ["C101", "R101"]
    assert table["fixed_cost_v1"].eq(1e5).all()  # 10^7 at scale 100
    assert table["fixed_cost_v2"].iloc[0] == pytest.approx(10 * d_ub, abs=1e-4)  # k * D_ub distance units
    assert table["ratio_pct"].iloc[0] == pytest.approx(100 * 10 * d_ub / 1e5, rel=1e-6)
    summary = fe.fixed_cost_ratio_summary(table, "en")
    assert summary.iloc[0]["n_instances"] == 2
    assert summary.iloc[0]["ratio_min_pct"] == summary.iloc[0]["ratio_max_pct"] == round(100 * 10 * d_ub / 1e5, 1)
    assert fe.fixed_cost_ratio_summary(table, "pt").shape == summary.shape


def test_evaluation_fixed_cost_table_uses_the_evaluation_loader(tmp_path, text):
    import zipfile

    archive = tmp_path / "s.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ("R101", "R102"):
            zf.writestr(f"{name.lower()}.txt", text.replace("R101", name))
    split = {"R101": "dev", "R102": "eval"}
    table = fe.evaluation_fixed_cost_table(archive, split, ["R102"], 10000, 10.0)
    assert list(table["instance"]) == ["R102"] and table["ratio_pct"].iloc[0] > 0
    with pytest.raises(ValueError, match="R101"):
        fe.evaluation_fixed_cost_table(archive, split, ["R101"])
