import numpy as np
import pandas as pd
import pytest
from test_results import SPLIT, make_grid_frame

from delivery_routing_vrptw import Reference, analysis, results, viz_results
from delivery_routing_vrptw.eda import localize_columns

CFG = {"fleet_strategy": "fixed_cost", "first_solution": "SAVINGS", "metaheuristic": "GUIDED_LOCAL_SEARCH"}


@pytest.fixture
def grid(tmp_path):
    path = tmp_path / "results.csv"
    make_grid_frame().to_csv(path, index=False)
    return results.load_dev_grid(path, SPLIT)


@pytest.fixture
def live(inst):
    refs = {"R101": Reference("R101", 3, 100.0, "test")}
    return analysis.run_baselines({"R101": inst}, refs, ["R101"])


def test_run_baselines_and_tables(live):
    assert set(live["method"]) == {"greedy_deadline", "nearest_neighbor_tw"}
    assert live["feasible"].all() and (live["delta_vehicles"] == 0).all()
    wide = analysis.baseline_table(live, "en")
    assert len(wide) == 1 and "Deadline-first greedy · Vehicles" in wide.columns
    pt = analysis.baseline_table(live, "pt")
    assert "Gulosa por prazo · Veículos" in pt.columns
    summary = analysis.baseline_family_summary(live, "en")
    assert summary["family"].tolist() == ["R1", "Total"]
    assert summary["Deadline-first greedy · Feasible"].tolist() == [1, 1]


def test_baselines_match_grid(live):
    grid = pd.DataFrame({"instance": ["R101"] * 2, "method": ["greedy_deadline", "nearest_neighbor_tw"],
                         "vehicles": [3, 3], "distance": live["distance"].to_numpy()})
    assert analysis.baselines_match_grid(live, grid) == (2, 2)
    grid.loc[0, "vehicles"] = 9
    assert analysis.baselines_match_grid(live, grid) == (1, 2)


def test_construction_comparison(grid):
    comp = analysis.construction_comparison(grid, "en")
    assert comp["construction"].tolist() == ["greedy_deadline", "nearest_neighbor_tw", "PATH_CHEAPEST_ARC",
                                             "SAVINGS"]
    saving = comp.set_index("construction").loc["SAVINGS"]
    assert saving["mean_delta_vehicles"] == 1.0 and saving["n_fleet_equal_ref"] == 0
    assert np.isnan(saving["median_gap_equal_fleet"])
    shown = analysis.construction_table(comp)
    assert "construction" not in shown.columns and shown["median_gap_equal_fleet"].iloc[0] == "-"
    fig = viz_results.plot_construction_comparison(comp, "pt")
    assert len(fig.axes) == 2


def test_grid_coverage(grid):
    cov = analysis.grid_coverage(grid, SPLIT, "en").set_index("metric")["value"]
    assert cov["Cells in the grid (total)"] == len(grid)
    assert cov["Instances outside the development set"] == 0
    assert cov["Cells with partial solver status"] == cov["Of which, construction without local search"] == 4
    assert cov["Distinct configurations in the tournament"] == 5


def test_ranking_table_and_heatmap():
    ranking = pd.DataFrame({
        "fleet_strategy": ["fixed_cost", "fixed_cost", "reduce_fleet"],
        "first_solution": ["SAVINGS", "PATH_CHEAPEST_ARC", "SAVINGS"],
        "metaheuristic": ["GUIDED_LOCAL_SEARCH", "TABU_SEARCH", "GUIDED_LOCAL_SEARCH"],
        "mean_rank": [1.23456, 2.0, 3.0], "mean_delta_vehicles": [0.5, 1.0, 1.0],
        "n_fleet_equal_ref": [2, 1, 1], "median_gap_equal_fleet": [1.0, 2.0, 3.0],
        "mean_runtime": [30.04, 20.0, 10.0], "n_instances": [2, 2, 2],
    })
    table = analysis.ranking_table(ranking)
    assert table["position"].tolist() == [1, 2, 3] and table["mean_rank"].iloc[0] == 1.235
    assert "n_instances" not in table.columns
    fig = viz_results.plot_rank_heatmap(ranking, "en")
    assert len(fig.axes) == 3  # two panels and the colour bar
    assert fig.axes[0].get_xlabel() == "Metaheuristic"


def test_frozen_vs_reference(grid):
    table = analysis.frozen_vs_reference(grid, CFG)
    assert table["instance"].tolist() == ["R101", "R102"]
    assert (table["delta_vs_ref"] == 0).all()
    assert table["gap_distance_pct"].tolist() == [5.0, -11.67]
    assert (table["best_vehicles"] <= table["vehicles"]).all()
    summary = analysis.frozen_family_summary(table, "pt")
    assert summary["family"].tolist() == ["R1", "Total"]
    assert summary["n_fleet_equal_ref"].tolist() == [2, 2]
    shown = localize_columns(table, "pt")
    assert "Δ veículos (vs SINTEF)" in shown.columns


def test_frozen_gap_is_dash_when_fleet_differs(grid):
    cfg = {"fleet_strategy": "fixed_cost", "first_solution": "PATH_CHEAPEST_ARC",
           "metaheuristic": "GUIDED_LOCAL_SEARCH"}
    table = analysis.frozen_vs_reference(grid, cfg)
    assert table["gap_distance_pct"].tolist()[1] == "-"
    summary = analysis.frozen_family_summary(table, "en")
    assert summary["n_fleet_equal_ref"].tolist() == [1, 1]
    none_equal = analysis.frozen_family_summary(table[table["instance"] == "R102"], "en")
    assert none_equal["median_gap_equal_fleet"].iloc[0] == "-"


def test_best_of_tournament(grid):
    best = analysis.best_of_tournament(grid)
    assert best["instance"].tolist() == ["R101", "R102"]
    assert best["metaheuristic"].tolist() == ["GUIDED_LOCAL_SEARCH"] * 2


def test_local_search_effect(grid):
    eff = analysis.local_search_effect(grid, CFG).set_index("first_solution")
    assert eff.loc["SAVINGS", "vehicles_saved_mean"] == 1.0 and eff.loc["SAVINGS", "n_better"] == 2
    assert eff.loc["PATH_CHEAPEST_ARC", "n_better"] == 2
    assert eff.loc["PATH_CHEAPEST_ARC", "delta_search"] == 0.5


def test_limitations_table(grid):
    lim = analysis.limitations_table(grid, "en").set_index("metric")["value"]
    assert lim["Cells where the first fleet reduction found no solution"] == "2 / 2"
    assert lim["Of which, stopped before 29 s"] == "4 / 4"
    assert lim["Replicates per cell (maximum)"] == "1"
    assert analysis.limitations_table(grid, "pt").shape == (8, 2)


def test_route_panels(inst):
    fig = viz_results.plot_route_panels(
        [(inst, "a", [[1, 2], [3, 4], [5]]), (inst, "b", [[1], [2], [3, 4, 5]])], "pt")
    assert len(fig.axes) == 2 and "3 veículos" in fig.axes[0].get_title()


def test_dev_instance_table_and_selected_config(inst):
    refs = {"R101": Reference("R101", 3, 100.0, "test")}
    table = analysis.dev_instance_table({"R101": inst}, refs)
    assert table.loc[0, "n_customers"] == 5 and table.loc[0, "ref_vehicles"] == 3
    selected = {
        "config": CFG,
        "selection_stats": {"mean_rank": 1.23456, "mean_delta_vehicles": 0.5, "n_fleet_equal_ref": 2.0,
                            "median_gap_equal_fleet": 1.5, "mean_runtime": 30.04, "n_instances": 2.0},
    }
    cfg = analysis.selected_config_table(selected, "en").set_index("metric")["value"]
    assert cfg["Fleet strategy"] == "fixed_cost" and cfg["Mean rank"] == 1.235
    assert cfg["No. of instances"] == 2


def test_baseline_routes_checks_and_panels(inst, live, grid):
    routes = analysis.baseline_routes(live, "R101", "greedy_deadline")
    assert sorted(c for r in routes for c in r) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="no unique row"):
        analysis.baseline_routes(live, "R101", "missing")
    checks = analysis.baseline_checks(live, grid.assign(vehicles=3, distance=0.0), "en")
    assert checks["value"].tolist() == ["2 / 2", "0 / 2"]
    panels = analysis.baseline_panels("R101", {"R101": inst}, live, grid, "en")
    assert [p[1] for p in panels][0] == "Deadline-first greedy" and len(panels) == 3
    fro = analysis.frozen_panels(["R101"], {"R101": inst}, grid, CFG, "pt")
    assert fro[0][1] == "Configuração congelada (fixed_cost / SAVINGS / GUIDED_LOCAL_SEARCH)"
    assert fro[0][2] == [[1], [2], [3]]



def test_time_curve_tables_and_plot():
    from test_results import make_time_frame

    df = make_time_frame()
    curve = analysis.time_curve_table(df)
    assert curve["budget_s"].tolist() == [10.0, 60.0]
    assert curve["mean_delta_vehicles"].tolist() == [0.5, 0.0]
    assert curve["n_fleet_equal_ref"].tolist() == [1, 2]
    assert curve["median_gap_equal_fleet"].iloc[0] == pytest.approx(5.0)
    assert curve["median_gap_equal_fleet"].iloc[1] == pytest.approx(2.25)
    assert curve["mean_runtime"].tolist() == [10.0, 60.0]
    shown = analysis.time_curve_table_display(curve)
    assert shown["budget_s"].tolist() == ["10", "60"]
    only_gap_undefined = analysis.time_curve_table_display(curve.assign(median_gap_equal_fleet=np.nan))
    assert (only_gap_undefined["median_gap_equal_fleet"] == "-").all()
    by_inst = analysis.time_curve_by_instance(df)
    assert by_inst.columns.tolist() == ["instance", "family", "ref_vehicles", "10 s", "60 s"]
    assert by_inst.set_index("instance").loc["R101", ["10 s", "60 s"]].tolist() == [4, 3]
    for lang in ("pt", "en"):
        fig = viz_results.plot_time_curve(df, lang)
        assert len(fig.axes) == 2


def test_time_curve_offsets_keep_lines_distinct():
    import numpy as np

    from delivery_routing_vrptw import viz_results as vr

    rows = []
    for inst in ("R101", "R102", "R103"):  # identical trajectories would coincide without offsets
        for budget in (10.0, 60.0):
            rows.append({"instance": inst, "family": "R1", "budget_s": budget, "vehicles": 4,
                         "ref_vehicles": 3, "delta_vehicles": 1})
    fig = vr.plot_time_curve(pd.DataFrame(rows), "en")
    ys = [line.get_ydata()[0] for line in fig.axes[1].get_lines() if line.get_linestyle() != "--"]
    assert len(set(np.round(ys, 6))) == 3
    assert max(abs(np.array(ys) - 4)) <= 0.15


def _reduce_grid(runtimes):
    rows = []
    for first, values in runtimes.items():
        for k, t in enumerate(values):
            rows.append({"tournament": True, "fleet_strategy": "reduce_fleet", "first_solution": first,
                         "runtime_s": t, "instance": f"X{k}"})
    rows.append({"tournament": True, "fleet_strategy": "fixed_cost", "first_solution": "SAVINGS",
                 "runtime_s": 30.0, "instance": "X0"})
    rows.append({"tournament": False, "fleet_strategy": "reduce_fleet", "first_solution": "SAVINGS",
                 "runtime_s": 99.0, "instance": "X0"})
    return pd.DataFrame(rows)


def test_reduce_fleet_timing():
    grid = _reduce_grid({"SAVINGS": [1.08, 10.2, 10.5, 12.0], "PATH_CHEAPEST_ARC": [15.0, 15.04, 9.6]})
    out = analysis.reduce_fleet_timing(grid).set_index("first_solution")
    sav, pca = out.loc["SAVINGS"], out.loc["PATH_CHEAPEST_ARC"]
    assert sav["n_cells"] == 4 and sav["runtime_min"] == 1.08 and sav["runtime_max"] == 12.0
    assert sav["runtime_median"] == pytest.approx(10.35)
    assert sav["n_fast_fail"] == 3 and sav["n_step_used"] == 0  # 12.0 is above 11 s but not a used step
    assert sav["budget_used_pct"] == pytest.approx(34.5)
    assert pca["n_fast_fail"] == 1 and pca["n_step_used"] == 2
    assert analysis.reduce_fleet_timing(grid, first_step_s=5, step_s=5, budget_s=20)["n_step_used"].tolist() == [
        3, 3,
    ]
    shown = localize_columns(out.reset_index(), "pt")
    assert "Tempo mediano (s)" in shown.columns
