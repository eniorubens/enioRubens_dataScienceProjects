import json

import pandas as pd
import pytest

from delivery_routing_vrptw import paths, results

SPLIT = {"R101": "dev", "R102": "dev", "R103": "eval"}
COLS = ["instance", "fleet_strategy", "first_solution", "metaheuristic", "tournament", "method", "family",
        "feasible", "vehicles", "distance", "runtime_s", "status", "attempts", "ref_vehicles", "ref_distance",
        "delta_vehicles", "gap_distance_pct", "distance_comparable", "routes"]


def make_grid_frame() -> pd.DataFrame:
    """Return a tiny synthetic grid (two dev instances, a few configurations and reference cells)."""
    ref = {"R101": (3, 100.0), "R102": (3, 120.0)}

    def row(inst, fleet, first, meta, tour, method, veh, dist, rt, status="ROUTING_SUCCESS", attempts=None):
        rv, rd = ref[inst]
        return {
            "instance": inst, "fleet_strategy": fleet, "first_solution": first, "metaheuristic": meta,
            "tournament": tour, "method": method, "family": "R1", "feasible": True, "vehicles": veh,
            "distance": dist, "runtime_s": rt, "status": status, "attempts": attempts, "ref_vehicles": rv,
            "ref_distance": rd, "delta_vehicles": veh - rv, "gap_distance_pct": 100 * (dist - rd) / rd,
            "distance_comparable": veh == rv, "routes": json.dumps([[i] for i in range(1, veh + 1)]),
        }

    rows = []
    for inst, bump in (("R101", 0), ("R102", 1)):
        for first in ("SAVINGS", "PATH_CHEAPEST_ARC"):
            veh = 4 if (first == "PATH_CHEAPEST_ARC" and inst == "R102") else 3
            for meta, plus in (("GUIDED_LOCAL_SEARCH", 0.0), ("TABU_SEARCH", 2.0)):
                rows.append(row(inst, "fixed_cost", first, meta, True, "ortools", veh,
                                105.0 + plus + bump + (1.0 if first == "PATH_CHEAPEST_ARC" else 0.0),
                                30.0 if meta == "GUIDED_LOCAL_SEARCH" else 12.0))
        rows.append(row(inst, "reduce_fleet", "SAVINGS", "GUIDED_LOCAL_SEARCH", True, "ortools_minimize_fleet", 3,
                        110.0 + bump, 10.0, attempts=json.dumps([[5, True, 3, 110.0], [2, False, 0, None]])))
        for first, veh in (("SAVINGS", 4), ("PATH_CHEAPEST_ARC", 5)):
            rows.append(row(inst, "fixed_cost", first, "NONE", False, "ortools", veh, 150.0, 0.1,
                            status="ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED"))
        rows.append(row(inst, "none", "none", "none", False, "greedy_deadline", 5, 300.0, 0.001, status=None))
        rows.append(row(inst, "none", "none", "none", False, "nearest_neighbor_tw", 4, 200.0, 0.01, status=None))
    return pd.DataFrame(rows, columns=COLS)


@pytest.fixture
def grid_csv(tmp_path):
    path = tmp_path / "results.csv"
    make_grid_frame().to_csv(path, index=False)
    return path


def test_load_dev_grid_parses_json(grid_csv):
    grid = results.load_dev_grid(grid_csv, SPLIT)
    assert isinstance(grid["routes"].iloc[0], list)
    reduce = grid[grid["fleet_strategy"] == "reduce_fleet"]
    assert reduce["attempts"].iloc[0][1][1] is False
    assert grid.loc[grid["fleet_strategy"] == "fixed_cost", "attempts"].isna().all()
    assert grid["tournament"].dtype == bool


def test_load_dev_grid_rejects_eval(grid_csv):
    with pytest.raises(ValueError, match="R101"):
        results.load_dev_grid(grid_csv, {"R101": "eval", "R102": "dev"})
    with pytest.raises(ValueError, match="R102"):
        results.load_dev_grid(grid_csv, {"R101": "dev"})


def test_load_gate(tmp_path):
    path = tmp_path / "gate.csv"
    pd.DataFrame({"instance": ["R101", "R102"], "method": ["a", "b"]}).to_csv(path, index=False)
    assert len(results.load_gate(path, SPLIT)) == 2
    pd.DataFrame({"instance": ["R103"], "method": ["a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="R103"):
        results.load_gate(path, SPLIT)


def test_load_ranking_and_config(tmp_path):
    rpath = tmp_path / "ranking.csv"
    pd.DataFrame({"mean_rank": [1.0]}).to_csv(rpath, index=False)
    assert results.load_ranking(rpath)["mean_rank"].iloc[0] == 1.0
    cpath = tmp_path / "cfg.json"
    cpath.write_text(json.dumps({"config": {"a": "b"}, "instances": ["R101"]}), encoding="utf-8")
    assert results.load_selected_config(cpath)["config"] == {"a": "b"}
    assert results.load_selected_config(cpath, SPLIT)["instances"] == ["R101"]
    cpath.write_text(json.dumps({"config": {}, "instances": ["R103"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        results.load_selected_config(cpath, SPLIT)


def test_get_routes(grid_csv):
    grid = results.load_dev_grid(grid_csv, SPLIT)
    routes = results.get_routes(grid, "R101", method="ortools", first_solution="SAVINGS", metaheuristic="NONE")
    assert routes == [[1], [2], [3], [4]]
    with pytest.raises(ValueError, match="matches 0"):
        results.get_routes(grid, "R101", method="nothing")
    with pytest.raises(ValueError, match="matches 2"):
        results.get_routes(grid, "R101", method="ortools", metaheuristic="NONE")


def test_result_paths(tmp_path):
    p = paths.result_paths(tmp_path)
    assert p["dev_grid"] == tmp_path / "outputs" / "frozen" / "dev_grid" / "results.csv"
    assert p["gate"].name == "gate_results.csv" and p["selected_config"].name == "selected_config.json"


def test_dev_names_and_load_dev_instances(tmp_path, text):
    import zipfile

    assert results.dev_names({"C101": "eval", "R102": "dev", "R101": "dev"}) == ["R101", "R102"]
    archive = tmp_path / "s.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("r101.txt", text)
        zf.writestr("r102.txt", text.replace("R101", "R102"))
    refs = tmp_path / "refs.csv"
    refs.write_text("instance,vehicles,distance,reference,note\nR101,3,100.0,t,\nR102,3,110.0,t,\n",
                    encoding="utf-8")
    instances, references = results.load_dev_instances(archive, refs, {"R101": "dev", "R102": "eval"})
    assert list(instances) == ["R101"] and list(references) == ["R101"]


def make_time_frame() -> pd.DataFrame:
    """Return a synthetic time-sensitivity frame: two dev instances, budgets 10 and 60."""
    rows = []
    for inst, family, ref_v, ref_d, by_budget in (
        ("R101", "R1", 3, 100.0, {10.0: (4, 90.0), 60.0: (3, 102.0)}),
        ("R102", "R1", 3, 120.0, {10.0: (3, 126.0), 60.0: (3, 123.0)}),
    ):
        for budget, (veh, dist) in by_budget.items():
            rows.append({
                "instance": inst, "family": family, "budget_s": budget, "feasible": True, "vehicles": veh,
                "distance": dist, "runtime_s": budget, "status": "ROUTING_SUCCESS", "ref_vehicles": ref_v,
                "ref_distance": ref_d, "delta_vehicles": veh - ref_v,
                "gap_distance_pct": 100 * (dist - ref_d) / ref_d, "distance_comparable": veh == ref_v,
                "routes": json.dumps([[i] for i in range(1, veh + 1)]),
            })
    return pd.DataFrame(rows)


def test_load_time_sensitivity(tmp_path):
    path = tmp_path / "time.csv"
    make_time_frame().to_csv(path, index=False)
    df = results.load_time_sensitivity(path, SPLIT)
    assert len(df) == 4 and isinstance(df["routes"].iloc[0], list)
    with pytest.raises(ValueError, match="R102"):
        results.load_time_sensitivity(path, {"R101": "dev", "R102": "eval"})


def test_result_paths_time_sensitivity(tmp_path):
    p = paths.result_paths(tmp_path)
    assert p["time_sensitivity"] == tmp_path / "outputs" / "frozen" / "time_sensitivity" / "results.csv"


def test_eval_loaders_and_paths(tmp_path, text):
    import zipfile

    split = {"R101": "dev", "R102": "eval", "R103": "eval"}
    frame = pd.DataFrame({"instance": ["R102", "R103"], "routes": [json.dumps([[1]]), json.dumps([[1, 2]])]})
    path = tmp_path / "eval.csv"
    frame.to_csv(path, index=False)
    loaded = results.load_evaluation(path, split)
    assert loaded["routes"].iloc[1] == [[1, 2]]
    frame.assign(instance=["R102", "R101"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="R101"):
        results.load_evaluation(path, split)
    with pytest.raises(ValueError, match="R102"):
        results.assert_eval_only(["R102"], {"R102": "dev"})

    archive = tmp_path / "s.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ("R101", "R102"):
            zf.writestr(f"{name.lower()}.txt", text.replace("R101", name))
    assert list(results.load_eval_instances(archive, split, ["R102"])) == ["R102"]
    with pytest.raises(ValueError, match="R101"):
        results.load_eval_instances(archive, split, ["R101"])
    assert list(results.load_dev_inputs(archive, split)) == ["R101"]

    p = paths.result_paths(tmp_path)
    assert p["evaluation"].parts[-2:] == ("evaluation", "results.csv")
    assert p["window_scenarios_contended"].parent.name == "window_scenarios_run1_contended"
    assert p["window_scenarios"].parent.name == "window_scenarios"
    assert p["early_delivery"].parent.name == "early_delivery"


def test_layer_b_loaders(tmp_path):
    split = {"R101": "dev", "R103": "eval"}
    window = pd.DataFrame({"instance": ["R101"], "routes": [json.dumps([[1, 2]])]})
    early = pd.DataFrame({"instance": ["R101", "R101"], "tolerance_label": ["0", "1"],
                          "routes": [json.dumps([[1]]), json.dumps([[2]])]})
    window.to_csv(tmp_path / "w.csv", index=False)
    early.to_csv(tmp_path / "e.csv", index=False)
    assert results.load_window_scenarios(tmp_path / "w.csv", split)["routes"].iloc[0] == [[1, 2]]
    loaded = results.load_early_delivery(tmp_path / "e.csv", split)
    assert list(loaded["tolerance_label"]) == ["0", "1"] and isinstance(loaded["tolerance_label"].iloc[0], str)
    window.assign(instance="R103").to_csv(tmp_path / "w2.csv", index=False)
    with pytest.raises(ValueError, match="R103"):
        results.load_window_scenarios(tmp_path / "w2.csv", split)
    with pytest.raises(ValueError, match="R103"):
        results.load_early_delivery(tmp_path / "w2.csv", split)


def test_protocol_v2_loaders(tmp_path):
    split = {"R101": "dev", "R103": "eval"}
    frame = pd.DataFrame({"instance": ["R101"], "scale": [10000], "routes": [json.dumps([[1, 2]])]})
    frame.to_csv(tmp_path / "b.csv", index=False)
    assert results.load_v2_block(tmp_path / "b.csv", split)["routes"].iloc[0] == [[1, 2]]
    frame.assign(instance="R103").to_csv(tmp_path / "b2.csv", index=False)
    with pytest.raises(ValueError, match="R103"):
        results.load_v2_block(tmp_path / "b2.csv", split)
    record = {"block": "B1", "winner": 10000, "representatives": [{"instance": "R101"}]}
    (tmp_path / "d.json").write_text(json.dumps(record), encoding="utf-8")
    assert results.load_v2_decision(tmp_path / "d.json", split)["winner"] == 10000
    record["representatives"].append({"instance": "R103"})
    (tmp_path / "d2.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="R103"):
        results.load_v2_decision(tmp_path / "d2.json", split)
    p = paths.result_paths(tmp_path)
    assert p["v2_B1"].parts[-3:] == ("v2", "B1", "results.csv") and p["v2_B3_decision"].name == "decision.json"
    assert p["v2_B5_early"].parent.name == "B5_early"


def test_phase_c_and_phase_d_loaders(tmp_path):
    split = {"R101": "dev", "R103": "eval"}
    folder = tmp_path / "block"
    folder.mkdir()
    cells = pd.DataFrame({"instance": ["R101", "R101"], "level": ["1", "0.9"], "variant": [None, None],
                          "feasible": [True, False], "capacity_infeasible": [False, True], "routes": ["[]", "[]"]})
    cells.to_csv(folder / "results.csv", index=False)
    loaded = results.load_phase_c_cells(folder, split)
    assert loaded["level"].tolist() == ["1", "0.9"] and loaded["feasible"].tolist() == [True, False]
    assert loaded["variant"].tolist() == ["", ""] and loaded["capacity_infeasible"].tolist() == [False, True]
    cells.assign(instance="R103").to_csv(folder / "results.csv", index=False)
    with pytest.raises(ValueError, match="R103"):
        results.load_phase_c_cells(folder, split)

    c1 = tmp_path / "c1"
    c1.mkdir()
    pd.DataFrame({"instance": ["R103"], "plan_feasible": [True]}).to_csv(c1 / "results.csv", index=False)
    pd.DataFrame({"scope": ["overall"]}).to_csv(c1 / "summary_by_family_method.csv", index=False)
    pd.DataFrame({"slack_bin": ["0"], "late_rate": [1.0]}).to_csv(c1 / "slack_lateness.csv", index=False)
    pd.DataFrame({"scenario": ["service"]}).to_csv(c1 / "slack_correlation.csv", index=False)
    out = results.load_robustness(c1, split)
    assert set(out) == {"cells", "summary", "slack_lateness", "slack_correlation"} and out["cells"]["plan_feasible"].all()
    pd.DataFrame({"instance": ["R101"], "plan_feasible": [True]}).to_csv(c1 / "results.csv", index=False)
    with pytest.raises(ValueError, match="R101"):  # the robustness run uses evaluation plans only
        results.load_robustness(c1, split)

    comparison = pd.DataFrame({"instance": ["R103"], "v1_vehicles": [3]})
    comparison.to_csv(tmp_path / "cmp.csv", index=False)
    assert len(results.load_evaluation_comparison(tmp_path / "cmp.csv", split)) == 1
    comparison.assign(instance="R101").to_csv(tmp_path / "cmp2.csv", index=False)
    with pytest.raises(ValueError, match="R101"):
        results.load_evaluation_comparison(tmp_path / "cmp2.csv", split)
    p = paths.result_paths(tmp_path)
    assert p["phase_c_capacity"].parts[-3:] == ("C2", "capacity") or p["phase_c_capacity"].parts[-2:] == ("C2", "capacity")
    assert p["phase_c_removal"].parent.name == "C3" and p["evaluation_v2"].parent.name == "evaluation_v2"
