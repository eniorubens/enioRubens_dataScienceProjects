import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delivery_routing_vrptw import v2_view, viz_business


def block_rows(plans, level_col="scale", workers=8, k=10.0, strategy="fixed_cost", scale=10000):
    """plans: {(instance, level): [(vehicles, distance), ...]}, one tuple per replicate; reference = 10 vehicles."""
    rows = []
    for (inst, level), reps in plans.items():
        for r, (veh, dist) in enumerate(reps):
            row = {
                "instance": inst, "scale": scale, "k": k, "fleet_strategy": strategy, "workers": workers,
                "replicate": r, "feasible": True, "vehicles": veh, "distance": dist, "ref_vehicles": 10,
                "delta_vehicles": veh - 10, "gap_distance_pct": 1.0, "distance_comparable": veh == 10,
                "runtime_s": 1.0,
            }
            row[level_col] = level
            rows.append(row)
    return pd.DataFrame(rows)


def two_scales():
    return block_rows({
        ("A", 100): [(12, 120.0)] * 3, ("A", 10000): [(11, 125.0)] * 3,
        ("B", 100): [(10, 100.0)] * 3, ("B", 10000): [(10, 98.0)] * 3,
    }, scale=None)


def test_compared_rows_filters_the_blocks():
    b1 = block_rows({("A", 100): [(10, 1.0)] * 3, ("A", 10000): [(10, 1.0)] * 3})
    b2 = block_rows({("A", 1.0): [(10, 1.0)] * 3, ("A", 100.0): [(10, 1.0)] * 3}, level_col="k")
    b3 = block_rows({("A", "reduce_fleet_v2"): [(10, 1.0)] * 3}, level_col="fleet_strategy")
    blocks = {"B1": b1, "B2": b2, "B3": b3}
    rows, col = v2_view.compared_rows("B1", blocks)
    assert col == "scale" and set(rows["scale"]) == {100, 10000}
    rows, col = v2_view.compared_rows("B2", blocks)
    assert col == "k" and set(rows["k"]) == {1.0, 10.0, 100.0}  # B1's k = 10 rows at scale 10000 are reused
    rows, col = v2_view.compared_rows("B3", blocks)
    assert col == "fleet_strategy" and set(rows["fleet_strategy"]) == {"fixed_cost", "reduce_fleet_v2"}


def test_ranking_display_and_decision_table():
    ranking = pd.DataFrame({
        "scale": [10000, 100], "mean_rank": [1.3214, 1.6786], "mean_delta_vehicles": [0.5714, 0.6429],
        "n_fleet_equal_ref": [7, 7], "median_gap_equal_fleet": [1.2156, 1.5285], "mean_runtime": [120.0, 120.0],
        "n_instances": [14, 14],
    })
    out = v2_view.ranking_display(ranking, "scale", "pt")
    assert list(out.columns)[:3] == ["position", "scale", "mean_rank"]
    assert out["scale"].tolist() == ["escala 10000", "escala 100"] and out["mean_rank"].iloc[0] == 1.321
    record = {"level_column": "scale", "winner": 10000, "decided_by": "mean_rank",
              "ranking": ranking.to_dict(orient="records")}
    table = v2_view.decision_table({"B1": record}, "en")
    assert table.iloc[0]["winner"] == "scale 10000" and table.iloc[0]["decided_by"] == "mean rank"
    assert "scale 100: 1.68" in table.iloc[0]["levels_ranked"]
    strict = {"level_column": "fleet_strategy", "winner": "fixed_cost", "decided_by": "not_strictly_better",
              "ranking": [{"fleet_strategy": "fixed_cost", "mean_rank": 1.3}]}
    assert v2_view.decision_table({"B3": strict}, "pt").iloc[0]["decided_by"] == "não é estritamente melhor"


def test_level_tables():
    rows = two_scales()
    by_instance = v2_view.level_by_instance(rows, "scale", "en")
    assert list(by_instance["instance"]) == ["A", "B"]
    assert by_instance["Vehicles (scale 100)"].tolist() == [12, 10]
    assert by_instance["Vehicles (scale 10000)"].tolist() == [11, 10]
    comparison = v2_view.level_comparison(rows, "scale", 100, "en")
    row = comparison.iloc[0]
    assert (row["n_fewer"], row["n_more"], row["n_equal"]) == (1, 0, 1)
    assert row["distance_change_mean"] == pytest.approx(-2.0)  # only instance B has an equal fleet
    pair = v2_view.pairwise_instances(rows, "scale", 100, 10000, "en")
    assert list(pair["instance"]) == ["A", "B"]  # A: fleet changed; B: distance change of -2%
    a = pair[pair["instance"] == "A"].iloc[0]
    assert (a["vehicles_base"], a["vehicles_other"]) == (12, 11) and pd.isna(a["distance_change_pct"])
    reps = v2_view.replicates_table(rows, "A", "scale", "en")
    assert reps["vehicles_replicates"].tolist() == ["12 · 12 · 12", "11 · 11 · 11"]
    with pytest.raises(ValueError):
        v2_view.replicates_table(rows, "Z", "scale", "en")


def test_block_coverage():
    b = block_rows({("A", 10000): [(10, 1.0)] * 3}, workers=4)
    b["feasible"] = [True, True, False]
    out = v2_view.block_coverage({"B4": b}, "en").iloc[0]
    assert (out["block"], out["n_cells"], out["n_feasible"], out["workers"]) == (
        "B4 · machine load (4 processes)", 3, 2, 4)


def test_workers_by_instance_counts_cells():
    eight = block_rows({("A", 10000): [(10, 100.0), (10, 100.0), (10, 100.0)],
                        ("B", 10000): [(10, 50.0)] * 3}, workers=8)
    four = block_rows({("A", 10000): [(10, 100.0), (10, 99.0), (10, 99.0)],
                       ("B", 10000): [(10, 50.0)] * 3}, workers=4)
    out = v2_view.workers_by_instance(four, eight, "en").set_index("instance")
    assert out.loc["A", "n_distance_differs"] == 2 and out.loc["A", "n_fleet_differs"] == 0
    assert out.loc["A", "distance_change_pct"] == pytest.approx(-1.0)
    assert out.loc["B", "n_distance_differs"] == 0 and out.loc["B", "n_cells"] == 3


def window_rows(by_factor, warm="none"):
    out = []
    for inst, vehicles in by_factor.items():
        for i, (f, v) in enumerate(vehicles.items()):
            out.append({"instance": inst, "family": "R1", "factor": f, "mode": "symmetric", "feasible": True,
                        "vehicles": v, "distance": 1000.0 - 10 * i, "duration": 1.0, "waiting": 0.0, "replicate": 0,
                        "warm_start": warm if i == 0 else "used"})
    return pd.DataFrame(out)


def early_rows(levels, instance="R101", reps=3):
    out = []
    for lab, (veh, dist) in levels.items():
        for r in range(reps):
            out.append({
                "instance": instance, "family": "R1", "tolerance_label": lab, "replicate": r, "feasible": True,
                "vehicles": veh, "distance": dist, "total_duration": 100.0, "total_waiting": 5.0,
                "total_duration_min": 90.0, "total_waiting_min": 2.0, "makespan": 50.0,
                "warm_start": "none" if lab == "0" else "used",
            })
    return pd.DataFrame(out)


def test_layerb_tables_and_plot():
    w1 = window_rows({"R101": {1.0: 10, 1.25: 11, 1.5: 9}})
    w2 = window_rows({"R101": {1.0: 10, 1.25: 10, 1.5: 9}})
    e1 = early_rows({"0": (10, 1000.0), "0.5": (10, 1010.0), "full": (9, 900.0)})
    e2 = early_rows({"0": (10, 1000.0), "0.5": (10, 990.0), "full": (9, 900.0)})
    levels = v2_view.layerb_levels(w1, w2, e1, e2, "en")
    assert levels["level"].tolist() == ["1.25×", "1.5×", "50%", "full"]
    assert levels["v1_vehicles_saved_mean"].tolist() == [-1.0, 1.0, 0.0, 1.0]
    assert levels["v2_vehicles_saved_total"].tolist() == [0, 1, 0, 1]
    with_base = v2_view.layerb_levels(w1, w2, e1, e2, "en", include_base=True)
    assert with_base["level"].tolist() == ["1×", "1.25×", "1.5×", "0%", "50%", "full"]
    assert with_base["v2_vehicles_saved_total"].iloc[0] == 0 and with_base["v1_vehicles_mean"].iloc[0] == 10.0
    counts = v2_view.layerb_nonmonotonic(w1, w2, e1, e2, "en")
    by = counts.set_index(["scenario", "version"])
    assert by.loc[("Widened windows", "v1 (no warm start)"), "n_fleet"] == 1
    assert by.loc[("Widened windows", "v2 (warm start)"), "n_nonmonotonic"] == 0
    assert by.loc[("Early delivery", "v1 (no warm start)"), "n_distance"] == 1
    warm = v2_view.layerb_warm_counts(w2, e2, "pt").set_index("scenario")
    assert warm.loc["Janelas alargadas", "n_warm"] == 2 and warm.loc["Janelas alargadas", "n_cold"] == 1
    assert warm.loc["Entrega antecipada", "n_cells"] == 9
    fig = viz_business.plot_warm_start(levels, "pt")
    assert len(fig.axes) == 2
    plt.close(fig)
