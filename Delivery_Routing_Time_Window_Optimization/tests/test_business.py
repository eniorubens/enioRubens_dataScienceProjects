import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delivery_routing_vrptw import business as bz
from delivery_routing_vrptw import viz_business
from delivery_routing_vrptw.config import EarlyDeliverySection, ExperimentConfig
from delivery_routing_vrptw.route_compare import reallocation_table

SPLIT = {"C101": "dev", "R101": "dev", "R201": "eval"}


def window_rows():
    data = {  # (instance, factor) -> (vehicles, distance)
        ("C101", 1.0): (10, 100.0), ("C101", 1.5): (10, 99.0), ("C101", 2.0): (10, 98.0),
        ("R101", 1.0): (10, 200.0), ("R101", 1.5): (9, 180.0), ("R101", 2.0): (10, 170.0),
    }
    return pd.DataFrame([
        {"instance": i, "family": "C1" if i[0] == "C" else "R1", "factor": f, "mode": "symmetric",
         "feasible": True, "vehicles": v, "distance": d, "routes": [[1], [2]]}
        for (i, f), (v, d) in data.items()
    ])


def early_rows():
    plans = {"0": [[1, 2], [3, 4]], "0.5": [[1, 2], [3, 4]], "full": [[1, 3], [2, 4]]}
    out = []
    for inst in ("C101", "R101"):
        for lab, routes in plans.items():
            for rep in (0, 1):
                scale = {"0": 1.0, "0.5": 0.9, "full": 0.8}[lab]
                vehicles = 3 if (inst == "R101" and lab == "0") else len(routes)
                out.append({
                    "instance": inst, "tolerance_label": lab, "replicate": rep, "feasible": True,
                    "vehicles": vehicles, "distance": 100.0 * scale, "total_duration": 200.0 * scale,
                    "total_waiting": 50.0 * scale, "makespan": 100.0 * scale,
                    "total_duration_min": 190.0 * scale, "total_waiting_min": 40.0 * scale, "routes": routes,
                })
    return pd.DataFrame(out)


def test_tolerance_text():
    assert bz.tolerance_text("0.25", "en") == "25%"
    assert bz.tolerance_text("1", "en") == "100%"
    assert bz.tolerance_text("full", "pt") == "livre"
    assert bz.tolerance_text("noise", "pt") == "ruído"


def test_config_table():
    cfg = ExperimentConfig(early_delivery=EarlyDeliverySection(tolerances=(0.0, 0.5), replicates=2))
    out = bz.config_table(cfg, "en")
    values = dict(zip(out["metric"], out["value"], strict=True))
    assert values["Early delivery: tolerances (fraction of the width)"] == "0, 0.5, full"
    assert values["Early delivery: replicates"] == "2"
    assert values["Windows: width factors"] == "1, 1.25, 1.5, 2"


def test_run_checks_and_dev_only():
    out = bz.run_checks(window_rows(), early_rows(), SPLIT, "en")
    values = dict(zip(out["metric"], out["value"], strict=True))
    assert values["Widened windows: cells"] == "6" and values["Early delivery: cells"] == "12"
    assert values["Distinct instances"] == "2" and values["Instances outside the development set"] == "0"
    bad = window_rows().assign(instance="R201")
    with pytest.raises(ValueError, match="R201"):
        bz.run_checks(bad, early_rows(), SPLIT, "en")


def test_window_tables():
    summary = bz.window_results_summary(window_rows())
    assert "family" in summary.columns
    by_factor = bz.window_by_factor(summary)
    assert list(by_factor["factor"]) == ["1.5×", "2×"]
    assert by_factor["vehicles_saved_mean"].tolist() == [0.5, 0.0]
    assert by_factor["n_with_saving"].tolist() == [1, 0]
    assert by_factor["n_equal"].tolist() == [1, 2]
    assert "distance_change_mean" not in by_factor.columns  # never a pooled distance mean
    by_fleet = bz.window_distance_by_fleet(summary, "en")
    rows = {(r.factor, r.fleet_class): r for r in by_fleet.itertuples()}
    assert set(rows) == {("1.5×", "equal fleet"), ("1.5×", "reduced fleet"), ("2×", "equal fleet")}
    assert rows[("1.5×", "equal fleet")].distance_change_mean == pytest.approx(-1.0)
    assert rows[("1.5×", "reduced fleet")].distance_change_mean == pytest.approx(-10.0)
    assert rows[("2×", "equal fleet")].n_instances == 2
    assert rows[("2×", "equal fleet")].distance_change_mean == pytest.approx(-8.5)
    by_family = bz.window_by_family(summary, "en")
    assert list(by_family["family"]) == ["C1", "R1"]
    r1 = by_family[by_family["family"] == "R1"].iloc[0]
    assert r1["×1.5 · vehicles saved (mean)"] == 1.0
    assert r1["×1.5 · instances with equal fleet"] == 0 and pd.isna(r1["×1.5 · distance, equal fleet (%, mean)"])
    assert r1["×2 · distance, equal fleet (%, mean)"] == pytest.approx(-15.0)
    fig = viz_business.plot_window_effect(summary, "en")
    assert len(fig.axes) == 2
    plt.close("all")


def test_early_tables_and_departure_conventions():
    summary = bz.early_results_summary(early_rows())
    by_tol = bz.early_by_tolerance(summary, "en")
    assert list(by_tol["tolerance"]) == ["50%", "50%", "full", "full"]
    assert list(by_tol["fleet_class"]) == ["equal fleet", "reduced fleet"] * 2
    full = by_tol[(by_tol["tolerance"] == "full") & (by_tol["fleet_class"] == "equal fleet")].iloc[0]
    assert full["n_instances"] == 1 and full["vehicles_saved_mean"] == 0
    assert full["duration_pct_mean"] == pytest.approx(-20.0)
    assert full["duration_min_pct_mean"] == pytest.approx(-100 * (1 - 152 / 190), abs=0.06)
    gap = full["duration_min_pct_mean"] - full["duration_pct_mean"]
    assert full["inflation_pp"] == pytest.approx(gap, abs=0.1)
    reduced = by_tol[(by_tol["tolerance"] == "full") & (by_tol["fleet_class"] == "reduced fleet")].iloc[0]
    assert reduced["vehicles_saved_mean"] == 1.0
    fam = bz.early_by_family(summary, "vehicles_saved", "pt")
    assert list(fam["family"]) == ["C1", "R1"] and list(fam.columns[-2:]) == ["50%", "livre"]
    assert fam.loc[fam["family"] == "R1", "livre"].iloc[0] == 1.0
    assert fam.loc[fam["family"] == "C1", "livre"].iloc[0] == 0.0
    dur = bz.early_by_family(summary, "duration_min_pct", "en", fleet="equal")
    assert dur.loc[dur["family"] == "C1", "full"].iloc[0] == pytest.approx(-20.0, abs=0.1)
    assert pd.isna(dur.loc[dur["family"] == "R1", "full"].iloc[0])  # no equal-fleet instance in R1
    with pytest.raises(ValueError):
        bz.early_by_family(summary, "distance_pct", "en")
    with pytest.raises(ValueError):
        bz.early_by_family(summary, "distance_pct", "en", fleet="all")
    counts = bz.early_fleet_counts(summary, "en")
    r1 = counts[(counts["family"] == "R1") & (counts["tolerance"] == "full")].iloc[0]
    assert (r1["n_equal"], r1["n_reduced"], r1["n_increased"]) == (0, 1, 0)
    fig = viz_business.plot_early_effect(summary, "en")
    assert len(fig.axes) >= 3
    plt.close("all")


def test_fleet_class():
    assert [bz.fleet_class(v) for v in (1.0, 0.0, -1.0)] == ["reduced", "equal", "increased"]
    assert bz.fleet_class(float("nan")) == "missing"


def test_reallocation_tables_and_plot():
    table = reallocation_table(early_rows())
    display = bz.reallocation_display(table, "en")
    assert list(display["tolerance"])[-1] == "noise" and display["reassigned_pct"].iloc[-1] == 0.0
    full = display[display["tolerance"] == "full"].iloc[0]
    assert full["reassigned_pct"] == 50.0
    by_family = bz.reallocation_by_family(table, "en")
    assert list(by_family.columns) == ["family", "50%", "full", "noise"]
    fig = viz_business.plot_reallocation(table, "en")
    assert fig.axes[0].get_ylabel() == "Median (%)"
    plt.close("all")


def test_noise_floor_and_examples(inst):
    summary = bz.early_results_summary(early_rows())
    table = reallocation_table(early_rows())
    noise = bz.noise_floor_table(summary, table, "en")
    values = dict(zip(noise["metric"], noise["value"], strict=True))
    assert values["Replicates of the original plan per instance"] == "2"
    assert values["Instances with a different fleet across original-plan replicates"] == "0 / 2"
    assert bz.pick_early_example(summary) == "R101"  # the only instance that saves a vehicle at "full"
    panels = bz.before_after_panels("C101", {"C101": inst}, early_rows(), "en")
    assert [p[0].name for p in panels] == ["R101", "R101@efull"]
    assert "original plan" in panels[0][1] and "early delivery: full" in panels[1][1]
    with pytest.raises(ValueError):
        bz.before_after_panels("C101", {"C101": inst}, early_rows(), "en", replicate=7)


def test_run_comparison_and_nonmonotonic():
    official = window_rows()
    other = official.assign(vehicles=official["vehicles"] + [0, 0, 0, 1, 0, 0],
                            distance=official["distance"] + [0, 0, 0, 0, 0, 1e-9])
    out = bz.run_comparison_table(official, other, "en")
    values = dict(zip(out["metric"], out["value"], strict=True))
    assert values["Cells with a different fleet between the two runs"] == "1 / 6"
    assert values["Largest fleet difference (vehicles)"] == "1"
    assert values["Cells with a different distance between the two runs"] == "0 / 6"
    cases = bz.nonmonotonic_cases(
        bz.window_results_summary(window_rows()), bz.early_results_summary(early_rows()), "en"
    )
    assert list(cases["instance"]) == ["R101"] and cases["scenario"].iloc[0] == "Widened windows"
    assert (cases["vehicles"].iloc[0], cases["best_lower"].iloc[0]) == (10, 9)
    assert cases["best_lower_distance"].iloc[0] == 180.0 and cases["worsening"].iloc[0] == "fleet"


def test_nonmonotonic_is_lexicographic():
    win = window_rows()
    win.loc[(win["instance"] == "C101") & (win["factor"] == 1.5), "distance"] = 101.0  # same fleet, longer
    mask = (win["instance"] == "R101") & (win["factor"] == 2.0)
    win.loc[mask, ["vehicles", "distance"]] = [9, 190.0]  # same fleet as the best (9, 180), longer
    cases = bz.nonmonotonic_cases(bz.window_results_summary(win), bz.early_results_summary(early_rows()), "pt")
    rows = {(r.instance, r.level): r for r in cases.itertuples()}
    # C101 at 2x (98) beats the baseline (100) and is not flagged although 1.5x (101) was worse
    assert set(rows) == {("C101", "1.5×"), ("R101", "2×")}
    assert rows[("C101", "1.5×")].worsening == "distância (frota igual)"
    assert rows[("R101", "2×")].best_lower == 9 and rows[("R101", "2×")].best_lower_distance == 180.0
