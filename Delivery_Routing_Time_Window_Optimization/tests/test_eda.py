"""Tests of the exploratory-analysis helpers with synthetic instances only."""

import csv
import hashlib
import json
import math

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from conftest import make_text
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from delivery_routing_vrptw import (
    LABELS,
    Instance,
    Reference,
    eda,
    paths,
    read_cvrplib_references,
    route_length,
    sha256_file,
    truncated,
    viz_eda,
)
from delivery_routing_vrptw import io as vio
from delivery_routing_vrptw.distance import exact
from delivery_routing_vrptw.labels import get_labels

GOOD_ROUTES = [[1, 2], [3, 5], [4]]


@pytest.fixture
def pool():
    """Four instances of different families."""
    return {
        name: vio.parse_instance(make_text(name, capacity=cap))
        for name, cap in [("C101", 10), ("C201", 12), ("R101", 10), ("RC201", 15)]
    }


def refs_for(pool):
    """SINTEF-like references matching the pool."""
    return {n: Reference(n, 3, 123.45, "T") for n in pool}


def test_family_summary(pool):
    out = eda.family_summary(pool)
    assert list(out["family"]) == ["C1", "C2", "R1", "RC2"]
    row = out.set_index("family").loc["C1"]
    assert row["n_instances"] == 1 and row["n_customers"] == 5 and row["capacity"] == 10
    assert row["horizon"] == 100 and row["service_time"] == 2 and row["mean_total_demand"] == 25
    assert eda.family_summary(list(pool.values())).shape == out.shape


def test_contract_check_ok_and_failures(inst):
    ok = eda.contract_check([inst])
    assert bool(ok.loc[0, "all_ok"]) and ok.loc[0, "instance"] == "R101"
    bad = Instance(
        name="C101", vehicle_limit=3, capacity=3, x=inst.x, y=inst.y, demand=inst.demand + [1, 0, 0, 0, 0, 0],
        ready=inst.ready + [0, 0, 100, 0, 0, 0], due=inst.due + [0, 0, 0, 0, 0, 20], service=inst.service + [1, 0, 0, 0, 0, 0],
    )
    row = eda.contract_check([bad]).iloc[0]
    assert not row["depot_zero"] and not row["demand_le_capacity"]
    assert not row["ready_le_due"] and not row["windows_in_horizon"] and not row["all_ok"]
    broken = Instance(
        name="C101", vehicle_limit=3, capacity=10, x=inst.x[:3], y=inst.y, demand=inst.demand,
        ready=inst.ready, due=inst.due, service=inst.service,
    )
    assert not eda.contract_check([broken]).iloc[0]["ids_contiguous"]


def test_contract_summary(pool):
    out = eda.contract_summary(eda.contract_check(pool), "pt")
    assert out["passed"].tolist() == [4] * 6 and out["n_instances"].tolist() == [4] * 6
    assert out["rule"].iloc[-1] == LABELS["pt"]["columns"]["all_ok"]


def test_window_stats_and_summary(pool):
    stats = eda.window_stats(pool)
    row = stats.iloc[0]
    assert row["mean_width"] == pytest.approx(60.0) and row["median_width"] == 60.0
    assert row["width_share_horizon"] == pytest.approx(0.6)
    assert row["unconstrained_share"] == 0.0
    assert row["service_share_horizon"] == pytest.approx(0.016)
    summary = eda.window_summary(stats)
    assert list(summary["family"]) == ["C1", "C2", "R1", "RC2"]
    assert summary["width_share_horizon"].iloc[0] == pytest.approx(0.6)


def test_unconstrained_share(inst):
    wide = Instance(
        name="R101", vehicle_limit=3, capacity=10, x=inst.x, y=inst.y, demand=inst.demand,
        ready=np.zeros(6, dtype=int), due=np.array([100, 95, 90, 89, 10, 100]), service=inst.service,
    )
    assert eda.window_stats([wide]).iloc[0]["unconstrained_share"] == pytest.approx(3 / 5)


def test_capacity_lower_bound_and_fleet(inst, pool):
    assert eda.capacity_lower_bound(inst) == 3
    out = eda.fleet_bounds(pool, refs_for(pool))
    rows = out.set_index("instance")
    assert rows.loc["C101", "capacity_lb"] == 3 and rows.loc["C101", "binding"] == "capacity"
    assert rows.loc["C201", "capacity_lb"] == 3  # ceil(25/12)
    assert rows.loc["RC201", "capacity_lb"] == 2 and rows.loc["RC201", "binding"] == "time windows"
    assert rows.loc["RC201", "ratio"] == pytest.approx(1.5)
    missing = eda.fleet_bounds(pool, {"C101": Reference("C101", 3, 1.0, "T")})
    assert list(missing["instance"]) == ["C101"]
    summary = eda.fleet_summary(out)
    assert summary.set_index("family").loc["C1", "n_capacity_binding"] == 1


def brute_density(inst):
    d = exact(inst)
    n = inst.n_customers
    ok = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                leave = max(inst.ready[i], d[0, i]) + inst.service[i]
                ok += leave + d[i, j] <= inst.due[j]
    return ok / (n * (n - 1))


def test_compatibility_density(inst, pool):
    assert eda.compatibility_density(inst) == pytest.approx(brute_density(inst))
    assert 0 <= eda.compatibility_density(inst) <= 1
    assert math.isnan(eda.compatibility_density(inst.subset(1)))
    tight = Instance(
        name="R101", vehicle_limit=3, capacity=10, x=inst.x, y=inst.y, demand=inst.demand,
        ready=np.zeros(6, dtype=int), due=np.array([100, 7, 7, 7, 7, 7]), service=inst.service,
    )
    assert eda.compatibility_density(tight) == 0.0
    df = eda.compatibility_by_instance(pool)
    assert list(df.columns) == ["instance", "family", "density"] and len(df) == 4
    summary = eda.compatibility_summary(df)
    assert {"density", "density_min", "density_max"} <= set(summary.columns)


def conv_inputs(pool, routes=None, cost=None, distance=None):
    routes = GOOD_ROUTES if routes is None else routes
    inst = pool["R101"]
    trunc = truncated(inst)
    published = sum(route_length(r, trunc) for r in routes) if cost is None else cost
    true = sum(route_length(r, exact(inst)) for r in routes)
    sintef = {"R101": Reference("R101", 3, round(true, 2) if distance is None else distance, "T")}
    return {"R101": inst}, sintef, {"R101": (routes, round(published, 1))}


def test_convention_comparison_consistent(pool):
    insts, sintef, cvrplib = conv_inputs(pool)
    out = eda.convention_comparison(insts, sintef, cvrplib).iloc[0]
    assert out["feasible_truncated"] and out["feasible_exact"] and out["cost_reproduced"]
    assert out["delta_vehicles"] == 0 and out["equal_fleet_match"]
    assert out["distance_exact"] >= out["distance_truncated"]
    assert abs(out["distance_diff_pct"]) < 0.01
    assert out["cvrplib_routes"] == 3 and out["sintef_vehicles"] == 3


def test_convention_comparison_differences(pool):
    insts, _, _ = conv_inputs(pool)
    sintef = {"R101": Reference("R101", 2, 200.0, "T")}
    cvrplib = {"R101": (GOOD_ROUTES, 999.0)}
    out = eda.convention_comparison(insts, sintef, cvrplib).iloc[0]
    assert out["delta_vehicles"] == 1 and not out["cost_reproduced"] and not out["equal_fleet_match"]
    one = {"R101": ([[1, 2, 3, 4, 5]], 1.0)}
    bad = eda.convention_comparison(insts, sintef, one).iloc[0]
    assert not bad["feasible_exact"] and not bad["feasible_truncated"]
    assert eda.convention_comparison(insts, {}, cvrplib).empty


def test_convention_summary(pool):
    insts, sintef, cvrplib = conv_inputs(pool)
    conv = eda.convention_comparison(insts, sintef, cvrplib)
    out = eda.convention_summary(conv, "en")
    values = dict(zip(out["metric"], out["value"], strict=True))
    labels = LABELS["en"]["convention"]
    assert values[labels["n_instances"]] == "1" and values[labels["equal_fleet"]] == "1"
    assert values[labels["infeasible_exact"]] == "0" and values[labels["sintef_fewer"]] == "0"
    bad = eda.convention_comparison(insts, sintef, {"R101": ([[1, 2, 3, 4, 5]], 1.0)})
    text = dict(zip(*eda.convention_summary(bad, "pt").T.values, strict=True))
    assert text[LABELS["pt"]["convention"]["infeasible_exact"]] == "1: R101"


def test_convention_summary_ranges(pool):
    insts, _, cvrplib = conv_inputs(pool)
    sintef = {"R101": Reference("R101", 2, 100.0, "T")}
    conv = eda.convention_comparison(insts, sintef, cvrplib)
    pt = dict(zip(*eda.convention_summary(conv, "pt").T.values, strict=True))
    en = dict(zip(*eda.convention_summary(conv, "en").T.values, strict=True))
    key = "extra_distance_range"
    assert "," in pt[LABELS["pt"]["convention"][key]] and "." in en[LABELS["en"]["convention"][key]]
    assert pt[LABELS["pt"]["convention"]["max_vehicle_diff"]] == "1"


def test_loaded_counts_and_split(pool):
    out = eda.loaded_counts(pool, {"a": 1}, {}, "pt")
    assert out["count"].tolist() == [4, 1, 0]
    split = {"C101": "dev", "C102": "eval", "R101": "eval", "R102": "eval"}
    table = eda.split_summary(split)
    assert table.iloc[-1].tolist() == ["Total", 1, 3, 4]
    assert table["family"].tolist() == ["C1", "R1", "Total"]


def test_localize_columns(pool):
    fleet = eda.fleet_bounds(pool, refs_for(pool))
    pt = eda.localize_columns(fleet, "pt")
    assert "Família" in pt.columns and "janelas de tempo" in set(pt["Restrição determinante"])
    en = eda.localize_columns(fleet, "en")
    assert "Capacity lower bound" in en.columns
    assert eda.localize_columns(pd.DataFrame({"zzz": [1]}), "en").columns.tolist() == ["zzz"]


def test_all_column_keys_are_labelled(pool):
    insts, sintef, cvrplib = conv_inputs(pool)
    frames = [
        eda.family_summary(pool), eda.contract_check(pool), eda.window_stats(pool),
        eda.fleet_bounds(pool, refs_for(pool)), eda.compatibility_by_instance(pool),
        eda.convention_comparison(insts, sintef, cvrplib), eda.split_summary({"C101": "dev"}),
        eda.synthesis_table(
            eda.window_stats(pool), eda.fleet_bounds(pool, refs_for(pool)), eda.compatibility_by_instance(pool)),
        eda.fleet_summary(eda.fleet_bounds(pool, refs_for(pool))),
    ]
    for frame in frames:
        for col in frame.columns:
            assert col in LABELS["pt"]["columns"] and col in LABELS["en"]["columns"], col


def test_synthesis_table(pool):
    out = eda.synthesis_table(
        eda.window_stats(pool), eda.fleet_bounds(pool, refs_for(pool)), eda.compatibility_by_instance(pool))
    assert list(out["family"]) == ["C1", "C2", "R1", "RC2"]
    assert {"width_share_horizon", "capacity_lb", "ref_vehicles", "density", "ratio"} <= set(out.columns)


def test_manifest_and_provenance(tmp_path):
    payloads = {"a.zip": b"alpha", "b.7z": b"beta", "c.7z": b"gamma"}
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    digest = {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
    (tmp_path / "m1.md").write_text(f"# x\n- a.zip  sha256 {digest['a.zip']}\n- note without hash\n", encoding="utf-8")
    wrong = "0" * 64
    (tmp_path / "m2.md").write_text(
        f"- b.7z  <- http://x/1      sha256 {digest['b.7z']}\n- c.7z <- y sha256 {wrong}\n", encoding="utf-8")
    assert eda.manifest_hashes(tmp_path / "m1.md") == {"a.zip": digest["a.zip"]}
    spec = {
        "sintef_zip": tmp_path / "a.zip", "sintef_manifest": tmp_path / "m1.md",
        "cvrplib_instances": tmp_path / "b.7z", "cvrplib_solutions": tmp_path / "c.7z",
        "cvrplib_manifest": tmp_path / "m2.md",
    }
    out = eda.provenance_table(spec)
    assert out["match"].tolist() == [True, True, False]
    missing = dict(spec, cvrplib_manifest=tmp_path / "m1.md")
    assert eda.provenance_table(missing)["match"].tolist() == [True, False, False]


def test_sha256_and_cvrplib_reader(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc" * 10)
    assert sha256_file(f, chunk_size=7) == hashlib.sha256(b"abc" * 10).hexdigest()
    csv_path = tmp_path / "c.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "routes", "cost", "source"])
        writer.writerow(["r101", json.dumps([[1, 2], [3]]), "12.3", "CVRPLIB"])
    assert read_cvrplib_references(csv_path) == {"R101": ([[1, 2], [3]], 12.3)}


def test_paths(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert paths.project_root(deep) == tmp_path.resolve()
    assert paths.project_root(tmp_path) == tmp_path.resolve()
    data = paths.data_paths(tmp_path)
    assert data["root"] == tmp_path and data["sintef_zip"].name == "solomon-100.zip"
    assert data["cvrplib_bks"].name == "cvrplib_bks_100.csv" and data["split"].name == "split.csv"
    assert paths.data_paths()["root"] == paths.project_root()
    with pytest.raises(FileNotFoundError):
        paths.project_root(tmp_path.anchor)


def test_plots(pool):
    insts, sintef, cvrplib = conv_inputs(pool)
    conv = eda.convention_comparison(insts, sintef, cvrplib)
    window = eda.window_stats(pool)
    fleet = eda.fleet_bounds(pool, refs_for(pool))
    density = eda.compatibility_by_instance(pool)
    for lang in ("pt", "en"):
        fig = viz_eda.plot_family_layouts(pool, ["C101", "C201", "R101", "RC201"], lang)
        assert isinstance(fig, Figure) and len(fig.axes) >= 4
        fig = viz_eda.plot_family_layouts(pool, ["C101"], lang)
        assert isinstance(fig, Figure)
        fig = viz_eda.plot_family_layouts(pool, ["C101", "C201", "R101", "RC201"], lang)
        clims = {tuple(c.get_clim()) for ax in fig.axes for c in ax.collections if c.get_array() is not None}
        assert clims == {(0.0, 1.0)}  # one common scale: share of the horizon
        shares = [c.get_array() for ax in fig.axes for c in ax.collections if c.get_array() is not None]
        assert all(0.0 <= float(a.min()) and float(a.max()) <= 1.0 for a in shares)
        assert fig.axes[-1].get_ylabel() == get_labels(lang)["plots"]["layout_colorbar"]
        assert isinstance(viz_eda.plot_window_comparison(pool, ["C101", "R101"], lang), Figure)
        for func, frame in [
            (viz_eda.plot_window_widths, window), (viz_eda.plot_fleet_bounds, fleet),
            (viz_eda.plot_compatibility, density), (viz_eda.plot_convention_tradeoff, conv),
        ]:
            ax = func(frame, lang)
            assert isinstance(ax, Axes) and ax.get_title()
            _, own = plt.subplots()
            assert func(frame, lang, ax=own) is own
        plt.close("all")
    plt.close("all")
