"""Tests of the 2026-11-19 phase-C corrections: C3 representative aggregation, noise distribution, C2.2 drop spread."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from delivery_routing_vrptw import phase_c, phase_c_view

ROOT = Path(__file__).resolve().parents[1]
PLAN_SAME = [[1, 2], [3, 4]]
PLAN_OTHER = [[1, 3], [2, 4]]


def _c3_row(instance, variant, replicate, vehicles, distance, share, routes=PLAN_SAME):
    return {
        "block": "removal", "instance": instance, "variant": variant, "replicate": replicate, "feasible": True,
        "vehicles": vehicles, "distance": distance, "sample": "[9]", "routes": json.dumps(routes),
        "delta_vehicles": 0, "delta_distance_pct": distance - 10, "runtime_s": 30.0,
        "realloc_customers_reassigned": 0, "realloc_customers_reassigned_share": share,
        "realloc_pairs_kept": 1 - share, "realloc_pair_agreement": 1.0, "realloc_arcs_kept": 1 - share,
    }


def make_c3() -> pd.DataFrame:
    rows = []
    # A: warm representative (2nd of 3 by vehicles, distance) is replicate 1, share 0.1; pooled median is 0.5
    for rep, (veh, dist, share) in enumerate([(3, 10.0, 0.5), (3, 12.0, 0.1), (4, 9.0, 0.9)]):
        rows.append(_c3_row("A", "warm", rep, veh, dist, share))
        rows.append(_c3_row("A", "cold", rep, 3, 10.0 + rep, 0.4))
    # B: cold plans differ between replicate 0 and the others -> positive noise in 2 of 3 comparisons
    for rep, routes in enumerate([PLAN_SAME, PLAN_OTHER, PLAN_OTHER]):
        rows.append(_c3_row("B", "warm", rep, 3, 10.0 + rep, 0.2))
        rows.append(_c3_row("B", "cold", rep, 3, 10.0 + rep, 0.3, routes))
    rows.append({**_c3_row("A", "plan90", 0, 3, 10.0, 0.0), "block": "insertion"})
    return pd.DataFrame(rows)


def test_representative_picks_one_real_replicate_per_instance_and_variant():
    rep = phase_c.reopt_representatives(make_c3())
    assert len(rep) == 4 and set(rep["variant"]) == {"warm", "cold"}  # plan90 ignored
    a = rep[(rep["instance"] == "A") & (rep["variant"] == "warm")].iloc[0]
    assert a["replicate"] == 1 and a["realloc_customers_reassigned_share"] == pytest.approx(0.1)
    assert (rep["n_replicates"] == 3).all()


def test_noise_distribution_reports_positive_comparisons():
    noise = phase_c.noise_table(make_c3())
    d = phase_c.noise_distribution(noise).iloc[0]
    assert d["n_pairs"] == 6 and d["n_positive"] == 2 and d["median_share"] == 0.0
    assert d["max_share"] > 0 and d["instances"].startswith("B ") and d["instances"].endswith("x2")


def test_aggregate_representative_differs_from_pooled():
    rows = make_c3()
    noise = phase_c.noise_table(rows)
    rep = phase_c.reopt_aggregate_representative(rows, noise).set_index("variant")
    pooled = phase_c.reopt_aggregate(rows, noise).set_index("variant")
    assert rep.loc["warm", "n"] == 2 and pooled.loc["warm", "n"] == 6
    assert rep.loc["warm", "realloc_customers_reassigned_share"] == pytest.approx(0.15)  # median of (0.1, 0.2)
    assert pooled.loc["warm", "realloc_customers_reassigned_share"] == pytest.approx(0.2)
    assert rep.loc["noise", "noise_n_positive"] == 2 and rep.loc["noise", "noise_max_share"] > 0


def test_views():
    rows = make_c3()
    table = phase_c_view.reopt_table_representative(rows, rows, "en")
    assert list(table["variant"]) == ["Warm start", "Cold start"] * 2
    assert {"reassigned_pct", "pooled_reassigned_pct", "n_instances"} <= set(table.columns)
    noise = phase_c_view.reopt_noise_table(rows, rows, "pt")
    assert noise["noise_n_positive"].tolist() == [2, 2] and noise["n_pairs"].tolist() == [6, 6]


def _fleet_row(instance, level, variant, replicate, dropped):
    return {"instance": instance, "level": level, "variant": variant, "replicate": replicate, "feasible": True,
            "n_dropped": len(dropped), "n_dropped_priority": 0, "dropped_ids": json.dumps(dropped)}


def test_dropped_spread_reproduces_the_rc108_pattern():
    rows = pd.DataFrame(
        [_fleet_row("RC108", "K-2", "uniform", r, ids) for r, ids in enumerate(
            [list(range(9)), list(range(7)), list(range(1, 8))])]
        + [_fleet_row("C101", "K-1", "uniform", r, [5]) for r in range(3)]
        + [_fleet_row("R110", "K-2", "priority", r, ids) for r, ids in enumerate([[1, 2], [1, 3], [1, 2]])]
    )
    spread = phase_c.dropped_spread(rows).set_index(["instance", "level", "variant"])
    rc = spread.loc[("RC108", "K-2", "uniform")]
    assert rc["dropped_values"] == "9, 7, 7" and rc["dropped_range"] == 2 and rc["n_distinct_sets"] == 3
    assert spread.loc[("C101", "K-1", "uniform"), "n_distinct_sets"] == 1
    assert spread.loc[("R110", "K-2", "priority"), "dropped_range"] == 0  # same count, different customers
    shown = phase_c_view.dropped_spread_table(rows, "en")
    assert list(shown["instance"]) == ["RC108", "R110"]
    summary = phase_c_view.dropped_spread_summary(rows, "en")
    assert summary["count"].tolist() == [3, 1, 2]


def test_derive_script_writes_new_files_only(tmp_path):
    spec = importlib.util.spec_from_file_location("derive_pc", ROOT / "scripts" / "derive_phase_c_corrections.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["derive_pc"] = mod
    spec.loader.exec_module(mod)
    folder = tmp_path / "C3" / "removal"
    folder.mkdir(parents=True)
    make_c3().to_csv(folder / "results.csv", index=False)
    before = (folder / "results.csv").read_bytes()
    path = mod.derive_c3(folder)
    assert path.name == "aggregate_representative.csv" and (folder / "results.csv").read_bytes() == before
    assert list(pd.read_csv(path)["variant"]) == ["warm", "cold", "noise"]
