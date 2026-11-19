import json

import numpy as np
import pandas as pd
import pytest

from delivery_routing_vrptw import v2
from delivery_routing_vrptw.config import V2Section, load_config

SPLIT = {"A1": "dev", "A2": "dev", "A3": "dev", "E1": "eval", "E2": "eval"}


def make_rows(plans, level_col="scale", extra=None, ref=10):
    """plans: {(instance, level): [(vehicles, distance), ...]} one tuple per replicate."""
    rows = []
    for (inst, level), reps in plans.items():
        for r, (veh, dist) in enumerate(reps):
            row = {
                "instance": inst, "scale": 100, "k": 10.0, "fleet_strategy": "fixed_cost", "replicate": r,
                "feasible": veh is not None, "vehicles": veh if veh is not None else float("nan"),
                "distance": dist if veh is not None else float("nan"), "ref_vehicles": ref,
                "delta_vehicles": (veh - ref) if veh is not None else float("nan"),
                "gap_distance_pct": 100 * (dist - 1000) / 1000 if veh is not None else float("nan"),
                "distance_comparable": veh == ref, "runtime_s": 1.0,
            }
            row.update(extra or {})
            row[level_col] = level
            rows.append(row)
    return pd.DataFrame(rows)


def test_select_dev_instances_refuses_eval():
    assert v2.select_dev_instances(SPLIT) == ["A1", "A2", "A3"]
    assert v2.select_dev_instances(SPLIT, ["a2"]) == ["A2"]
    with pytest.raises(PermissionError):
        v2.select_dev_instances(SPLIT, instance_set="eval")
    with pytest.raises(PermissionError):
        v2.select_dev_instances(SPLIT, instance_set="all")
    with pytest.raises(PermissionError):
        v2.select_dev_instances(SPLIT, ["A1", "E1"])
    with pytest.raises(ValueError):
        v2.select_dev_instances({"E1": "eval"})


def test_block_cells():
    cfg = V2Section()
    names = ["A1", "A2"]
    b1 = v2.block_cells("B1", names, cfg)
    assert len(b1) == 2 * 2 * 3 and {c["scale"] for c in b1} == {100, 10000} and {c["k"] for c in b1} == {10.0}
    b2 = v2.block_cells("B2", names, cfg, scale=10000)
    assert {c["k"] for c in b2} == {1.0, 100.0} and len(b2) == 12
    b3 = v2.block_cells("B3", names, cfg, scale=10000, k=10)
    assert {c["fleet_strategy"] for c in b3} == {"reduce_fleet_v2"} and len(b3) == 6
    b4 = v2.block_cells("B4", names, cfg, scale=10000, k=10, fleet_strategy="fixed_cost", replicates=2)
    assert len(b4) == 4
    assert v2.block_workers("B4", cfg) == 4 and v2.block_workers("B1", cfg) == 8
    for bad in (
        lambda: v2.block_cells("B2", names, cfg),
        lambda: v2.block_cells("B3", names, cfg, scale=100),
        lambda: v2.block_cells("B4", names, cfg, scale=100, k=10),
        lambda: v2.block_cells("B9", names, cfg),
    ):
        with pytest.raises(ValueError):
            bad()


def test_cell_key_normalises_types():
    cell = {"instance": "A1", "scale": "100", "k": "10.0", "fleet_strategy": "fixed_cost", "replicate": "2"}
    assert v2.cell_key(cell) == ("A1", 100, "10", "fixed_cost", 2)


def test_representative_is_position_two_of_three():
    rows = make_rows({("A1", 100): [(5, 90.0), (4, 80.0), (4, 70.0)]})
    rep = v2.representative_table(rows, "scale")
    assert len(rep) == 1
    # lexicographic order: (4, 70) rep 2, (4, 80) rep 1, (5, 90) rep 0 -> position 2 is replicate 1
    assert int(rep.iloc[0]["replicate"]) == 1 and rep.iloc[0]["distance"] == 80.0
    inf = make_rows({("A1", 100): [(None, 0.0), (6, 1.0), (5, 2.0)]})
    assert int(v2.representative_table(inf, "scale").iloc[0]["vehicles"]) == 6
    uneven = make_rows({("A1", 100): [(5, 1.0)] * 3, ("A1", 10000): [(5, 1.0)] * 2})
    with pytest.raises(ValueError):
        v2.representative_table(uneven, "scale")


def plans(better_at, instances=("A1", "A2", "A3")):
    out = {}
    for i, inst in enumerate(instances):
        base = 1000.0 + i
        out[(inst, 100)] = [(10, base + 5)] * 3
        out[(inst, 10000)] = [(10, base + 5 if better_at is None else base)] * 3
    return out


def test_b1_clear_winner_by_mean_rank():
    rows = make_rows(plans(better_at=10000))
    dec, ranking = v2.decision_record("B1", rows)
    assert dec["winner"] == 10000 and dec["decided_by"] == "mean_rank"
    assert ranking.iloc[0]["mean_rank"] == 1.0 and ranking.iloc[1]["mean_rank"] == 2.0
    assert dec["n_instances"] == 3
    json.dumps(dec)


def test_b1_scale_100_wins_when_clearly_better():
    p = plans(better_at=100)
    for key in list(p):
        if key[1] == 100:
            p[key] = [(9, 1000.0)] * 3
    dec, _ = v2.decision_record("B1", make_rows(p))
    assert dec["winner"] == 100 and dec["decided_by"] == "mean_rank"


def test_b1_exact_tie_prefers_10000():
    dec, ranking = v2.decision_record("B1", make_rows(plans(better_at=None)))
    assert dec["winner"] == 10000 and dec["decided_by"] == "protocol_preference"
    assert list(ranking["mean_rank"]) == [1.5, 1.5]


def test_tiebreak_by_delta_vehicles_when_ranks_tie():
    # opposite orders on two instances -> equal mean rank
    p = {
        ("A1", 100): [(9, 1000.0)] * 3, ("A1", 10000): [(10, 1000.0)] * 3,
        ("A2", 100): [(11, 900.0)] * 3, ("A2", 10000): [(10, 900.0)] * 3,
    }
    ranking, _ = v2.rank_levels(make_rows(p), "scale")
    assert list(ranking["mean_rank"]) == [1.5, 1.5]
    dec, _ = v2.decision_record("B1", make_rows(p))
    # mean delta vehicles is 0 for both levels; the median gap among fleet-equal instances decides
    assert dec["winner"] == 10000 and dec["decided_by"] == "tiebreak_median_gap"
    p2 = {
        ("A1", 100): [(9, 1000.0)] * 3, ("A1", 10000): [(10, 1000.0)] * 3,
        ("A2", 100): [(12, 900.0)] * 3, ("A2", 10000): [(10, 900.0)] * 3,
    }
    dec2, ranking2 = v2.decision_record("B1", make_rows(p2))
    assert list(ranking2["mean_rank"]) == [1.5, 1.5]
    # mean delta vehicles: scale 100 -> (-1 + 2) / 2 = 0.5; scale 10000 -> 0, so 10000 wins by tie-break
    assert dec2["winner"] == 10000 and dec2["decided_by"] == "tiebreak_delta_vehicles"
    p3 = {
        ("A1", 100): [(8, 1000.0)] * 3, ("A1", 10000): [(10, 1000.0)] * 3,
        ("A2", 100): [(11, 900.0)] * 3, ("A2", 10000): [(10, 900.0)] * 3,
    }
    dec3, _ = v2.decision_record("B1", make_rows(p3))
    # mean delta vehicles: scale 100 -> (-2 + 1) / 2 = -0.5 < 0, so scale 100 wins by tie-break
    assert dec3["winner"] == 100 and dec3["decided_by"] == "tiebreak_delta_vehicles"


def test_b2_tie_keeps_k10_and_better_k_wins():
    base = {}
    for inst in ("A1", "A2"):
        for k in (1.0, 10.0, 100.0):
            base[(inst, k)] = [(10, 1000.0)] * 3
    tie, _ = v2.decision_record("B2", make_rows(base, "k"), scale=100)
    assert tie["winner"] == 10.0 and tie["decided_by"] == "protocol_preference"
    base[("A1", 100.0)] = [(9, 990.0)] * 3
    base[("A2", 100.0)] = [(10, 990.0)] * 3
    win, _ = v2.decision_record("B2", make_rows(base, "k"), scale=100)
    assert win["winner"] == 100.0 and win["decided_by"] == "mean_rank"


def test_b2_uses_only_rows_of_the_chosen_scale():
    chosen = pd.concat([
        make_rows({("A1", 10.0): [(10, 1000.0)] * 3, ("A2", 10.0): [(10, 1000.0)] * 3}, "k",
                  extra={"scale": 10000}),
        make_rows({("A1", 1.0): [(10, 1000.0)] * 3, ("A2", 1.0): [(10, 1000.0)] * 3}, "k",
                  extra={"scale": 10000}),
    ])
    other = make_rows({("A1", 10.0): [(1, 1.0)] * 3, ("A2", 10.0): [(1, 1.0)] * 3}, "k", extra={"scale": 100})
    _, ranking = v2.decision_record("B2", pd.concat([chosen, other]), scale=10000)
    assert set(ranking["k"]) == {1.0, 10.0}
    assert (ranking["mean_rank"] == 1.5).all()


def test_b3_requires_strictly_better_mean_rank():
    fc = {("A1", "fixed_cost"): [(10, 1000.0)] * 3, ("A2", "fixed_cost"): [(10, 1000.0)] * 3}
    same = {("A1", "reduce_fleet_v2"): [(10, 1000.0)] * 3, ("A2", "reduce_fleet_v2"): [(10, 1000.0)] * 3}
    dec, _ = v2.decision_record("B3", make_rows({**fc, **same}, "fleet_strategy"), scale=100, k=10)
    assert dec["winner"] == "fixed_cost" and dec["decided_by"] == "not_strictly_better"
    better = {("A1", "reduce_fleet_v2"): [(9, 1000.0)] * 3, ("A2", "reduce_fleet_v2"): [(10, 1000.0)] * 3}
    dec, _ = v2.decision_record("B3", make_rows({**fc, **better}, "fleet_strategy"), scale=100, k=10)
    assert dec["winner"] == "reduce_fleet_v2" and dec["decided_by"] == "strictly_better_mean_rank"
    # a lower mean delta of vehicles alone (equal mean rank) is not enough
    swap_fc = {("A1", "fixed_cost"): [(9, 1000.0)] * 3, ("A2", "fixed_cost"): [(11, 1000.0)] * 3}
    swap_new = {("A1", "reduce_fleet_v2"): [(10, 1000.0)] * 3, ("A2", "reduce_fleet_v2"): [(10, 1000.0)] * 3}
    dec, _ = v2.decision_record("B3", make_rows({**swap_fc, **swap_new}, "fleet_strategy"), scale=100, k=10)
    assert dec["winner"] == "fixed_cost"
    worse = {("A1", "reduce_fleet_v2"): [(11, 1000.0)] * 3, ("A2", "reduce_fleet_v2"): [(10, 900.0)] * 3}
    dec, _ = v2.decision_record("B3", make_rows({**fc, **worse}, "fleet_strategy"), scale=100, k=10)
    assert dec["winner"] == "fixed_cost"


def test_levels_must_cover_same_instances():
    rows = make_rows({
        ("A1", 100): [(10, 1.0)] * 3, ("A1", 10000): [(10, 1.0)] * 3, ("A2", 100): [(10, 1.0)] * 3,
    })
    with pytest.raises(ValueError):
        v2.rank_levels(rows, "scale")


def test_instance_check_reports_per_level_vehicles():
    p = {("R110", 100): [(11, 1.0), (11, 1.0), (12, 1.0)], ("R110", 10000): [(11, 1.0), (10, 1.0), (10, 1.0)]}
    out = v2.instance_check(make_rows(p, ref=10), "R110", "scale")
    assert out["10000"]["representative_vehicles"] == 10 and out["100"]["representative_vehicles"] == 11
    assert out["10000"]["ref_vehicles"] == 10
    assert v2.instance_check(make_rows(p), "ZZZ", "scale") is None
    both = {**p, ("A1", 100): [(10, 1.0)] * 3, ("A1", 10000): [(10, 1.0)] * 3}
    dec, _ = v2.decision_record("B1", make_rows(both))
    assert dec["r110"]["10000"]["representative_vehicles"] == 10


def test_json_safe_handles_nan_and_numpy():
    assert v2.json_safe({"a": np.int64(3), "b": float("nan"), "c": [np.float64(1.5)]}) == {
        "a": 3, "b": None, "c": [1.5]}


def window_rows(vehicles_by_factor, instance="R101"):
    out = []
    for i, (f, v) in enumerate(vehicles_by_factor.items()):
        out.append({"instance": instance, "family": "R1", "factor": f, "mode": "symmetric", "feasible": True,
                    "vehicles": v, "distance": 1000.0 - 10 * i, "duration": 1.0, "waiting": 0.0,
                    "replicate": 0, "warm_start": "used"})
    return pd.DataFrame(out)


def test_layerb_windows_summary_and_nonmonotonic():
    parts = v2.layerb_summary("windows", window_rows({1.0: 10, 1.25: 10, 1.5: 9}))
    assert parts["nonmonotonic"].empty
    assert list(parts["summary"]["vehicles_saved"]) == [0, 0, 1]
    nm = v2.layerb_summary("windows", window_rows({1.0: 10, 1.25: 11, 1.5: 9}))["nonmonotonic"]
    assert len(nm) == 1 and nm.iloc[0]["instance"] == "R101"
    with pytest.raises(ValueError):
        v2.layerb_summary("nope", window_rows({1.0: 10}))


def early_rows(levels, instance="R101", reps=3):
    out = []
    for lab, (veh, dist) in levels.items():
        for r in range(reps):
            out.append({
                "instance": instance, "family": "R1", "tolerance_label": lab, "replicate": r, "feasible": True,
                "vehicles": veh, "distance": dist, "total_duration": 100.0, "total_waiting": 5.0,
                "total_duration_min": 90.0, "total_waiting_min": 2.0, "makespan": 50.0, "warm_start": "used",
            })
    return pd.DataFrame(out)


def test_layerb_early_summary_and_comparison():
    v1 = early_rows({"0": (10, 1000.0), "0.5": (10, 1010.0), "full": (9, 900.0)})
    new = early_rows({"0": (10, 1000.0), "0.5": (10, 990.0), "full": (9, 900.0)})
    parts = v2.layerb_summary("early", v1)
    assert len(parts["nonmonotonic"]) == 1  # 0.5 is longer than the baseline at equal fleet
    assert not parts["by_level"].empty and not parts["fleet_split"].empty
    comp = v2.compare_layerb("early", pd.concat([v1, early_rows({"0": (10, 1.0)}, instance="R102")]), new)
    nm = comp["nonmonotonic"].set_index("version")
    assert nm.loc["v1", "n_nonmonotonic"] == 1 and nm.loc["v2", "n_nonmonotonic"] == 0
    assert nm.loc["v1", "n_distance"] == 1 and nm.loc["v1", "n_fleet"] == 0
    levels = comp["levels"].set_index("level")
    assert levels.loc["full", "v2_vehicles_saved_mean"] == 1 and levels.loc["full", "v1_n_instances"] == 1


def test_compare_layerb_windows():
    v1 = window_rows({1.0: 10, 1.25: 10, 1.5: 10})
    new = window_rows({1.0: 10, 1.25: 10, 1.5: 9})
    lv = v2.compare_layerb("windows", v1, new)["levels"].set_index("level")
    assert lv.loc[1.5, "v1_vehicles_saved_total"] == 0 and lv.loc[1.5, "v2_vehicles_saved_total"] == 1


def test_v2_section_validation_and_toml(tmp_path):
    with pytest.raises(ValueError):
        V2Section(k_levels=(1.0, 100.0))  # must include k_default
    for bad in ({"scales": ()}, {"scales": (0,)}, {"first_fraction": 1.0}, {"time_limit_s": 0},
                {"replicates": 0}, {"b4_workers": 0}, {"drop_penalty_factor": 0}, {"k_levels": ()}):
        with pytest.raises(ValueError):
            V2Section(**bad)
    path = tmp_path / "c.toml"
    path.write_text("[v2]\nscales = [100]\nk_levels = [10, 50]\nk_default = 10\nreplicates = 2\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.v2.scales == (100,) and cfg.v2.k_levels == (10, 50) and cfg.v2.replicates == 2
    assert json.dumps(cfg.to_dict())
    path.write_text("[v2]\nbogus = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_project_toml_v2_section_matches_protocol():
    cfg = load_config()
    assert cfg.v2 == V2Section()
    assert cfg.v2.scales == (100, 10000) and cfg.v2.k_levels == (1, 10, 100) and cfg.v2.time_limit_s == 120
    assert cfg.v2.workers == 8 and cfg.v2.b4_workers == 4 and cfg.v2.step_time_s == 20
    assert cfg.v2.b5_time_limit_s == 30 and cfg.v2.replicates == 3
    assert abs(cfg.v2.first_fraction - 1 / 3) < 1e-15
    assert cfg.solver.fleet_strategy == "fixed_cost" and cfg.solver.time_limit_s == 30  # v1 untouched


def test_run_metadata_has_versions():
    meta = v2.run_metadata({"a": 1}, "cfg.toml", "proto.md", ["A1"], block="B1")
    assert meta["protocol"] == "proto.md" and meta["ortools"] and meta["block"] == "B1"
    assert meta["instances"] == ["A1"]
