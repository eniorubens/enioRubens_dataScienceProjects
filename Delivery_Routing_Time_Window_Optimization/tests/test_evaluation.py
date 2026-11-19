import math

import pandas as pd
import pytest

from delivery_routing_vrptw.evaluation import (
    aggregate_table,
    compare_versions,
    instance_table,
    median_position,
    sort_replicates,
    v2_eval_guard,
)


def row(instance, method, replicate, vehicles, distance, ref_v, ref_d, family=None, feasible=True):
    family = family or ("C1" if instance.startswith("C") else "R1")
    if not feasible:
        return {
            "instance": instance, "family": family, "method": method, "replicate": replicate,
            "feasible": False, "vehicles": math.nan, "distance": math.nan, "runtime_s": 1.0,
            "ref_vehicles": ref_v, "ref_distance": ref_d, "delta_vehicles": math.nan,
            "gap_distance_pct": math.nan, "distance_comparable": False, "beats_reference": False,
        }
    delta = vehicles - ref_v
    gap = 100 * (distance - ref_d) / ref_d
    return {
        "instance": instance, "family": family, "method": method, "replicate": replicate,
        "feasible": True, "vehicles": vehicles, "distance": distance, "runtime_s": 1.0,
        "ref_vehicles": ref_v, "ref_distance": ref_d, "delta_vehicles": delta,
        "gap_distance_pct": gap, "distance_comparable": delta == 0,
        "beats_reference": delta < 0 or (delta == 0 and distance < ref_d - 0.005),
    }


def ort(instance, reps, ref_v=10, ref_d=100.0, **kw):
    return [row(instance, "ortools", i + 1, v, d, ref_v, ref_d, **kw) if v is not None
            else row(instance, "ortools", i + 1, 0, 0, ref_v, ref_d, feasible=False, **kw)
            for i, (v, d) in enumerate(reps)]


def test_median_position():
    assert [median_position(n) for n in (1, 2, 3, 4, 5)] == [0, 0, 1, 1, 2]
    with pytest.raises(ValueError):
        median_position(0)


def test_sort_and_median_selection():
    reps = [(11, 90.0), (10, 120.0), (10, 105.0), (12, 80.0), (10, 101.0)]
    df = pd.DataFrame(ort("C101", reps))
    s = sort_replicates(df)
    assert list(s["replicate"]) == [5, 3, 2, 1, 4]
    t = instance_table(df)
    assert len(t) == 1
    r = t.iloc[0]
    assert r["median_replicate"] == 2 and r["vehicles"] == 10 and r["distance"] == 120.0
    assert r["best_of_R_replicate"] == 5 and r["best_of_R_distance"] == 101.0
    assert r["vehicles_min"] == 10 and r["vehicles_max"] == 12 and r["vehicles_range"] == 2
    assert r["distance_min"] == 80.0 and r["distance_max"] == 120.0
    assert r["distance_range_pct"] == pytest.approx(50.0)
    assert r["n_feasible"] == 5 and r["n_replicates"] == 5


def test_ties_use_replicate_id():
    df = pd.DataFrame(ort("C101", [(10, 100.0)] * 3 + [(9, 150.0), (11, 80.0)]))
    t = instance_table(df).iloc[0]
    # order: rep4 (9), reps 1,2,3 (10), rep5 (11); median = 3rd = rep2
    assert t["median_replicate"] == 2 and t["best_of_R_replicate"] == 4


def test_infeasible_replicate_goes_last():
    df = pd.DataFrame(ort("C101", [(None, None), (10, 100.0), (10, 110.0), (10, 120.0), (None, None)]))
    t = instance_table(df).iloc[0]
    assert t["n_feasible"] == 3 and t["median_replicate"] == 4 and t["distance"] == 120.0
    # majority infeasible -> median replicate infeasible
    df2 = pd.DataFrame(ort("C101", [(None, None)] * 3 + [(10, 100.0)] * 2))
    t2 = instance_table(df2).iloc[0]
    assert not t2["feasible"] and math.isnan(t2["vehicles"]) and not t2["beats_reference"]
    assert t2["best_of_R_vehicles"] == 10


def test_gap_only_when_comparable():
    rows = ort("C101", [(11, 90.0)] * 5, ref_v=10) + ort("R101", [(10, 110.0)] * 5, ref_v=10)
    t = instance_table(pd.DataFrame(rows)).set_index("instance")
    assert math.isnan(t.loc["C101", "gap_distance_pct"]) and not t.loc["C101", "distance_comparable"]
    assert t.loc["R101", "gap_distance_pct"] == pytest.approx(10.0)
    assert not t.loc["R101", "beats_reference"]
    # an inconsistent gap value in a non-comparable row is still masked
    bad = pd.DataFrame(rows)
    bad["gap_distance_pct"] = 5.0
    assert math.isnan(instance_table(bad).set_index("instance").loc["C101", "gap_distance_pct"])


def test_aggregate_with_baselines():
    rows = (
        ort("C101", [(10, 95.0)] * 5, family="C1")        # equal fleet, beats on distance
        + ort("C102", [(11, 120.0)] * 5, family="C1")     # +1 vehicle
        + ort("R101", [(9, 300.0), (10, 90.0), (10, 100.0), (10, 100.0), (10, 100.0)], family="R1")
    )
    rows += [
        row("C101", "greedy_deadline", 1, 12, 130.0, 10, 100.0, "C1"),
        row("C102", "greedy_deadline", 1, 13, 140.0, 10, 100.0, "C1"),
        row("R101", "greedy_deadline", 1, 10, 120.0, 10, 100.0, "R1"),
        row("C101", "nearest_neighbor_tw", 1, 10, 100.0, 10, 100.0, "C1"),
    ]
    agg = aggregate_table(pd.DataFrame(rows))
    o = agg[(agg["scope"] == "overall") & (agg["method"] == "ortools")].iloc[0]
    # medians: C101 (10,95) ; C102 (11,120) ; R101 sorted -> 9, 10(90), 10(100)x3 -> 3rd = (10,100)
    assert o["n_instances"] == 3 and o["n_fleet_equal"] == 2 and o["n_beats_reference"] == 1
    assert o["mean_delta_vehicles"] == pytest.approx(1 / 3)
    assert o["median_gap_fleet_equal"] == pytest.approx((-5.0 + 0.0) / 2)
    # best of R: R101 reaches 9 vehicles -> beats reference
    assert o["best_of_R_n_beats_reference"] == 2 and o["best_of_R_n_fleet_equal"] == 1
    assert o["best_of_R_mean_delta_vehicles"] == pytest.approx((0 + 1 - 1) / 3)
    c1 = agg[(agg["scope"] == "C1") & (agg["method"] == "ortools")].iloc[0]
    assert c1["n_instances"] == 2 and c1["n_fleet_equal"] == 1
    g = agg[(agg["scope"] == "overall") & (agg["method"] == "greedy_deadline")].iloc[0]
    assert g["n_instances"] == 3 and g["n_fleet_equal"] == 1 and g["n_beats_reference"] == 0
    assert g["mean_delta_vehicles"] == pytest.approx(5 / 3)
    assert g["median_gap_fleet_equal"] == pytest.approx(20.0) and math.isnan(g["best_of_R_n_fleet_equal"])
    n = agg[(agg["scope"] == "overall") & (agg["method"] == "nearest_neighbor_tw")].iloc[0]
    assert n["n_instances"] == 1 and n["n_fleet_equal"] == 1 and n["median_gap_fleet_equal"] == 0.0
    assert set(agg["scope"]) == {"overall", "C1", "R1"}


def test_no_baselines():
    agg = aggregate_table(pd.DataFrame(ort("C101", [(10, 100.0)] * 3)))
    assert set(agg["method"]) == {"ortools"}


# --------------------------------------------------------------------------- v1 x v2
def _version_rows(spec):
    """spec: instance -> (replicates, ref_v, ref_d)."""
    rows = []
    for inst, (reps, ref_v, ref_d) in spec.items():
        fam = "RC1" if inst.startswith("RC") else None
        rows += ort(inst, reps, ref_v=ref_v, ref_d=ref_d, family=fam)
    return pd.DataFrame(rows)


def _compare():
    v1 = _version_rows({
        "C101": ([(10, 100.0)] * 3, 10, 100.0),     # identical in v2
        "C102": ([(10, 100.0)] * 3, 10, 100.0),     # +0.3% in v2: within noise
        "R101": ([(11, 200.0)] * 3, 10, 190.0),     # v2 saves one vehicle
        "R106": ([(13, 220.0)] * 3, 11, 210.0),     # v2 saves one vehicle
        "RC101": ([(12, 300.0)] * 3, 11, 290.0),    # same fleet, v2 -2%
        "RC102": ([(12, 300.0)] * 3, 11, 290.0),    # v2 infeasible
    })
    v2 = _version_rows({
        "C101": ([(10, 100.0)] * 3, 10, 100.0),
        "C102": ([(10, 100.3)] * 3, 10, 100.0),
        "R101": ([(10, 195.0)] * 3, 10, 190.0),
        "R106": ([(12, 230.0)] * 3, 11, 210.0),
        "RC101": ([(12, 294.0)] * 3, 11, 290.0),
        "RC102": ([(None, None)] * 3, 11, 290.0),
        "C201": ([(3, 600.0)] * 3, 3, 600.0),        # only in v2: ignored
    })
    return compare_versions(v1, v2)


def test_compare_versions_by_instance():
    d, _ = _compare()
    assert list(d["instance"]) == ["C101", "C102", "R101", "R106", "RC101", "RC102"]
    d = d.set_index("instance")
    assert d["delta_vehicles_v2_minus_v1"].loc[["C101", "R101", "R106"]].tolist() == [0, -1, -1]
    assert d["fleet_equal_versions"].tolist() == [True, True, False, False, True, False]
    assert d["distance_diff_pct"].loc["C101"] == 0
    assert d["distance_diff_pct"].loc["RC101"] == pytest.approx(-2.0)
    assert d["distance_diff_pct"].loc["C102"] == pytest.approx(0.3)
    # distance difference only when fleets are equal
    assert d["distance_diff_pct"].isna().loc[["R101", "R106", "RC102"]].all()
    assert d["within_noise"].tolist() == [True, True, False, False, False, False]
    assert d["v2_feasible"].loc["RC102"] == False  # noqa: E712
    assert pd.isna(d["delta_vehicles_v2_minus_v1"].loc["RC102"])


def test_noise_flag_threshold():
    v1 = _version_rows({"C101": ([(10, 100.0)] * 3, 10, 100.0), "C102": ([(10, 100.0)] * 3, 10, 100.0)})
    v2 = _version_rows({"C101": ([(10, 100.5)] * 3, 10, 100.0), "C102": ([(10, 99.6)] * 3, 10, 100.0)})
    d, _ = compare_versions(v1, v2)
    assert d["within_noise"].tolist() == [False, True]  # |0.5| is not < 0.5; |-0.4| is


def test_compare_versions_aggregate():
    _, agg = _compare()
    assert list(agg["scope"]) == ["overall", "C1", "R1", "RC1"]
    o = agg.set_index("scope").loc["overall"]
    assert o["n_instances"] == 6
    assert o["v1_mean_delta_vehicles"] == pytest.approx((0 + 0 + 1 + 2 + 1 + 1) / 6)
    assert o["v1_n_fleet_equal"] == 2
    assert o["v2_n_fleet_equal"] == 3          # C101, C102, R101 (RC101 has +1; RC102 infeasible)
    assert o["n_fleet_equal_versions"] == 3
    assert o["n_within_noise"] == 2
    fam = agg.set_index("scope")
    assert fam.loc["R1", "mean_delta_vehicles_v2_minus_v1"] == pytest.approx(-1.0)
    assert fam.loc["C1", "v2_median_gap_fleet_equal"] == pytest.approx(0.15)


def test_v2_eval_guard():
    p = "reports/protocolo-fase-d.md"
    v2_eval_guard("dev", False, False, p)  # smoke on dev needs no flags
    for confirm, post_hoc, missing in ((False, False, "--confirm-eval and --post-hoc"),
                                       (True, False, "--post-hoc"), (False, True, "--confirm-eval")):
        with pytest.raises(PermissionError, match=missing) as exc:
            v2_eval_guard("eval", confirm, post_hoc, p)
        assert p in str(exc.value)
    with pytest.raises(PermissionError):
        v2_eval_guard("all", True, False, p)
    v2_eval_guard("eval", True, True, p)
