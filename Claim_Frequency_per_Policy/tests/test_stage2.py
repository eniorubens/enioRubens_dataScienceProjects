"""Testes sinteticos da etapa 2: particao, perfis, segmentos, metricas."""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from freq_sinistros import design, metrics, partition, profiles, segments
from freq_sinistros.acquire import hash_file
from freq_sinistros.protocol import nested_train_sample, protocol_config

HEADER = """@relation toy
@attribute IDpol numeric
@attribute ClaimNb numeric
@attribute Exposure numeric
@attribute Area {'A','B','C'}
@attribute VehPower numeric
@attribute VehAge numeric
@attribute DrivAge numeric
@attribute BonusMalus numeric
@attribute VehBrand {'B1','B2'}
@attribute VehGas string
@attribute Density numeric
@attribute Region {'R11','R21'}
@data
"""


def synthetic_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "IDpol": np.arange(1, n + 1, dtype="int64") * 3,
        "ClaimNb": rng.poisson(0.1, n).astype(float),
        "Exposure": np.round(rng.uniform(0.01, 1.2, n), 2),
        "Area": pd.Categorical(rng.choice(["A", "B", "C"], n)),
        "VehPower": rng.integers(4, 8, n).astype(float),
        "VehAge": rng.integers(0, 25, n).astype(float),
        "DrivAge": rng.integers(18, 80, n).astype(float),
        "BonusMalus": rng.integers(50, 120, n).astype(float),
        "VehBrand": pd.Categorical(rng.choice(["B1", "B2"], n)),
        "VehGas": pd.Categorical(rng.choice(["Diesel", "Regular"], n)),
        "Density": rng.integers(1, 5, n).astype(float),
        "Region": pd.Categorical(rng.choice(["R11", "R21"], n)),
    })


# ---------------------------------------------------------- particao
def test_canonical_id_accepts_scientific_and_rejects_bad_values():
    assert partition.canonical_id("1e+05") == 100000
    assert partition.canonical_id(100000.0) == 100000
    assert partition.canonical_id(np.int64(7)) == 7
    for bad in ("1.5", float("nan"), None, "x", "-3", 2.5, float("inf"),
                True):
        with pytest.raises(ValueError):
            partition.canonical_id(bad)


def test_bucket_matches_the_documented_rule():
    digest = hashlib.sha256(b"freMTPL2freq-v1-etapa2|42").digest()
    expected = int.from_bytes(digest[:8], "big") % 10000
    assert partition.bucket(42) == expected
    b = expected
    name = ("train" if b < 7000 else "validation" if b < 8500 else "test")
    assert partition.partition_of(42) == name


def test_assignments_cover_are_disjoint_and_order_invariant():
    ids = pd.Series(np.arange(1, 20001, dtype="int64"))
    a = partition.build_assignments(ids)
    shuffled = ids.sample(frac=1.0, random_state=3)
    b = partition.build_assignments(shuffled)
    assert partition.assignment_hash(a) == partition.assignment_hash(b)
    assert partition.assignment_hash(a) == partition.assignment_hash(
        partition.build_assignments(ids))
    checks = partition.check_partition(a, ids)
    assert checks["full_coverage"] and checks["disjoint"]
    assert checks["sizes_sum_to_population"] and checks["assigned_ids_unique"]
    share = partition.partition_sizes(a).set_index("partition")["share"]
    assert abs(share["train"] - 0.70) < 0.02
    assert abs(share["validation"] - 0.15) < 0.02
    assert abs(share["test"] - 0.15) < 0.02


def test_repeated_or_invalid_ids_are_refused():
    with pytest.raises(ValueError, match="repetidos"):
        partition.build_assignments(pd.Series([1, 2, 2]))
    with pytest.raises(ValueError, match="repetidos"):
        partition.build_assignments(pd.Series(["100000", "1e+05"]))
    with pytest.raises(ValueError):
        partition.build_assignments(pd.Series([1.0, float("nan")]))
    with pytest.raises(ValueError):
        partition.build_assignments(pd.Series([1.0, 2.5]))


def test_check_partition_detects_missing_and_overlap():
    ids = pd.Series([1, 2, 3])
    assignments = pd.DataFrame({"IDpol": [1, 2],
                                "partition": ["train", "test"]})
    assert not partition.check_partition(assignments, ids)["full_coverage"]
    dup = pd.DataFrame({"IDpol": [1, 1, 2, 3],
                        "partition": ["train", "test", "train", "test"]})
    result = partition.check_partition(dup, ids)
    assert result["overlap_ids"] == 1 and not result["disjoint"]


# ------------------------------------------------------------ perfis
def test_profile_key_is_stable_and_marks_missing():
    frame = synthetic_frame(5)
    frame.loc[0, "VehAge"] = np.nan
    frame.loc[1, "Density"] = 2.5
    keys = profiles.profile_keys(frame)
    assert keys[0].split("|")[2] == "__MISSING__"
    assert keys[1].split("|")[6] == "2.5"
    assert keys[2].split("|")[2] == str(int(frame.loc[2, "VehAge"]))
    assert len(keys[2].split("|")) == len(profiles.PROFILE_COLUMNS)
    assert (profiles.profile_keys(frame) == keys).all()


def test_grouped_partition_keeps_keys_together_and_excludes_test():
    frame = synthetic_frame(3000)
    frame["Density"] = 1.0  # forca muitas chaves repetidas
    assignments = partition.build_assignments(frame["IDpol"])
    part = design.attach_partition(frame, assignments)
    dev_mask = part.isin(["train", "validation"])
    grouped = profiles.grouped_assignments(frame, dev_mask)
    assert len(grouped) == int(dev_mask.sum())
    assert set(grouped["IDpol"]) == set(frame.loc[dev_mask, "IDpol"])
    summary = profiles.grouped_summary(frame, grouped)
    assert summary["key_overlap_between_groups"] == 0
    assert summary["group_train_rows"] + summary[
        "group_validation_rows"] == summary["dev_rows"]
    assert summary["distinct_keys"] < summary["dev_rows"]


# ---------------------------------------------------------- segmentos
def test_vehicle_age_bands_have_explicit_limits():
    ages = pd.Series([0, 1, 2, 5, 6, 10, 11, 20, 21, 100, -1, np.nan])
    got = segments.vehicle_age_band(ages).tolist()
    assert got == ["0-1", "0-1", "2-5", "2-5", "6-10", "6-10", "11-20",
                   "11-20", "21+", "21+", segments.VEHAGE_INVALID,
                   segments.VEHAGE_INVALID]


def test_region_missing_category_and_support_flag():
    region = pd.Series(pd.Categorical(["R11", None, "R21"]))
    assert segments.region_segment(region).tolist() == [
        "R11", "__MISSING__", "R21"]
    assert segments.low_support(499, 1000.0)
    assert segments.low_support(1000, 24.9)
    assert not segments.low_support(500, 25.0)


def test_segment_tables_reconcile_with_totals():
    frame = synthetic_frame(800)
    frame.loc[0, "Region"] = np.nan
    frame.loc[1, "VehAge"] = np.nan
    table = segments.segment_table(frame, with_outcome=True)
    recon = segments.reconcile_segments(table, frame, True)
    assert recon["ok"].all()
    assert set(recon["family"]) == set(segments.FAMILIES)
    no_outcome = segments.segment_table(frame, with_outcome=False)
    assert "claims" not in no_outcome.columns


# ---------------------------------------------- saidas sem uso do teste
def test_development_outputs_do_not_depend_on_test_outcomes():
    frame = synthetic_frame(4000)
    assignments = partition.build_assignments(frame["IDpol"])
    part = design.attach_partition(frame, assignments)
    base = design.development_outputs(frame, assignments)
    altered = frame.copy()
    test_rows = (part == "test").to_numpy()
    altered.loc[test_rows, "ClaimNb"] = 99.0
    altered.loc[test_rows, "Exposure"] = 7.0
    changed = design.development_outputs(altered, assignments)
    assert json.dumps(design._jsonable(base["train_overview"])) == (
        json.dumps(design._jsonable(changed["train_overview"])))
    for key in ("segment_support", "segment_reconciliation", "sizes",
                "grouped_assignments"):
        pd.testing.assert_frame_equal(base[key], changed[key])
    for key in ("profile_coincidence", "grouped_summary",
                "validation_overview", "test_overview"):
        assert base[key] == changed[key]
    assert set(base["test_overview"]) == {"ids", "note"}


def test_validation_support_columns_have_no_claims():
    frame = synthetic_frame(2000)
    out = design.development_outputs(
        frame, partition.build_assignments(frame["IDpol"]))
    columns = out["segment_support"].columns
    assert not any(c.startswith("validation_claims") for c in columns)
    assert "train_claims" in columns


# ------------------------------------------------------------ metricas
def test_poisson_deviance_zero_term_and_perfect_fit():
    y = np.array([0.0, 2.0])
    mu = np.array([0.5, 2.0])
    expected = 2 * ((0 - (0 - 0.5)) + (2 * math.log(1.0) - 0)) / 2
    assert metrics.mean_poisson_deviance(y, mu) == pytest.approx(expected)
    assert metrics.mean_poisson_deviance(y, y + 1e-12) == pytest.approx(
        1e-12, abs=1e-9)


def test_invalid_predictions_are_refused_not_dropped():
    for mu in ([0.0, 1.0], [np.nan, 1.0], [-1.0, 1.0], [np.inf, 1.0]):
        with pytest.raises(ValueError):
            metrics.mean_poisson_deviance([0.0, 1.0], mu)
    with pytest.raises(ValueError):
        metrics.mean_poisson_deviance([0.5, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        metrics.check_inputs([0.0], [1.0], [0.0])


def test_pmfs_normalise_and_nb2_tends_to_poisson():
    y = np.arange(0, 80, dtype=float)
    mu = np.full_like(y, 1.7)
    assert np.exp(metrics.poisson_logpmf(y, mu)).sum() == pytest.approx(1)
    assert np.exp(metrics.nb2_logpmf(y, mu, 0.5)).sum() == pytest.approx(1)
    near = metrics.nb2_logpmf(y, mu, 1e-8)
    assert near == pytest.approx(metrics.poisson_logpmf(y, mu), rel=1e-6)
    p0 = metrics.zero_prob(mu[:1], alpha=0.5)
    assert p0[0] == pytest.approx(
        math.exp(metrics.nb2_logpmf([0.0], [1.7], 0.5)[0]))
    assert metrics.zero_prob(mu[:1])[0] == pytest.approx(math.exp(-1.7))
    with pytest.raises(ValueError):
        metrics.nb2_logpmf(y, mu, 0.0)


def test_zero_and_rate_calibration_use_pred_minus_obs():
    cal = metrics.zero_calibration([0, 1, 0, 2], [0.6, 0.5, 0.7, 0.2])
    assert cal["zeros_observed"] == 2.0
    assert cal["zeros_predicted"] == pytest.approx(2.0)
    seg = pd.Series(["a", "a", "b"])
    table = metrics.calibration_table(
        seg, [1.0, 0.0, 2.0], [0.5, 0.5, 1.0], [0.5, 0.5, 1.0])
    a = table[table["segment"] == "a"].iloc[0]
    assert a["observed_rate_per_100"] == pytest.approx(100.0)
    assert a["predicted_rate_per_100"] == pytest.approx(100.0)
    b = table[table["segment"] == "b"].iloc[0]
    assert b["rate_difference_pred_minus_obs"] == pytest.approx(-100.0)
    assert table["low_support"].all()


def test_bootstrap_is_deterministic_paired_and_handles_undefined():
    rng = np.random.default_rng(1)
    y = rng.poisson(0.3, 400).astype(float)
    mu_a = np.full(400, 0.3)
    mu_b = np.full(400, 0.35)

    def stat(idx):
        return {
            "dev_a": metrics.mean_poisson_deviance(y[idx], mu_a[idx]),
            "dev_b": metrics.mean_poisson_deviance(y[idx], mu_b[idx]),
            "diff": metrics.mean_poisson_deviance(y[idx], mu_b[idx])
            - metrics.mean_poisson_deviance(y[idx], mu_a[idx]),
        }

    one = metrics.bootstrap_intervals(stat, 400, replicas=60, seed=5)
    two = metrics.bootstrap_intervals(stat, 400, replicas=60, seed=5)
    pd.testing.assert_frame_equal(one, two)
    row = one.set_index("indicator").loc["diff"]
    assert row["ci_low"] < row["ci_high"]
    assert row["estimate"] > 0  # mu_b = 0.35 e pior que mu_a = 0.3 aqui
    assert one["interval_reported"].all()

    def undefined(idx):
        return {"x": float("nan") if idx.size and idx[0] % 2 == 0 else 1.0}

    res = metrics.bootstrap_intervals(undefined, 10, replicas=40, seed=2)
    assert res.loc[0, "undefined_replicas"] > 0
    assert not res.loc[0, "interval_reported"]


def test_cluster_bootstrap_resamples_whole_groups():
    codes = np.array([0, 0, 1, 1, 1, 2])
    for idx in metrics.cluster_bootstrap_indices(codes, 3, 20):
        counts = np.bincount(codes[idx], minlength=3)
        assert counts[0] % 2 == 0 and counts[1] % 3 == 0


# ---------------------------------------------------- amostra e protocolo
def test_nested_sample_is_order_invariant_and_prefix_consistent():
    ids = pd.Series(np.arange(1, 501, dtype="int64"))
    small = nested_train_sample(ids, 50)
    large = nested_train_sample(ids, 120)
    shuffled = nested_train_sample(ids.sample(frac=1.0, random_state=4), 50)
    assert small.tolist() == shuffled.tolist()
    assert set(small) <= set(large) and len(large) == 120
    assert len(nested_train_sample(ids, None)) == 500


def test_protocol_config_is_serialisable_and_not_production():
    config = protocol_config()
    assert config["production_eligible"] is False
    assert config["uncertainty"]["replicas"] == 1000
    assert config["segments"]["low_support"]["rows_lt"] == 500
    json.dumps(config)


# -------------------------------------------- execucao completa sintetica
def _toy_project(tmp_path, n=400):
    frame = synthetic_frame(n)
    lines = []
    for r in frame.itertuples(index=False):
        lines.append(
            f"{r.IDpol},{int(r.ClaimNb)},{r.Exposure},'{r.Area}',"
            f"{int(r.VehPower)},{int(r.VehAge)},{int(r.DrivAge)},"
            f"{int(r.BonusMalus)},'{r.VehBrand}','{r.VehGas}',"
            f"{int(r.Density)},'{r.Region}'")
    raw = tmp_path / "data" / "raw" / "toy.arff"
    raw.parent.mkdir(parents=True)
    raw.write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    digest = hash_file(raw)
    out = tmp_path / "outputs" / "etapa_01"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({"artifact": {
        "local_path": "data/raw/toy.arff", "sha256": digest["sha256"],
        "size_bytes": digest["size_bytes"]}}), "utf-8")
    return raw


def test_run_partition_and_verify_roundtrip(tmp_path):
    _toy_project(tmp_path)
    assert design.run_partition(tmp_path) == 0
    config = json.loads(
        (tmp_path / "outputs" / "etapa_02" / "partition_config.json")
        .read_text("utf-8"))
    assert config["checks"]["full_coverage"] and config["checks"]["disjoint"]
    assert config["checks"]["invariant_to_row_order"]
    assert config["checks"]["token_1e+05_canonical_id"] == 100000
    assert config["production_eligible"] is False
    assignments = pd.read_csv(tmp_path / "data" / "splits"
                              / "partition_assignments.csv")
    assert list(assignments.columns) == ["IDpol", "partition"]
    overview = json.loads(
        (tmp_path / "outputs" / "etapa_02" / "overview.json")
        .read_text("utf-8"))
    assert set(overview["test"]) == {"ids", "note"}
    assert design.verify_partition(tmp_path) == 0
    # adulterar a atribuicao local deve ser detectado
    path = tmp_path / "data" / "splits" / "partition_assignments.csv"
    path.write_text(path.read_text().replace("train", "test", 1))
    assert design.verify_partition(tmp_path) == 1


def test_run_partition_refuses_modified_arff(tmp_path):
    raw = _toy_project(tmp_path)
    raw.write_text(raw.read_text() + "999999,0,0.5,'A',5,3,40,60,'B1',"
                   "'Diesel',2,'R11'\n")
    assert design.run_partition(tmp_path) == 2
    assert not (tmp_path / "outputs" / "etapa_02").exists()


def test_sort_segments_follows_declared_order():
    table = pd.DataFrame({
        "family": ["vehicle_age", "region", "vehicle_age", "exposure_band"],
        "segment": ["11-20", "R11", "2-5", "(0, 0.02]"]})
    ordered = segments.sort_segments(table)
    assert ordered["segment"].tolist() == ["(0, 0.02]", "2-5", "11-20",
                                           "R11"]
