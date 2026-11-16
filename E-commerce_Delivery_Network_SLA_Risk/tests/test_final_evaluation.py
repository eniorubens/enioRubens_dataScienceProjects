"""Tests for the single pre-registered final evaluation (synthetic data only)."""

from __future__ import annotations

import itertools
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import ecom_delivery_sla.final_evaluation as fe
from ecom_delivery_sla.final_evaluation import (
    classify_preregistered_outcome,
    paired_metric_difference,
    reliability_table,
    run_final_evaluation,
    split_final_windows,
    weekly_block_bootstrap,
)
from ecom_delivery_sla.modeling import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    classification_metrics,
    run_classification_tournament,
)

FAST = {"iterations": 20}


@pytest.fixture
def tmp_path():
    """Isolated directory via ``tempfile`` (pytest's own base dir may be locked)."""
    with tempfile.TemporaryDirectory(prefix="final_eval_") as directory:
        yield Path(directory)


def _frame(seed: int = 0) -> pd.DataFrame:
    """Synthetic modeling frame with train, June, test and post-test purchases."""
    rng = np.random.default_rng(seed)
    parts = [
        pd.date_range("2018-01-01", "2018-05-31 23:00", periods=420),
        pd.date_range("2018-06-01", "2018-06-30 23:00", periods=60),
        pd.date_range("2018-07-01", "2018-08-31 23:00", periods=260),
        pd.date_range("2018-09-01", "2018-09-20", periods=20),
    ]
    stamps = pd.DatetimeIndex(np.concatenate([p.to_numpy() for p in parts]))
    rows = len(stamps)
    frame = pd.DataFrame({name: rng.normal(size=rows) for name in NUMERIC_FEATURES})
    frame["promised_lead_days"] = rng.uniform(8, 30, size=rows)
    frame["customer_state"] = rng.choice(["SP", "RJ", "MG"], size=rows)
    frame["seller_state"] = rng.choice(["SP", "RJ"], size=rows)
    frame["primary_payment_type"] = rng.choice(["credit_card", "boleto"], size=rows)
    frame["order_id"] = [f"o{i}" for i in range(rows)]
    frame["order_purchase_timestamp"] = stamps
    frame["actual_lead_days"] = 12 + 2 * frame["item_count"] + rng.normal(scale=2.0, size=rows)
    frame["delivery_delay_days"] = frame["actual_lead_days"] - frame["promised_lead_days"]
    frame["sla_breach"] = (frame["delivery_delay_days"] > frame["delivery_delay_days"].quantile(0.7)).astype(int)
    return frame


def _run(tmp_path, frame, version="F1", **kwargs):
    options = {"n_boot": 20, "catboost_params": FAST, "cache_root": tmp_path / "runs", "model_dir": tmp_path / "models"}
    options.update(kwargs)
    return run_final_evaluation(frame, allow_test=True, version=version, **options)


def _ledger(tmp_path) -> list[dict]:
    path = tmp_path / "runs" / "final_evaluation" / "LEDGER.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# -- windows -----------------------------------------------------------------
def test_windows_exclude_june_and_respect_boundaries():
    frame = _frame()
    boundary = pd.DataFrame(
        {"order_purchase_timestamp": pd.to_datetime(
            ["2018-05-31 23:59:59", "2018-06-01 00:00:00", "2018-06-30 23:59:59",
             "2018-07-01 00:00:00", "2018-08-31 23:59:59", "2018-09-01 00:00:00"])}
    )
    windows = split_final_windows(pd.concat([frame, boundary], ignore_index=True))
    train_time = windows["train"]["order_purchase_timestamp"]
    test_time = windows["test"]["order_purchase_timestamp"]

    assert (train_time < pd.Timestamp("2018-06-01")).all()
    assert pd.Timestamp("2018-05-31 23:59:59") in set(train_time)
    assert not ((train_time >= "2018-06-01").any() or (test_time < "2018-06-01").any())
    assert ((test_time >= "2018-07-01") & (test_time < "2018-09-01")).all()
    assert pd.Timestamp("2018-07-01") in set(test_time)
    assert pd.Timestamp("2018-09-01") not in set(test_time)
    june = frame["order_purchase_timestamp"].between("2018-06-01", "2018-06-30 23:59:59")
    assert len(windows["train"]) + len(windows["test"]) + june.sum() + (
        frame["order_purchase_timestamp"] >= "2018-09-01"
    ).sum() + 3 == len(frame) + len(boundary)


def test_windows_require_nonempty_partitions():
    frame = _frame()
    with pytest.raises(ValueError, match="non-empty"):
        split_final_windows(frame[frame["order_purchase_timestamp"] < "2018-06-01"])


# -- single-use guard ----------------------------------------------------------
def test_run_refuses_without_allow_test_and_writes_nothing(tmp_path):
    frame = _frame()
    with pytest.raises(ValueError, match="allow_test"):
        run_final_evaluation(frame, cache_root=tmp_path / "runs", model_dir=tmp_path / "models")
    with pytest.raises(ValueError, match="allow_test"):
        run_final_evaluation(frame, allow_test=False, cache_root=tmp_path / "runs", model_dir=tmp_path / "models")
    assert not (tmp_path / "runs").exists() and not (tmp_path / "models").exists()


# -- bootstrap -----------------------------------------------------------------
def _boot_inputs(seed: int = 1, rows: int = 400):
    rng = np.random.default_rng(seed)
    y = (rng.random(rows) < 0.3).astype(int)
    weeks = np.repeat([f"2018-W{n:02d}" for n in range(27, 37)], rows // 10)
    scores = {
        "good": np.clip(0.3 * y + rng.random(rows) * 0.7, 0, 1),
        "baseline_weak": rng.random(rows),
    }
    return y, scores, weeks


def test_bootstrap_is_reproducible_with_same_seed_and_differs_otherwise():
    y, scores, weeks = _boot_inputs()
    fns = {"ap": lambda yy, ss: classification_metrics(yy, ss)["average_precision"]}
    first = weekly_block_bootstrap(y, scores, weeks, n_boot=60, seed=42, metric_fns=fns)
    again = weekly_block_bootstrap(y, scores, weeks, n_boot=60, seed=42, metric_fns=fns)
    other = weekly_block_bootstrap(y, scores, weeks, n_boot=60, seed=7, metric_fns=fns)

    pd.testing.assert_frame_equal(first, again)
    assert not np.allclose(first[["ci_low", "ci_high"]], other[["ci_low", "ci_high"]])
    assert (first["estimate"] == other["estimate"]).all()
    assert (first["ci_low"] <= first["ci_high"]).all()


def test_default_bootstrap_metrics_and_paired_difference_match_summary():
    y, scores, weeks = _boot_inputs()
    summary = weekly_block_bootstrap(y, scores, weeks, n_boot=30, seed=42)
    assert set(summary["metric"]) == set(fe.BOOTSTRAP_METRICS)
    assert set(summary["model"]) == {"good", "baseline_weak"}

    paired = paired_metric_difference(y, scores, weeks, model="good", reference="baseline_weak", n_boot=30, seed=42)
    point = summary.set_index(["model", "metric"])["estimate"]
    expected = point[("good", "average_precision")] - point[("baseline_weak", "average_precision")]
    assert paired.loc[0, "difference"] == pytest.approx(expected)
    assert paired.loc[0, "ci_low"] <= paired.loc[0, "ci_high"]
    assert paired.loc[0, "outcome"] in fe.OUTCOME_LABELS


def test_resamples_are_whole_weekly_blocks():
    weeks = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 2)
    blocks = fe._week_blocks(weeks)
    assert [len(b) for b in blocks] == [3, 5, 2]
    for picks, rows in fe._block_resamples(blocks, n_boot=25, seed=3):
        assert len(rows) == sum(len(blocks[i]) for i in picks)
        counts = np.bincount(rows, minlength=len(weeks))
        for number, block in enumerate(blocks):
            multiplicity = int((picks == number).sum())
            # every row of a week appears exactly as many times as the week was drawn
            assert (counts[block] == multiplicity).all()

    # the row count of every replicate is a sum of whole week sizes (3, 5, 2)
    seen = set()
    fns = {"n": lambda yy, ss: seen.add(len(yy)) or float(len(yy))}
    y = np.array([0, 1] * 5)
    weekly_block_bootstrap(y, {"m": np.linspace(0, 1, 10)}, weeks, n_boot=50, seed=3, metric_fns=fns)
    sums = {sum(c) for c in itertools.product([3, 5, 2], repeat=3)}
    assert seen <= sums | {10}


def test_degenerate_resamples_are_discarded_and_counted():
    weeks = np.repeat(["w1", "w2", "w3"], 10)
    y = np.array([0] * 20 + [0, 1] * 5)  # only w3 has both classes
    fns = {"mean": lambda yy, ss: float(np.mean(ss))}
    result = weekly_block_bootstrap(y, {"m": np.linspace(0, 1, 30)}, weeks, n_boot=200, seed=0, metric_fns=fns)

    discarded = int(result.loc[0, "n_discarded"])
    assert discarded > 0
    assert result.loc[0, "n_boot"] == 200
    assert result.loc[0, "n_valid"] == 200 - discarded
    # w3 is absent from a resample with probability (2/3)**3
    assert discarded == pytest.approx(200 * (2 / 3) ** 3, abs=30)


def test_classify_preregistered_outcome_three_labels():
    assert classify_preregistered_outcome(0.02, 0.005, 0.04) == "confirmado"
    assert classify_preregistered_outcome(-0.02, -0.04, -0.005) == "não confirmado"
    assert classify_preregistered_outcome(0.02, -0.01, 0.05) == "inconclusivo"
    assert classify_preregistered_outcome(-0.02, -0.05, 0.01) == "inconclusivo"
    assert classify_preregistered_outcome(0.01, 0.0, 0.03) == "inconclusivo"
    assert classify_preregistered_outcome(float("nan"), 0.0, 0.0) == "inconclusivo"


def test_reliability_table_bins_counts_and_rates():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.05, 0.15, 0.55, 0.95, 1.0, 0.55])
    table = reliability_table(y, p, bins=10)

    assert len(table) == 10
    assert table["n"].sum() == 6
    assert table.loc[0, "n"] == 1 and table.loc[0, "mean_predicted"] == pytest.approx(0.05)
    assert table.loc[5, "n"] == 2 and table.loc[5, "observed_rate"] == pytest.approx(0.5)
    assert table.loc[9, "n"] == 2 and table.loc[9, "observed_rate"] == pytest.approx(1.0)
    assert np.isnan(table.loc[2, "mean_predicted"]) and table.loc[2, "n"] == 0


# -- full run ------------------------------------------------------------------
def test_run_computes_tables_saves_model_and_one_ledger_line(tmp_path):
    frame = _frame()
    orders = pd.DataFrame(
        {
            "order_status": ["delivered", "shipped", "delivered"],
            "order_purchase_timestamp": ["2018-07-03", "2018-07-20", "2018-08-02"],
            "order_delivered_customer_date": ["2018-07-25", None, "2018-08-10"],
            "order_estimated_delivery_date": ["2018-07-15", "2018-08-05", "2018-08-20"],
        }
    )
    tables, meta = _run(tmp_path, frame, orders=orders)

    assert meta["source"] == "computed" and meta["post_hoc"] is False
    expected_keys = {
        "provenance", "populations", "classification_metrics", "paired_difference",
        "reliability", "top_k", "regression", "psi", "monthly_prevalence", "censoring",
    }
    assert set(tables) == expected_keys
    windows = split_final_windows(frame)
    populations = tables["populations"].set_index("split")
    assert populations.loc["train", "orders"] == len(windows["train"])
    assert populations.loc["test", "orders"] == len(windows["test"])
    months = tables["monthly_prevalence"]["purchase_month"]
    assert "2018-06" not in set(months) and "2018-09" not in set(months)
    metrics = tables["classification_metrics"]
    assert set(metrics["model"]) == {
        "CatBoost", "baseline_prevalence", "baseline_negative_promised_lead_days",
        "baseline_logistic_promise_plus_state",
    }
    assert (metrics["n_boot"] == 20).all()
    paired = tables["paired_difference"].iloc[0]
    assert paired["reference"] == "baseline_logistic_promise_plus_state"
    assert set(tables["regression"]["model"]) == {
        "CatBoost", "baseline_train_mean", "baseline_median_actual_minus_promise",
        "baseline_state_median_actual_minus_promise",
    }
    assert set(tables["psi"]["feature"]) <= set(NUMERIC_FEATURES)
    assert tables["psi"].set_index("feature").loc["promised_lead_days", "highlight"] == "promise"

    ledger = _ledger(tmp_path)
    assert len(ledger) == 1
    assert ledger[0]["version"] == "F1" and ledger[0]["post_hoc"] is False
    assert ledger[0]["train_orders"] == len(windows["train"])
    assert ledger[0]["test_orders"] == len(windows["test"])
    assert ledger[0]["fingerprint"] == meta["fingerprint"]

    model_dir = tmp_path / "models" / "F1"
    pipeline = joblib.load(model_dir / "model.joblib")
    model_meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert model_meta["version"] == "F1" and model_meta["production_eligible"] is False
    test = windows["test"]
    probabilities = pipeline.predict_proba(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    reloaded_ap = classification_metrics(test["sla_breach"].astype(int), probabilities)["average_precision"]
    saved_ap = metrics.set_index(["model", "metric"]).loc[("CatBoost", "average_precision"), "estimate"]
    assert reloaded_ap == pytest.approx(saved_ap)


def test_candidate_matches_the_tournament_catboost_with_default_configuration(tmp_path):
    full = _frame()
    windows_full = split_final_windows(full)
    frame = pd.concat([windows_full["train"].tail(160), windows_full["test"].head(90)])
    windows = split_final_windows(frame)
    tournament, _ = run_classification_tournament(
        {"train": windows["train"], "validation": windows["test"]}, models=("CatBoost",)
    )
    tables, _ = _run(tmp_path, frame, catboost_params=None, n_boot=3)
    estimate = tables["classification_metrics"].set_index(["model", "metric"]).loc[
        ("CatBoost", "average_precision"), "estimate"
    ]
    assert estimate == pytest.approx(tournament.loc[0, "average_precision"])


def test_second_call_loads_frozen_result_without_recomputing(tmp_path, monkeypatch):
    frame = _frame()
    calls = {"n": 0}
    original = fe._compute_final_tables

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fe, "_compute_final_tables", counting)
    first, first_meta = _run(tmp_path, frame)
    assert calls["n"] == 1

    # same inputs: loaded silently
    again, again_meta = _run(tmp_path, frame)
    assert calls["n"] == 1 and again_meta["source"] == "loaded"
    assert again_meta["fingerprint_diverged"] is False

    # different configuration and data (different fingerprint): still loaded, with a warning
    changed = frame.copy()
    changed.loc[changed.index[0], "total_price"] += 1.0
    with pytest.warns(UserWarning, match="frozen"):
        loaded, loaded_meta = _run(tmp_path, changed, n_boot=999)
    assert calls["n"] == 1
    assert loaded_meta["source"] == "loaded" and loaded_meta["fingerprint_diverged"] is True
    assert loaded_meta["fingerprint"] == first_meta["fingerprint"]
    assert loaded_meta["current_fingerprint"] != first_meta["fingerprint"]
    pd.testing.assert_frame_equal(
        loaded["classification_metrics"], first["classification_metrics"]
    )
    assert len(_ledger(tmp_path)) == 1


def test_new_version_is_post_hoc_and_adds_a_ledger_line(tmp_path):
    frame = _frame()
    _run(tmp_path, frame, version="F1")
    tables, meta = _run(tmp_path, frame, version="F2")

    assert meta["post_hoc"] is True and meta["source"] == "computed"
    assert bool(tables["provenance"].loc[0, "post_hoc"]) is True
    assert meta["config"]["post_hoc"] is True
    ledger = _ledger(tmp_path)
    assert [line["version"] for line in ledger] == ["F1", "F2"]
    assert [line["post_hoc"] for line in ledger] == [False, True]
    model_meta = json.loads((tmp_path / "models" / "F2" / "metadata.json").read_text(encoding="utf-8"))
    assert model_meta["post_hoc"] is True
    assert (tmp_path / "models" / "F1" / "model.joblib").exists()


def test_invalid_version_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid version"):
        _run(tmp_path, _frame(), version="../F1")
