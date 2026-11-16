"""Tests for the post-hoc F2 evaluation (synthetic data only; the real CSVs are never read)."""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import ecom_delivery_sla.posthoc as ph
from ecom_delivery_sla.final_evaluation import _block_resamples, _week_blocks, run_final_evaluation
from ecom_delivery_sla.modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ecom_delivery_sla.posthoc import (
    DELIVERY_COLUMN,
    PROMISE_FEATURE,
    REFERENCE_NAME,
    VARIANT_NAMES,
    attach_delivery_dates,
    daily_block_key,
    gate_summary_from_folds,
    known_label_mask,
    label_availability_table,
    promise_regime_table,
    promise_vs_recent_median,
    recalibration_split,
    recency_weights,
    run_development_gate,
    run_posthoc_evaluation,
)

FAST = {"iterations": 15}
UPDATE = pd.Timestamp("2018-08-01")
TS = "order_purchase_timestamp"


@pytest.fixture
def tmp_path():
    """Isolated directory via ``tempfile`` (pytest's own base dir may be locked)."""
    with tempfile.TemporaryDirectory(prefix="posthoc_") as directory:
        yield Path(directory)


def _frame(seed: int = 0, per_month: int = 90) -> pd.DataFrame:
    """Synthetic frame, 2017-01 to 2018-09, with delivery dates (about 23% breaches)."""
    rng = np.random.default_rng(seed)
    parts = []
    for month in pd.period_range("2017-01", "2018-08", freq="M"):
        parts.append(
            pd.date_range(month.start_time, month.end_time.floor("s"), periods=per_month).to_numpy()
        )
    parts.append(pd.date_range("2018-09-01", "2018-09-20", periods=30).to_numpy())
    stamps = pd.DatetimeIndex(np.concatenate(parts))
    rows = len(stamps)
    frame = pd.DataFrame({name: rng.normal(size=rows) for name in NUMERIC_FEATURES})
    frame["promised_lead_days"] = rng.uniform(8, 30, size=rows)
    frame["customer_state"] = rng.choice(["SP", "RJ", "MG"], size=rows)
    frame["seller_state"] = rng.choice(["SP", "RJ"], size=rows)
    frame["primary_payment_type"] = rng.choice(["credit_card", "boleto"], size=rows)
    frame["order_id"] = [f"o{i}" for i in range(rows)]
    frame[TS] = stamps
    delay = rng.normal(loc=-3.0, scale=4.0, size=rows) + 0.15 * (frame["promised_lead_days"] - 19)
    frame["actual_lead_days"] = frame["promised_lead_days"] + delay
    frame["delivery_delay_days"] = delay
    frame["sla_breach"] = (delay > 0).astype(int)
    return frame


def _supervised(frame: pd.DataFrame) -> pd.DataFrame:
    delivered = frame[TS] + pd.to_timedelta(frame["actual_lead_days"], unit="D")
    return pd.DataFrame({"order_id": frame["order_id"], DELIVERY_COLUMN: delivered})


# -- known labels ------------------------------------------------------------
def test_known_label_mask_excludes_deliveries_on_or_after_as_of():
    frame = pd.DataFrame(
        {
            TS: pd.to_datetime(["2018-07-01", "2018-07-20", "2018-07-31", "2018-08-02", "2018-07-10"]),
            DELIVERY_COLUMN: pd.to_datetime(
                ["2018-07-30 23:59:59", "2018-08-01 00:00:00", "2018-08-05 00:00:00", "2018-08-10 00:00:00", None]
            ),
        }
    )
    mask = known_label_mask(frame, "2018-08-01")
    assert mask.tolist() == [True, False, False, False, False]
    assert known_label_mask(frame, "2018-08-06").tolist() == [True, True, True, False, False]


def test_known_label_mask_requires_columns():
    with pytest.raises(KeyError):
        known_label_mask(pd.DataFrame({TS: pd.to_datetime(["2018-01-01"])}), "2018-08-01")


def test_attach_delivery_dates_joins_by_order_and_requires_source():
    frame = _frame(per_month=5)
    supervised = _supervised(frame).sample(frac=1.0, random_state=1)
    joined = attach_delivery_dates(frame, supervised)
    expected = _supervised(frame).set_index("order_id")[DELIVERY_COLUMN]
    assert (joined.set_index("order_id")[DELIVERY_COLUMN] == expected).all()
    assert DELIVERY_COLUMN not in frame.columns  # the input is untouched
    with pytest.raises(KeyError):
        attach_delivery_dates(frame, None)
    with pytest.raises(ValueError):
        attach_delivery_dates(frame, pd.concat([supervised, supervised.iloc[:1]]))


def test_delivery_date_is_not_a_model_feature():
    assert DELIVERY_COLUMN not in NUMERIC_FEATURES
    assert DELIVERY_COLUMN not in CATEGORICAL_FEATURES
    assert PROMISE_FEATURE not in NUMERIC_FEATURES  # catalog default is unchanged


# -- recency weights ---------------------------------------------------------
def test_recency_weights_halve_every_half_life():
    purchases = pd.to_datetime(["2018-08-01", "2018-05-03", "2018-02-02", "2018-07-31"])
    weights = recency_weights(purchases, "2018-08-01", 90)
    ages = np.array([0.0, 90.0, 180.0, 1.0])
    np.testing.assert_allclose(weights, 0.5 ** (ages / 90.0))
    np.testing.assert_allclose(weights[:3], [1.0, 0.5, 0.25], atol=1e-12)
    with pytest.raises(ValueError):
        recency_weights(pd.to_datetime(["2018-08-02"]), "2018-08-01")
    with pytest.raises(ValueError):
        recency_weights(purchases, "2018-08-01", 0)


class _RecordingLogit(LogisticRegression):
    """Logistic regression that records every ``sample_weight`` it receives."""

    seen: list = []

    def fit(self, X, y, sample_weight=None):
        type(self).seen.append(None if sample_weight is None else np.asarray(sample_weight).copy())
        return super().fit(X, y, sample_weight=sample_weight)


def test_weights_reach_the_estimator_through_the_calibration():
    frame = _frame(per_month=20)
    frame = frame[frame[TS] < "2018-06-01"]
    weights = recency_weights(frame[TS], "2018-08-01")
    _RecordingLogit.seen = []
    ph._fit_calibrated_model(
        frame, numeric_features=list(NUMERIC_FEATURES), sample_weight=weights, estimator=_RecordingLogit(max_iter=500)
    )
    assert _RecordingLogit.seen and all(w is not None for w in _RecordingLogit.seen)
    # each calibration fold receives the weights of its own training rows
    assert all(w.min() >= weights.min() and w.max() <= weights.max() for w in _RecordingLogit.seen)
    _RecordingLogit.seen = []
    ph._fit_calibrated_model(frame, numeric_features=list(NUMERIC_FEATURES), estimator=_RecordingLogit(max_iter=500))
    assert all(w is None for w in _RecordingLogit.seen)


def test_weights_change_the_catboost_fit():
    frame = _frame(per_month=30).iloc[:600].reset_index(drop=True)
    skewed = np.where(np.arange(len(frame)) < 300, 1e-6, 1.0)
    options = {"numeric_features": list(NUMERIC_FEATURES), "catboost_params": FAST}
    plain = ph._fit_calibrated_model(frame, **options)
    weighted = ph._fit_calibrated_model(frame, sample_weight=skewed, **options)
    assert not np.allclose(plain.predict_proba(frame)[:, 1], weighted.predict_proba(frame)[:, 1])


# -- recalibration (variant D) ----------------------------------------------
def test_recalibration_split_uses_only_the_recent_window():
    known = pd.DataFrame(
        {DELIVERY_COLUMN: pd.to_datetime(["2018-06-01 00:00", "2018-07-01 00:00", "2018-07-02 00:00", "2018-07-31 23:00"])}
    )
    fit_mask, cal_mask = recalibration_split(known, "2018-08-01", 30)
    assert fit_mask.tolist() == [True, True, False, False]  # strictly before 2018-07-02
    assert cal_mask.tolist() == [False, False, True, True]
    assert not (fit_mask & cal_mask).any()


def test_isotonic_calibration_is_fitted_only_on_the_calibration_rows():
    frame = _frame(per_month=40)
    dev = frame[frame[TS] < "2018-06-01"].reset_index(drop=True)
    fit_set, cal_set = dev.iloc[:-250], dev.iloc[-250:]
    model = ph._fit_recalibrated_model(
        fit_set, cal_set, numeric_features=list(NUMERIC_FEATURES), catboost_params=FAST
    )
    base = model.estimator.estimator  # FrozenEstimator -> fitted pipeline
    raw_cal = base.predict_proba(cal_set)[:, 1]
    thresholds = model.calibrated_classifiers_[0].calibrators[0].X_thresholds_
    assert np.isin(thresholds, raw_cal).all()
    raw_fit = base.predict_proba(fit_set)[:, 1]
    assert not np.isin(thresholds, np.setdiff1d(raw_fit, raw_cal)).any()
    # the preprocessor was fitted on the fit set only
    medians = fit_set[NUMERIC_FEATURES[0]].median()
    imputer = base.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == pytest.approx(medians)


def test_recalibration_requires_both_classes():
    frame = _frame(per_month=20).iloc[:200]
    one_class = frame.assign(sla_breach=0)
    with pytest.raises(ValueError, match="both classes"):
        ph._fit_recalibrated_model(frame, one_class, numeric_features=list(NUMERIC_FEATURES), catboost_params=FAST)


# -- promise_vs_recent_median -----------------------------------------------
def _promise_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    stamps = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.uniform(0, 90, size=240), unit="D").round("h")
    states = rng.choice(["SP", "RJ", "MG", "AC"], size=240, p=[0.5, 0.3, 0.19, 0.01])
    promise = rng.uniform(5, 40, size=240).round(1)
    promise[rng.choice(240, size=8, replace=False)] = np.nan
    return pd.DataFrame({TS: stamps, "customer_state": states, "promised_lead_days": promise})


def _brute_force(frame: pd.DataFrame) -> np.ndarray:
    out = []
    for _, row in frame.iterrows():
        window = frame[(frame[TS] < row[TS]) & (frame[TS] >= row[TS] - pd.Timedelta(days=30))]
        same = window.loc[window["customer_state"] == row["customer_state"], "promised_lead_days"].dropna()
        pool = same if len(same) else window["promised_lead_days"].dropna()
        median = pool.median() if len(pool) else np.nan
        out.append(row["promised_lead_days"] / median if median > 0 else np.nan)
    return np.array(out)


def test_promise_feature_matches_brute_force_with_ties_and_rare_states():
    frame = _promise_frame()
    frame.loc[frame.index[:6], TS] = frame.loc[frame.index[0], TS]  # identical timestamps
    shuffled = frame.sample(frac=1.0, random_state=5)
    result = promise_vs_recent_median(shuffled)
    assert result.name == PROMISE_FEATURE
    assert result.index.equals(shuffled.index)
    np.testing.assert_allclose(result.to_numpy(), _brute_force(shuffled), equal_nan=True)


def test_promise_feature_ignores_later_purchases_and_the_order_itself():
    frame = _promise_frame().sort_values(TS).reset_index(drop=True)
    base = promise_vs_recent_median(frame)
    cut = 120
    tampered = frame.copy()
    tampered.loc[cut:, "promised_lead_days"] = 1e6  # every later promise changes
    after = promise_vs_recent_median(tampered)
    # purchases up to the cut (and strictly before the first changed timestamp) are unchanged
    unaffected = frame.index[frame[TS] <= frame.loc[cut, TS]].difference([cut])
    unaffected = [i for i in unaffected if frame.loc[i, TS] < frame.loc[cut, TS]]
    np.testing.assert_allclose(base.loc[unaffected], after.loc[unaffected], equal_nan=True)
    # changing only an order's own promise rescales its ratio but not its denominator
    own = frame.copy()
    row = 100
    own.loc[row, "promised_lead_days"] = frame.loc[row, "promised_lead_days"] * 2
    ratio = promise_vs_recent_median(own)
    assert ratio.loc[row] == pytest.approx(base.loc[row] * 2)
    np.testing.assert_allclose(
        ratio.drop(index=row).loc[frame[TS] < frame.loc[row, TS]],
        base.drop(index=row).loc[frame[TS] < frame.loc[row, TS]],
        equal_nan=True,
    )


def test_promise_feature_falls_back_to_global_then_nan():
    frame = pd.DataFrame(
        {
            TS: pd.to_datetime(["2018-01-01", "2018-01-05", "2018-01-10", "2018-03-01"]),
            "customer_state": ["SP", "SP", "RJ", "RJ"],
            "promised_lead_days": [10.0, 20.0, 30.0, 12.0],
        }
    )
    result = promise_vs_recent_median(frame)
    assert np.isnan(result.iloc[0])  # nothing before
    assert result.iloc[1] == pytest.approx(20.0 / 10.0)  # SP, median of the SP purchase
    assert result.iloc[2] == pytest.approx(30.0 / 15.0)  # no RJ yet: global median of {10, 20}
    assert np.isnan(result.iloc[3])  # window empty (previous purchases are > 30 days old)


# -- gate ---------------------------------------------------------------------
def _gate_table(ap: dict[str, float], ece: dict[str, float]) -> pd.DataFrame:
    rows = []
    for variant in ap:
        for fold in ("2018-04", "2018-05"):
            rows.append(
                {"fold": fold, "variant": variant, "average_precision": ap[variant], "ece_10bins": ece[variant]}
            )
    return pd.DataFrame(rows)


def test_gate_summary_classifies_pass_and_fail():
    ap = {REFERENCE_NAME: 0.200, "C_recency": 0.190, "D_recalibration": 0.189, "E_promise_feature": 0.250}
    ece = {REFERENCE_NAME: 0.050, "C_recency": 0.060, "D_recalibration": 0.050, "E_promise_feature": 0.0601}
    summary = gate_summary_from_folds(_gate_table(ap, ece)).set_index("variant")
    assert bool(summary.loc["C_recency", "passed"])  # ratio exactly 0.95, ECE margin exactly 0.01
    assert not bool(summary.loc["D_recalibration", "passed"])  # AP ratio 0.945
    assert bool(summary.loc["D_recalibration", "passes_ece"]) and not bool(summary.loc["D_recalibration", "passes_ap"])
    assert not bool(summary.loc["E_promise_feature", "passed"])  # AP fine, ECE just above the margin
    assert bool(summary.loc["E_promise_feature", "passes_ap"]) and not bool(summary.loc["E_promise_feature", "passes_ece"])
    assert summary.loc["D_recalibration", "status"] == ph.STATUS_FAILED
    assert REFERENCE_NAME not in summary.index
    with pytest.raises(KeyError):
        gate_summary_from_folds(_gate_table({"C_recency": 0.1}, {"C_recency": 0.1}))


def test_development_gate_runs_on_folds_and_never_reads_the_evaluation_period():
    frame = _frame(per_month=60)
    folds, summary = run_development_gate(frame, catboost_params=FAST)
    assert sorted(folds["fold"].unique()) == [f"{p}" for p in pd.period_range("2017-10", "2018-06", freq="M")]
    assert set(folds["variant"]) == {REFERENCE_NAME, "C_recency", "D_recalibration", "E_promise_feature"}
    assert set(summary["variant"]) == {"C_recency", "D_recalibration", "E_promise_feature"}
    assert summary["passed"].dtype == bool
    first = folds[folds["fold"] == "2017-10"].iloc[0]
    assert first["update_date"] == pd.Timestamp("2017-09-01")
    d_rows = folds[folds["variant"] == "D_recalibration"]
    assert (d_rows["n_calibration"] > 0).all()
    # removing every purchase from 2018-07-01 on changes nothing: the gate is development-only
    trimmed_folds, trimmed_summary = run_development_gate(frame[frame[TS] < "2018-07-01"], catboost_params=FAST)
    pd.testing.assert_frame_equal(folds, trimmed_folds)
    pd.testing.assert_frame_equal(summary, trimmed_summary)


# -- blocks ------------------------------------------------------------------
def test_daily_blocks_keep_each_day_whole():
    stamps = pd.to_datetime(
        ["2018-08-01 03:00", "2018-08-01 23:59", "2018-08-02 00:00", "2018-08-03 12:00", "2018-08-03 13:00"]
    )
    key = daily_block_key(stamps)
    assert key.tolist() == ["2018-08-01", "2018-08-01", "2018-08-02", "2018-08-03", "2018-08-03"]
    blocks = _week_blocks(key)
    assert sorted(len(b) for b in blocks) == [1, 2, 2]
    assert all(len(set(key[b])) == 1 for b in blocks)
    sizes = {key[b[0]]: len(b) for b in blocks}
    for picks, rows in _block_resamples(blocks, 5, 42):
        assert len(rows) == sum(len(blocks[i]) for i in picks)
        days, counts = np.unique(key[rows], return_counts=True)
        for day, count in zip(days, counts, strict=True):
            assert count % sizes[day] == 0


def test_label_availability_exposes_the_slow_delivery_bias():
    frame = pd.DataFrame(
        {
            TS: pd.to_datetime(["2018-06-10", "2018-06-20", "2018-07-10", "2018-07-20", "2018-07-25"]),
            DELIVERY_COLUMN: pd.to_datetime(["2018-06-25", "2018-08-20", "2018-07-20", "2018-07-28", "2018-08-30"]),
            "sla_breach": [0, 1, 0, 0, 1],
        }
    )
    table = label_availability_table(frame, "2018-08-01").set_index("purchase_month")
    assert table.loc["2018-06", "known_labels"] == 1 and table.loc["2018-06", "unknown_labels"] == 1
    assert table.loc["2018-07", "known_labels"] == 2 and table.loc["2018-07", "unknown_labels"] == 1
    assert table.loc["2018-07", "prevalence_known"] == 0.0
    assert table.loc["2018-07", "prevalence_unknown"] == 1.0
    assert table.loc["2018-07", "share_known"] == pytest.approx(2 / 3)


# -- end to end ---------------------------------------------------------------
@pytest.fixture(scope="module")
def f2():
    """Run F1 (synthetic) and then F2 once, recording what each fit received."""
    with tempfile.TemporaryDirectory(prefix="posthoc_e2e_") as directory:
        root = Path(directory)
        frame = _frame()
        supervised = _supervised(frame)
        run_final_evaluation(
            frame,
            allow_test=True,
            version="F1",
            cache_root=root / "runs",
            model_dir=root / "models" / "catboost_final",
            n_boot=10,
            catboost_params=FAST,
        )
        runs = root / "runs" / "final_evaluation"
        f1_files = {p.name: p.read_bytes() for p in (runs / "F1").iterdir()}
        ledger_before = (runs / "LEDGER.jsonl").read_text(encoding="utf-8")
        records = {"calibrated": [], "recalibrated": []}
        original = (ph._fit_calibrated_model, ph._fit_recalibrated_model)

        def spy_calibrated(train, **kwargs):
            records["calibrated"].append((train.copy(), kwargs))
            return original[0](train, **kwargs)

        def spy_recalibrated(fit_set, calibration_set, **kwargs):
            records["recalibrated"].append((fit_set.copy(), calibration_set.copy(), kwargs))
            return original[1](fit_set, calibration_set, **kwargs)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(ph, "_fit_calibrated_model", spy_calibrated)
        monkey.setattr(ph, "_fit_recalibrated_model", spy_recalibrated)
        # force a mixed gate: C passes, D and E fail (the real gate is tested separately)
        real_gate = ph.run_development_gate

        def forced_gate(modeling_frame, **kwargs):
            folds, summary = real_gate(modeling_frame, **kwargs)
            summary = summary.copy()
            summary["passed"] = summary["variant"].eq("C_recency")
            summary["status"] = np.where(summary["passed"], ph.STATUS_EVALUATED, ph.STATUS_FAILED)
            return folds, summary

        monkey.setattr(ph, "run_development_gate", forced_gate)
        options = {
            "supervised_orders": supervised,
            "allow_test": True,
            "version": "F2",
            "cache_root": root / "runs",
            "f1_model_path": root / "models" / "catboost_final" / "F1" / "model.joblib",
            "n_boot": 12,
            "catboost_params": FAST,
        }
        try:
            tables, meta = run_posthoc_evaluation(frame, **options)
        finally:
            monkey.undo()
        yield {
            "root": root,
            "runs": runs,
            "frame": frame,
            "supervised": supervised,
            "tables": tables,
            "meta": meta,
            "records": records,
            "options": options,
            "f1_files": f1_files,
            "ledger_before": ledger_before,
        }


def test_f2_tables_marking_and_provenance(f2):
    tables, meta = f2["tables"], f2["meta"]
    assert set(tables) == {
        "provenance",
        "variants",
        "populations",
        "gate_folds",
        "gate_summary",
        "classification_metrics",
        "paired_difference",
        "reliability",
        "top_k",
        "label_availability",
    }
    assert meta["post_hoc"] is True and meta["source"] == "computed"
    provenance = tables["provenance"].iloc[0]
    assert bool(provenance["post_hoc"]) is True
    assert provenance["reading"] == "exploratório"
    assert bool(provenance["production_eligible"]) is False
    assert provenance["gate_passed"] == "C" and provenance["gate_failed"] == "D,E"
    assert provenance["eval_blocks"] == 31
    stored = json.loads((f2["runs"] / "F2" / "metadata.json").read_text(encoding="utf-8"))
    assert stored["config"]["post_hoc"] is True and stored["config"]["bootstrap_blocks"] == "daily"


def test_known_training_sets_never_contain_labels_delivered_on_or_after_the_update(f2):
    august_fits = [
        (train, kwargs)
        for train, kwargs in f2["records"]["calibrated"]
        if train[TS].max() >= pd.Timestamp("2018-07-01")
    ]
    assert len(august_fits) == 2  # B and C (E and D failed the forced gate)
    for train, _ in august_fits:
        assert (pd.to_datetime(train[DELIVERY_COLUMN]) < UPDATE).all()
        assert (pd.to_datetime(train[TS]) < UPDATE).all()
    expected = f2["frame"].merge(f2["supervised"], on="order_id")
    known = expected[expected[DELIVERY_COLUMN] < UPDATE]
    assert set(august_fits[0][0]["order_id"]) == set(known["order_id"])
    assert len(known) < (expected[TS] < UPDATE).sum()  # some old-purchase orders were still undelivered


def test_recency_weights_are_forwarded_for_variant_c(f2):
    august_fits = [
        (train, kwargs)
        for train, kwargs in f2["records"]["calibrated"]
        if train[TS].max() >= pd.Timestamp("2018-07-01")
    ]
    plain, weighted = august_fits
    assert plain[1].get("sample_weight") is None
    np.testing.assert_allclose(weighted[1]["sample_weight"], recency_weights(weighted[0][TS], UPDATE))
    assert PROMISE_FEATURE not in weighted[1]["numeric_features"]


def test_failed_variants_are_reported_but_not_scored(f2):
    tables = f2["tables"]
    variants = tables["variants"].set_index("variant")
    assert variants.loc["C_recency", "scored"] and variants.loc["A_f1_frozen", "scored"]
    assert not variants.loc["D_recalibration", "scored"] and not variants.loc["E_promise_feature", "scored"]
    metrics = tables["classification_metrics"]
    failed = metrics[metrics["model"].isin(["D_recalibration", "E_promise_feature"])]
    assert (failed["status"] == ph.STATUS_FAILED).all() and failed["estimate"].isna().all()
    assert not f2["records"]["recalibrated"] or all(
        fit[TS].max() < pd.Timestamp("2018-07-01") for fit, _, _ in f2["records"]["recalibrated"]
    )
    scored_models = set(tables["reliability"]["model"]) | set(tables["top_k"]["model"])
    assert "D_recalibration" not in scored_models and "E_promise_feature" not in scored_models
    paired = tables["paired_difference"]
    assert set(paired.loc[paired["status"] == ph.STATUS_FAILED, "model"]) == {"D_recalibration", "E_promise_feature"}
    assert (paired["reading"] == "exploratório").all()
    assert not {"outcome", "outcome_en"} & set(paired.columns)
    assert set(paired["reference"]) == {ph.REFERENCE_BASELINE, ph.BASELINE_RULE}
    assert set(tables["gate_summary"]["variant"]) == {"C_recency", "D_recalibration", "E_promise_feature"}


def test_f2_evaluation_window_and_baselines(f2):
    populations = f2["tables"]["populations"]
    evaluation = populations[populations["role"].str.startswith("evaluation")].iloc[0]
    frame = f2["frame"]
    august = frame[(frame[TS] >= "2018-08-01") & (frame[TS] < "2018-09-01")]
    assert evaluation["orders"] == len(august)
    assert evaluation["last_purchase"] < pd.Timestamp("2018-09-01")
    models = set(f2["tables"]["classification_metrics"]["model"])
    assert {"baseline_logistic_promise_plus_state", "baseline_negative_promised_lead_days", "baseline_prevalence"} <= models
    availability = f2["tables"]["label_availability"]
    assert availability["purchase_month"].tolist() == ["2018-06", "2018-07"]


def test_f2_does_not_touch_f1_and_adds_one_ledger_line(f2):
    runs = f2["runs"]
    assert {p.name: p.read_bytes() for p in (runs / "F1").iterdir()} == f2["f1_files"]
    before = f2["ledger_before"].splitlines()
    after = (runs / "LEDGER.jsonl").read_text(encoding="utf-8").splitlines()
    assert after[: len(before)] == before and len(after) == len(before) + 1
    line = json.loads(after[-1])
    assert line["version"] == "F2" and line["post_hoc"] is True
    assert sorted(p.name for p in (f2["root"] / "models").iterdir()) == ["catboost_final"]
    assert sorted(p.name for p in (f2["root"] / "models" / "catboost_final").iterdir()) == ["F1"]


def test_f2_is_frozen_second_call_loads_without_recomputing(f2, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a frozen F2 must not be recomputed")

    monkeypatch.setattr(ph, "_compute_posthoc_tables", boom)
    monkeypatch.setattr(ph, "run_development_gate", boom)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tables, meta = run_posthoc_evaluation(f2["frame"], **f2["options"])
    assert meta["source"] == "loaded" and meta["post_hoc"] is True
    assert meta["fingerprint_diverged"] is False
    pd.testing.assert_frame_equal(tables["gate_summary"], f2["tables"]["gate_summary"])
    ledger = (f2["runs"] / "LEDGER.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == len(f2["ledger_before"].splitlines()) + 1


def test_f2_divergent_fingerprint_warns_and_still_loads(f2, monkeypatch):
    monkeypatch.setattr(ph, "_compute_posthoc_tables", lambda *a, **k: pytest.fail("recomputed"))
    changed = f2["frame"].copy()
    changed.loc[changed.index[0], "total_price"] += 1.0
    with pytest.warns(UserWarning, match="frozen"):
        tables, meta = run_posthoc_evaluation(changed, **{**f2["options"], "n_boot": 99})
    assert meta["source"] == "loaded" and meta["fingerprint_diverged"] is True and meta["post_hoc"] is True
    assert bool(tables["provenance"]["post_hoc"].iloc[0])
    ledger = (f2["runs"] / "LEDGER.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == len(f2["ledger_before"].splitlines()) + 1


def test_all_variants_failing_the_gate_scores_only_a_and_b(tmp_path, f2, monkeypatch):
    monkeypatch.setattr(ph, "GATE_AP_RATIO", 1e9)
    options = {**f2["options"], "cache_root": tmp_path / "runs", "n_boot": 5}
    tables, _ = run_posthoc_evaluation(f2["frame"], **options)
    variants = tables["variants"].set_index("code")
    assert variants["scored"].to_dict() == {"A": True, "B": True, "C": False, "D": False, "E": False}
    assert not tables["gate_summary"]["passed"].any()
    assert set(tables["reliability"]["model"]) == {
        "A_f1_frozen",
        "B_retrain",
        "baseline_prevalence",
        "baseline_logistic_promise_plus_state",
    }
    assert tables["provenance"]["gate_passed"].iloc[0] == ""


def test_all_variants_passing_scores_every_variant(tmp_path, f2, monkeypatch):
    monkeypatch.setattr(ph, "GATE_AP_RATIO", 0.0)
    monkeypatch.setattr(ph, "GATE_ECE_MARGIN", 10.0)
    options = {**f2["options"], "cache_root": tmp_path / "runs", "n_boot": 5}
    tables, _ = run_posthoc_evaluation(f2["frame"], **options)
    assert tables["variants"]["scored"].all()
    assert set(VARIANT_NAMES.values()) <= set(tables["reliability"]["model"])
    paired = tables["paired_difference"]
    assert set(paired["status"]) == {ph.STATUS_EVALUATED}
    assert paired["difference_sign"].isin(["positive", "negative", "zero"]).all()


# -- guards -------------------------------------------------------------------
def test_requires_allow_test_and_rejects_f1_and_bad_versions(tmp_path):
    frame = _frame(per_month=5)
    options = {"supervised_orders": _supervised(frame), "cache_root": tmp_path, "f1_model_path": tmp_path / "m.joblib"}
    with pytest.raises(ValueError, match="allow_test"):
        run_posthoc_evaluation(frame, **options)
    with pytest.raises(ValueError, match="allow_test"):
        run_posthoc_evaluation(frame, allow_test=False, **options)
    with pytest.raises(ValueError, match="F1"):
        run_posthoc_evaluation(frame, allow_test=True, version="F1", **options)
    with pytest.raises(ValueError, match="invalid version"):
        run_posthoc_evaluation(frame, allow_test=True, version="../x", **options)
    assert not (tmp_path / "final_evaluation").exists()


def test_missing_f1_model_or_delivery_dates_fail_before_computing(tmp_path):
    frame = _frame(per_month=5)
    with pytest.raises(KeyError, match=DELIVERY_COLUMN):
        run_posthoc_evaluation(frame, allow_test=True, cache_root=tmp_path, f1_model_path=tmp_path / "m.joblib")
    with pytest.raises(FileNotFoundError):
        run_posthoc_evaluation(
            frame,
            supervised_orders=_supervised(frame),
            allow_test=True,
            cache_root=tmp_path,
            f1_model_path=tmp_path / "missing.joblib",
        )
    assert not (tmp_path / "final_evaluation").exists()


def test_promise_regime_table_reads_labels_only_before_july_2018():
    frame = _frame(per_month=40)
    poisoned = frame.copy()
    late = poisoned[TS] >= "2018-07-01"
    poisoned.loc[late, "sla_breach"] = 1 - poisoned.loc[late, "sla_breach"]  # must not matter
    table = promise_regime_table(frame)
    pd.testing.assert_frame_equal(table, promise_regime_table(poisoned))
    late_rows = table[table["purchase_month"] >= "2018-07"]
    assert not late_rows["labels_used"].any()
    assert late_rows[["breach_prevalence", "auc_negative_promise"]].isna().all().all()
    assert late_rows["promise_median"].notna().all()
    dev = table[table["purchase_month"] == "2018-03"].iloc[0]
    month = frame[frame[TS].dt.to_period("M").astype(str) == "2018-03"]
    assert dev["breach_prevalence"] == pytest.approx(month["sla_breach"].mean())
    assert 0.0 <= dev["auc_negative_promise"] <= 1.0
