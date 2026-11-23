"""Testes sinteticos da etapa 4: restauracao, congelamento e zeros."""

import json

import numpy as np
import pandas as pd
import pytest

from freq_sinistros import compare, freeze, metrics, models, prep, stage4
from tests.test_stage3 import simulate


def fitted():
    frame = simulate(6000, seed=11, alpha=0.4)
    pre = prep.Preprocessor(rare_min_rows=40).fit(frame.iloc[:4500])
    x, _ = pre.transform(frame.iloc[:4500])
    y = frame["ClaimNb"].to_numpy()[:4500]
    e = frame["Exposure"].to_numpy()[:4500]
    fit = models.fit_poisson(x, y, e, pre.columns)
    glob = models.fit_global(y, e)
    return frame, pre, fit, glob


def frozen_from(pre, fit, glob):
    return json.loads(json.dumps({
        "preprocessor": pre.summary(),
        "estimator": {"columns": list(pre.columns),
                      "coefficients": [float(b) for b in fit.beta]},
        "global_rate": {"rate": float(glob.rate)}}))


def test_preprocessor_roundtrip_is_exact_including_unknowns():
    frame, pre, _, _ = fitted()
    state = json.loads(json.dumps(pre.summary()))
    restored = prep.Preprocessor.from_summary(state)
    other = frame.iloc[4500:].copy()
    other.loc[other.index[0], "Region"] = np.nan
    other.loc[other.index[1], "DrivAge"] = np.nan
    a, ra = pre.transform(other)
    b, rb = restored.transform(other)
    assert restored.columns == pre.columns
    assert np.array_equal(a, b) and ra == rb


def test_frozen_prediction_matches_fit_and_offset_scales():
    frame, pre, fit, glob = fitted()
    frozen = frozen_from(pre, fit, glob)
    val = frame.iloc[4500:]
    mu = freeze.predict_frozen(frozen, val)
    x, _ = pre.transform(val)
    expected = fit.predict(x, val["Exposure"].to_numpy())
    assert np.array_equal(mu["poisson"], expected)
    assert np.allclose(mu["global"], glob.rate * val["Exposure"])
    doubled = val.copy()
    doubled["Exposure"] = 2 * val["Exposure"]
    assert np.allclose(freeze.predict_frozen(frozen, doubled)["poisson"],
                       2 * mu["poisson"], rtol=1e-12)


def test_restore_rejects_column_order_and_size_mismatch():
    _, pre, fit, glob = fitted()
    frozen = frozen_from(pre, fit, glob)
    swapped = json.loads(json.dumps(frozen))
    cols = swapped["estimator"]["columns"]
    cols[1], cols[2] = cols[2], cols[1]
    with pytest.raises(ValueError, match="ordem"):
        freeze.restore(swapped)
    short = json.loads(json.dumps(frozen))
    short["estimator"]["coefficients"] = short["estimator"][
        "coefficients"][:-1]
    with pytest.raises(ValueError, match="coeficientes"):
        freeze.restore(short)


def test_reference_comparison_detects_changes():
    frame, pre, fit, glob = fitted()
    frozen = frozen_from(pre, fit, glob)
    val = frame.iloc[4500:]
    mu = freeze.predict_frozen(frozen, val)
    ref = freeze.aggregate_reference(val, mu)
    assert freeze.compare_references(ref, ref)["ok"]
    changed = json.loads(json.dumps(ref))
    changed["candidates"]["poisson"]["predicted_claims"] *= 1.001
    result = freeze.compare_references(ref, changed)
    assert not result["ok"] and result["max_rel_diff"] > 1e-4


def eval_with_free():
    frame = simulate(3000, seed=7, alpha=0.5)
    pre = prep.Preprocessor(rare_min_rows=30).fit(frame.iloc[:2000])
    fits, _, _ = compare.fit_candidates(frame.iloc[:2000], pre,
                                        {"context": "t"})
    valid = frame.iloc[2000:].reset_index(drop=True)
    mu, _ = compare.predict_all(fits, pre, valid)
    mu["poisson_free_exposure"] = mu["poisson"] * np.where(
        valid["Exposure"].to_numpy() < 0.3, 1.5, 0.95)
    ev = compare.EvalSet(valid, mu, {"nb2": fits["nb2"].alpha}, [])
    return ev, fits


def test_zero_diagnostic_gate_and_reconciliation():
    ev, _ = eval_with_free()
    iv = metrics.bootstrap_intervals(ev.statistic, ev.n, replicas=40,
                                     seed=3)
    iv.insert(0, "population", "validation")
    iv.insert(0, "degree", "train_full")
    free = iv.copy()
    free["population"] = "validation_free_exposure_diagnostic"
    out3 = {"intervals": pd.concat([iv, free], ignore_index=True)}
    table, recon = stage4.zero_diagnostic(ev, out3)
    assert recon["ok"].all()
    assert set(table["candidate"]) == set(stage4.CANDIDATES)
    overall = table[table["band"] == "overall"]
    assert (overall["rows"] == ev.n).all()
    gate = stage4.zero_gate(table)
    assert set(gate) == set(stage4.CANDIDATES)
    bands = stage4.exposure_band_table(ev, out3)
    assert bands["rows"].iloc[:-1].sum() == ev.n
    mz = stage4.mean_vs_zero_table(ev)
    assert (mz["band"] == "total").sum() == 2


def test_zero_gate_requires_total_and_two_supported_bands():
    rows = []
    for band, hi in (("overall", -1.0), ("a", -1.0), ("b", -0.5),
                     ("c", 3.0)):
        rows.append({"candidate": "m", "band": band, "ci_high": hi,
                     "ci_low": hi - 5, "low_support": False,
                     "diff_pred_minus_obs": hi - 2})
    table = pd.DataFrame(rows)
    assert stage4.zero_gate(table)["m"]["numeric_requirement_met"]
    table.loc[table["band"] == "b", "ci_high"] = 1.0
    assert not stage4.zero_gate(table)["m"]["numeric_requirement_met"]
    table.loc[table["band"] == "b", "ci_high"] = -0.5
    table.loc[table["band"] == "b", "low_support"] = True
    assert not stage4.zero_gate(table)["m"]["numeric_requirement_met"]


def test_stage5_script_keeps_test_reserved_and_short_band_visible():
    script = stage4.stage5_script("x.json")
    assert script["production_eligible"] is False
    assert script["shortest_band_visible"] == "(0, 0.02]"
    assert "(0, 0.02]" in script["exposure_bands"]
    assert "uma unica execucao" in script["test_access"]
