import numpy as np
import pandas as pd
import pytest

from retail_multilevel import p1


def sim_rows(n_s=4, n_p=5, n_w=24, seed=0, slope_sd=0.0):
    """Tiny crossed panel; the pair (last store, last UPC) only exists from week 12 on.

    The week of the year repeats every 12 weeks so that a 12-week fit window already spans a full cycle.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(1.5, 0.3, n_p)
    u_s, u_p = rng.normal(0, 0.3, n_s), rng.normal(0, 0.5, n_p)
    slope = -2.0 + rng.normal(0, slope_sd, n_p) if slope_sd else np.full(n_p, -2.0)
    rows = []
    for s in range(n_s):
        for p in range(n_p):
            for w in range(n_w):
                if s == n_s - 1 and p == n_p - 1 and w < 12:
                    continue
                disc = -rng.choice([0.0, 0.1, 0.2], p=[0.6, 0.2, 0.2])
                feat, disp = int(rng.random() < 0.15), int(rng.random() < 0.15)
                tpr = int(disc < 0 and not feat and not disp)
                lb = base[p] + rng.normal(0, 0.03)
                y = 2 + u_s[s] + u_p[p] + slope[p] * disc + 0.4 * feat + 0.3 * disp + rng.normal(0, 0.3)
                rows.append((s, p, f"{s}_{p}", w, (w % 12) * 4 + 1, y, lb, lb + disc, feat, disp, tpr))
    cols = ["STORE_NUM", "UPC", "PAIR_ID", "WEEK", "WEEK_OF_YEAR", "LOG_UNITS", "LOG_BASE_PRICE", "LOG_PRICE",
            "FEATURE", "DISPLAY", "TPR_ONLY"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


SMALL_FOLDS = {"A": {"fit": (0, 11), "val": (12, 17)}, "B": {"fit": (0, 17), "val": (18, 23)}}


# ---------------------------------------------------------------------------------------------- metrics
def test_pair_mean_mae_equal_weight_per_pair():
    ae = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    pair = np.array(["a", "a", "a", "a", "b"])
    assert p1.pair_mean_mae(ae, pair) == pytest.approx(3.0)  # row-weighted would be 1.8


def test_history_stratum_labels():
    out = p1.history_stratum(np.array([1, 25, 26, 77, 78, 150]))
    assert list(out) == ["<26", "<26", "26-77", "26-77", ">=78", ">=78"]


def _ev():
    # pairs: p1 seen/covered/short, p2 seen/not covered/mid, p3 seen/covered/long, u unseen, n none
    ev = pd.DataFrame(
        {
            "PAIR_ID": ["p1", "p1", "p2", "p3", "p3", "u", "n"],
            "WEEK": [1, 2, 1, 1, 2, 1, 1],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "POP": ["seen", "seen", "seen", "seen", "seen", "unseen", "none"],
            "HISTORY": [10, 10, 40, 90, 90, 0, 0],
            "COVERED": [True, True, False, True, True, False, False],
        }
    )
    ev["STRATUM"] = p1.history_stratum(ev["HISTORY"]).to_numpy()
    return ev


def test_metric_table_populations_and_strata():
    ev = _ev()
    nan = np.nan
    preds = {
        "m": np.array([1.0, 1.0, 2.0, 3.0, 5.0, 4.0, nan]),
        p1.OLS_POOLED: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan]),
        p1.OLS_PAIR: np.array([2.0, 2.0, nan, 2.0, 2.0, nan, nan]),
    }
    t = p1.metric_table(ev, preds, "A").set_index(["model", "population", "stratum"])
    # population a, all strata: pair MAEs p1=1, p2=2, p3=4 -> mean 7/3; rows: (1+1+2+3+5)/5
    a = t.loc[("m", p1.POP_SEEN, "all")]
    assert a["mae_pair_mean"] == pytest.approx(7 / 3) and a["n_pairs"] == 3 and a["n_rows"] == 5
    assert a["mae_row"] == pytest.approx(12 / 5) and a["rmse"] == pytest.approx(np.sqrt((1 + 1 + 4 + 9 + 25) / 5))
    assert t.loc[("m", p1.POP_SEEN, "<26"), "n_pairs"] == 1
    assert t.loc[("m", p1.POP_SEEN, ">=78"), "mae_pair_mean"] == pytest.approx(4.0)
    # population b: identical covered rows (p1 and p3) for every model, p2 excluded
    b_m = t.loc[("m", p1.POP_OLS, "all")]
    b_o = t.loc[(p1.OLS_PAIR, p1.POP_OLS, "all")]
    assert b_m["n_rows"] == b_o["n_rows"] == 4 and b_m["n_pairs"] == 2
    assert b_m["mae_pair_mean"] == pytest.approx((1 + 4) / 2) and b_o["mae_pair_mean"] == pytest.approx(2.0)
    assert t.loc[("m", p1.POP_OLS, "26-77"), "n_pairs"] == 0  # empty stratum is reported as empty
    assert np.isnan(t.loc[("m", p1.POP_OLS, "26-77"), "mae_pair_mean"])
    # per-pair OLS is not evaluated in a or c; unseen only has 'all'
    assert (p1.OLS_PAIR, p1.POP_SEEN, "all") not in t.index
    assert (p1.OLS_PAIR, p1.POP_UNSEEN, "all") not in t.index
    assert t.loc[("m", p1.POP_UNSEEN, "all"), "mae_pair_mean"] == pytest.approx(4.0)
    assert [i for i in t.index if i[1] == p1.POP_UNSEEN and i[0] == "m"] == [("m", p1.POP_UNSEEN, "all")]


def test_metric_table_rejects_nan_inside_population():
    ev = _ev()
    with pytest.raises(ValueError, match="NaN"):
        p1.metric_table(ev, {"m": np.full(len(ev), np.nan)}, "A")


def test_weekly_diff_sign_and_counts():
    ev = _ev()
    cand = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
    inc = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    d = p1.weekly_diff(ev, cand, inc, "A", ("C1", "C0"))
    assert list(d["WEEK"]) == [1, 2] and list(d["n_rows"]) == [3, 2]
    assert np.allclose(d["mae_diff"], -0.5) and (d["candidate"] == "C1").all()


# ------------------------------------------------------------------------------------------ OLS coverage
def _pair(pid, n, disc=None, feat=0, disp=None, tpr=None, seed=0):
    rng = np.random.default_rng(seed)
    disc = rng.choice([0.0, -0.1, -0.2], n) if disc is None else np.full(n, disc)
    disp = rng.integers(0, 2, n) if disp is None else disp
    tpr = rng.integers(0, 2, n) if tpr is None else tpr
    feat = np.full(n, feat) if np.isscalar(feat) else feat
    y = 1 + 2 * disc + 0.5 * disp + 0.3 * tpr + 0.7 * feat
    return pd.DataFrame(
        {"PAIR_ID": pid, "LOG_UNITS": y, "LOG_DISC": disc, "FEATURE": feat, "DISPLAY": disp, "TPR_ONLY": tpr}
    )


def _val(rows):
    return pd.DataFrame(rows, columns=["PAIR_ID", "LOG_DISC", "FEATURE", "DISPLAY", "TPR_ONLY"])


def test_pair_ols_coverage_rule():
    disp = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    fit = pd.concat(
        [
            _pair("noFeat", 10, seed=1),  # FEATURE never occurs
            _pair("flatDisc", 10, disc=0.0, seed=2),  # LOG_DISC constant
            _pair("short", 5, seed=3),  # fewer than 8 weeks
            _pair("collin", 10, disp=disp, tpr=disp, seed=4),  # TPR_ONLY == DISPLAY -> TPR dropped
            _pair("ok", 10, feat=np.array([0, 1] * 5), seed=5),
        ],
        ignore_index=True,
    )
    val = _val(
        [
            ("noFeat", -0.1, 1, 0, 0),  # 0: FEATURE=1 on a not-estimated term -> uncovered
            ("noFeat", -0.1, 0, 1, 1),  # 1: FEATURE=0 -> covered
            ("flatDisc", -0.2, 0, 0, 0),  # 2: LOG_DISC != 0 on a constant term -> uncovered
            ("flatDisc", 0.0, 0, 1, 0),  # 3: covered
            ("short", 0.0, 0, 0, 0),  # 4: < 8 weeks -> uncovered
            ("collin", 0.0, 0, 0, 1),  # 5: TPR_ONLY=1 dropped by collinearity -> uncovered
            ("collin", -0.1, 0, 1, 0),  # 6: covered
            ("new", 0.0, 0, 0, 0),  # 7: pair absent from the fit window -> uncovered
            ("ok", -0.1, 1, 1, 0),  # 8: covered, exact linear prediction
        ]
    )
    pred, covered, info = p1.pair_ols_predict(fit, val)
    assert list(covered) == [False, True, False, True, False, False, True, False, True]
    assert np.isnan(pred[~covered]).all() and not np.isnan(pred[covered]).any()
    assert pred[8] == pytest.approx(1 + 2 * -0.1 + 0.5 + 0.3 * 0 + 0.7)
    assert pred[1] == pytest.approx(1 + 2 * -0.1 + 0.5 + 0.3)
    i = info.set_index("PAIR_ID")
    assert not i.loc["short", "has_ols"] and i.loc["short", "n_weeks"] == 5
    assert not i.loc["noFeat", "est_FEATURE"] and i.loc["noFeat", "est_DISPLAY"]
    assert not i.loc["flatDisc", "est_LOG_DISC"]
    assert i.loc["collin", "est_DISPLAY"] and not i.loc["collin", "est_TPR_ONLY"]


def test_coverage_row_counts():
    ev = _ev()
    info = pd.DataFrame(
        {"PAIR_ID": ["p1", "p2", "p3"], "n_weeks": [10, 40, 90], "has_ols": [True, True, True],
         "est_FEATURE": [True, False, True]}
    )  # fmt: skip
    row = p1.coverage_row(ev, info, "A")
    assert row["n_seen_pairs"] == 3 and row["n_seen_rows"] == 5 and row["n_covered_rows"] == 4
    assert row["n_covered_pairs"] == 2 and row["pct_pairs_covered"] == pytest.approx(200 / 3)
    assert row["pct_rows_covered"] == pytest.approx(80.0)
    assert row["n_unseen_pairs"] == 1 and row["n_none_rows"] == 1
    assert row["n_pairs_not_estimated_FEATURE"] == 1


# --------------------------------------------------------------------------------------- fold data / refs
def test_prepare_fold_learns_on_fit_window_only_and_guards_final_window():
    rows = sim_rows(seed=3)
    fit, val = p1.prepare_fold(rows, (0, 11), (12, 17))
    assert fit["WEEK"].max() == 11 and val["WEEK"].min() == 12 and val["WEEK"].max() == 17
    assert set(fit["MEAN_SOURCE"]) == {"pair"}
    expected = fit.groupby("PAIR_ID")["LOG_BASE_PRICE"].mean()
    seen = val[val["MEAN_SOURCE"] == "pair"]
    assert np.allclose(seen["LOG_BASE_MEAN"], seen["PAIR_ID"].map(expected))
    assert (val.loc[val["PAIR_ID"] == "3_4", "MEAN_SOURCE"] == "upc").all()  # unseen pair -> UPC mean
    # changing validation values never changes anything learned on the fit window
    rows2 = rows.copy()
    rows2.loc[rows2["WEEK"] >= 12, "LOG_BASE_PRICE"] += 5
    fit2, _ = p1.prepare_fold(rows2, (0, 11), (12, 17))
    pd.testing.assert_frame_equal(fit, fit2)
    with pytest.raises(ValueError, match="beyond"):
        p1.prepare_fold(rows, (0, 77), (78, 104))
    with pytest.raises(ValueError, match="start after"):
        p1.prepare_fold(rows, (0, 12), (12, 17))


def test_pooled_ols_recovers_and_nan_for_none():
    rows = sim_rows(seed=4, n_w=40)
    fit, val = p1.prepare_fold(rows, (0, 29), (30, 39))
    val.loc[val.index[0], "MEAN_SOURCE"] = "none"
    pred = p1.pooled_ols_predict(fit, val)
    assert np.isnan(pred[0]) and not np.isnan(pred[1:]).any()


# ------------------------------------------------------------------------------------------ selection rule
def _mae(a, b):
    return {"A": a, "B": b}


def test_decide_step_accepts_clear_gain():
    r = p1.decide_step(1, "C1", "C0", _mae((0.9, 1.0), (0.8, 0.9)), {"A": True, "B": True})
    assert r["decision"] == "accepted" and r["converged_ok"]
    assert r["rel_gain_A"] == pytest.approx(0.1) and r["mean_rel_gain"] == pytest.approx((0.1 + 1 / 9) / 2)


def test_decide_step_rejects_inconsistent_gain():
    r = p1.decide_step(1, "C1", "C0", _mae((0.9, 1.0), (1.01, 1.0)), {"A": True, "B": True})
    assert r["decision"] == "rejected" and "fold B: MAE not lower" in r["reason"]


def test_decide_step_rejects_small_mean_gain():
    r = p1.decide_step(1, "C1", "C0", _mae((0.997, 1.0), (0.999, 1.0)), {"A": True, "B": True})  # mean 0.2%
    assert r["decision"] == "rejected" and "< 0.5%" in r["reason"]
    ok = p1.decide_step(1, "C1", "C0", _mae((0.994, 1.0), (0.994, 1.0)), {"A": True, "B": True})
    assert ok["decision"] == "accepted"


def test_decide_step_rejects_nonconvergence_even_with_gain():
    r = p1.decide_step(2, "C2", "C1", _mae((0.5, 1.0), (0.5, 1.0)), {"A": True, "B": False})
    assert r["decision"] == "rejected" and not r["converged_ok"] and "fold B: not converged" in r["reason"]


def test_candidate_random_part():
    assert p1.candidate_random_part("add", "(0 + X | G)", "(1 | G)") == "(1 | G) + (0 + X | G)"
    assert p1.candidate_random_part("replace", "(1 + X | G)", "(1 | G)") == "(1 + X | G)"
    with pytest.raises(ValueError):
        p1.candidate_random_part("other", "x", "y")


def _fake(offsets, bad=()):
    """fit_predict stub: prediction = y + offset, so the pair MAE equals the offset."""

    def fit_predict(fit_df, val_df, random_part, name, fold):
        calls.append((name, fold, random_part))
        conv = {"model": name, "fold": fold, "n_messages": 0, "messages": "", "singular": False,
                "optimizer": "bobyqa", "seconds": 0.0, "npar": 1, "nobs": len(fit_df),
                "converged_ok": (name, fold) not in bad}  # fmt: skip
        pred = val_df["LOG_UNITS"].to_numpy() + offsets[name][fold]
        pred[(val_df["MEAN_SOURCE"] == "none").to_numpy()] = np.nan
        return pred, conv

    calls = []
    return fit_predict, calls


def test_run_p1_stops_at_first_failure_and_does_not_fit_later_candidates():
    rows = sim_rows(seed=5)
    offsets = {
        "C0": {"A": 0.5, "B": 0.5}, "C1": {"A": 0.4, "B": 0.4}, "C2": {"A": 0.41, "B": 0.41},
        "C3": {"A": 0.1, "B": 0.1}, "C4": {"A": 0.1, "B": 0.1},
    }  # fmt: skip
    fp, calls = _fake(offsets)
    out = p1.run_p1(rows, folds=SMALL_FOLDS, fit_predict=fp)
    log = out["selection_log"].set_index("candidate")
    assert log.loc["C1", "decision"] == "accepted" and log.loc["C2", "decision"] == "rejected"
    assert log.loc["C3", "decision"] == "not fitted" and log.loc["C4", "reason"] == "not fitted: sequence stopped"
    assert log.loc["C2", "compared_to"] == "C1"
    assert {c[0] for c in calls} == {"C0", "C1", "C2"}
    assert out["selected"]["model"] == "C1" and "(0 + LOG_DISC | UPC)" in out["selected"]["random_part"]
    assert list(out["selection_log"].columns) == p1.SELECTION_COLS
    assert set(out["convergence"]["model"]) == {"C0", "C1", "C2"} and len(out["convergence"]) == 6
    m = out["metrics"]
    c1 = m[(m["model"] == "C1") & (m["fold"] == "A") & (m["population"] == p1.POP_SEEN) & (m["stratum"] == "all")]
    assert c1["mae_pair_mean"].iloc[0] == pytest.approx(0.4)
    assert set(out["weekly_diff"]["candidate"]) == {"C1", "C2"}
    assert np.allclose(out["weekly_diff"].query("candidate == 'C1'")["mae_diff"], -0.1)
    assert len(out["coverage"]) == 2 and out["coverage"]["n_unseen_rows"].iloc[0] > 0


def test_run_p1_all_accepted_builds_cumulative_random_part():
    rows = sim_rows(seed=6)
    offsets = {n: {"A": 0.5 - 0.1 * i, "B": 0.5 - 0.1 * i} for i, n in enumerate(["C0", "C1", "C2", "C3", "C4"])}
    fp, calls = _fake(offsets)
    out = p1.run_p1(rows, folds=SMALL_FOLDS, fit_predict=fp)
    assert out["selected"]["model"] == "C4"
    rp = dict(((n, f), r) for n, f, r in calls)
    assert rp[("C2", "A")] == "(1 + LOG_DISC | UPC) + (1 + LOG_DISC | STORE_NUM) + (1 | STORE_NUM:UPC)"
    assert rp[("C3", "A")].endswith("+ (0 + LOG_DISC | STORE_NUM:UPC)") and rp[("C3", "A")].startswith("(1 + LOG_DISC")
    assert rp[("C4", "B")].endswith("(0 + FEATURE | UPC) + (0 + DISPLAY | UPC)") and "STORE_NUM:UPC)" in rp[("C4", "B")]


def test_run_p1_convergence_failure_stops_sequence():
    rows = sim_rows(seed=7)
    offsets = {n: {"A": 0.5 - 0.1 * i, "B": 0.5 - 0.1 * i} for i, n in enumerate(["C0", "C1", "C2", "C3", "C4"])}
    fp, calls = _fake(offsets, bad={("C1", "B")})
    out = p1.run_p1(rows, folds=SMALL_FOLDS, fit_predict=fp)
    log = out["selection_log"].set_index("candidate")
    assert log.loc["C1", "decision"] == "rejected" and not log.loc["C1", "converged_ok"]
    assert out["selected"]["model"] == "C0" and {c[0] for c in calls} == {"C0", "C1"}


def test_run_p1_never_touches_final_window():
    rows = sim_rows(seed=8, n_w=30)
    fp, _ = _fake({n: {"A": 0.1, "B": 0.1} for n in ["C0", "C1", "C2", "C3", "C4"]})
    bad = {"A": {"fit": (0, 11), "val": (12, 17)}, "B": {"fit": (0, 17), "val": (18, 200)}}
    with pytest.raises(ValueError, match="beyond"):
        p1.run_p1(rows, folds=bad, fit_predict=fp)


def test_protocol_config_is_json_ready():
    import json

    cfg = p1.protocol_config()
    json.dumps(cfg)
    assert cfg["min_rel_gain"] == 0.005 and cfg["pair_ols"]["min_weeks"] == 8 and cfg["max_week_allowed"] == 103
    assert cfg["folds"]["A"] == {"fit": [0, 51], "val": [52, 77]}


# ----------------------------------------------------------------------------------------------- R test
@pytest.mark.r
def test_end_to_end_c0_vs_c1_with_r(monkeypatch):
    pytest.importorskip("rpy2")
    monkeypatch.setattr(p1, "CANDIDATES", p1.CANDIDATES[:1])
    rows = sim_rows(n_s=8, n_p=6, n_w=24, seed=11, slope_sd=0.8)
    out = p1.run_p1(rows, folds=SMALL_FOLDS)
    conv = out["convergence"]
    assert set(conv["model"]) == {"C0", "C1"} and len(conv) == 4
    assert (conv["nobs"] > 0).all() and (conv["seconds"] > 0).all()
    log = out["selection_log"]
    assert log.loc[0, "candidate"] == "C1" and log.loc[0, "decision"] in {"accepted", "rejected"}
    assert np.isfinite(log.loc[0, ["mae_fold_A", "mae_fold_B", "mean_rel_gain"]].astype(float)).all()
    m = out["metrics"]
    assert {"C0", "C1", p1.OLS_POOLED, p1.OLS_PAIR} <= set(m["model"])
    seen = m[(m["population"] == p1.POP_SEEN) & (m["stratum"] == "all") & (m["fold"] == "A")].set_index("model")
    assert seen.loc["C0", "mae_pair_mean"] < 1.0 and seen.loc["C0", "n_rows"] == seen.loc["C1", "n_rows"]
    unseen = m[(m["population"] == p1.POP_UNSEEN) & (m["fold"] == "A") & (m["model"] == "C0")]
    assert unseen["n_rows"].iloc[0] > 0 and np.isfinite(unseen["mae_pair_mean"].iloc[0])
