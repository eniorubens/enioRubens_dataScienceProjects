import math

import numpy as np
import pandas as pd
import pytest

from retail_multilevel import interpret, scenarios

TERMS = ["(Intercept)", "LOG_DISC", "FEATURE", "DISPLAY", "TPR_ONLY"]


def _fixef():
    return pd.DataFrame({"term": TERMS, "estimate": [3.0, -1.0, math.log(2), math.log(1.5), math.log(0.5)]})


def _vcov():
    v = np.diag([0.5, 0.01, 0.04, 0.02, 0.03])
    v[1, 2] = v[2, 1] = 0.005
    v[1, 4] = v[4, 1] = -0.004
    v[2, 3] = v[3, 2] = 0.003
    return pd.DataFrame(v, index=TERMS, columns=TERMS)


def test_contrast_linear_combination_and_se_known_vcov():
    scen = [("a", 0.2, 1, 0, 0), ("b", 0.1, 0, 0, 1)]
    out = interpret.scenario_contrast(_fixef(), _vcov(), scen)
    v = _vcov().to_numpy()
    beta = _fixef()["estimate"].to_numpy()
    for i, (x, d) in enumerate([([math.log(0.8), 1, 0, 0], 0.2), ([math.log(0.9), 0, 0, 1], 0.1)]):
        xv = np.zeros(5)
        xv[1:] = x
        dlog = xv @ beta
        se = math.sqrt(xv @ v @ xv)
        assert out["dlog"].iloc[i] == pytest.approx(dlog)
        assert out["se"].iloc[i] == pytest.approx(se)
        assert out["effect_pct"].iloc[i] == pytest.approx(100 * math.expm1(dlog))
        assert out["ci_low_pct"].iloc[i] == pytest.approx(100 * math.expm1(dlog - 1.96 * se))
        assert out["ci_high_pct"].iloc[i] == pytest.approx(100 * math.expm1(dlog + 1.96 * se))
        assert out["discount"].iloc[i] == d


def test_contrast_uses_covariance_not_marginal_bounds():
    scen = [("a", 0.2, 1, 0, 0)]
    joint = interpret.scenario_contrast(_fixef(), _vcov(), scen)
    diag = pd.DataFrame(np.diag(np.diag(_vcov().to_numpy())), index=TERMS, columns=TERMS)
    indep = interpret.scenario_contrast(_fixef(), diag, scen)
    assert joint["se"].iloc[0] != pytest.approx(indep["se"].iloc[0])


def test_regular_week_is_exactly_zero():
    out = interpret.scenario_contrast(_fixef(), _vcov(), [("regular", 0.0, 0, 0, 0)])
    assert out[["dlog", "se", "effect_pct", "ci_low_pct", "ci_high_pct"]].to_numpy().tolist() == [[0.0] * 5]


def test_tpr_exclusivity_enforced_and_defaults_respect_it():
    with pytest.raises(ValueError, match="exclusive"):
        interpret.scenario_contrast(_fixef(), _vcov(), [("bad", 0.2, 1, 0, 1)])
    with pytest.raises(ValueError, match="exclusive"):
        interpret.scenario_contrast(_fixef(), _vcov(), [("bad", 0.2, 0, 1, 1)])
    with pytest.raises(ValueError, match="0 or 1"):
        interpret.scenario_contrast(_fixef(), _vcov(), [("bad", 0.2, 2, 0, 0)])
    assert all(not (tpr and (f or d)) for _, _, f, d, tpr in interpret.DEFAULT_SCENARIOS)
    out = interpret.scenario_contrast(_fixef(), _vcov(), interpret.DEFAULT_SCENARIOS)
    assert len(out) == 8


def test_contrast_between_matches_difference_with_joint_se():
    a, b = ("tpr", 0.2, 0, 0, 1), ("disp", 0.2, 0, 1, 0)
    res = interpret.contrast_between(_fixef(), _vcov(), a, b)
    x = np.array([0.0, 0.0, 1.0, -1.0])  # DISPLAY minus TPR_ONLY; the common discount cancels
    v = _vcov().loc[TERMS[1:], TERMS[1:]].to_numpy()
    beta = _fixef().set_index("term")["estimate"]
    assert res["dlog"] == pytest.approx(beta["DISPLAY"] - beta["TPR_ONLY"])
    assert res["se"] == pytest.approx(math.sqrt(x @ v @ x))
    assert res["ratio"] == pytest.approx(math.exp(res["dlog"]))
    assert res["ratio_low"] < res["ratio"] < res["ratio_high"]


def test_reconciles_with_scenario_effects_on_stored_fixef():
    eff = interpret.scenario_effects(_fixef(), interpret.DEFAULT_SCENARIOS)
    con = interpret.scenario_contrast(_fixef(), _vcov(), interpret.DEFAULT_SCENARIOS)
    assert con["effect_pct"].to_numpy() == pytest.approx(eff["effect_pct"].to_numpy(), abs=1e-12)
    assert list(con["scenario"]) == list(eff["scenario"])


def _frame():
    rows = [
        # (feature, display, tpr, discount, base_imputed, fallback, pair, week)
        (0, 0, 1, 0.20, False, False, "p1", 1),  # tpr20 supported, observed base
        (0, 0, 1, 0.23, False, False, "p2", 2),  # boundary: inside the window
        (0, 0, 1, 0.231, False, False, "p3", 3),  # just outside
        (0, 0, 1, 0.18, True, False, "p4", 4),  # supported, carried-forward base
        (0, 0, 0, 0.0, True, True, "p5", 5),  # regular, fallback
        (0, 0, 0, 0.0, False, False, "p5", 6),  # regular, observed
        (1, 0, 1, 0.20, False, False, "p6", 7),  # feature + TPR matches no exact scenario here
        (1, 0, 0, 0.20, False, False, "p7", 8),  # feature 20%
        (0, 1, 0, 0.0, True, True, "p8", 9),  # display 0% via fallback
    ]
    cols = ["FEATURE", "DISPLAY", "TPR_ONLY", "DISCOUNT", "BASE_IMPUTED", "BASE_FALLBACK", "PAIR_ID", "WEEK"]
    return pd.DataFrame(rows, columns=cols)


def test_support_counts_exact_modality_window_and_boundary():
    scen = [("regular", 0.0, 0, 0, 0), ("tpr20", 0.20, 0, 0, 1), ("feat20", 0.20, 1, 0, 0), ("disp0", 0.0, 0, 1, 0)]
    out = scenarios.support_table(_frame(), scen).set_index("scenario")
    assert out.loc["tpr20", "n_rows"] == 3 and out.loc["tpr20", "n_pairs"] == 3  # p1, p2 (boundary), p4
    assert out.loc["tpr20", "n_rows_observed_base"] == 2 and out.loc["tpr20", "n_pairs_observed_base"] == 2
    assert out.loc["tpr20", "n_rows_fallback"] == 0
    assert out.loc["tpr20", "share"] == pytest.approx(3 / 9)
    assert out.loc["tpr20", "share_observed_base"] == pytest.approx(2 / 6)
    assert out.loc["regular", "n_rows"] == 2 and out.loc["regular", "n_pairs"] == 1
    assert out.loc["regular", "n_rows_fallback"] == 1 and out.loc["regular", "n_rows_observed_base"] == 1
    assert out.loc["feat20", "n_rows"] == 1  # the feature + TPR row is not an exact match
    assert out.loc["disp0", "n_rows"] == 1 and out.loc["disp0", "n_rows_fallback"] == 1
    assert out.loc["disp0", "n_rows_observed_base"] == 0
    assert not out["outside_support"].any()


def test_zero_support_flagged_and_kept():
    scen = [("regular", 0.0, 0, 0, 0), ("both25", 0.25, 1, 1, 0), ("tpr50", 0.50, 0, 0, 1)]
    out = scenarios.support_table(_frame(), scen).set_index("scenario")
    assert list(out.index) == ["regular", "both25", "tpr50"]
    assert bool(out.loc["both25", "outside_support"]) and bool(out.loc["tpr50", "outside_support"])
    assert out.loc["both25", ["n_rows", "n_pairs", "n_rows_observed_base", "n_rows_fallback"]].tolist() == [0] * 4
    assert not bool(out.loc["regular", "outside_support"])


def test_support_group_cut_counts_within_group():
    df = _frame().assign(G=["a"] * 5 + ["b"] * 4)
    out = scenarios.support_table(df, [("tpr20", 0.20, 0, 0, 1)], group="G")
    assert list(out.columns)[0] == "G"
    by = out.set_index("G")
    assert by.loc["a", "n_rows"] == 3 and by.loc["b", "n_rows"] == 0
    assert by.loc["a", "share"] == pytest.approx(3 / 5)
    assert bool(by.loc["b", "outside_support"])


def test_temporal_guard():
    df = _frame()
    scenarios.assert_training_only(df)
    late = df.assign(WEEK=[1, 2, 3, 4, 5, 6, 7, 8, 104])
    with pytest.raises(ValueError, match="test period"):
        scenarios.assert_training_only(late)
    with pytest.raises(ValueError, match="test period"):
        scenarios.support_table(late, [("regular", 0.0, 0, 0, 0)])


def test_support_rows_and_pairs_agree_with_interpret_scenario_support():
    df = _frame()
    ours = scenarios.support_table(df, interpret.DEFAULT_SCENARIOS)
    ref = interpret.scenario_support(df, interpret.DEFAULT_SCENARIOS, tol=0.03)
    assert list(ours["n_rows"]) == list(ref["n_rows"])
    assert list(ours["n_pairs"]) == list(ref["n_pairs"])


@pytest.mark.r
def test_vcov_from_r_matches_standard_errors():
    pytest.importorskip("rpy2")
    from retail_multilevel.multilevel import fit_lmer, to_r_frame

    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 15)
    x1, x2 = rng.normal(size=300), rng.normal(size=300)
    y = 1 + 0.5 * x1 - 0.3 * x2 + rng.normal(size=20)[g] + rng.normal(size=300)
    d = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})
    fit = fit_lmer(to_r_frame(d, ["g"]), "y ~ x1 + x2 + (1 | g)", "m", reml=True)
    v = scenarios.vcov_from_r(fit)
    assert list(v.index) == list(v.columns) == list(fit.fixef["term"])
    assert np.allclose(v.to_numpy(), v.to_numpy().T)
    assert np.sqrt(np.diag(v)) == pytest.approx(fit.fixef["std_error"].to_numpy(), rel=1e-6)
