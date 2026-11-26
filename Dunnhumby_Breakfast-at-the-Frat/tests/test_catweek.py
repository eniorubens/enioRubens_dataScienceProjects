import numpy as np
import pandas as pd
import pytest

from retail_multilevel import catweek
from retail_multilevel.features import add_model_features, pair_means
from retail_multilevel.multilevel import LmerFit


def _fit(name, fixef_shift=0.0, messages=(), loglik=-100.0, npar=10, sec=1.0):
    terms = catweek.KEY_TERMS
    fixef = pd.DataFrame(
        {"term": terms, "estimate": np.arange(len(terms)) + fixef_shift, "std_error": np.full(len(terms), 0.1)}
    )
    vc = pd.DataFrame(
        {
            "group": ["STORE_NUM:UPC", "Residual"],
            "term": ["(Intercept)", ""],
            "variance": [1.0, 3.0],
            "sd": [1.0, 3**0.5],
        }
    )
    return LmerFit(name, "f", True, 100, npar, loglik, 0.0, 0.0, vc, fixef, list(messages), None, sec)


def _frame(n_weeks=6):
    rows = [
        {"CATEGORY": c, "WEEK": w, "PAIR_ID": f"{c}{u}", "STORE_NUM": 1, "UPC": u}
        for c in ("A", "B")
        for u in (1, 2)
        for w in range(n_weeks)
    ]
    return pd.DataFrame(rows)


def test_formulas():
    ref, sens = catweek.reference_formula(), catweek.sensitivity_formula()
    assert sens == ref + " + (1 | CATWEEK)"
    assert "(1 | STORE_NUM:UPC)" in ref
    assert "CATWEEK" not in ref
    assert ref.startswith("LOG_UNITS ~ 1 + LOG_DISC")


def test_add_catweek_levels_and_counts():
    df = catweek.add_catweek(_frame())
    assert df["CATWEEK"].nunique() == 12
    assert df["CATWEEK"].iloc[0] == "A|0"
    c = catweek.counts(df)
    assert c["n_rows"] == 24 and c["n_pairs"] == 4 and c["n_categories"] == 2
    assert c["n_catweek_levels"] == 12 and c["max_week"] == 5 and c["n_missing_category"] == 0


def test_missing_category_raises():
    df = _frame()
    df.loc[0, "CATEGORY"] = None
    with pytest.raises(ValueError):
        catweek.add_catweek(df)


def test_temporal_guard():
    df = _frame()
    catweek.check_training(df)
    df.loc[0, "WEEK"] = 104
    with pytest.raises(RuntimeError):
        catweek.check_training(df)


def test_fixef_compare_assembly():
    out = catweek.fixef_compare(_fit("a"), _fit("b", fixef_shift=0.5))
    assert list(out["term"]) == catweek.KEY_TERMS
    assert np.allclose(out["diff"], 0.5)
    assert np.allclose(out["se_ratio"], 1.0)
    assert list(out.columns) == ["term", "est_ref", "se_ref", "est_catweek", "se_catweek", "diff", "se_ratio"]


def test_varcomp_compare_shares():
    out = catweek.varcomp_compare({"a": _fit("a"), "b": _fit("b")})
    assert set(out["model"]) == {"a", "b"}
    a = out[out["model"] == "a"].set_index("group")
    assert a.loc["Residual", "share"] == pytest.approx(0.75)


def test_fit_with_retry_uses_fallback_only_on_convergence_messages():
    calls = []

    def fitter(r_df, formula, name, reml, control):
        calls.append(control)
        msgs = ["Model failed to converge"] if len(calls) == 1 else []
        return _fit(name, messages=msgs)

    kept, rows = catweek.fit_with_retry(None, "f", "m", "ctl", fitter)
    assert calls == ["ctl", catweek.FALLBACK_CONTROL]
    assert [r["used"] for r in rows] == [False, True] and rows[1]["optimizer"] == "nloptwrap"
    assert not catweek.convergence_messages(kept)

    calls.clear()
    kept, rows = catweek.fit_with_retry(None, "f", "m", "ctl", lambda *a, **k: _fit("m", messages=["singular fit"]))
    assert len(rows) == 1 and rows[0]["singular"] and catweek.is_singular(kept)


def test_fit_both_passes_identical_rows():
    seen = []

    def frame_builder(d, factors):
        assert "CATWEEK" in factors
        return d

    def fitter(r_df, formula, name, reml, control):
        seen.append((r_df, formula))
        return _fit(name)

    df = _frame()
    df = df.assign(
        **{c: 0.0 for c in __import__("retail_multilevel.fixedeffects", fromlist=["x"])._R_COLS if c not in df.columns}
    )
    fits, conv, n = catweek.fit_both(df, frame_builder=frame_builder, fitter=fitter)
    assert n == len(df) and set(fits) == {catweek.REFERENCE, catweek.SENSITIVITY}
    assert seen[0][0] is seen[1][0]
    assert seen[0][1] != seen[1][1] and len(conv) == 2


def test_residual_summaries_shapes():
    rng = np.random.default_rng(0)
    df = _frame(12)
    res = df.assign(resid=rng.normal(size=len(df)))
    out = catweek.residual_summaries({"x": res, "y": res})
    assert set(out["lag_corr"]["model"]) == {"x", "y"} and len(out["lag_corr"]) == 8
    assert list(out["week_summary"]["model"]) == ["x", "y"]
    assert "share_category_week" in out["week_summary"].columns
    assert out["week_shocks"]["model"].iloc[0] == "x"


def test_descriptive_lr():
    out = catweek.descriptive_lr(_fit("a", loglik=-110, npar=10), _fit("b", loglik=-100, npar=11))
    assert out["stat"] == pytest.approx(20.0) and out["df"] == 1
    assert out["p_value"] == pytest.approx(out["p_naive"] / 2)


@pytest.mark.r
def test_end_to_end_planted_catweek():
    pytest.importorskip("rpy2")
    rng = np.random.default_rng(5)
    n_s, n_w = 12, 30
    upcs = {"A": [1, 2, 3], "B": [4, 5, 6]}
    sd_cw = 0.5
    shock = {(c, w): rng.normal(0, sd_cw) for c in upcs for w in range(n_w)}
    rows = []
    for s in range(n_s):
        for c, us in upcs.items():
            for u in us:
                for w in range(n_w):
                    lb = 1.0 + 0.1 * u + rng.normal(0, 0.03)
                    disc = -rng.choice([0.0, 0.1, 0.2], p=[0.7, 0.2, 0.1])
                    y = 2 + 0.3 * u + 0.2 * s / n_s - 2.0 * disc + shock[(c, w)] + rng.normal(0, 0.3)
                    rows.append(
                        {
                            "STORE_NUM": s, "UPC": u, "CATEGORY": c, "PAIR_ID": f"{s}_{u}", "WEEK": w,
                            "WEEK_OF_YEAR": w % 52 + 1, "LOG_UNITS": y, "LOG_BASE_PRICE": lb,
                            "LOG_PRICE": lb + disc, "FEATURE": int(rng.random() < 0.1),
                            "DISPLAY": int(rng.random() < 0.1), "TPR_ONLY": int(disc < 0),
                        }
                    )  # fmt: skip
    raw = pd.DataFrame(rows)
    df = add_model_features(raw, pair_means(raw))
    fits, conv, n = catweek.fit_both(df)
    ref, sens = fits[catweek.REFERENCE], fits[catweek.SENSITIVITY]
    assert ref.nobs == sens.nobs == n == len(df)
    cw = sens.varcomp.set_index("group").loc["CATWEEK", "variance"]
    assert cw == pytest.approx(sd_cw**2, rel=0.5)
    res = {k: catweek.conditional_residuals(f, df) for k, f in fits.items()}
    summ = catweek.residual_summaries(res)
    ws = summ["week_summary"].set_index("model")
    assert ws.loc[catweek.SENSITIVITY, "share_category_week"] < ws.loc[catweek.REFERENCE, "share_category_week"]
