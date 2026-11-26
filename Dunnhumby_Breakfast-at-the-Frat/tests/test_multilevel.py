import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from retail_multilevel.multilevel import LmerFit, fits_table, icc, lr_table, lr_test


def make_fit(name, loglik, npar, reml=True, nobs=100):
    return LmerFit(
        name=name,
        formula="y ~ 1",
        reml=reml,
        nobs=nobs,
        npar=npar,
        loglik=loglik,
        aic=-2 * loglik + 2 * npar,
        bic=-2 * loglik + np.log(nobs) * npar,
        varcomp=pd.DataFrame(columns=["group", "term", "variance", "sd"]),
        fixef=pd.DataFrame(columns=["term", "estimate", "std_error"]),
    )


def test_lr_test_df1_boundary_halves_p():
    r = lr_test(make_fit("a", -100.0, 2), make_fit("b", -96.0, 3))
    assert r["stat"] == pytest.approx(8.0)
    assert r["df"] == 1
    assert r["p_naive"] == pytest.approx(chi2.sf(8.0, 1))
    assert r["p_value"] == pytest.approx(0.5 * chi2.sf(8.0, 1))
    assert set(r) == {"restricted", "full", "stat", "df", "p_value", "p_naive"}
    assert (r["restricted"], r["full"]) == ("a", "b")


def test_lr_test_df2_mixture_and_no_boundary():
    a, b = make_fit("a", -100.0, 2), make_fit("b", -95.0, 4)
    r = lr_test(a, b)
    assert r["df"] == 2
    assert r["p_value"] == pytest.approx(0.5 * chi2.sf(10.0, 1) + 0.5 * chi2.sf(10.0, 2))
    r2 = lr_test(a, b, boundary=False)
    assert r2["p_value"] == pytest.approx(chi2.sf(10.0, 2))


def test_lr_test_negative_stat_is_clipped():
    a, b = make_fit("a", -100.0, 2), make_fit("b", -101.0, 3)
    r = lr_test(a, b)
    assert r["stat"] == 0.0
    assert r["p_naive"] == pytest.approx(1.0)
    assert r["p_value"] == pytest.approx(0.5)


def test_lr_test_validation_errors():
    a = make_fit("a", -100.0, 2)
    with pytest.raises(ValueError, match="REML"):
        lr_test(a, make_fit("b", -90.0, 3, reml=False))
    with pytest.raises(ValueError, match="observations"):
        lr_test(a, make_fit("b", -90.0, 3, nobs=99))
    with pytest.raises(ValueError, match="parameters"):
        lr_test(a, make_fit("b", -90.0, 2))
    with pytest.raises(ValueError, match="parameters"):
        lr_test(make_fit("a", -100.0, 3), make_fit("b", -90.0, 2))


def test_icc_shares_sum_to_one_and_ignore_slopes():
    vc = pd.DataFrame(
        {
            "group": ["store", "store", "prod", "Residual"],
            "term": ["(Intercept)", "price", "(Intercept)", ""],
            "variance": [0.25, 0.04, 1.0, 0.09],
            "sd": [0.5, 0.2, 1.0, 0.3],
        }
    )
    out = icc(vc)
    assert list(out.columns) == ["group", "variance", "share"]
    assert list(out["group"]) == ["store", "prod", "Residual"]
    assert out["share"].sum() == pytest.approx(1.0)
    assert out["share"].iloc[1] == pytest.approx(1.0 / 1.34)


def test_lr_table_and_fits_table_shapes():
    fits = [make_fit("a", -100.0, 2), make_fit("b", -96.0, 3), make_fit("c", -90.0, 4)]
    t = lr_table(fits, [("a", "b"), ("b", "c"), ("a", "c")])
    assert t.shape == (3, 6)
    assert list(t.columns) == ["restricted", "full", "stat", "df", "p_value", "p_naive"]
    ft = fits_table(fits)
    assert len(ft) == 3
    assert {"name", "formula", "reml", "nobs", "npar", "loglik", "aic", "bic", "n_messages", "seconds"} <= set(
        ft.columns
    )


# ---------------------------------------------------------------- R-backed tests


@pytest.fixture(scope="module")
def rml():
    pytest.importorskip("rpy2")
    from retail_multilevel import multilevel

    return multilevel


@pytest.fixture(scope="module")
def crossed(rml):
    rng = np.random.default_rng(42)
    n_store, n_prod, n_rep = 30, 20, 8
    u_s = rng.normal(0, 0.5, n_store)
    u_p = rng.normal(0, 1.0, n_prod)
    store = np.repeat(np.arange(n_store), n_prod * n_rep)
    prod = np.tile(np.repeat(np.arange(n_prod), n_rep), n_store)
    y = 1.0 + u_s[store] + u_p[prod] + rng.normal(0, 0.3, store.size)
    df = pd.DataFrame({"store": store, "prod": prod, "y": y})
    r_df = rml.to_r_frame(df, ["store", "prod"])
    fits = {
        "lm": rml.fit_lm(r_df, "y ~ 1", "lm"),
        "s": rml.fit_lmer(r_df, "y ~ 1 + (1 | store)", "s"),
        "sp": rml.fit_lmer(r_df, "y ~ 1 + (1 | store) + (1 | prod)", "sp"),
    }
    return r_df, fits, df


@pytest.mark.r
def test_varcomp_columns_and_recovery(rml, crossed):
    _, fits, df = crossed
    sp = fits["sp"]
    assert list(sp.varcomp.columns) == ["group", "term", "variance", "sd"]
    assert list(sp.fixef.columns) == ["term", "estimate", "std_error"]
    assert "(Intercept)" in set(sp.fixef["term"])
    sd = dict(zip(sp.varcomp["group"], sp.varcomp["sd"], strict=True))
    assert sd["store"] == pytest.approx(0.5, abs=0.2)
    assert sd["prod"] == pytest.approx(1.0, abs=0.4)
    assert sd["Residual"] == pytest.approx(0.3, abs=0.03)
    assert sp.varcomp.loc[sp.varcomp["group"] == "Residual", "term"].iloc[0] == ""
    shares = icc(sp.varcomp)
    assert shares["share"].sum() == pytest.approx(1.0)
    assert shares.set_index("group").loc["prod", "share"] > shares.set_index("group").loc["store", "share"]
    assert shares.set_index("group").loc["Residual", "share"] < 0.2


@pytest.mark.r
def test_counts_and_lr_tests(rml, crossed):
    _, fits, df = crossed
    assert [fits[k].nobs for k in ("lm", "s", "sp")] == [len(df)] * 3
    assert (fits["lm"].npar, fits["s"].npar, fits["sp"].npar) == (2, 3, 4)
    assert fits["lm"].messages == []
    r = lr_test(fits["lm"], fits["s"])
    assert r["df"] == 1
    assert r["p_value"] < 1e-6
    t = lr_table(list(fits.values()), [("lm", "s"), ("s", "sp")])
    assert t.shape == (2, 6)
    assert (t["p_value"] < 1e-6).all()
    assert fits_table(fits.values()).shape[0] == 3


@pytest.mark.r
def test_ranef_frame(rml, crossed):
    _, fits, _ = crossed
    out = rml.ranef_frame(fits["sp"], "store")
    assert list(out.columns) == ["level", "term", "blup", "cond_sd"]
    assert len(out) == 30
    assert np.isfinite(out["cond_sd"]).all()
    assert (out["cond_sd"] > 0).all()
    assert len(rml.ranef_frame(fits["sp"], "prod")) == 20


@pytest.mark.r
def test_fit_lmer_with_control_and_ml(rml, crossed):
    r_df, _, _ = crossed
    f = rml.fit_lmer(r_df, "y ~ 1 + (1 | store)", "ctl", reml=False, control='lmerControl(optimizer="bobyqa")')
    assert f.reml is False
    assert f.npar == 3
    assert f.seconds > 0
    assert "(Intercept)" in set(f.fixef["term"])
