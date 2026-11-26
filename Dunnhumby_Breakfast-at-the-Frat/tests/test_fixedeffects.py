import numpy as np
import pandas as pd
import pytest

from retail_multilevel.features import add_model_features, pair_means
from retail_multilevel.fixedeffects import FE_MODELS, LR_PAIRS, RANDOM, pseudo_r2


def test_constants_nested_order():
    names = [n for n, _ in FE_MODELS]
    assert names == ["F0_null", "F1_price", "F2_promo", "F3_season"]
    assert RANDOM == "(1 | STORE_NUM) + (1 | UPC) + (1 | STORE_NUM:UPC)"
    assert LR_PAIRS == list(zip(names[:-1], names[1:], strict=True))
    for (_, small), (_, big) in zip(FE_MODELS[:-1], FE_MODELS[1:], strict=True):
        assert big.startswith(small)


def test_pseudo_r2_pure():
    a = pd.DataFrame({"group": ["s", "Residual"], "term": ["(Intercept)", ""], "variance": [1.0, 2.0], "sd": [1, 1]})
    b = pd.DataFrame({"group": ["s", "Residual"], "term": ["(Intercept)", ""], "variance": [1.5, 1.0], "sd": [1, 1]})
    out = pseudo_r2(a, b).set_index("group")
    assert out.loc["s", "reduction"] == pytest.approx(-0.5)
    assert out.loc["Residual", "reduction"] == pytest.approx(0.5)


TRUE = {"LOG_DISC": -2.0, "LOG_BASE_DEV": -1.0, "FEATURE": 0.4, "DISPLAY": 0.3, "TPR_ONLY": 0.1}


@pytest.fixture(scope="module")
def tables():
    pytest.importorskip("rpy2")
    from retail_multilevel.fixedeffects import fit_fixed_effects

    rng = np.random.default_rng(11)
    n_s, n_p, n_w = 20, 12, 30
    u_s = rng.normal(0, 0.3, n_s)
    u_p = rng.normal(0, 0.6, n_p)
    u_sp = rng.normal(0, 0.2, (n_s, n_p))
    base_p = rng.normal(1.5, 0.4, n_p)
    s = np.repeat(np.arange(n_s), n_p * n_w)
    p = np.tile(np.repeat(np.arange(n_p), n_w), n_s)
    w = np.tile(np.arange(n_w), n_s * n_p)
    lb = base_p[p] + rng.normal(0, 0.05, s.size) + 0.1 * (rng.random(n_s * n_p)[s * n_p + p] > 0.5) * 0
    disc = -rng.choice([0.0, 0.1, 0.2, 0.3], size=s.size, p=[0.6, 0.15, 0.15, 0.1])
    feat = (rng.random(s.size) < 0.15).astype(int)
    disp = (rng.random(s.size) < 0.15).astype(int)
    tpr = (disc < 0).astype(int)
    woy = w % 52 + 1
    y = (
        2.0
        + u_s[s]
        + u_p[p]
        + u_sp[s, p]
        + TRUE["LOG_DISC"] * disc
        + TRUE["LOG_BASE_DEV"] * (lb - pd.Series(lb).groupby(s * n_p + p).transform("mean").to_numpy())
        + TRUE["FEATURE"] * feat
        + TRUE["DISPLAY"] * disp
        + TRUE["TPR_ONLY"] * tpr
        + 0.1 * np.sin(2 * np.pi * woy / 52.18)
        + rng.normal(0, 0.3, s.size)
    )
    df = pd.DataFrame(
        {
            "STORE_NUM": s,
            "UPC": p,
            "PAIR_ID": pd.Series(s).astype(str) + "_" + pd.Series(p).astype(str),
            "WEEK": w,
            "WEEK_OF_YEAR": woy,
            "LOG_UNITS": y,
            "LOG_BASE_PRICE": lb,
            "LOG_PRICE": lb + disc,
            "FEATURE": feat,
            "DISPLAY": disp,
            "TPR_ONLY": tpr,
        }
    )
    feats = add_model_features(df, pair_means(df))
    return fit_fixed_effects(feats)


@pytest.mark.r
def test_tables_and_lr(tables):
    expected = {"fits", "lr", "fixef", "varcomp_final", "varcomp_steps", "pseudo_r2", "ols_pooled"}
    assert expected <= set(tables)
    assert list(tables["fits"]["name"]) == [n for n, _ in FE_MODELS]
    assert not tables["fits"]["reml"].any()
    lr = tables["lr"]
    assert list(lr["df"]) == [3, 3, 7]
    assert (lr["stat"] > 0).all()
    assert list(lr.columns) == ["restricted", "full", "stat", "df", "p_value", "p_naive"]
    assert (lr["p_value"] == lr["p_naive"]).all()
    assert list(tables["fixef"].columns) == ["term", "estimate", "std_error", "z", "ci_low", "ci_high"]


@pytest.mark.r
def test_recovers_coefficients(tables):
    fx = tables["fixef"].set_index("term")
    for term, true in TRUE.items():
        assert fx.loc[term, "estimate"] == pytest.approx(true, abs=0.15), term
    assert (fx["ci_low"] < fx["estimate"]).all()


@pytest.mark.r
def test_pseudo_r2_and_ols(tables):
    pr = tables["pseudo_r2"]
    assert len(pr) == 4
    assert set(pr["group"]) == {"STORE_NUM:UPC", "UPC", "STORE_NUM", "Residual"}
    assert list(pr.columns) == ["group", "var_null", "var_final", "reduction"]
    assert pr.set_index("group").loc["Residual", "reduction"] > 0
    ols = tables["ols_pooled"]
    assert set(ols["model"]) == {"ols_pooled", "ols_upc_fe"}
    assert not ols["term"].str.startswith("factor(UPC)").any()
    assert set(ols["model"].unique()) == {"ols_pooled", "ols_upc_fe"}
    assert set(tables["varcomp_steps"]["model"]) == {n for n, _ in FE_MODELS}
