import numpy as np
import pandas as pd
import pytest

from retail_multilevel.nullmodels import LR_PAIRS, NULL_MODELS, crosscheck_table, statsmodels_crosscheck


def test_null_models_names_and_pairs():
    names = [n for n, _ in NULL_MODELS]
    assert names == ["M0_ols", "M1_store", "M2_product", "M3_crossed", "M4_crossed_pair"]
    assert len(LR_PAIRS) == 5
    assert all(a in names and b in names and names.index(a) < names.index(b) for a, b in LR_PAIRS)


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(7)
    n_s, n_p, n_rep = 25, 15, 6
    u_s = rng.normal(0, 0.4, n_s)
    u_p = rng.normal(0, 0.9, n_p)
    u_sp = rng.normal(0, 0.3, (n_s, n_p))
    s = np.repeat(np.arange(n_s), n_p * n_rep)
    p = np.tile(np.repeat(np.arange(n_p), n_rep), n_s)
    y = 1.0 + u_s[s] + u_p[p] + u_sp[s, p] + rng.normal(0, 0.5, s.size)
    return pd.DataFrame({"y": y, "store": s, "prod": p})


@pytest.fixture(scope="module")
def tables(sim):
    pytest.importorskip("rpy2")
    from retail_multilevel.nullmodels import fit_null_models

    return fit_null_models(sim, y="y", store="store", product="prod")


@pytest.mark.r
def test_fits_icc_and_lr(tables):
    assert list(tables["fits"]["name"]) == [n for n, _ in NULL_MODELS]
    icc4 = tables["icc"].query("model == 'M4_crossed_pair'").set_index("group")
    assert len(icc4) == 4
    assert icc4["share"].sum() == pytest.approx(1.0)
    total = 0.4**2 + 0.9**2 + 0.3**2 + 0.5**2
    assert icc4.loc["prod", "share"] == pytest.approx(0.81 / total, abs=0.15)
    assert icc4.loc["Residual", "share"] == pytest.approx(0.25 / total, abs=0.08)
    assert len(tables["lr"]) == 5
    assert (tables["lr"]["p_value"] < 0.01).all()
    assert set(tables["fixef"]["model"]) == set(tables["fits"]["name"])


@pytest.mark.r
def test_ranef_and_crosscheck(tables, sim):
    assert len(tables["ranef_store"]) == 25
    assert len(tables["ranef_product"]) == 15
    assert len(tables["ranef_pair"]) == 25 * 15
    sm = statsmodels_crosscheck(sim, y="y", group="store")
    cc = crosscheck_table(sm, tables["varcomp"])
    assert len(cc) == 2
    assert (cc["rel_diff"] < 0.01).all()
