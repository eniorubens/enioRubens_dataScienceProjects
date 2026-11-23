"""Testes sinteticos da etapa 3: preparo, estimadores, EDA e notebooks."""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from freq_sinistros import (  # noqa: E402
    artifacts,
    compare,
    eda,
    labels,
    metrics,
    models,
    notebook_checks,
    plots,
    prep,
)
from freq_sinistros.data import load_partitions  # noqa: E402
from tests.test_stage2 import synthetic_frame  # noqa: E402


def simulate(n=4000, seed=0, alpha=0.0):
    """Dados com taxa log-linear conhecida e exposicao variavel."""
    rng = np.random.default_rng(seed)
    frame = synthetic_frame(n, seed)
    frame["Exposure"] = np.round(rng.uniform(0.05, 1.0, n), 2)
    x = (frame["Area"].astype(str) == "B").to_numpy() * 0.5
    mu = frame["Exposure"].to_numpy() * np.exp(-1.2 + x)
    if alpha > 0:
        lam = rng.gamma(1 / alpha, alpha * mu)
    else:
        lam = mu
    frame["ClaimNb"] = rng.poisson(lam).astype(float)
    return frame


# ---------------------------------------------------------------- prep
def test_preprocessor_learns_only_from_fit_rows_and_maps_unknowns():
    frame = simulate(3000)
    fit_rows = frame.iloc[:2000].copy()
    fit_rows["Region"] = pd.Categorical(["R11"] * 2000)
    pre = prep.Preprocessor(rare_min_rows=50).fit(fit_rows)
    other = frame.iloc[2000:]
    matrix, report = pre.transform(other)
    assert matrix.shape == (len(other), len(pre.columns))
    assert report["unknown_levels"]["Region"] == int(
        (other["Region"] != "R11").sum())
    assert np.isfinite(matrix).all()
    assert "const" == pre.columns[0] and (matrix[:, 0] == 1).all()
    assert not any("IDpol" in c for c in pre.columns)


def test_rare_levels_are_pooled_and_missing_values_imputed():
    frame = simulate(1500)
    frame.loc[:3, "VehPower"] = 99.0   # nivel raro
    pre = prep.Preprocessor(rare_min_rows=100).fit(frame)
    assert "99" in pre.pooled["VehPower"] and pre.has_other["VehPower"]
    test = frame.iloc[:5].copy()
    test.loc[test.index[0], "DrivAge"] = np.nan
    test.loc[test.index[1], "Region"] = np.nan
    matrix, report = pre.transform(test)
    assert report["imputed_numeric"]["DrivAge"] == 1
    assert report["unknown_levels"]["Region"] >= 1
    assert np.isfinite(matrix).all()


def test_bonus_malus_only_in_sensitivity_design():
    frame = simulate(1500)
    main = prep.Preprocessor().fit(frame)
    bonus = prep.Preprocessor(with_bonus_malus=True).fit(frame)
    assert not any(c.startswith("BonusMalus") for c in main.columns)
    assert any(c.startswith("BonusMalus") for c in bonus.columns)
    with pytest.raises(RuntimeError):
        prep.Preprocessor().transform(frame)
    with pytest.raises(ValueError):
        prep.Preprocessor().fit(frame.iloc[0:0])


def test_design_checks_flag_empty_and_rank_deficiency():
    frame = simulate(1500)
    pre = prep.Preprocessor().fit(frame)
    matrix, _ = pre.transform(frame)
    checks = prep.design_checks(matrix, pre.columns)
    assert checks["all_finite"] and checks["full_column_rank"]
    bad = np.hstack([matrix, matrix[:, [1]]])
    assert not prep.design_checks(bad, pre.columns + ["dup"])[
        "full_column_rank"]


# ------------------------------------------------------------ estimadores
def test_poisson_irls_matches_statsmodels_and_doubles_with_exposure():
    sm = pytest.importorskip("statsmodels.api")
    frame = simulate(3000)
    pre = prep.Preprocessor(rare_min_rows=30).fit(frame)
    x, _ = pre.transform(frame)
    y = frame["ClaimNb"].to_numpy()
    e = frame["Exposure"].to_numpy()
    fit = models.fit_poisson(x, y, e, pre.columns)
    ref = sm.GLM(y, x, family=sm.families.Poisson(),
                 offset=np.log(e)).fit()
    assert fit.converged
    assert np.allclose(fit.beta, ref.params, atol=1e-6)
    assert fit.loglik == pytest.approx(ref.llf, rel=1e-9)
    one = fit.predict(x[:50], e[:50])
    two = fit.predict(x[:50], 2 * e[:50])
    assert np.allclose(two, 2 * one, rtol=1e-12)


def test_nb2_recovers_dispersion_and_doubling_holds():
    frame = simulate(20000, seed=3, alpha=0.8)
    pre = prep.Preprocessor(rare_min_rows=30).fit(frame)
    x, _ = pre.transform(frame)
    y = frame["ClaimNb"].to_numpy()
    e = frame["Exposure"].to_numpy()
    pois = models.fit_poisson(x, y, e, pre.columns)
    nb = models.fit_nb2(x, y, e, pre.columns, start=pois)
    assert nb.converged and not nb.extras["alpha_at_poisson_limit"]
    assert 0.5 < nb.alpha < 1.2
    assert nb.loglik > pois.loglik
    assert nb.extras["max_abs_score_beta_per_row"] < 1e-5
    assert np.allclose(nb.predict(x[:20], 2 * e[:20]),
                       2 * nb.predict(x[:20], e[:20]))


def test_nb2_near_poisson_limit_is_stable():
    frame = simulate(8000, seed=5, alpha=0.0)
    pre = prep.Preprocessor(rare_min_rows=30).fit(frame)
    x, _ = pre.transform(frame)
    y = frame["ClaimNb"].to_numpy()
    e = frame["Exposure"].to_numpy()
    pois = models.fit_poisson(x, y, e, pre.columns)
    nb = models.fit_nb2(x, y, e, pre.columns, start=pois)
    assert np.isfinite(nb.loglik) and np.isfinite(nb.beta).all()
    assert nb.alpha < 0.2
    assert abs(nb.loglik - pois.loglik) < 5.0
    mu = nb.predict(x, e)
    near = metrics.nb2_logpmf(y, mu, 1e-9)
    assert np.isfinite(near).all()
    assert near == pytest.approx(metrics.poisson_logpmf(y, mu), abs=1e-6)


def test_global_rate_and_dispersion_diagnostics():
    y = np.array([0.0, 1.0, 0.0, 2.0])
    e = np.array([0.5, 1.0, 0.5, 1.0])
    g = models.fit_global(y, e)
    assert g.rate == pytest.approx(3 / 3.0)
    assert np.allclose(g.predict(None, e), e)
    frame = simulate(6000, seed=2, alpha=1.0)
    pre = prep.Preprocessor(rare_min_rows=30).fit(frame)
    x, _ = pre.transform(frame)
    fit = models.fit_poisson(x, frame["ClaimNb"].to_numpy(), frame[
        "Exposure"].to_numpy(), pre.columns)
    mu = fit.predict(x, frame["Exposure"].to_numpy())
    diag = models.dispersion_diagnostics(frame["ClaimNb"].to_numpy(), mu,
                                         x.shape[1])
    assert diag["degrees_of_freedom"] == len(frame) - x.shape[1]
    assert diag["cameron_trivedi_alpha"] > 0.3


def test_resource_monitor_reports_time_and_memory():
    with models.ResourceMonitor(interval=0.005) as monitor:
        block = np.ones((2000, 2000))
        block.sum()
    assert monitor.wall_s > 0 and monitor.peak_rss_mib > 0
    assert monitor.delta_rss_mib >= 0


# ------------------------------------------------------------------ EDA
def test_driver_age_bands_and_rate_table_denominators():
    ages = pd.Series([18, 24, 25, 74, 75, 100, 17, np.nan])
    got = eda.driver_age_band(ages).tolist()
    assert got[:6] == ["18-24", "18-24", "25-34", "65-74", "75+", "75+"]
    assert got[6:] == [eda.DRIVAGE_INVALID] * 2
    frame = simulate(1200)
    table = eda.rate_table(frame, eda.driver_age_band(frame["DrivAge"]),
                           eda.DRIVAGE_LABELS)
    assert table["rows"].sum() == len(frame)
    assert table["claims"].sum() == frame["ClaimNb"].sum()
    row = table.iloc[0]
    assert row["rate_per_100"] == pytest.approx(
        100 * row["claims"] / row["exposure"])


def test_short_exposure_check_detects_higher_short_rates():
    rng = np.random.default_rng(1)
    n = 60000
    frame = synthetic_frame(n, 1)
    frame["Exposure"] = np.where(rng.random(n) < 0.3, 0.05, 1.0)
    rate = np.where(frame["Exposure"] < 0.5, 0.6, 0.1)  # nao proporcional
    frame["ClaimNb"] = rng.poisson(rate * frame["Exposure"]).astype(float)
    bands, strata, summary = eda.short_exposure_check(frame)
    assert summary["strata_supported_in_both"] > 0
    assert summary["median_ratio_short_over_ref"] > 3
    assert summary["pooled_short_ratio_observed_expected"] > 3
    assert bands["ratio_observed_expected"].notna().any()
    assert len(strata) > 0


def test_extremes_and_histogram_and_checks():
    frame = simulate(2000)
    frame.loc[0, "ClaimNb"] = 9.0
    frame.loc[1, "Exposure"] = 1.0
    assert eda.extreme_cases(frame)["claimnb_gt_4_rows"] >= 1
    hist = eda.exposure_histogram(frame)
    assert hist["rows"].sum() == len(frame)
    check = eda.unknown_level_check(frame, n=1000)
    assert check["design_all_finite"]
    assert check["fit_rows"] + check["heldout_train_rows"] == len(frame)


# ------------------------------------------------ comparacao e regras
def small_eval():
    frame = simulate(3000, seed=7, alpha=0.5)
    pre = prep.Preprocessor(rare_min_rows=30).fit(frame.iloc[:2000])
    fits, records, _ = compare.fit_candidates(frame.iloc[:2000], pre,
                                              {"context": "t"})
    valid = frame.iloc[2000:].reset_index(drop=True)
    mu, _ = compare.predict_all(fits, pre, valid)
    ev = compare.EvalSet(valid, mu, {"nb2": fits["nb2"].alpha},
                         compare.PAIRS)
    return ev, records


def test_evalset_statistic_matches_direct_metrics_and_reconciles():
    ev, records = small_eval()
    assert all(r["converged"] for r in records)
    stat = ev.statistic(np.arange(ev.n))
    assert stat["dev|poisson"] == pytest.approx(
        metrics.mean_poisson_deviance(ev.y, ev.mu["poisson"]))
    assert stat["d_dev|nb2-poisson"] == pytest.approx(
        stat["dev|nb2"] - stat["dev|poisson"])
    assert stat["ratediff|global|overall"] == pytest.approx(
        100 * (ev.mu["global"].sum() - ev.y.sum()) / ev.e.sum())
    table = compare.reconcile(ev, "t", "validation")
    assert table["ok"].all()
    iv = metrics.bootstrap_intervals(ev.statistic, ev.n, replicas=30,
                                     seed=1)
    assert {"d_dev|poisson-global", "d_ll|nb2-poisson"} <= set(
        iv["indicator"])


def test_rules_and_stability_logic():
    ev, _ = small_eval()
    iv = compare.interval_table(ev, "t", "validation") if False else (
        metrics.bootstrap_intervals(ev.statistic, ev.n, replicas=40,
                                    seed=2))
    met = compare.metric_rows(ev, "t", "validation")
    rules = compare.evaluate_rules(met, iv)
    assert rules["preferred_by_rules"] in {"global", "poisson", "nb2"}
    assert "etapa 4" in rules["note"]
    signs = {"a": {"d_dev_poisson_minus_global": -1.0,
                   "d_ll_nb2_minus_poisson": 2.0,
                   "d_dev_nb2_minus_global": -1.0},
             "b": {"d_dev_poisson_minus_global": -0.5,
                   "d_ll_nb2_minus_poisson": 1.0,
                   "d_dev_nb2_minus_global": 0.2}}
    out = compare._stability(signs, ["a", "b"])
    assert out["d_dev_poisson_minus_global"]["same_sign_across_viable"]
    assert not out["d_dev_nb2_minus_global"]["same_sign_across_viable"]
    single = compare._stability(signs, ["a"])
    assert not single["d_ll_nb2_minus_poisson"]["verified"]


def test_select_sample_is_nested_and_hash_stable():
    frame = simulate(500)
    a = compare.select_sample(frame, 100)
    b = compare.select_sample(frame.sample(frac=1.0, random_state=1), 100)
    assert compare.ids_hash(a["IDpol"]) == compare.ids_hash(b["IDpol"])
    assert len(compare.select_sample(frame, None)) == 500


# ------------------------------------------------------------- dados
def test_loader_refuses_the_test_partition(tmp_path):
    with pytest.raises(ValueError, match="reservado"):
        load_partitions(tmp_path, keep=("train", "test"))


def test_labels_and_plots_smoke():
    assert labels.text("rows", "pt_br") == "registros"
    assert labels.text("rows", "en_us") == "records"
    with pytest.raises(ValueError):
        labels.text("rows", "fr_fr")
    frame = simulate(3000)
    dist = eda.audit.claimnb_distribution(frame)
    hist = eda.exposure_histogram(frame)
    table = eda.rate_table(frame, eda.driver_age_band(frame["DrivAge"]),
                           eda.DRIVAGE_LABELS)
    bands, _, _ = eda.short_exposure_check(frame)
    ev, _ = small_eval()
    iv = metrics.bootstrap_intervals(ev.statistic, ev.n, replicas=20,
                                     seed=1)
    cal = compare.calibration_rows(ev, "t", "validation", iv)
    zer = compare.zero_rows(ev, "t", "validation", iv)
    for lang in labels.LANGUAGES:
        for fig in (plots.plot_claimnb(dist, lang),
                    plots.plot_exposure(hist, lang),
                    plots.plot_rates(table, lang, "region"),
                    plots.plot_short_check(bands, lang),
                    plots.plot_calibration_bands(cal, lang),
                    plots.plot_zero_diff(zer, lang),
                    plots.plot_forest(iv, lang, ["d_dev|poisson-global"])):
            assert fig.axes
            for ax in fig.axes:
                assert ax.get_title()[0].isupper()
                assert not ax.spines["top"].get_visible()
                assert not ax.spines["right"].get_visible()
            plots.plt.close(fig)
    fits = pd.DataFrame({"context": ["main"] * 3,
                         "degree": ["100"] * 3,
                         "candidate": ["global", "poisson", "nb2"],
                         "wall_s": [0.1, 0.2, 0.3]})
    plots.plt.close(plots.plot_cost(fits, "pt_br"))


def test_localize_columns():
    table = pd.DataFrame({"rows": [1], "candidate": ["a"], "other": [2]})
    assert "registros" in artifacts.localize(table, "pt_br").columns
    assert "records" in artifacts.localize(table, "en_us").columns
    with pytest.raises(ValueError):
        artifacts.localize(table, "fr_fr")


def test_project_root_finds_pyproject():
    assert (artifacts.project_root() / "pyproject.toml").exists()


# ------------------------------------------------------------ notebooks
def make_nb(lang, code_extra="x = 1", ref="Referências", cite="Smith"):
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text}

    def code(text):
        return {"cell_type": "code", "metadata": {}, "source": text,
                "outputs": [], "execution_count": None}

    return {"cells": [
        md("# Titulo"), code(f'LANG = "{lang}"\n{code_extra}'),
        md(f"Texto ({cite} & Lee, 2020)."), md(f"## {ref}\n- Smith & Lee "
                                                "(2020).")],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def test_notebook_pair_validation_accepts_parity():
    pt, en = make_nb("pt_br"), make_nb("en_us", ref="References")
    assert notebook_checks.validate_pair(pt, en) == []


def test_notebook_pair_validation_detects_problems():
    pt = make_nb("pt_br", code_extra="x = 1")
    en = make_nb("en_us", code_extra="x = 2", ref="References")
    assert any("difere" in p for p in notebook_checks.validate_pair(pt, en))
    bad = make_nb("pt_br", code_extra="df = pd.read_csv('a')")
    assert any("proibida" in p for p in
               notebook_checks.validate_structure(bad, "pt_br"))
    noref = make_nb("pt_br", cite="Jones")
    assert any("sem referencia" in p for p in
               notebook_checks.validate_structure(noref, "pt_br"))
    wrong_end = make_nb("pt_br", ref="Outra")
    assert any("ultima celula" in p for p in
               notebook_checks.validate_structure(wrong_end, "pt_br"))


def test_notebook_directory_validation_and_status(tmp_path):
    for lang, ref in (("pt_br", "Referências"), ("en_us", "References")):
        folder = tmp_path / "notebooks" / lang
        folder.mkdir(parents=True)
        (folder / "01.ipynb").write_text(
            json.dumps(make_nb(lang, ref=ref)), "utf-8")
    (tmp_path / "notebooks" / "pt_br" / "solo.ipynb").write_text(
        json.dumps(make_nb("pt_br")), "utf-8")
    result = notebook_checks.validate_directory(tmp_path)
    assert result["01.ipynb"] == []
    assert result["solo.ipynb"] == ["sem par no outro idioma"]
    status = notebook_checks.execution_status(make_nb("pt_br"))
    assert status == {"code_cells": 1, "executed": 0, "errors": 0}
