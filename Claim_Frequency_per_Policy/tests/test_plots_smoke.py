"""Testes de fumaca dos graficos, com tabelas minimas feitas a mao."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from freq_sinistros import plots  # noqa: E402
from freq_sinistros.labels import text  # noqa: E402

LANGS = ["pt_br", "en_us"]
BANDS = ["(0, 0.1]", "(0.1, 0.5]", "(0.5, 1]"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _check(fig, n_axes=1):
    assert isinstance(fig, Figure)
    assert len(fig.axes) == n_axes
    for ax in fig.axes:
        title = ax.get_title()
        assert title and title[0] == title[0].upper()
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
    return fig


def _band_diffs(candidates=("global", "poisson", "nb2")):
    rows = []
    for k, cand in enumerate(candidates):
        for j, band in enumerate(BANDS):
            diff = 0.1 * (j - k)
            rows.append({"candidate": cand, "segment": band, "band": band,
                         "family": "exposure_band",
                         "rate_difference_pred_minus_obs": diff,
                         "diff_pred_minus_obs": diff,
                         "ci_low": diff - 0.2, "ci_high": diff + 0.2})
    # um IC nao finito deve ser tratado como erro zero
    rows[0]["ci_low"] = np.nan
    frame = pd.DataFrame(rows)
    other = frame.iloc[:2].assign(family="region")
    return pd.concat([frame, other], ignore_index=True)


@pytest.mark.parametrize("lang", LANGS)
def test_claimnb_and_exposure(lang):
    dist = pd.DataFrame({"claimnb": [0, 1, 2], "rows": [90, 8, 2],
                         "share_of_valid_rows": [0.9, 0.08, 0.02]})
    fig = _check(plots.plot_claimnb(dist, lang))
    assert fig.axes[0].get_yscale() == "log"
    assert fig.axes[0].get_ylabel() == text("rows", lang)
    hist = pd.DataFrame({"kind": ["bin", "bin", "exactly 1"],
                         "upper": [0.05, 0.10, 1.0], "rows": [5, 7, 30]})
    fig = _check(plots.plot_exposure(hist, lang))
    assert fig.axes[0].get_legend() is not None


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("low", [[False, False, False], [False, True, True]])
def test_rates_with_and_without_low_support(lang, low):
    table = pd.DataFrame({"segment": ["a", "b", "c"], "low_support": low,
                          "rate_per_100": [5.0, np.nan, 7.5]})
    fig = _check(plots.plot_rates(table, lang, "region"))
    assert len(fig.axes[0].get_xticklabels()) == 3
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert (text("low_support", lang) in labels) == any(low)


@pytest.mark.parametrize("lang", LANGS)
def test_short_check_and_cost(lang):
    bands = pd.DataFrame({"band": BANDS,
                          "ratio_observed_expected": [3.0, 1.2, 0.9]})
    _check(plots.plot_short_check(bands, lang))
    fits = pd.DataFrame([
        {"context": "main", "degree": d, "candidate": c, "wall_s": 1.0 + k}
        for d in ("100000", "train_full")
        for k, c in enumerate(("global", "poisson", "nb2"))]
        + [{"context": "sensitivity", "degree": "100000",
            "candidate": "poisson", "wall_s": 9.0}])
    fig = _check(plots.plot_cost(fits, lang))
    assert len(fig.axes[0].patches) == 6


@pytest.mark.parametrize("lang", LANGS)
def test_calibration_bands_and_zero_diff(lang):
    cal = _band_diffs()
    fig = _check(plots.plot_calibration_bands(cal, lang))
    assert len(fig.axes[0].get_xticklabels()) == len(BANDS)
    fig = _check(plots.plot_calibration_bands(cal, lang, ("poisson",)))
    assert len(fig.axes[0].patches) == len(BANDS)
    zeros = cal[cal["family"] == "exposure_band"]
    zeros = pd.concat([zeros, zeros.iloc[:1].assign(band="overall")])
    fig = _check(plots.plot_zero_diff(zeros, lang))
    assert "overall" not in [t.get_text()
                             for t in fig.axes[0].get_xticklabels()]
    _check(plots.plot_zero_diff(zeros, lang, ("nb2", "poisson")))


@pytest.mark.parametrize("lang", LANGS)
def test_forest_and_paired_difference(lang):
    intervals = pd.DataFrame({
        "population": ["validation", "validation", "test",
                       "test_cluster_bootstrap"],
        "indicator": ["d_dev|poisson-global", "d_dev|nb2-poisson",
                      "d_dev|poisson-global", "d_dev|poisson-global"],
        "estimate": [-0.01, 0.0, -0.02, -0.02],
        "ci_low": [-0.02, -0.01, -0.03, -0.04],
        "ci_high": [0.0, 0.01, -0.01, 0.0]})
    first = intervals[intervals["population"] == "validation"]
    _check(plots.plot_forest(first, lang, ["d_dev|poisson-global",
                                           "d_dev|nb2-poisson"]))
    fig = _check(plots.plot_paired_difference(intervals, lang))
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
        text("bootstrap_main", lang), text("bootstrap_profile", lang)]


@pytest.mark.parametrize("lang", LANGS)
def test_exposure_rates_ignores_total_row(lang):
    bands = pd.DataFrame({
        "band": [*BANDS, "total"],
        "observed_rate_per_100": [30.0, 8.0, 6.0, 7.0],
        "predicted_rate_poisson": [12.0, 8.0, 6.5, 7.0],
        "predicted_rate_poisson_free_exposure": [25.0, 8.0, 6.2, 7.0]})
    fig = _check(plots.plot_exposure_rates(bands, lang))
    assert fig.axes[0].get_yscale() == "log"
    assert len(fig.axes[0].patches) == 3 * len(BANDS)


@pytest.mark.parametrize("lang", LANGS)
def test_stability_uses_only_validation(lang):
    decision = pd.DataFrame({
        "population": ["validation", "validation", "grouped_validation"],
        "degree": ["100000", "train_full", "train_full"],
        "d_dev_poisson_minus_global": [-0.01, -0.02, -0.03],
        "d_dev_poisson_minus_global_lo": [-0.02, -0.03, -0.04],
        "d_dev_poisson_minus_global_hi": [0.0, -0.01, -0.02]})
    fig = _check(plots.plot_stability(decision, lang))
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
        "100000", "train_full"]


@pytest.mark.parametrize("lang", LANGS)
def test_panel_with_and_without_low_support(lang):
    panel = pd.DataFrame({
        "family": ["region"] * 4 + ["area"],
        "candidate": ["poisson"] * 3 + ["global", "poisson"],
        "segment": ["R1", "R2", "R3", "R1", "A"],
        "pred_minus_obs": [0.5, -0.2, 0.1, 0.3, 0.0],
        "ci_low": [0.1, -0.5, np.nan, 0.0, -0.1],
        "ci_high": [0.9, 0.1, np.nan, 0.6, 0.1],
        "low_support": [False, False, True, False, False]})
    fig = _check(plots.plot_panel(panel, lang, "region"))
    assert len(fig.axes[0].get_yticklabels()) == 3
    assert fig.axes[0].get_legend() is not None
    fig = _check(plots.plot_panel(panel, lang, "region", "global"))
    assert len(fig.axes[0].get_yticklabels()) == 1
    _check(plots.plot_panel(panel, lang, "area"))


@pytest.mark.parametrize("lang", LANGS)
def test_validation_vs_test(lang):
    comparison = pd.DataFrame({
        "band": BANDS, "validation_diff": [0.4, 0.1, -0.1],
        "test_diff": [0.5, 0.0, -0.2],
        "test_ci_low": [0.2, np.nan, -0.4],
        "test_ci_high": [0.8, np.nan, 0.0]})
    fig = _check(plots.plot_validation_vs_test(comparison, lang))
    assert len(fig.axes[0].patches) == 2 * len(BANDS)


def test_finalize_capitalizes_title_and_hides_spines():
    fig, ax = plt.subplots()
    ax.set_title("titulo minusculo")
    out = plots._finalize_figure(fig)
    assert out is fig
    assert ax.get_title() == "Titulo minusculo"


def test_show_displays_then_closes(monkeypatch):
    display_module = pytest.importorskip("IPython.display")
    shown = []
    monkeypatch.setattr(display_module, "display", shown.append)
    fig, _ = plt.subplots()
    plots.show(fig)
    assert shown == [fig]
    assert not plt.fignum_exists(fig.number)


def test_unknown_language_is_rejected():
    dist = pd.DataFrame({"claimnb": [0], "rows": [1],
                         "share_of_valid_rows": [1.0]})
    with pytest.raises(ValueError, match="idioma"):
        plots.plot_claimnb(dist, "fr_fr")
