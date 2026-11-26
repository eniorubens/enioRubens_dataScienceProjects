"""Synthetic tests of the per-pair OLS discount slopes and of their plot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from retail_multilevel import pair_slopes, viz  # noqa: E402
from retail_multilevel.i18n import LANGS, t  # noqa: E402


def _pair_rows(pair, upc, n_weeks, slope, rng, discount=True, constant_units=False):
    rows = []
    for w in range(n_weeks):
        disc = -float(rng.choice([0.0, 0.1, 0.2], p=[0.5, 0.25, 0.25])) if discount else 0.0
        feat, disp = int(rng.random() < 0.15), int(rng.random() < 0.15)
        tpr = int(disc < 0 and not feat and not disp)
        base = 1.5
        y = 0.0 if constant_units else 2 + slope * disc + 0.4 * feat + 0.3 * disp + rng.normal(0, 0.05)
        rows.append((pair, upc, w, w % 52 + 1, y, base, base + disc, feat, disp, tpr))
    return rows


def _panel():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        rows += _pair_rows(f"s{i}_a", "a", 40, -2.0, rng)  # long history
    rows += _pair_rows("s0_b", "b", 20, -2.0, rng)  # short history
    rows += _pair_rows("s0_c", "c", 5, -2.0, rng)  # fewer than 8 weeks: not fitted
    rows += _pair_rows("s0_d", "d", 30, 0.0, rng, discount=False)  # no discount variation: no slope
    rows += _pair_rows("s0_e", "e", 30, 0.0, rng, constant_units=True)  # constant units: slope exactly zero
    cols = ["PAIR_ID", "UPC", "WEEK", "WEEK_OF_YEAR", "LOG_UNITS", "LOG_BASE_PRICE", "LOG_PRICE",
            "FEATURE", "DISPLAY", "TPR_ONLY"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def test_pair_slopes_recover_slope_and_coverage():
    out = pair_slopes.pair_slopes(_panel()).set_index("PAIR_ID")
    assert len(out) == 10
    assert not out.loc["s0_c", "has_ols"] and np.isnan(out.loc["s0_c", "slope"])
    assert out.loc["s0_d", "has_ols"] and np.isnan(out.loc["s0_d", "slope"])
    assert out.loc["s0_e", "slope"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc["s0_a", "stratum"] == "26-77" and out.loc["s0_b", "stratum"] == "<26"
    assert out.loc["s0_c", "stratum"] == "<26"
    fitted = out.loc[[f"s{i}_a" for i in range(6)], "slope"]
    assert (fitted - (-2.0)).abs().max() < 0.5


def test_summarize_slopes_counts_and_zero_is_not_positive():
    slopes = pd.DataFrame(
        {
            "slope": [-2.0, 1.0, np.nan, 1e-15, 3.0, -0.5],
            "stratum": ["<26", "<26", "<26", "26-77", ">=78", ">=78"],
        }
    )
    s = pair_slopes.summarize_slopes(slopes).set_index("stratum")
    assert list(s.index) == ["all", "<26", "26-77", ">=78"]
    assert s.loc["all", "n_pairs"] == 5 and s.loc["all", "n_positive"] == 2
    assert s.loc["all", "share_positive_pct"] == pytest.approx(40.0)
    assert s.loc["<26", "n_pairs"] == 2 and s.loc["<26", "share_positive_pct"] == pytest.approx(50.0)
    assert s.loc["26-77", "n_positive"] == 0  # rounding-level slope is not positive
    assert s.loc["all", "median"] == pytest.approx(1e-15)
    empty = pair_slopes.summarize_slopes(slopes.iloc[:0]).set_index("stratum")
    assert (empty["n_pairs"] == 0).all() and empty["median"].isna().all()


@pytest.mark.parametrize("lang", LANGS)
def test_plot_pair_slopes_labels_zero_line_and_outliers(lang):
    rng = np.random.default_rng(1)
    slopes = pd.DataFrame({"slope": np.r_[rng.normal(-1.5, 1.0, 300), [80.0]], "stratum": ["26-77"] * 301})
    slopes.loc[:10, "stratum"] = "<26"
    fig = viz.plot_pair_slopes(slopes, lang)
    ax = fig.axes[0]
    texts = [x.get_text() for x in fig.findobj(Text) if x.get_text().strip()]
    assert t("pair_slopes_title", lang) in texts
    assert t("pair_slopes_xlabel", lang) in texts
    assert any(line.get_xdata()[0] == 0 for line in ax.lines)
    assert any("n = 11" in x for x in texts)
    assert any(t("pair_slopes_outside", lang).format(n=int(x.split()[0])) == x for x in texts if x[0].isdigit())
    plt.close(fig)
