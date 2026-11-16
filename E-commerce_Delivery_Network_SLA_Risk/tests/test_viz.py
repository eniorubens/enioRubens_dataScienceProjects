import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecom_delivery_sla_viz import (  # noqa: E402
    get_labels,
    plot_ablation,
    plot_expanding_folds,
    plot_gain_over_baseline,
    plot_label_availability,
    plot_metric_by_fold,
    plot_metric_intervals,
    plot_promise_regime,
    plot_psi,
)

LANGUAGES = ["pt", "en"]


def assert_valid(figure, language_titles=None):
    assert isinstance(figure, go.Figure)
    assert figure.to_json()
    return figure.layout.title.text


def unchanged(frame, build):
    before = frame.copy(deep=True)
    figure = build(frame)
    pd.testing.assert_frame_equal(frame, before)
    return figure


def titles_differ(build):
    pt, en = build("pt"), build("en")
    assert pt.layout.title.text != en.layout.title.text
    return pt, en


@pytest.fixture
def fold_summary():
    rows = []
    for index, month in enumerate(["2017-10", "2017-11", "2017-12"], start=1):
        start = pd.Timestamp(f"{month}-01")
        rows.append(
            {
                "fold": index,
                "train_start": pd.Timestamp("2017-01-02"),
                "train_end": start - pd.Timedelta(days=31),
                "validation_start": start,
                "validation_end": start + pd.Timedelta(days=27),
                "train_orders": 1000 * index,
                "validation_orders": 300 + index,
                "train_prevalence": 0.07 + index / 1000,
                "validation_prevalence": 0.09 - index / 1000,
            }
        )
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


@pytest.fixture
def psi_table():
    return pd.DataFrame(
        {
            "feature": ["purchase_month", "days_to_christmas", "freight_value", "promised_lead_days", "price", "zero"],
            "psi": [2.5, 1.2, 0.05, 0.3, 0.001, 0.0],
            "highlight": ["other", "other", "other", "promise", "other", "other"],
        }
    )


def test_expanding_folds(fold_summary):
    figure = unchanged(fold_summary, lambda f: plot_expanding_folds(f, language="pt"))
    assert_valid(figure)
    scatter = [t for t in figure.data if isinstance(t, go.Scatter)]
    bars = [t for t in figure.data if isinstance(t, go.Bar)]
    assert len(scatter) == 3 * len(fold_summary)
    assert len(bars) == 2
    assert all(t.orientation == "h" for t in bars)
    assert list(bars[0].y) == ["Fold 1", "Fold 2", "Fold 3"]
    assert figure.layout.yaxis.autorange == "reversed"
    assert sum(1 for t in scatter if t.showlegend) == 3
    pt, en = titles_differ(lambda lang: plot_expanding_folds(fold_summary, language=lang))
    assert pt.data[0].name != en.data[0].name


def test_psi_neutral_and_top_n(psi_table):
    neutral = ["purchase_month", "days_to_christmas"]
    figure = unchanged(psi_table, lambda f: plot_psi(f, neutral_features=neutral, language="en"))
    assert_valid(figure)
    bars = [t for t in figure.data if isinstance(t, go.Bar)]
    assert len(bars) == 2
    grey = [t for t in bars if t.marker.color == "#9CA3AF"][0]
    assert set(grey.y) == set(neutral)
    assert grey.name == "calendar (high PSI by construction)"
    assert figure.layout.xaxis.type == "log"
    assert len([t for t in figure.data if isinstance(t, go.Scatter)]) == 2
    assert all(v > 0 for t in bars for v in t.x)
    assert list(figure.layout.yaxis.categoryarray)[0] == "purchase_month"
    top = plot_psi(psi_table, top_n=3)
    assert sum(len(t.y) for t in top.data if isinstance(t, go.Bar)) == 3
    full = plot_psi(psi_table)
    assert sum(len(t.y) for t in full.data if isinstance(t, go.Bar)) == len(psi_table)
    pt = plot_psi(psi_table, neutral_features=neutral, language="pt")
    assert pt.layout.title.text != figure.layout.title.text
    assert [t.name for t in pt.data if isinstance(t, go.Bar)][-1] == "calendário (PSI alto por construção)"


def test_psi_highlight_column(psi_table):
    figure = plot_psi(psi_table, highlight_col="highlight", neutral_features=["purchase_month"], language="en")
    names = [t.name for t in figure.data if isinstance(t, go.Bar)]
    assert "promised lead time" in names
    assert plot_psi(psi_table.iloc[0:0]).to_json()


@pytest.fixture
def metrics():
    return pd.DataFrame(
        {
            "model": ["catboost", "baseline_a", "baseline_b", "gated"],
            "metric": ["average_precision"] * 3 + ["average_precision"],
            "estimate": [0.20, 0.15, 0.10, np.nan],
            "ci_low": [0.15, 0.10, 0.07, np.nan],
            "ci_high": [0.25, 0.20, 0.13, np.nan],
            "status": ["ok", "ok", "ok", "reprovada no gate"],
        }
    ).pipe(lambda df: pd.concat([df, df.assign(metric="roc_auc")], ignore_index=True))


@pytest.fixture
def paired():
    return pd.DataFrame(
        {
            "model": ["catboost", "catboost", "gated", "gated"],
            "reference": ["logistic", "promise", "logistic", "promise"],
            "metric": ["average_precision"] * 4,
            "difference": [0.05, -0.02, np.nan, np.nan],
            "ci_low": [0.01, -0.05, np.nan, np.nan],
            "ci_high": [0.09, 0.01, np.nan, np.nan],
        }
    )


def test_metric_intervals_single_panel(metrics):
    figure = unchanged(metrics, lambda f: plot_metric_intervals(f))
    assert_valid(figure)
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [0.20, 0.15, 0.10]
    assert list(figure.layout.yaxis.ticktext) == ["catboost", "baseline_a", "baseline_b"]


def test_metric_intervals_paired_skips_nan(metrics, paired):
    figure = plot_metric_intervals(metrics, paired=paired, language="en")
    before = paired.copy(deep=True)
    plot_metric_intervals(metrics, paired=paired)
    pd.testing.assert_frame_equal(paired, before)
    assert len(figure.data) == 3  # estimates + two references
    assert [t.name for t in figure.data[1:]] == ["Reference: logistic", "Reference: promise"]
    assert all(len(t.x) == 1 for t in figure.data[1:])
    assert figure.data[1].marker.symbol != figure.data[2].marker.symbol
    assert any(shape.x0 == 0 and shape.x1 == 0 for shape in figure.layout.shapes)
    pt = plot_metric_intervals(metrics, paired=paired, language="pt")
    assert pt.layout.title.text != figure.layout.title.text


def test_metric_intervals_other_metric(metrics):
    figure = plot_metric_intervals(metrics, metric="roc_auc", language="en")
    assert len(figure.data[0].x) == 3
    assert "ROC AUC" in figure.layout.xaxis.title.text


@pytest.fixture
def gate_folds():
    rows = []
    for fold in (3, 1, 2):
        for variant, offset in (("F1_config", 0.0), ("C", 0.01), ("E", -0.01)):
            rows.append({"fold": fold, "variant": variant, "average_precision": 0.1 + fold / 100 + offset})
    return pd.DataFrame(rows)


def test_metric_by_fold(gate_folds):
    figure = unchanged(
        gate_folds,
        lambda f: plot_metric_by_fold(f, "fold", "variant", "average_precision", reference="F1_config", language="en"),
    )
    assert_valid(figure)
    assert len(figure.data) == 3
    reference = figure.data[-1]
    assert reference.line.dash == "dash"
    assert "reference" in reference.name
    assert list(reference.x) == [1, 2, 3]
    assert all(t.line.dash == "solid" for t in figure.data[:-1])
    titles_differ(lambda lang: plot_metric_by_fold(gate_folds, "fold", "variant", "average_precision", language=lang))
    assert len(plot_metric_by_fold(gate_folds, "fold", "variant", "average_precision").data) == 3
    with pytest.raises(KeyError):
        plot_metric_by_fold(gate_folds, "fold", "variant", "average_precision", reference="missing")


def test_promise_regime():
    table = pd.DataFrame(
        {
            "purchase_month": ["2018-05", "2018-06", "2018-07"],
            "orders": [10, 11, 12],
            "promise_median": [24.0, 23.0, 19.0],
            "auc_negative_promise": [0.5, 0.6, np.nan],
        }
    )
    figure = unchanged(table, plot_promise_regime)
    assert_valid(figure)
    kinds = [type(t).__name__ for t in figure.data]
    assert kinds == ["Bar", "Scatter", "Scatter"]
    assert np.isnan(figure.data[1].y[2])
    assert list(figure.data[2].y) == [0.5, 0.5]
    assert figure.data[1].yaxis == "y2"
    titles_differ(lambda lang: plot_promise_regime(table, language=lang))


def test_label_availability():
    table = pd.DataFrame(
        {
            "purchase_month": ["2018-06", "2018-07"],
            "orders": [100, 200],
            "known_labels": [99, 135],
            "unknown_labels": [1, 65],
            "prevalence_known": [0.0084, 0.0031],
            "prevalence_unknown": [np.nan, 0.0979],
        }
    )
    figure = unchanged(table, plot_label_availability)
    assert_valid(figure)
    assert len(figure.data) == 2 and figure.layout.barmode == "stack"
    assert list(figure.data[0].y) == [99, 135]
    assert figure.data[1].text[0] == ""
    assert "9.8%" in figure.data[1].text[1]
    titles_differ(lambda lang: plot_label_availability(table, language=lang))


@pytest.fixture
def tournament():
    return pd.DataFrame(
        {
            "model": ["catboost", "xgboost", "baseline_promise", "baseline_prev"],
            "average_precision_mean": [0.22, 0.20, 0.19, 0.07],
            "average_precision_std": [0.02, 0.03, 0.02, 0.01],
            "is_baseline": [False, False, True, True],
            "average_precision_gain_vs_baseline": [0.139, 0.05, np.nan, np.nan],
            "average_precision_folds_won": [8, 5, np.nan, np.nan],
            "average_precision_best_baseline": ["baseline_promise", "baseline_promise", "", ""],
        }
    )


def test_gain_over_baseline(tournament):
    figure = unchanged(tournament, lambda f: plot_gain_over_baseline(f, threshold=0.05, language="en"))
    assert_valid(figure)
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == ["catboost", "xgboost"]
    assert figure.data[0].y[0] == pytest.approx(13.9)
    assert figure.data[0].error_y.array is None
    assert list(figure.data[0].text) == ["8", "5"]
    assert any(abs(shape.y0 - 5.0) < 1e-9 for shape in figure.layout.shapes)
    titles_differ(lambda lang: plot_gain_over_baseline(tournament, language=lang))


def test_gain_over_baseline_regression_metric():
    summary = pd.DataFrame({"model": ["a", "b"], "mae_gain_vs_baseline": [0.04, np.nan]})
    figure = plot_gain_over_baseline(summary, metric="mae", threshold=0.03)
    assert list(figure.data[0].x) == ["a"]


def test_metric_by_fold_models_subset_and_lower_is_better(gate_folds):
    figure = unchanged(
        gate_folds,
        lambda f: plot_metric_by_fold(
            f, "fold", "variant", "average_precision", reference="F1_config", models=["C"], lower_is_better=True,
            language="en",
        ),
    )
    assert [t.name for t in figure.data] == ["C", "F1_config (reference)"]
    assert "lower is better" in figure.layout.yaxis.title.text
    pt = plot_metric_by_fold(gate_folds, "fold", "variant", "average_precision", models=["E"], lower_is_better=True)
    assert [t.name for t in pt.data] == ["E"]
    assert "menor é melhor" in pt.layout.yaxis.title.text
    plain = plot_metric_by_fold(gate_folds, "fold", "variant", "average_precision")
    assert "lower" not in plain.layout.yaxis.title.text


def _fold_frame(metric, base, a, b):
    rows = []
    for fold, (vb, va, vc) in enumerate(zip(base, a, b, strict=True), start=1):
        rows += [
            {"fold": fold, "model": "baseline_x", metric: vb},
            {"fold": fold, "model": "a", metric: va},
            {"fold": fold, "model": "b", metric: vc},
        ]
    return pd.DataFrame(rows)


def test_gain_per_fold_error_and_folds_won_higher_is_better():
    results = _fold_frame("average_precision", [0.10, 0.20, 0.10], [0.11, 0.18, 0.12], [0.10, 0.22, 0.09])
    summary = pd.DataFrame(
        {
            "model": ["a", "b", "baseline_x"],
            "n_folds": [3, 3, 3],
            "average_precision_gain_vs_baseline": [0.03, 0.01, np.nan],
            "average_precision_best_baseline": ["baseline_x", "baseline_x", ""],
        }
    )
    figure = unchanged(results, lambda r: plot_gain_over_baseline(summary, fold_results=r, language="en"))
    bar = figure.data[0]
    assert list(bar.x) == ["a", "b"]
    assert bar.y[0] == pytest.approx(3.0)
    assert bar.error_y.array[0] == pytest.approx(np.std([0.1, -0.1, 0.2], ddof=1) * 100)
    assert list(bar.text) == ["2/3", "1/3"]


def test_gain_per_fold_lower_is_better_and_summary_won_column():
    results = _fold_frame("mae", [4.0, 4.0, 5.0], [3.0, 4.0, 4.0], [5.0, 4.0, 5.0])
    summary = pd.DataFrame(
        {
            "model": ["a", "b", "baseline_x"],
            "mae_gain_vs_baseline": [0.15, -0.1, np.nan],
            "mae_best_baseline": ["baseline_x", "baseline_x", ""],
            "mae_folds_won": [2, 0, np.nan],
        }
    )
    figure = plot_gain_over_baseline(summary, metric="mae", threshold=0.03, fold_results=results)
    bar = figure.data[0]
    assert list(bar.x) == ["a", "b"]
    assert bar.y[0] == pytest.approx(15.0)
    assert bar.error_y.array[0] == pytest.approx(np.std([0.25, 0.0, 0.2], ddof=1) * 100)
    assert bar.error_y.array[1] == pytest.approx(np.std([-0.25, 0.0, 0.0], ddof=1) * 100)
    assert list(bar.text) == ["2/3", "0/3"]
    computed = plot_gain_over_baseline(summary.drop(columns=["mae_folds_won"]), metric="mae", fold_results=results)
    assert list(computed.data[0].text) == ["2/3", "0/3"]


@pytest.fixture
def ablation():
    return pd.DataFrame(
        {
            "variant": ["mean", "mean", "min_max", "min_max", "none", "none"],
            "model": ["catboost", "xgboost"] * 3,
            "average_precision_mean": [0.20, 0.19, 0.21, 0.18, 0.19, np.nan],
            "average_precision_std": [0.02, 0.02, 0.03, 0.02, 0.02, np.nan],
            "average_precision_delta_vs_mean": [np.nan, np.nan, 0.05, -0.05, -0.05, np.nan],
            "multi_seller_average_precision_mean": [0.1, 0.1, 0.12, 0.09, 0.1, 0.1],
            "multi_seller_average_precision_std": [0.01] * 6,
        }
    )


def test_ablation(ablation):
    figure = unchanged(ablation, lambda f: plot_ablation(f, language="en"))
    assert_valid(figure)
    assert len(figure.data) == 2
    assert list(figure.data[1].x) == ["mean", "min_max"]  # NaN row skipped
    assert figure.data[0].error_y.array is not None
    sub = plot_ablation(ablation, subgroup=True)
    assert len(sub.data[0].x) == 3
    titles_differ(lambda lang: plot_ablation(ablation, language=lang))
    with pytest.raises(KeyError):
        plot_ablation(ablation.drop(columns=["average_precision_mean"]))


def test_labels_reject_unknown_language():
    assert set(get_labels("pt")) == set(get_labels("en"))
    for section in get_labels("pt"):
        assert set(get_labels("pt")[section]) == set(get_labels("en")[section])
    with pytest.raises(ValueError):
        get_labels("es")


def test_viz_package_does_not_import_training_package():
    code = (
        "import sys; sys.path.insert(0, 'src'); import ecom_delivery_sla_viz; "
        "print('ecom_delivery_sla' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_viz_package_source_has_no_training_package_imports():
    for path in (SRC / "ecom_delivery_sla_viz").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import ecom_delivery_sla\n" not in source + "\n"
        assert "from ecom_delivery_sla " not in source
        assert "from ecom_delivery_sla." not in source
        assert "import ecom_delivery_sla." not in source
