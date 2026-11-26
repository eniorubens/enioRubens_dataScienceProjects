"""Tests of the GPBoost challenger on synthetic data."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gpboost")
pytest.importorskip("optuna")

from retail_multilevel import gpb  # noqa: E402
from retail_multilevel.features import add_model_features, pair_means, upc_means  # noqa: E402


def _panel(n_weeks=60, n_stores=3, n_upcs=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_stores + 1):
        for u in range(100, 100 + n_upcs):
            for w in range(n_weeks):
                disc = -rng.uniform(0, 0.3) * (rng.random() < 0.3)
                base = 1.0 + 0.1 * u / 100
                rows.append({
                    "STORE_NUM": s, "UPC": u, "PAIR_ID": f"{s}:{u}", "WEEK": w, "WEEK_OF_YEAR": w % 52 + 1,
                    "LOG_BASE_PRICE": base, "LOG_PRICE": base + disc, "FEATURE": int(rng.random() < 0.1),
                    "DISPLAY": 0, "TPR_ONLY": int(disc < 0), "IN_MODEL": True,
                    "LOG_UNITS": 2 + 0.3 * s - 1.5 * disc + rng.normal(0, 0.2),
                })  # fmt: skip
    return pd.DataFrame(rows)


def test_pair_mean_mae_equal_weight_per_pair():
    df = pd.DataFrame({"PAIR_ID": ["a", "a", "a", "b"], "LOG_UNITS": [0.0, 0.0, 0.0, 0.0]})
    pred = np.array([1.0, 1.0, 1.0, 3.0])
    assert gpb.pair_mean_mae(df, pred) == pytest.approx(2.0)  # (1 + 3) / 2, not the row mean 1.5
    assert np.isnan(gpb.pair_mean_mae(df.iloc[:0], pred[:0]))


def test_folds_never_touch_final_window_and_are_disjoint():
    panel = _panel(n_weeks=120)
    for fold in gpb.FOLDS:
        fit, val = gpb.fold_windows(panel, fold)
        assert fit["WEEK"].max() < val["WEEK"].min()
        assert val["WEEK"].max() < 104
    with pytest.raises(ValueError):
        gpb.assert_no_final_window(panel)


def test_end_to_end_fit_predict_with_unseen_group():
    panel = _panel(n_weeks=60)
    fit_panel = panel[panel["WEEK"] < 50]
    fit = add_model_features(fit_panel, pair_means(fit_panel), upc_means(fit_panel))
    booster = gpb.fit_model(fit, {"learning_rate": 0.1, "num_leaves": 8, "min_data_in_leaf": 20}, 20)
    val = fit.tail(30).copy()
    pred = gpb.predict(booster, val)
    assert pred.shape == (30,) and np.isfinite(pred).all()
    assert gpb.pair_mean_mae(val, pred) < 1.0
    # unseen store/UPC/pair: random effects are zero, so only the tree part is predicted
    a, b = val.copy(), val.copy()
    a[["STORE_NUM", "UPC", "PAIR_ID"]] = [999, 998, "999:998"]
    b[["STORE_NUM", "UPC", "PAIR_ID"]] = [997, 996, "997:996"]
    assert np.allclose(gpb.predict(booster, a), gpb.predict(booster, b))
    assert not np.allclose(gpb.predict(booster, a), pred)
