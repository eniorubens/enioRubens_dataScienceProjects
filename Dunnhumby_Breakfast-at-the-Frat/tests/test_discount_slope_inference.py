import numpy as np
import pandas as pd
import pytest

from retail_multilevel.discount_slope_bootstrap import bootstrap, bootstrap_summary
from retail_multilevel.discount_slope_lr import fit_observed, likelihood_ratio, prepare_training


@pytest.fixture
def training_rows():
    return pd.DataFrame(
        {
            "PAIR_ID": ["1_a", "1_a", "2_a", "2_a"],
            "STORE_NUM": [1, 1, 2, 2],
            "UPC": ["a", "a", "a", "a"],
            "WEEK": [0, 1, 0, 1],
            "WEEK_OF_YEAR": [2, 3, 2, 3],
            "LOG_UNITS": np.log([7, 9, 11, 13]),
            "LOG_PRICE": np.log([2, 2, 7, 7]),
            "LOG_BASE_PRICE": np.log([2, 4, 7, 7]),
            "FEATURE": [0, 1, 0, 0],
            "DISPLAY": [0, 0, 0, 0],
            "TPR_ONLY": [0, 0, 0, 0],
            "IN_MODEL": [True, True, True, True],
        }
    )


def test_discount_slope_preprocessing_computes_training_pair_means(training_rows):
    result = prepare_training(training_rows)
    assert result.loc[0, "LOG_BASE_MEAN"] == pytest.approx(np.log(2 * 4) / 2)
    assert result.loc[1, "LOG_DISC"] == pytest.approx(np.log(0.5))
    assert result.loc[2, "LOG_BASE_DEV"] == pytest.approx(0)


def test_discount_slope_preprocessing_rejects_a_held_out_week(training_rows):
    training_rows.loc[1, "WEEK"] = 104
    with pytest.raises((ValueError, RuntimeError)):
        prepare_training(training_rows)


def _draws(observed):
    return pd.DataFrame(
        {
            "replicate": [1, 2, 3],
            "seed": [21, 22, 23],
            "lr": [0.0, observed, 0.1],
            "valid": [True, True, True],
            "singular_c0": [False, False, False],
            "singular_c1": [True, False, True],
            "seconds": [1.0, 1.0, 1.0],
            "message": ["", "", ""],
        }
    )


def test_bootstrap_counts_ties_and_valid_singular_fits():
    draws = _draws(2.0)
    result = bootstrap_summary(2.0, draws, planned=3)
    assert result["complete"]
    assert result["valid"] == 3
    assert result["exceedances"] == 1
    assert result["p_value"] == pytest.approx(0.5)
    assert result["null_tail_ci_low"] < 1 / 3 < result["null_tail_ci_high"]


def test_bootstrap_withholds_p_value_for_partial_or_failed_draws():
    draws = _draws(2.0)
    partial = bootstrap_summary(2.0, draws, planned=39)
    assert not partial["complete"]
    assert partial["p_value"] is None

    draws.loc[0, ["lr", "valid", "message"]] = [np.nan, False, "convergence failure"]
    failed = bootstrap_summary(2.0, draws, planned=3)
    assert failed["attempted"] == 3 and failed["valid"] == 2
    assert not failed["complete"]
    assert failed["p_value"] is None


@pytest.mark.r
def test_crossed_bootstrap_refits_reproduce_a_draw_on_the_same_design():
    pytest.importorskip("rpy2")
    rng = np.random.default_rng(82)
    store, upc, week = np.indices((8, 9, 20)).reshape(3, -1)
    pair = store * 9 + upc
    n = len(pair)
    base = rng.normal(1, 0.4, 72)[pair] + rng.normal(0, 0.08, n)
    log_discount = -rng.choice([0.0, 0.1, 0.2, 0.4], n)
    promotion = rng.integers(0, 4, n)
    train = pd.DataFrame(
        {
            "STORE_NUM": store,
            "UPC": upc,
            "PAIR_ID": pair,
            "WEEK": week,
            "WEEK_OF_YEAR": week + 2,
            "LOG_PRICE": base + log_discount,
            "LOG_BASE_PRICE": base,
            "FEATURE": (promotion == 1).astype(int),
            "DISPLAY": (promotion == 2).astype(int),
            "TPR_ONLY": (promotion == 3).astype(int),
            "LOG_UNITS": (
                4
                + rng.normal(0, 0.6, 8)[store]
                + rng.normal(0, 0.6, 9)[upc]
                + rng.normal(0, 0.5, 72)[pair]
                + (-1.4 + rng.normal(0, 0.7, 8)[store] + rng.normal(0, 0.7, 9)[upc]) * log_discount
                + rng.normal(0, 0.25, n)
            ),
        }
    )
    c0, c1 = fit_observed(prepare_training(train))
    assert c0.nobs == c1.nobs == n and likelihood_ratio(c0, c1) > 0
    saved = []
    first = bootstrap(c0, c1, [(1, 9407)], on_result=saved.append).iloc[0]
    second = bootstrap(c0, c1, [(1, 9407)]).iloc[0]
    assert first["valid"] and second["valid"]
    assert first["lr"] == pytest.approx(second["lr"], abs=1e-8)
    assert saved == [first.to_dict()]
