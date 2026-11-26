import numpy as np
import pandas as pd
import pytest

from retail_multilevel.sensitivity import demean_two_way, two_way_fe_ols


def _tiny(seed=3, n_pairs=6, n_weeks=5, drop=4):
    rng = np.random.default_rng(seed)
    grid = pd.MultiIndex.from_product([range(n_pairs), range(n_weeks)], names=["PAIR_ID", "WEEK"]).to_frame(index=False)
    grid = grid.drop(index=rng.choice(len(grid), drop, replace=False)).reset_index(drop=True)  # unbalanced
    grid["x1"] = rng.normal(size=len(grid)) + grid["PAIR_ID"] * 0.3
    grid["x2"] = rng.normal(size=len(grid)) + grid["WEEK"] * 0.2
    return grid, rng


def test_alternating_projection_matches_dummy_ols():
    df, rng = _tiny()
    df["y"] = 1.0 + 0.7 * df["x1"] - 0.4 * df["x2"] + rng.normal(size=len(df))
    table, _ = two_way_fe_ols(df, "y", ["x1", "x2"])
    dummies = pd.concat(
        [
            pd.get_dummies(df["PAIR_ID"], prefix="p", dtype=float),
            pd.get_dummies(df["WEEK"], prefix="w", dtype=float).iloc[:, 1:],
        ],
        axis=1,
    )
    x = np.column_stack([df[["x1", "x2"]].to_numpy(), dummies.to_numpy()])
    beta = np.linalg.lstsq(x, df["y"].to_numpy(), rcond=None)[0]
    assert np.allclose(table["estimate"].to_numpy(), beta[:2], atol=1e-8)
    dm, _ = demean_two_way(df[["x1", "x2"]].to_numpy(), df["PAIR_ID"].to_numpy(), df["WEEK"].to_numpy())
    assert np.allclose(pd.DataFrame(dm).groupby(df["PAIR_ID"].to_numpy()).mean().to_numpy(), 0, atol=1e-8)
    assert np.allclose(pd.DataFrame(dm).groupby(df["WEEK"].to_numpy()).mean().to_numpy(), 0, atol=1e-8)


def test_recovers_known_betas_with_unobserved_effects():
    rng = np.random.default_rng(11)
    n_pairs, n_weeks = 300, 40
    pair = np.repeat(np.arange(n_pairs), n_weeks)
    week = np.tile(np.arange(n_weeks), n_pairs)
    keep = rng.random(len(pair)) > 0.1
    pair, week = pair[keep], week[keep]
    a, g = rng.normal(0, 2, n_pairs), rng.normal(0, 1, n_weeks)
    x1 = rng.normal(size=len(pair)) + a[pair]  # correlated with the pair effect (what FE removes)
    x2 = rng.normal(size=len(pair)) + g[week]
    y = 2.0 * x1 - 1.5 * x2 + a[pair] * 3 + g[week] + rng.normal(0, 0.1, len(pair))
    df = pd.DataFrame({"PAIR_ID": pair, "WEEK": week, "x1": x1, "x2": x2, "y": y})
    table, info = two_way_fe_ols(df, "y", ["x1", "x2"])
    est = table.set_index("term")
    assert abs(est.loc["x1", "estimate"] - 2.0) < 4 * est.loc["x1", "se_cluster"] + 1e-3
    assert abs(est.loc["x2", "estimate"] + 1.5) < 4 * est.loc["x2", "se_cluster"] + 1e-3
    assert est.loc["x1", "estimate"] == pytest.approx(2.0, abs=0.01)
    assert est.loc["x2", "estimate"] == pytest.approx(-1.5, abs=0.01)
    assert (table["se_cluster"] > 0).all() and info["n_sweeps"] >= 1
    assert table.loc[0, ["nobs", "n_pairs", "n_weeks"]].tolist() == [len(df), n_pairs, n_weeks]
    assert (table["ci_low"] < table["estimate"]).all() and (table["estimate"] < table["ci_high"]).all()


def test_noiseless_recovery_is_exact():
    df, rng = _tiny(seed=5, n_pairs=8, n_weeks=6, drop=6)
    df["y"] = 3 * df["x1"] + 0.5 * df["x2"] + rng.normal(size=8)[df["PAIR_ID"]] + rng.normal(size=6)[df["WEEK"]]
    table, _ = two_way_fe_ols(df, "y", ["x1", "x2"])
    assert np.allclose(table["estimate"], [3.0, 0.5], atol=1e-8)


def test_twoway_se_equals_manual_sandwich_on_tiny_example():
    df, rng = _tiny(seed=9, n_pairs=5, n_weeks=4, drop=3)
    df["y"] = 0.8 * df["x1"] - 0.3 * df["x2"] + rng.normal(size=len(df))
    table, info = two_way_fe_ols(df, "y", ["x1", "x2"])
    # manual: dummy-variable OLS, then the Cameron-Gelbach-Miller sandwich written out explicitly
    dummies = pd.concat(
        [
            pd.get_dummies(df["PAIR_ID"], prefix="p", dtype=float),
            pd.get_dummies(df["WEEK"], prefix="w", dtype=float).iloc[:, 1:],
        ],
        axis=1,
    )
    xfull = np.column_stack([df[["x1", "x2"]].to_numpy(), dummies.to_numpy()])
    yv = df["y"].to_numpy()
    beta = np.linalg.lstsq(xfull, yv, rcond=None)[0]
    u = yv - xfull @ beta
    dm, _ = demean_two_way(df[["x1", "x2"]].to_numpy(), df["PAIR_ID"].to_numpy(), df["WEEK"].to_numpy())
    bread = np.linalg.inv(dm.T @ dm)
    n, k = dm.shape
    g_p, g_w = df["PAIR_ID"].nunique(), df["WEEK"].nunique()
    base = (n - 1) / (n - k - (g_w - 1))

    def meat(groups):
        total = np.zeros((k, k))
        for _, idx in df.groupby(groups).indices.items():
            sg = (dm[idx] * u[idx, None]).sum(axis=0)
            total += np.outer(sg, sg)
        return total

    v_pair = bread @ (g_p / (g_p - 1) * base * meat("PAIR_ID")) @ bread
    v_week = bread @ (g_w / (g_w - 1) * base * meat("WEEK")) @ bread
    v_row = bread @ (n / (n - 1) * base * meat(["PAIR_ID", "WEEK"])) @ bread
    v_two = v_pair + v_week - v_row
    if info["twoway_psd_fixed"]:
        val, vec = np.linalg.eigh(v_two)
        v_two = (vec * np.clip(val, 0, None)) @ vec.T
    assert np.allclose(table["estimate"], beta[:2], atol=1e-8)
    assert np.allclose(table["se_cluster"], np.sqrt(np.diag(v_pair)), rtol=1e-8)
    assert np.allclose(table["se_week"], np.sqrt(np.diag(v_week)), rtol=1e-8)
    assert np.allclose(table["se_twoway"], np.sqrt(np.diag(v_two)), rtol=1e-8)
    assert (table["ci_low_twoway"] < table["estimate"]).all() and (table["estimate"] < table["ci_high_twoway"]).all()
    assert table.loc[0, "n_pairs"] == g_p and info["n_clusters_week"] == g_w


def test_twoway_se_is_not_smaller_than_one_way_under_week_shocks_and_serial_correlation():
    n_pairs, n_weeks, reps = 60, 30, 30
    pair = np.repeat(np.arange(n_pairs), n_weeks)
    week = np.tile(np.arange(n_weeks), n_pairs)
    ratios_p, ratios_w = [], []
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        x = np.zeros((n_pairs, n_weeks))
        x[:, 0] = rng.normal(size=n_pairs)
        for t in range(1, n_weeks):  # persistent regressor, so serial error correlation reaches the scores
            x[:, t] = 0.7 * x[:, t - 1] + rng.normal(size=n_pairs)
        slope_shock = rng.normal(0, 0.4, n_weeks)  # common week shock to the slope: survives the week effects
        e = np.zeros((n_pairs, n_weeks))
        e[:, 0] = rng.normal(size=n_pairs)
        for t in range(1, n_weeks):  # AR(1) serial correlation within the pair
            e[:, t] = 0.8 * e[:, t - 1] + rng.normal(size=n_pairs)
        y = (1.0 + slope_shock[None, :]) * x + e
        df = pd.DataFrame({"PAIR_ID": pair, "WEEK": week, "x": x.ravel(), "y": y.ravel()})
        table, _ = two_way_fe_ols(df, "y", ["x"])
        ratios_p.append(table.loc[0, "se_twoway"] / table.loc[0, "se_cluster"])
        ratios_w.append(table.loc[0, "se_twoway"] / table.loc[0, "se_week"])
    assert np.mean(ratios_p) >= 1.0
    assert np.mean(ratios_w) >= 1.0


def test_fe_fit_arrays_matches_two_way_fe_ols_and_exposes_full_covariances():
    from retail_multilevel.sensitivity import fe_fit_arrays

    df, rng = _tiny(seed=13, n_pairs=7, n_weeks=5, drop=3)
    df["y"] = 0.5 * df["x1"] - 0.2 * df["x2"] + rng.normal(size=len(df))
    table, info = two_way_fe_ols(df, "y", ["x1", "x2"])
    fit = fe_fit_arrays(
        df["y"].to_numpy(), df[["x1", "x2"]].to_numpy(), df["PAIR_ID"].to_numpy(), df["WEEK"].to_numpy()
    )
    assert np.allclose(fit["beta"], table["estimate"])
    assert np.allclose(np.sqrt(np.diag(fit["cov_pair"])), table["se_cluster"])
    assert np.allclose(np.sqrt(np.diag(fit["cov_week"])), table["se_week"])
    assert np.allclose(np.sqrt(np.diag(fit["cov_twoway"])), table["se_twoway"])
    assert fit["psd_fixed"] == info["twoway_psd_fixed"] and fit["cov_twoway"].shape == (2, 2)
    checked = fe_fit_arrays(
        df["y"].to_numpy(),
        df[["x1", "x2"]].to_numpy(),
        df["PAIR_ID"].to_numpy(),
        df["WEEK"].to_numpy(),
        rank_check=True,
    )
    assert checked["design_rank"] == 2 and not checked["rank_deficient"]
    assert np.allclose(checked["cov_twoway"], fit["cov_twoway"])
