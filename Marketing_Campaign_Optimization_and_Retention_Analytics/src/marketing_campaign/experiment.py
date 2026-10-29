"""A/B test of two proportions, sample-size design and power simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import (
    proportion_effectsize,
    proportions_ztest,
)

from marketing_campaign.config import (
    ALPHA,
    MC_N_SIMULATIONS,
    MINIMUM_LIFT_PP,
    POWER,
    RANDOM_STATE,
)

Z_95 = 1.96


def run_ab_test(df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Compare the conversion of every pair of variants with a z-test.

    Each pair is evaluated once, in the order given by comparing the
    variant names (``control`` before ``personalization``).

    Parameters
    ----------
    df : pd.DataFrame
        Canonical user-level base with ``variant`` and ``converted_flag``.
    alpha : float, default=0.05
        Significance level used for the ``Significance`` column.

    Returns
    -------
    pd.DataFrame
        One row per pair with the conversion rates, the z statistic and
        p-value, the absolute and relative difference of B versus A and
        its 95% interval in percentage points. Empty when there are fewer
        than two variants.
    """
    if "variant" not in df.columns or df["variant"].nunique() < 2:
        return pd.DataFrame()

    variants = df["variant"].unique()
    rows = []
    for variant_a in variants:
        for variant_b in variants:
            if variant_a >= variant_b:
                continue
            group_a = df[df["variant"] == variant_a]
            group_b = df[df["variant"] == variant_b]
            conv_a, count_a = group_a["converted_flag"].sum(), len(group_a)
            conv_b, count_b = group_b["converted_flag"].sum(), len(group_b)
            rate_a, rate_b = conv_a / count_a, conv_b / count_b

            z_stat, p_value = proportions_ztest(
                [int(conv_a), int(conv_b)], [int(count_a), int(count_b)]
            )
            diff = rate_b - rate_a
            uplift = (diff / rate_a) * 100 if rate_a > 0 else np.nan
            std_error = np.sqrt(
                rate_a * (1 - rate_a) / count_a
                + rate_b * (1 - rate_b) / count_b
            )
            rows.append(
                {
                    "Variant_A": variant_a,
                    "Variant_B": variant_b,
                    "Conv_A_%": round(rate_a * 100, 2),
                    "Conv_B_%": round(rate_b * 100, 2),
                    "Z_Stat": round(z_stat, 4),
                    "P_Value": round(p_value, 6),
                    "Significance": "YES" if p_value < alpha else "NO",
                    "Effect_%": round((rate_a - rate_b) * 100, 2),
                    "Diff_B_vs_A_pp": round(diff * 100, 2),
                    "Uplift_B_vs_A_%": round(uplift, 2),
                    "IC95_Diff_B_vs_A_pp": (
                        f"[{(diff - Z_95 * std_error) * 100:.2f}, "
                        f"{(diff + Z_95 * std_error) * 100:.2f}]"
                    ),
                }
            )
    return pd.DataFrame(rows)


def required_sample_size(
    baseline_rate: float,
    minimum_lift_pp: float = MINIMUM_LIFT_PP,
    alpha: float = ALPHA,
    power: float = POWER,
) -> int:
    """Return the users needed per group to detect a minimum lift.

    Parameters
    ----------
    baseline_rate : float
        Control conversion rate, as a fraction.
    minimum_lift_pp : float, default=2.0
        Smallest absolute lift worth detecting, in percentage points.
    alpha : float, default=0.05
        Type I error of the two-sided test.
    power : float, default=0.80
        Desired statistical power.

    Returns
    -------
    int
        Sample size per group, rounded up (Cohen's h for proportions).
    """
    treated_rate = min(baseline_rate + minimum_lift_pp / 100, 0.999)
    effect_size = proportion_effectsize(baseline_rate, treated_rate)
    size = zt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    return int(np.ceil(size))


def simulate_power(
    baseline_rate: float,
    sample_per_group: int,
    minimum_lift_pp: float = MINIMUM_LIFT_PP,
    n_simulations: int = MC_N_SIMULATIONS,
    alpha: float = ALPHA,
    seed: int = RANDOM_STATE,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate the power of the z-test by Monte Carlo simulation.

    Parameters
    ----------
    baseline_rate : float
        Control conversion rate, as a fraction.
    sample_per_group : int
        Users per group in each simulated experiment.
    minimum_lift_pp : float, default=2.0
        Absolute lift of the treatment, in percentage points.
    n_simulations : int, default=10000
        Number of simulated experiments.
    alpha : float, default=0.05
        Significance level of the test.
    seed : int, default=42
        Seed of the random generator; ignored when ``rng`` is given.
    rng : numpy.random.Generator, optional
        Generator to draw from. Passing the same one to
        :func:`simulate_power_curve` continues its random stream.

    Returns
    -------
    float
        Share of simulated experiments in which the null hypothesis of
        equal conversion was rejected.
    """
    treated_rate = min(baseline_rate + minimum_lift_pp / 100, 0.999)
    rng = rng if rng is not None else np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_simulations):
        control = rng.binomial(1, baseline_rate, sample_per_group)
        treated = rng.binomial(1, treated_rate, sample_per_group)
        if control.sum() == 0 and treated.sum() == 0:
            continue
        try:
            _, p_value = proportions_ztest(
                [control.sum(), treated.sum()],
                [len(control), len(treated)],
            )
        except Exception:
            continue
        if p_value < alpha:
            rejections += 1
    return rejections / n_simulations


def simulate_power_curve(
    baseline_rate: float,
    sample_sizes: list[int] | np.ndarray,
    minimum_lift_pp: float = MINIMUM_LIFT_PP,
    n_simulations: int = 1_000,
    alpha: float = ALPHA,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Estimate the power of the z-test for several sample sizes.

    Parameters
    ----------
    baseline_rate : float
        Control conversion rate, as a fraction.
    sample_sizes : list of int or numpy.ndarray
        Users per group to evaluate.
    minimum_lift_pp : float, default=2.0
        Absolute lift of the treatment, in percentage points.
    n_simulations : int, default=1000
        Simulated experiments per sample size.
    alpha : float, default=0.05
        Significance level of the test.
    rng : numpy.random.Generator, optional
        Generator to draw from. Defaults to one seeded with the project
        seed.

    Returns
    -------
    list of float
        The share of rejections of the null hypothesis for each size.
    """
    treated_rate = min(baseline_rate + minimum_lift_pp / 100, 0.999)
    rng = rng if rng is not None else np.random.default_rng(RANDOM_STATE)
    powers = []
    for size in sample_sizes:
        rejections = 0
        for _ in range(n_simulations):
            control = rng.binomial(1, baseline_rate, size)
            treated = rng.binomial(1, treated_rate, size)
            try:
                _, p_value = proportions_ztest(
                    [control.sum(), treated.sum()], [size, size]
                )
            except Exception:
                continue
            if p_value < alpha:
                rejections += 1
        powers.append(rejections / n_simulations)
    return powers


def estimate_runtime_days(df: pd.DataFrame, required_per_group: int) -> int:
    """Estimate the days needed to reach the sample of both groups.

    Parameters
    ----------
    df : pd.DataFrame
        User-level base with ``date_served`` and ``user_id``.
    required_per_group : int
        Users needed per group.

    Returns
    -------
    int
        Days at the recent daily pace (mean of the last 30 days with
        data), never less than 7 so that every weekday is covered.
    """
    daily_users = (
        df.groupby(df["date_served"].dt.date)["user_id"]
        .nunique()
        .tail(30)
        .mean()
    )
    days = np.ceil((required_per_group * 2) / max(daily_users, 1))
    return max(int(days), 7)
