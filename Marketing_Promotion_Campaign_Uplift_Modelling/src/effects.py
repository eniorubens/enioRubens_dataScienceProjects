"""Estimação de ATE (naive, sob randomização) com intervalos de confiança."""
import numpy as np
import pandas as pd
from scipy import stats


def ate_binary(df, treatment_col, outcome, treated_arm, control_arm, alpha=0.05):
    """ATE para outcome binário, com IC via aproximação normal."""
    t = df[df[treatment_col] == treated_arm][outcome]
    c = df[df[treatment_col] == control_arm][outcome]
    p_t, p_c = t.mean(), c.mean()
    n_t, n_c = len(t), len(c)
    ate = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "ate": ate, "se": se,
        "ci_low": ate - z * se, "ci_high": ate + z * se,
        "p_value": 2 * (1 - stats.norm.cdf(abs(ate / se))),
    }


def ate_continuous(df, treatment_col, outcome, treated_arm, control_arm, alpha=0.05):
    """ATE para outcome contínuo, com IC via Welch t-test."""
    t = df[df[treatment_col] == treated_arm][outcome]
    c = df[df[treatment_col] == control_arm][outcome]
    ate = t.mean() - c.mean()
    se = np.sqrt(t.var() / len(t) + c.var() / len(c))
    _, p = stats.ttest_ind(t, c, equal_var=False)
    df_welch = (t.var() / len(t) + c.var() / len(c)) ** 2 / (
        (t.var() / len(t)) ** 2 / (len(t) - 1) + (c.var() / len(c)) ** 2 / (len(c) - 1)
    )
    crit = stats.t.ppf(1 - alpha / 2, df_welch)
    return {
        "ate": ate, "se": se,
        "ci_low": ate - crit * se, "ci_high": ate + crit * se,
        "p_value": p,
    }


def ate_table(df, treatment_col, treated_arms, control_arm, binary_outcomes, continuous_outcome, alpha=0.05):
    """Tabela consolidada de ATE (binários + contínuo) para cada braço tratado vs. controle."""
    results = []
    for arm in treated_arms:
        for outcome in binary_outcomes:
            r = ate_binary(df, treatment_col, outcome, arm, control_arm, alpha)
            r.update({"treatment": arm, "outcome": outcome})
            results.append(r)
        r = ate_continuous(df, treatment_col, continuous_outcome, arm, control_arm, alpha)
        r.update({"treatment": arm, "outcome": continuous_outcome})
        results.append(r)

    return pd.DataFrame(results)[
        ["treatment", "outcome", "ate", "se", "ci_low", "ci_high", "p_value"]
    ].round(4)
