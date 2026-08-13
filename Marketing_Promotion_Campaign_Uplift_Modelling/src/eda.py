"""EDA e randomization audit: distribuição do tratamento, balance (SMD) e testes formais."""
import numpy as np
import pandas as pd
from scipy import stats


def treatment_distribution(df, treatment_col, arms):
    """Distribuição de contagem/proporção do tratamento e teste chi-quadrado vs. uniforme."""
    counts = df[treatment_col].value_counts().reindex(arms)
    pct = counts / counts.sum()
    table = pd.DataFrame({"n": counts, "pct": pct.round(4)})

    expected = np.full(len(arms), len(df) / len(arms))
    chi2, p = stats.chisquare(counts.values, expected)
    return table, chi2, p


def smd_continuous(treated, control):
    """SMD para variáveis contínuas (Cohen's d-style)."""
    pooled_std = np.sqrt((treated.std() ** 2 + control.std() ** 2) / 2)
    return (treated.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0.0


def smd_binary(treated, control):
    """SMD para variáveis binárias (proporções)."""
    p_t, p_c = treated.mean(), control.mean()
    pooled = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
    return (p_t - p_c) / pooled if pooled > 0 else 0.0


def build_smd_table(df, treatment_col, control_arm, treated_arm, cont_vars, bin_vars, cat_vars):
    """Constrói tabela de SMDs para uma comparação par-a-par entre braços.

    Níveis de `cat_vars` são enumerados a partir do `df` completo (não dos
    subconjuntos tratado/controle), garantindo a mesma ordem de variável em
    qualquer comparação par-a-par sobre o mesmo `df`.
    """
    rows = []
    ctrl = df[df[treatment_col] == control_arm]
    trt = df[df[treatment_col] == treated_arm]

    for v in cont_vars:
        rows.append({
            "variable": v, "type": "continuous",
            "mean_control": ctrl[v].mean(), "mean_treated": trt[v].mean(),
            "smd": smd_continuous(trt[v], ctrl[v]),
        })
    for v in bin_vars:
        rows.append({
            "variable": v, "type": "binary",
            "mean_control": ctrl[v].mean(), "mean_treated": trt[v].mean(),
            "smd": smd_binary(trt[v], ctrl[v]),
        })
    for v in cat_vars:
        for level in df[v].unique():
            ctrl_ind = (ctrl[v] == level).astype(int)
            trt_ind = (trt[v] == level).astype(int)
            rows.append({
                "variable": f"{v} = {level}", "type": "binary",
                "mean_control": ctrl_ind.mean(), "mean_treated": trt_ind.mean(),
                "smd": smd_binary(trt_ind, ctrl_ind),
            })
    return pd.DataFrame(rows)


def formal_balance_tests(df, treatment_col, arms, cont_vars, cat_or_bin_vars, alpha=0.05):
    """Testes formais de balance N-way: ANOVA F (contínuas) e chi-quadrado (categóricas/binárias)."""
    rows = []
    for v in cont_vars:
        groups = [df[df[treatment_col] == a][v] for a in arms]
        f, p = stats.f_oneway(*groups)
        rows.append({
            "variable": v, "test": "ANOVA F", "statistic": f, "p_value": p,
            "flag": "WARN" if p < alpha else "ok",
        })
    for v in cat_or_bin_vars:
        ct = pd.crosstab(df[treatment_col], df[v])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append({
            "variable": v, "test": "chi-squared", "statistic": chi2, "p_value": p,
            "flag": "WARN" if p < alpha else "ok",
        })
    return pd.DataFrame(rows)


def outcome_summary_by_arm(df, treatment_col, arms):
    """Tabela de contagem, taxas (visit/conversion) e spend médio/total por braço."""
    summary = df.groupby(treatment_col, observed=True).agg(
        n=(treatment_col, "count"),
        visit_rate=("visit", "mean"),
        conversion_rate=("conversion", "mean"),
        spend_mean=("spend", "mean"),
        spend_sum=("spend", "sum"),
    ).round(4)
    return summary.reindex(arms)
