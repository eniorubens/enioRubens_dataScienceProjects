"""Avaliação de uplift: Qini, AUUC, uplift@k, bootstrap de IC, envelope aleatório.

Populado incrementalmente: métricas básicas e correlação de rankings em S3,
bootstrap de IC e envelope de rankings aleatórios em S6.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

from .config import SEED
from .effects import ate_binary
from .learners import (
    build_meta_learner_encoder,
    encode_meta_learner_features,
    fit_causal_forest,
    fit_propensity_baseline,
    fit_single_meta_learner,
    fit_uplift_random_forest,
    fit_uplift_tree,
    predict_causal_forest_uplift,
    predict_propensity_score,
    predict_single_meta_learner,
    predict_uplift_random_forest_uplift,
    predict_uplift_tree_uplift,
)


def evaluate_ranking(y_true, score, treatment, k=0.3):
    """Qini AUC, uplift AUC e uplift@k para um ranking de score/uplift na validação.

    `score` pode ser uma probabilidade de resposta (baseline de propensão) ou
    um CATE estimado (meta-learner) — a métrica só depende do ranking.
    """
    return {
        "qini_auc": qini_auc_score(y_true, score, treatment),
        "uplift_auc": uplift_auc_score(y_true, score, treatment),
        f"uplift_at_{int(k * 100)}pct": uplift_at_k(y_true, score, treatment, strategy="overall", k=k),
    }


def spearman_ranking_correlation(score_a, score_b):
    """Correlação de Spearman entre dois rankings/scores (ex.: propensão vs. uplift)."""
    corr, p_value = spearmanr(score_a, score_b)
    return {"spearman_corr": corr, "p_value": p_value}


def evaluate_multiple_rankings(y_true, scores, treatment, k=0.3):
    """`evaluate_ranking` para vários scores (ex.: um por meta-learner).

    `scores`: dict {nome: array de score/uplift}. Retorna um DataFrame
    indexado pelo nome, uma linha por score.
    """
    rows = {name: evaluate_ranking(y_true, score, treatment, k=k) for name, score in scores.items()}
    return pd.DataFrame(rows).T


def permutation_noise_floor(train_df, val_df, treatment_col, outcome_col, encoder,
                             learner_name="S", n_reps=20, seed=SEED):
    """Chão de ruído (S4.4): embaralha `treatment_col` em `train_df`, reajusta um
    único meta-learner (S por padrão — o mais barato dos quatro) e mede
    `std(CATE)` sobre `val_df` real. Repete `n_reps` vezes.

    **O que este diagnóstico testa, precisamente.** Embaralhar o tratamento
    destrói tanto a heterogeneidade quanto o próprio ATE — o null efetivo é
    H0: τ(x)=0 para todo x (mundo totalmente nulo, sem nenhuma associação
    tratamento-outcome), não H0: τ(x)=ATE (efeito homogêneo, porém não nulo).
    Se o desvio-padrão do CATE real cai acima desta distribuição nula, isso
    mostra que há estrutura além de ruído puro de amostragem/overfitting —
    mas **não** distingue, por si só, entre "o efeito é heterogêneo" e "o
    efeito é homogêneo mas não nulo": um estimador com interações implícitas
    (como uma árvore/LightGBM) pode produzir `std(CATE)` não nulo por ruído de
    estimação mesmo sob um ATE verdadeiramente constante, e esse ruído de
    estimação tende a ser diferente sob ATE=0 (o mundo permutado) e sob
    ATE≠0 (o mundo real) independente de heterogeneidade genuína. Um teste
    formal de heterogeneidade exigiria um null que preserve o ATE e destrua
    só a dependência em X — não implementado aqui (ver limitações de S4).

    Usa só treino/validação — nunca o teste selado. `train_df` não é
    modificado (o embaralhamento gera um array novo a cada repetição).
    """
    rng = np.random.default_rng(seed)
    X_train = encode_meta_learner_features(train_df, encoder)
    X_val = encode_meta_learner_features(val_df, encoder)
    y_train = train_df[outcome_col].to_numpy(dtype=float)
    real_treatment = train_df[treatment_col].to_numpy()

    stds = np.empty(n_reps)
    for i in range(n_reps):
        permuted_treatment = rng.permutation(real_treatment)
        model = fit_single_meta_learner(learner_name, X_train, permuted_treatment, y_train, seed=seed)
        stds[i] = model.predict(X_val).ravel().std()
    return stds


def gates_by_cate_quintile(df, cate_col, outcome_col, treatment_col, n_groups=5):
    """GATES simplificado (S4.4): agrupa `df` por quintil de `cate_col` e estima
    o ATE (via `effects.ate_binary`) dentro de cada grupo, com IC.

    Comparar ICs individuais entre o quintil de topo e o de base (se
    sobrepõem ou não) é um teste informal — `gates_delta_bootstrap` faz a
    versão direta (Δ_GATES com IC próprio). Só treino/validação.
    """
    groups = pd.qcut(df[cate_col], q=n_groups, labels=False, duplicates="drop")
    rows = []
    for g in sorted(groups.dropna().unique()):
        subset = df.loc[groups == g]
        result = ate_binary(subset, treatment_col, outcome_col, treated_arm=1, control_arm=0)
        result["group"] = int(g)
        result["n"] = len(subset)
        result["cate_mean"] = subset[cate_col].mean()
        rows.append(result)
    return pd.DataFrame(rows)[["group", "n", "cate_mean", "ate", "se", "ci_low", "ci_high", "p_value"]]


def gates_delta_bootstrap(df, cate_col, outcome_col, treatment_col, n_groups=5, n_boot=2000, seed=SEED, alpha=0.05):
    """Δ_GATES = ATE(quintil de topo) − ATE(quintil de base), com IC via bootstrap.

    Complementa `gates_by_cate_quintile`: em vez de inferir heterogeneidade
    checando se os ICs analíticos de cada quintil se sobrepõem (teste
    indireto e conservador), estima diretamente a distribuição de Δ_GATES por
    reamostragem com reposição **dentro de cada quintil** (grupos fixos,
    definidos pelo ranking de CATE já observado) — apropriado para um
    desenho randomizado, já que reamostrar linhas preserva a mecânica de
    atribuição aleatória do tratamento dentro de cada estrato; não é preciso
    re-simular o sorteio de tratamento em si.

    IC por percentil; p-valor bilateral = 2×min(P(Δ_boot≤0), P(Δ_boot≥0)).
    """
    groups = pd.qcut(df[cate_col], q=n_groups, labels=False, duplicates="drop")
    top_group, bottom_group = int(groups.max()), int(groups.min())
    top_vals = df.loc[groups == top_group, [treatment_col, outcome_col]].to_numpy(dtype=float)
    bottom_vals = df.loc[groups == bottom_group, [treatment_col, outcome_col]].to_numpy(dtype=float)

    def _ate(vals):
        treated = vals[vals[:, 0] == 1, 1]
        control = vals[vals[:, 0] == 0, 1]
        return treated.mean() - control.mean()

    observed_delta = _ate(top_vals) - _ate(bottom_vals)

    rng = np.random.default_rng(seed)
    n_top, n_bottom = len(top_vals), len(bottom_vals)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        top_sample = top_vals[rng.integers(0, n_top, n_top)]
        bottom_sample = bottom_vals[rng.integers(0, n_bottom, n_bottom)]
        deltas[i] = _ate(top_sample) - _ate(bottom_sample)

    ci_low, ci_high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_value = min(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()), 1.0)
    return {
        "delta_gates": observed_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_boot": n_boot,
        "top_group": top_group,
        "bottom_group": bottom_group,
        "n_top": n_top,
        "n_bottom": n_bottom,
    }


def repeated_stratified_holdout(train_df, treatment_col, outcome_col, candidates,
                                 n_reps=15, test_size=0.25, seed_offset=1000):
    """Repeated stratified holdout (S4.9) — `n_reps` splits 75/25 dentro de
    `train_df`, estratificados por (treatment_col x outcome_col), seeds
    `seed_offset + rep`. **Não é k-fold/OOF**: entre repetições, uma mesma
    linha pode aparecer 0, 1 ou mais vezes nos conjuntos de avaliação — daí o
    nome. Mede estabilidade de Qini AUC/uplift AUC/uplift@30% de um conjunto
    pequeno de candidatos, com os MESMOS splits/seeds reusados entre todos os
    candidatos passados na mesma chamada — permite comparação pareada por
    split (`paired_deltas`). Usa só `train_df` — nunca toca `val_df` nem o
    teste selado.

    `candidates`: dict {nome: (kind, learner_name, base_learner_factory,
    use_fixed_p)}.
      - `kind='meta'`: ajusta um meta-learner via `fit_single_meta_learner`/
        `predict_single_meta_learner`; `learner_name` em {'S','T','X','R'}.
        `use_fixed_p=True` passa `p=2/3` fixo no fit/predict (só relevante
        para X/R — ver `fit_single_meta_learner`).
      - `kind='propensity'`: ajusta o baseline via `fit_propensity_baseline`/
        `predict_propensity_score` (`learner_name`/`base_learner_factory`/
        `use_fixed_p` ignorados).
      - `kind='causal_forest'`: ajusta via `fit_causal_forest`/
        `predict_causal_forest_uplift` (S5; `learner_name`/
        `base_learner_factory`/`use_fixed_p` ignorados).
      - `kind='uplift_tree'`: ajusta via `fit_uplift_tree`/
        `predict_uplift_tree_uplift` (S5; `learner_name`/
        `base_learner_factory`/`use_fixed_p` ignorados).
      - `kind='uplift_rf'`: ajusta via `fit_uplift_random_forest`/
        `predict_uplift_random_forest_uplift` (S5; `learner_name`/
        `base_learner_factory`/`use_fixed_p` ignorados).

    Os três novos `kind` de S5 reusam exatamente o mesmo `X_fit`/`X_eval`
    (encoder ajustado só em `fit_df`, dentro do loop de `rep`, antes do loop
    de candidatos — ver abaixo) já usado pelos candidatos `'meta'`/
    `'propensity'`, preservando tanto a propriedade de um único split
    compartilhado por repetição quanto a ausência de leakage do encoder.

    Retorna DataFrame longo (rep, candidate, qini_auc, uplift_auc,
    uplift_at_30pct).
    """
    def _strata(d):
        return d[treatment_col].astype(str) + "__outcome_" + d[outcome_col].astype(str)

    rows = []
    for rep in range(n_reps):
        strata = _strata(train_df)
        fit_idx, eval_idx = train_test_split(
            train_df.index.values, test_size=test_size, stratify=strata, random_state=seed_offset + rep,
        )
        fit_df = train_df.loc[fit_idx]
        eval_df = train_df.loc[eval_idx]

        encoder_rep = build_meta_learner_encoder(fit_df)
        X_fit = encode_meta_learner_features(fit_df, encoder_rep)
        X_eval = encode_meta_learner_features(eval_df, encoder_rep)
        treatment_fit = fit_df[treatment_col].to_numpy()
        y_fit = fit_df[outcome_col].to_numpy(dtype=float)
        p_fit_fixed = np.full(len(fit_df), 2 / 3)
        p_eval_fixed = np.full(len(eval_df), 2 / 3)

        for cand_name, (kind, learner_name, factory, use_fixed_p) in candidates.items():
            if kind == "meta":
                p_fit_arg = p_fit_fixed if use_fixed_p else None
                model = fit_single_meta_learner(
                    learner_name, X_fit, treatment_fit, y_fit, base_learner_factory=factory, p=p_fit_arg,
                )
                p_pred_arg = p_eval_fixed if use_fixed_p else None
                score_eval = predict_single_meta_learner(learner_name, model, X_eval, p=p_pred_arg)
            elif kind == "propensity":
                prop_model = fit_propensity_baseline(fit_df, treatment_col, outcome_col)
                score_eval = predict_propensity_score(prop_model, eval_df)
            elif kind == "causal_forest":
                cf_model = fit_causal_forest(X_fit, treatment_fit, y_fit)
                score_eval = predict_causal_forest_uplift(cf_model, X_eval)
            elif kind == "uplift_tree":
                ut_model = fit_uplift_tree(X_fit, treatment_fit, y_fit)
                score_eval = predict_uplift_tree_uplift(ut_model, X_eval)
            elif kind == "uplift_rf":
                urf_model = fit_uplift_random_forest(X_fit, treatment_fit, y_fit)
                score_eval = predict_uplift_random_forest_uplift(urf_model, X_eval)
            else:
                raise ValueError(f"kind desconhecido: {kind!r}")
            metrics = evaluate_ranking(eval_df[outcome_col].values, score_eval, eval_df[treatment_col].values)
            rows.append({"rep": rep, "candidate": cand_name, **metrics})

    return pd.DataFrame(rows)


def _stratified_bootstrap_indices(arm_labels, rng):
    """Índices de uma reamostragem com reposição, estratificada por
    `arm_labels` (o braço randomizado original — `segment`/`TREATMENT_COL`,
    não o treatment pooled): reamostra dentro de cada braço observado
    separadamente, preservando exatamente o tamanho observado de cada braço
    na réplica. **Não estratifica por outcome** — dentro de cada braço, a
    reamostragem é uniforme sobre as linhas, então a proporção de outcome de
    cada réplica varia livremente por acaso (ver S6, Fase G: "não
    estratifique por outcome"). Usada por `bootstrap_qini_comparison`, e
    exposta separadamente para ser testável de forma isolada.
    """
    arm_labels = np.asarray(arm_labels)
    parts = []
    for arm in np.unique(arm_labels):
        idx_arm = np.flatnonzero(arm_labels == arm)
        parts.append(idx_arm[rng.integers(0, len(idx_arm), len(idx_arm))])
    return np.concatenate(parts)


def bootstrap_qini_comparison(y_true, treatment_pooled, arm_labels, scores, deltas,
                               n_boot=2000, seed=SEED, alpha=0.05):
    """Bootstrap pareado pré-especificado para S6 (Fase G): IC 95% por
    percentil do Qini AUC absoluto de cada candidato em `scores`, e das
    diferenças pareadas pedidas em `deltas`.

    **Estratificação.** Reamostra com reposição dentro de cada braço
    randomizado original (`arm_labels` — `segment`/`TREATMENT_COL`, não o
    treatment pooled), preservando o tamanho observado de cada braço em toda
    réplica (`_stratified_bootstrap_indices`) — apropriado para respeitar o
    desenho randomizado original. **Não estratifica por outcome.** O
    treatment pooled (`treatment_pooled`) é então indexado pela mesma
    reamostragem, reconstruindo o tratamento pooled da réplica a partir do
    braço reamostrado.

    **Pareamento.** A mesma amostra bootstrap de índices (uma por réplica) é
    usada simultaneamente para `y_true`, `treatment_pooled` e **todos** os
    arrays em `scores` — não uma reamostragem independente por candidato —
    o que torna os deltas entre candidatos genuinamente pareados por
    construção (dois arrays de score idênticos produzem delta exatamente
    zero em toda réplica).

    `scores`: dict {nome: array alinhado a `y_true`/`treatment_pooled`/
    `arm_labels`}. `deltas`: lista de tuplas `(nome_a, nome_b)` — cada uma
    produz a diferença Qini(nome_a) − Qini(nome_b); nenhum par é assumido
    implicitamente, todos devem ser passados explicitamente pelo chamador
    (ver `preregistration.json` para os 5 pares pré-especificados de S6).

    Retorna dict com `point_estimate`/`ci_low`/`ci_high` por candidato (chave
    `"candidates"`) e por delta (chave `"deltas"`, indexada por
    `"nome_a - nome_b"`), mais `n_boot`/`seed`. Não usa os repeated holdouts
    de S4/S5 — amostra diretamente de `y_true`/`treatment_pooled`/`scores`
    fornecidos pelo chamador.
    """
    y_true = np.asarray(y_true, dtype=float)
    treatment_pooled = np.asarray(treatment_pooled)
    arm_labels = np.asarray(arm_labels)
    names = list(scores.keys())
    score_arrays = {name: np.asarray(arr) for name, arr in scores.items()}

    point_estimates = {
        name: qini_auc_score(y_true, score_arrays[name], treatment_pooled) for name in names
    }

    rng = np.random.default_rng(seed)
    qini_boot = {name: np.empty(n_boot) for name in names}
    for b in range(n_boot):
        boot_idx = _stratified_bootstrap_indices(arm_labels, rng)
        y_b = y_true[boot_idx]
        t_b = treatment_pooled[boot_idx]
        for name in names:
            qini_boot[name][b] = qini_auc_score(y_b, score_arrays[name][boot_idx], t_b)

    def _percentile_ci(arr):
        low, high = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(low), float(high)

    candidates_result = {}
    for name in names:
        ci_low, ci_high = _percentile_ci(qini_boot[name])
        candidates_result[name] = {
            "point_estimate": float(point_estimates[name]),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    deltas_result = {}
    for name_a, name_b in deltas:
        delta_boot = qini_boot[name_a] - qini_boot[name_b]
        ci_low, ci_high = _percentile_ci(delta_boot)
        deltas_result[f"{name_a} - {name_b}"] = {
            "point_estimate": float(point_estimates[name_a] - point_estimates[name_b]),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return {"n_boot": n_boot, "seed": seed, "candidates": candidates_result, "deltas": deltas_result}


def repeated_holdout_summary(results_df, metric="qini_auc"):
    """Resumo por candidato (mean/median/std/min/max/win_rate) de um resultado
    de `repeated_stratified_holdout`, ordenado por média decrescente.

    `win_rate` é uma estatística descritiva auxiliar: usa `idxmax()` por
    repetição, então um empate exato de métrica entre dois candidatos atribui
    a vitória inteira a só um deles (sem fracionamento) — irrelevante na
    prática com métricas contínuas como Qini AUC, mas vale registrar."""
    summary = results_df.groupby("candidate")[metric].agg(["mean", "median", "std", "min", "max"])
    n_reps = results_df["rep"].nunique()
    win_counts = results_df.loc[results_df.groupby("rep")[metric].idxmax()]["candidate"].value_counts()
    summary["win_rate"] = (win_counts / n_reps).reindex(summary.index).fillna(0.0)
    return summary.sort_values("mean", ascending=False)


def paired_deltas(results_df, baseline_candidate, metric="qini_auc"):
    """Diferenças pareadas por split, Δ = métrica(candidato) − métrica(baseline_candidate),
    de um resultado de `repeated_stratified_holdout` (mesmos splits reusados
    entre candidatos). Retorna DataFrame indexado por candidato com
    delta_mean/delta_median/delta_std/prop_delta_positive (proporção
    empírica de splits em que o candidato supera o baseline — **não** um
    p-value; não confundir com os p-values diagnósticos de Wilcoxon/t-test
    reportados separadamente no notebook)."""
    pivot = results_df.pivot(index="rep", columns="candidate", values=metric)
    rows = {}
    for col in pivot.columns:
        if col == baseline_candidate:
            continue
        delta = pivot[col] - pivot[baseline_candidate]
        rows[col] = {
            "delta_mean": delta.mean(),
            "delta_median": delta.median(),
            "delta_std": delta.std(),
            "prop_delta_positive": (delta > 0).mean(),
        }
    return pd.DataFrame(rows).T
