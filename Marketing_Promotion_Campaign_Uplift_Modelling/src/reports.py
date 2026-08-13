"""Tabelas e figuras consumidas pelos notebooks.

Populado incrementalmente a partir de S3.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from .config import BIN_VARS, CAT_VARS, CONT_VARS, FEATURE_COLS, POOLED_TREATMENT_COL, SEED
from .evaluation import evaluate_ranking
from .learners import (
    build_meta_learner_encoder,
    encode_meta_learner_features,
    fit_single_meta_learner,
    predict_single_meta_learner,
)

S7_OUTCOMES = ("visit", "conversion", "spend")
S7_PROFILE_VARS = ("recency", "history_segment", "channel", "zip_code", "newbie", "mens", "womens", "history")
# Hillstrom's pooled treatment is randomized: two treatment arms versus one
# control arm, so P(T=1 | X) is known to be 2/3 for every development row.
# This is a design constant, not a tuned modeling hyperparameter.
S7_POOLED_PROPENSITY = 2 / 3


def fit_development_xtree_scores(train_df, val_df, outcomes=S7_OUTCOMES, seed=SEED):
    """Fit one shallow X-learner per outcome on train and score validation.

    This is an exploratory development-only ranking used by S7. It does not
    read or score the sealed test and it does not reuse S6 confirmatory
    artifacts. The shallow tree base learner mirrors the already documented
    development diagnostic from S4 while keeping the S7 question descriptive.
    Because Hillstrom is a pooled randomized experiment with two treatment
    arms and one control arm, the known propensity is 2/3 for every row.
    Passing that design constant explicitly prevents causalml from fitting a
    nuisance propensity model; it is not tuning or retrospective optimization.
    """
    encoder = build_meta_learner_encoder(train_df)
    X_train = encode_meta_learner_features(train_df, encoder)
    X_val = encode_meta_learner_features(val_df, encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    p_train = np.full(len(train_df), S7_POOLED_PROPENSITY, dtype=float)
    p_val = np.full(len(val_df), S7_POOLED_PROPENSITY, dtype=float)

    scores = {}
    for outcome in outcomes:
        model = fit_single_meta_learner(
            "X",
            X_train,
            treatment,
            train_df[outcome].to_numpy(dtype=float),
            seed=seed,
            base_learner_factory=lambda: DecisionTreeRegressor(max_depth=4, random_state=seed),
            p=p_train,
        )
        scores[outcome] = predict_single_meta_learner("X", model, X_val, p=p_val)
    return scores


def add_score_quantiles(df, scores, n_groups=5):
    """Return ``df`` with score and quantile columns for each outcome score."""
    out = df.copy()
    labels = list(range(1, n_groups + 1))
    for outcome, score in scores.items():
        score_col = f"score_{outcome}"
        quantile_col = f"q_{outcome}"
        out[score_col] = np.asarray(score, dtype=float)
        out[quantile_col] = pd.qcut(out[score_col].rank(method="first"), q=n_groups, labels=labels).astype("Int64")
    return out


def top_bottom_profile(df, quantile_col="q_visit", variables=S7_PROFILE_VARS):
    """Profile the highest vs. lowest score quantiles.

    Continuous variables are summarized by means; binary/categorical variables
    are expanded into level shares. ``delta_top_minus_bottom`` is descriptive,
    not a causal estimand.
    """
    bottom = df[quantile_col].min()
    top = df[quantile_col].max()
    rows = []
    for variable in variables:
        if variable in CONT_VARS:
            bottom_value = df.loc[df[quantile_col] == bottom, variable].mean()
            top_value = df.loc[df[quantile_col] == top, variable].mean()
            rows.append({
                "variable": variable,
                "level": "mean",
                "bottom_quantile": float(bottom_value),
                "top_quantile": float(top_value),
                "delta_top_minus_bottom": float(top_value - bottom_value),
            })
        else:
            levels = sorted(df[variable].dropna().astype(str).unique())
            for level in levels:
                bottom_value = df.loc[df[quantile_col] == bottom, variable].astype(str).eq(level).mean()
                top_value = df.loc[df[quantile_col] == top, variable].astype(str).eq(level).mean()
                rows.append({
                    "variable": variable,
                    "level": level,
                    "bottom_quantile": float(bottom_value),
                    "top_quantile": float(top_value),
                    "delta_top_minus_bottom": float(top_value - bottom_value),
                })
    result = pd.DataFrame(rows)
    return result.sort_values("delta_top_minus_bottom", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def quantile_outcome_table(df, score_outcome="visit", outcomes=S7_OUTCOMES):
    """Observed outcome means by score quantile for the selected ranking."""
    quantile_col = f"q_{score_outcome}"
    rows = []
    for group, group_df in df.groupby(quantile_col, observed=True):
        row = {"quantile": int(group), "n": len(group_df), f"score_{score_outcome}_mean": group_df[f"score_{score_outcome}"].mean()}
        for outcome in outcomes:
            row[f"{outcome}_mean"] = group_df[outcome].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True)


def funnel_ranking_summary(df, outcomes=S7_OUTCOMES):
    """Evaluate and compare development rankings across funnel outcomes."""
    metric_rows = []
    for outcome in outcomes:
        score = df[f"score_{outcome}"].to_numpy(dtype=float)
        top_mask = score >= np.quantile(score, 0.70)
        top_df = df.loc[top_mask]
        treated_mean = top_df.loc[top_df[POOLED_TREATMENT_COL] == 1, outcome].mean()
        control_mean = top_df.loc[top_df[POOLED_TREATMENT_COL] == 0, outcome].mean()
        metrics = {
            "qini_auc": np.nan,
            "uplift_auc": np.nan,
            "uplift_at_30pct": np.nan,
            "incremental_mean_top_30pct": float(treated_mean - control_mean),
        }
        values = set(pd.Series(df[outcome]).dropna().unique())
        if values == {0, 1} or values == {0.0, 1.0}:
            metrics.update(evaluate_ranking(
                df[outcome].to_numpy(dtype=float),
                score,
                df[POOLED_TREATMENT_COL].to_numpy(),
            ))
        metric_rows.append({"outcome": outcome, **metrics})
    metrics_df = pd.DataFrame(metric_rows)

    corr_rows = []
    for left in outcomes:
        for right in outcomes:
            corr, p_value = spearmanr(df[f"score_{left}"], df[f"score_{right}"])
            corr_rows.append({
                "score_a": left,
                "score_b": right,
                "spearman_corr": float(corr),
                "p_value": float(p_value),
            })
    corr_df = pd.DataFrame(corr_rows)
    return metrics_df, corr_df


def fit_high_uplift_surrogate(df, quantile_col="q_visit", max_depth=3, seed=SEED):
    """Fit a shallow descriptive surrogate for membership in the top quantile.

    The target is the estimated high-uplift group, not the causal outcome. This
    is only a readable approximation of the ranking and should be labelled as a
    post-confirmatory exploratory surrogate.
    """
    top = df[quantile_col].max()
    y = df[quantile_col].eq(top).astype(int)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_VARS),
            ("num", "passthrough", CONT_VARS + BIN_VARS),
        ]
    )
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=200, random_state=seed, class_weight="balanced")
    pipe = Pipeline([("preprocess", preprocess), ("tree", model)])
    pipe.fit(df[FEATURE_COLS], y)
    pred = pipe.predict(df[FEATURE_COLS])
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    rules = export_text(pipe.named_steps["tree"], feature_names=list(feature_names), max_depth=max_depth)
    return {
        "positive_rate": float(y.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "rules": rules,
    }


def build_s7_heterogeneity_report(train_df, val_df, full_df=None, seed=SEED):
    """Build all tables used by Notebook 06 without touching the sealed test."""
    scores = fit_development_xtree_scores(train_df, val_df, seed=seed)
    scored_val = add_score_quantiles(val_df, scores)
    profile = top_bottom_profile(scored_val)
    quantile_outcomes = quantile_outcome_table(scored_val)
    funnel_metrics, funnel_spearman = funnel_ranking_summary(scored_val)
    surrogate = fit_high_uplift_surrogate(scored_val)
    conversion_n = int((full_df["conversion"] == 1).sum()) if full_df is not None else int((scored_val["conversion"] == 1).sum())
    conversion_denom = len(full_df) if full_df is not None else len(scored_val)
    return {
        "scored_validation": scored_val,
        "profile": profile,
        "quantile_outcomes": quantile_outcomes,
        "funnel_metrics": funnel_metrics,
        "funnel_spearman": funnel_spearman,
        "surrogate": surrogate,
        "conversion_positives": conversion_n,
        "conversion_rate": conversion_n / conversion_denom,
    }
