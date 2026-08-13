"""Exploratory policy learning, IPW evaluation and ROI simulation for S8.

All model fitting in this module is development-only.  The public helpers take
an explicit validation frame for policy scoring/evaluation and never load S6
artifacts or the sealed partition.
"""
from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from .config import (
    ARMS,
    BIN_VARS,
    CAT_VARS,
    CONTROL_ARM,
    FEATURE_COLS,
    POOLED_TREATMENT_COL,
    SEED,
    TREATMENT_COL,
)
from .learners import (
    build_meta_learner_encoder,
    encode_meta_learner_features,
    fit_propensity_baseline,
    fit_single_meta_learner,
    fit_uplift_tree,
    predict_propensity_score,
    predict_single_meta_learner,
    predict_uplift_tree_uplift,
)

POOLED_PROPENSITY = 2 / 3
ARM_PROPENSITY = 1 / 3
POLICY_NAMES = ("no_contact", "treat_all", "random", "propensity", "uplift_tree", "x_tree")
ACTION_CODES = {CONTROL_ARM: 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
ACTION_LABELS = {0: CONTROL_ARM, 1: "Mens E-Mail", 2: "Womens E-Mail"}


def _as_float_array(values, name):
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_budget(budget):
    budget = float(budget)
    if not 0 <= budget <= 1:
        raise ValueError("budget must be between 0 and 1")
    return budget


def contact_count(n_rows, budget):
    """Number of contacts allowed by a fractional budget.

    Flooring makes the constraint conservative and deterministic.
    """
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    return int(np.floor(n_rows * _validate_budget(budget) + 1e-12))


def budget_mask(scores, budget):
    """Select the largest scores under a budget, with index-based tie breaks."""
    scores = _as_float_array(scores, "scores")
    n_select = contact_count(len(scores), budget)
    mask = np.zeros(len(scores), dtype=bool)
    if n_select == 0:
        return mask
    # lexsort uses the last key as primary: descending score, ascending index.
    order = np.lexsort((np.arange(len(scores)), -scores))
    mask[order[:n_select]] = True
    return mask


def make_binary_policy_masks(scores, budgets, seed=SEED):
    """Create reproducible binary policy masks for every policy and budget."""
    scores = {name: _as_float_array(value, name) for name, value in scores.items()}
    if not scores:
        raise ValueError("scores must contain at least one ranking")
    n_rows = len(next(iter(scores.values())))
    if any(len(value) != n_rows for value in scores.values()):
        raise ValueError("all score arrays must have the same length")
    random_scores = np.random.default_rng(seed).random(n_rows)
    rankings = {**scores, "random": random_scores}
    masks = {}
    for policy, ranking in rankings.items():
        masks[policy] = {float(budget): budget_mask(ranking, budget) for budget in budgets}
    masks["no_contact"] = {float(budget): np.zeros(n_rows, dtype=bool) for budget in budgets}
    masks["treat_all"] = {float(budget): np.ones(n_rows, dtype=bool) for budget in budgets}
    return masks


def binary_ipw_incremental_value(outcome, treatment, policy_mask, propensity=POOLED_PROPENSITY):
    """Estimate incremental value versus no contact for a binary policy by IPW."""
    outcome = _as_float_array(outcome, "outcome")
    treatment = np.asarray(treatment, dtype=int).ravel()
    policy_mask = np.asarray(policy_mask, dtype=bool).ravel()
    if not (len(outcome) == len(treatment) == len(policy_mask)):
        raise ValueError("outcome, treatment and policy_mask must have equal length")
    if not 0 < propensity < 1:
        raise ValueError("propensity must be strictly between 0 and 1")
    contribution = treatment * outcome / propensity - (1 - treatment) * outcome / (1 - propensity)
    return float(np.mean(policy_mask * contribution))


def stratified_bootstrap_indices(arm_labels, rng):
    """Draw a bootstrap sample preserving the observed count of each arm."""
    labels = np.asarray(arm_labels)
    pieces = []
    for label in pd.unique(labels):
        indices = np.flatnonzero(labels == label)
        pieces.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(pieces)


def evaluate_binary_policies(
    val_df,
    policy_masks,
    outcomes=("visit", "conversion", "spend"),
    n_boot=400,
    seed=SEED,
):
    """Evaluate frozen binary policies and paired stratified bootstrap CIs.

    ``treat_all`` is intentionally retained as an unconditional benchmark. It
    is marked ``budget_feasible=False`` whenever the requested budget is below
    100%, because it contacts everyone rather than obeying the cap.
    """
    if not isinstance(n_boot, (int, np.integer)) or int(n_boot) <= 0:
        raise ValueError("n_boot must be a positive integer")
    rows = []
    for policy, masks_by_budget in policy_masks.items():
        for budget, mask in masks_by_budget.items():
            for outcome in outcomes:
                value = binary_ipw_incremental_value(
                    val_df[outcome], val_df[POOLED_TREATMENT_COL], mask
                )
                rows.append({
                    "policy": policy,
                    "budget": float(budget),
                    "outcome": outcome,
                    "contact_rate": float(np.mean(mask)),
                    "contacts": int(np.sum(mask)),
                    "budget_feasible": bool(np.mean(mask) <= float(budget) + 1e-12),
                    "incremental_value": value,
                })
    result = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    boot = {tuple(row): [] for row in result[["policy", "budget", "outcome"]].itertuples(index=False, name=None)}
    propensity_keys = {(float(budget), outcome): [] for budget in policy_masks.get("propensity", {}) for outcome in outcomes}
    labels = val_df[TREATMENT_COL].astype(str).to_numpy()
    outcome_arrays = {outcome: _as_float_array(val_df[outcome], outcome) for outcome in outcomes}
    treatment = val_df[POOLED_TREATMENT_COL].to_numpy()
    for _ in range(int(n_boot)):
        index = stratified_bootstrap_indices(labels, rng)
        for policy, masks_by_budget in policy_masks.items():
            for budget, mask in masks_by_budget.items():
                selected = mask[index]
                for outcome in outcomes:
                    boot[(policy, float(budget), outcome)].append(
                        binary_ipw_incremental_value(outcome_arrays[outcome][index], treatment[index], selected)
                    )
        for budget, mask in policy_masks.get("propensity", {}).items():
            for outcome in outcomes:
                propensity_keys[(float(budget), outcome)].append(
                    binary_ipw_incremental_value(outcome_arrays[outcome][index], treatment[index], mask[index])
                )
    ci_rows = []
    for row in result.itertuples(index=False):
        values = np.asarray(boot[(row.policy, float(row.budget), row.outcome)])
        low, high = np.quantile(values, [0.025, 0.975])
        if row.policy == "propensity" or (float(row.budget), row.outcome) not in propensity_keys:
            delta = np.zeros_like(values)
        else:
            delta = values - np.asarray(propensity_keys[(float(row.budget), row.outcome)])
        delta_low, delta_high = np.quantile(delta, [0.025, 0.975])
        ci_rows.append({**row._asdict(), "ci_low": float(low), "ci_high": float(high),
                        "delta_vs_propensity": float(row.incremental_value - result.loc[
                            (result.policy == "propensity") & (result.budget == row.budget) & (result.outcome == row.outcome),
                            "incremental_value"].iloc[0]) if row.policy != "propensity" else 0.0,
                        "delta_ci_low": float(delta_low), "delta_ci_high": float(delta_high)})
    return pd.DataFrame(ci_rows)


def fit_binary_policy_scores(train_df, val_df, outcome="visit", seed=SEED):
    """Fit development-only propensity, historical UpliftTree and X+Tree scores."""
    encoder = build_meta_learner_encoder(train_df)
    x_train = encode_meta_learner_features(train_df, encoder)
    x_val = encode_meta_learner_features(val_df, encoder)
    treatment = train_df[POOLED_TREATMENT_COL].to_numpy()
    y = train_df[outcome].to_numpy(dtype=float)
    propensity_model = fit_propensity_baseline(train_df, POOLED_TREATMENT_COL, outcome, seed=seed)
    propensity_score = predict_propensity_score(propensity_model, val_df)
    uplift_model = fit_uplift_tree(x_train, treatment, y, seed=seed)
    uplift_score = predict_uplift_tree_uplift(uplift_model, x_val)
    x_model = fit_single_meta_learner(
        "X", x_train, treatment, y, seed=seed,
        base_learner_factory=lambda: DecisionTreeRegressor(max_depth=4, random_state=seed),
        p=np.full(len(train_df), POOLED_PROPENSITY),
    )
    x_score = predict_single_meta_learner("X", x_model, x_val, p=np.full(len(val_df), POOLED_PROPENSITY))
    return {"propensity": propensity_score, "uplift_tree": uplift_score, "x_tree": x_score}


def break_even_email_cost(margin, incremental_spend_per_customer, contact_rate):
    """Maximum cost per contact compatible with zero incremental profit."""
    margin = float(margin)
    spend = float(incremental_spend_per_customer)
    contact_rate = float(contact_rate)
    if contact_rate < 0 or margin < 0:
        raise ValueError("margin and contact_rate must be non-negative")
    if contact_rate == 0:
        return float("nan")
    return float(margin * spend / contact_rate)


def roi_metrics(incremental_spend_per_customer, contact_rate, gross_margin, email_cost, n_customers=1):
    """Return illustrative economics using observed spend as a revenue proxy.

    ``incremental_spend_per_customer`` is an observed-spend increment, not
    revenue already multiplied by margin. ``gross_margin`` converts the
    resulting revenue proxy into ``gross_profit``; it is an explicit scenario
    assumption because margin is absent from Hillstrom.
    """
    spend = float(incremental_spend_per_customer)
    contact_rate = float(contact_rate)
    gross_margin = float(gross_margin)
    email_cost = float(email_cost)
    n_customers = int(n_customers)
    if min(contact_rate, gross_margin, email_cost) < 0 or n_customers < 1:
        raise ValueError("economic inputs must be non-negative")
    incremental_revenue = spend * n_customers
    gross_profit = gross_margin * incremental_revenue
    cost = email_cost * contact_rate * n_customers
    net = gross_profit - cost
    roi = float(net / cost) if cost > 0 else np.nan
    return {
        "incremental_revenue": float(incremental_revenue),
        "gross_profit": float(gross_profit),
        "campaign_cost": float(cost),
        "net_profit": float(net),
        "roi": roi,
        "value_per_1000": float(net / n_customers * 1000),
        "break_even_cost": break_even_email_cost(gross_margin, spend, contact_rate),
    }


def build_roi_sensitivity(policy_curve, margins, email_costs, outcome="spend", n_customers=1):
    """Expand spend policy values over an explicitly illustrative scenario grid."""
    spend_curve = policy_curve[policy_curve["outcome"] == outcome].copy()
    rows = []
    for row in spend_curve.itertuples(index=False):
        for margin in margins:
            for email_cost in email_costs:
                rows.append({
                    "policy": row.policy, "budget": row.budget,
                    "budget_feasible": getattr(row, "budget_feasible", True),
                    "contact_rate": row.contact_rate, "incremental_spend_per_customer": row.incremental_value,
                    "gross_margin": float(margin), "email_cost": float(email_cost),
                    **roi_metrics(row.incremental_value, row.contact_rate, margin, email_cost, n_customers),
                })
    return pd.DataFrame(rows)


def _fit_arm_model(frame, outcome, seed):
    x = frame[FEATURE_COLS].copy()
    for col in CAT_VARS:
        x[col] = x[col].astype("category")
    if outcome == "spend":
        model = lgb.LGBMRegressor(random_state=seed, verbose=-1)
    else:
        model = lgb.LGBMClassifier(random_state=seed, verbose=-1)
    model.fit(x, frame[outcome].to_numpy())
    return model


def fit_three_way_models(train_df, outcomes=("visit", "conversion", "spend"), seed=SEED):
    """Fit one outcome model per observed arm, using only ``train_df``."""
    models = {}
    for arm in ARMS:
        arm_df = train_df[train_df[TREATMENT_COL].astype(str) == arm]
        models[arm] = {outcome: _fit_arm_model(arm_df, outcome, seed) for outcome in outcomes}
    return models


def predict_three_way_models(models, val_df, outcomes=("visit", "conversion", "spend")):
    """Predict each arm's potential outcome on validation rows."""
    x = val_df[FEATURE_COLS].copy()
    for col in CAT_VARS:
        x[col] = x[col].astype("category")
    predictions = {}
    for arm, arm_models in models.items():
        predictions[arm] = {}
        for outcome in outcomes:
            model = arm_models[outcome]
            predictions[arm][outcome] = (
                model.predict(x) if outcome == "spend" else model.predict_proba(x)[:, 1]
            )
    return predictions


def three_way_net_gains(predictions, gross_margin, email_cost, outcome="spend"):
    """Compute per-customer net gains for Mens/Womens versus No E-Mail.

    The policy target is economic net gain, not raw predicted spend:
    ``gross_margin * (predicted_spend_email - predicted_spend_control) -
    email_cost``. This is an illustrative scenario, not a company margin
    estimate. ``outcome`` is kept explicit to prevent accidentally using a
    binary outcome as currency.
    """
    if outcome != "spend":
        raise ValueError("economic gains require outcome='spend'")
    margin = float(gross_margin)
    cost = float(email_cost)
    if margin < 0 or cost < 0:
        raise ValueError("gross_margin and email_cost must be non-negative")
    control = _as_float_array(predictions[CONTROL_ARM][outcome], "control prediction")
    gains = []
    for arm in ("Mens E-Mail", "Womens E-Mail"):
        email_pred = _as_float_array(predictions[arm][outcome], f"{arm} prediction")
        if len(email_pred) != len(control):
            raise ValueError("all arm predictions must have equal length")
        gains.append(margin * (email_pred - control) - cost)
    return np.column_stack(gains)


def three_way_actions(net_gains, budget):
    """Choose the best positive net-gain email, then enforce the contact cap.

    A row remains No E-Mail whenever both email net gains are non-positive.
    Ties are deterministic: Mens E-Mail wins the tie between the two email
    actions because it is the first column.
    """
    gains = np.asarray(net_gains, dtype=float)
    if gains.ndim != 2 or gains.shape[1] != 2:
        raise ValueError("net_gains must have shape (n_rows, 2)")
    if not np.isfinite(gains).all():
        raise ValueError("net_gains contains non-finite values")
    best_arm = np.argmax(gains, axis=1)
    best_gain = gains[np.arange(len(gains)), best_arm]
    selected = budget_mask(best_gain, budget) & (best_gain > 0)
    return np.where(selected, best_arm + 1, 0).astype(int)


def targeted_arm_actions(gains, arm_code, budget):
    """Target one email arm by its predicted net gain under a budget."""
    gains = _as_float_array(gains, "gains")
    actions = np.zeros(len(gains), dtype=int)
    actions[budget_mask(gains, budget)] = int(arm_code)
    return actions


def all_arm_actions(n_rows, arm_code):
    """Literal all-arm benchmark; it is budget-feasible only at budget=1."""
    if arm_code not in (0, 1, 2):
        raise ValueError("arm_code must be 0 (control), 1 (Mens) or 2 (Womens)")
    return np.full(int(n_rows), int(arm_code), dtype=int)


def fixed_arm_actions(gains, arm_code, budget):
    """Backward-compatible alias for targeted single-arm allocation."""
    return targeted_arm_actions(gains, arm_code, budget)


def random_three_way_actions(n_rows, budget, seed=SEED):
    """Reproducible random contact and random email-arm assignment."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    actions = np.zeros(n_rows, dtype=int)
    n_contact = contact_count(n_rows, budget)
    selected = order[:n_contact]
    actions[selected] = rng.integers(1, 3, size=n_contact)
    return actions


def multi_arm_ipw_incremental_value(outcome, observed_arm, actions, probabilities=None, control_arm=CONTROL_ARM):
    """Estimate policy value versus no email under a three-arm RCT."""
    outcome = _as_float_array(outcome, "outcome")
    observed_arm = np.asarray(observed_arm).astype(str).ravel()
    actions = np.asarray(actions, dtype=int).ravel()
    if not (len(outcome) == len(observed_arm) == len(actions)):
        raise ValueError("outcome, observed_arm and actions must have equal length")
    probabilities = probabilities or {arm: ARM_PROPENSITY for arm in ARMS}
    contribution = np.zeros(len(outcome), dtype=float)
    for action_code, arm in ACTION_LABELS.items():
        if action_code == 0:
            continue
        if arm not in probabilities or not 0 < probabilities[arm] <= 1:
            raise ValueError("probabilities must contain valid values for every arm")
        treated = (actions == action_code) & (observed_arm == arm)
        control = (actions == action_code) & (observed_arm == control_arm)
        contribution += treated * outcome / probabilities[arm]
        contribution -= control * outcome / probabilities[control_arm]
    return float(np.mean(contribution))


def evaluate_three_way_policies(
    val_df,
    actions_by_policy,
    outcomes=("visit", "conversion", "spend"),
    n_boot=400,
    seed=SEED,
):
    """Evaluate frozen three-way actions by multi-arm IPW with paired CIs."""
    if not isinstance(n_boot, (int, np.integer)) or int(n_boot) <= 0:
        raise ValueError("n_boot must be a positive integer")
    rows = []
    for policy, actions_by_budget in actions_by_policy.items():
        for budget, actions in actions_by_budget.items():
            actions = np.asarray(actions, dtype=int)
            for outcome in outcomes:
                rows.append({
                    "policy": policy, "budget": float(budget), "outcome": outcome,
                    "no_email_rate": float(np.mean(actions == 0)),
                    "mens_rate": float(np.mean(actions == 1)),
                    "womens_rate": float(np.mean(actions == 2)),
                    "contact_rate": float(np.mean(actions != 0)),
                    "budget_feasible": bool(np.mean(actions != 0) <= float(budget) + 1e-12),
                    "incremental_value": multi_arm_ipw_incremental_value(
                        val_df[outcome], val_df[TREATMENT_COL], actions
                    ),
                })
    result = pd.DataFrame(rows)
    labels = val_df[TREATMENT_COL].astype(str).to_numpy()
    outcome_arrays = {outcome: _as_float_array(val_df[outcome], outcome) for outcome in outcomes}
    rng = np.random.default_rng(seed)
    boot = {tuple(row): [] for row in result[["policy", "budget", "outcome"]].itertuples(index=False, name=None)}
    for _ in range(int(n_boot)):
        index = stratified_bootstrap_indices(labels, rng)
        for policy, actions_by_budget in actions_by_policy.items():
            for budget, actions in actions_by_budget.items():
                for outcome in outcomes:
                    boot[(policy, float(budget), outcome)].append(
                        multi_arm_ipw_incremental_value(
                            outcome_arrays[outcome][index],
                            labels[index],
                            np.asarray(actions, dtype=int)[index],
                        )
                    )
    result["ci_low"] = [float(np.quantile(boot[(row.policy, float(row.budget), row.outcome)], 0.025))
                         for row in result.itertuples(index=False)]
    result["ci_high"] = [float(np.quantile(boot[(row.policy, float(row.budget), row.outcome)], 0.975))
                          for row in result.itertuples(index=False)]
    return result


def fit_three_way_surrogate(val_df, actions, max_depth=3, seed=SEED):
    """Fit an interpretable descriptive tree for the learned action."""
    target = np.asarray(actions, dtype=int)
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_VARS),
        ("num", "passthrough", ["recency", "history"] + BIN_VARS),
    ])
    # Balance the descriptive target so rare recommended actions are not
    # discarded by a majority-class split.
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=200,
        class_weight="balanced",
        random_state=seed,
    )
    pipeline = Pipeline([( "preprocess", preprocess), ("tree", tree)])
    pipeline.fit(val_df[FEATURE_COLS], target)
    pred = pipeline.predict(val_df[FEATURE_COLS])
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    rules = export_text(tree, feature_names=[_clean_rule_name(name) for name in feature_names], max_depth=max_depth)
    return {
        "fidelity": float(accuracy_score(target, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "action_distribution": pd.Series(target).value_counts(normalize=True).sort_index().to_dict(),
        "rules": rules,
    }


def _clean_rule_name(name):
    return re.sub(r"^(cat|num)__", "", str(name))
