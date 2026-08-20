# ADR-005: S4 Error and Representation Challenge

## Status

Accepted for development on 2026-08-15.

## Context

The S3 word TF-IDF baseline reached macro-F1 `0.7003926722`, but the
`debt_credit_management` class reached precision `0.1738726174`, recall
`0.4192825112`, and F1 `0.2458100559`. The learning curve did not show a
stable improvement from additional training volume. S4 therefore tests whether
the error is related to representation or to the strength of class weighting.

## Decision

S4 compares four frozen development candidates:

1. Word TF-IDF with balanced weights as the S3 reference.
2. Word TF-IDF with square-root balanced weights.
3. `char_wb` TF-IDF with balanced weights.
4. `char_wb` TF-IDF with square-root balanced weights.

The square-root scheme starts with the usual balanced weight
`n / (k * n_i)`, takes its square root, and divides by
`sum_j(p_j * sqrt(weight_j))`. Its weighted sample mean is therefore one.
Each representation is fitted once and shared by its two weight schemes.

The development gates are frozen in `config/s4_experiment.json`:

- global macro-F1 at least `0.6900`;
- critical-class F1 at least `0.2715`;
- critical-class precision at least `0.20`.

A candidate must pass all three gates. Eligible candidates are ranked by
critical-class F1 and then macro-F1. If none passes, the result is
`NO_ELIGIBLE_CHALLENGER`.

## Boundary

The S4 scientific cache must contain only unique groups from `train` and
`validation`. Estimators fit only on `train`; validation predictions are
processed in bounded batches and reduced to a fixed confusion matrix. The
diagnostics report critical false negatives, critical false positives, and
the largest global off-diagonal confusions. The operational all-text view is
outside this stage.

`test`, `stress`, and `monitor` are sealed and are not read, scored, or used
for candidate selection. S4 does not persist an estimator as a deployable
model. MLflow remains an optional tracker only; the default is `NullTracker`.

The artifact is incremental and atomic. It starts with `complete=false`, is
updated after each candidate, and is cacheable only after `complete=true`.
Failures leave an `ERROR` artifact with the completed candidates and are never
treated as valid evidence.

## Executed development evidence

The real S4 artifact at `temp/s4/s4_results.json` is complete and was
published without refitting or reopening any sealed partition. The run took
`2193.0 s`, reached `7.24 GB` peak RSS, and left `0.01 GB` of system memory
available. It fitted two vectorizers over `345552` train rows and evaluated
`245980` validation rows.

No candidate passed all three gates. The selection status is
`NO_ELIGIBLE_CHALLENGER` and `recommended_candidate` is `null`. The
`word_balanced_reference` result is retained only as diagnostic focus because
it has the highest critical-class F1; it is not selected or promoted. The
`char_wb_sqrt_balanced` candidate illustrates the precision-recall trade-off:
critical precision is high, but critical recall and F1 remain below the
gates.

The versioned publication manifest is
[`config/s4_results.json`](../config/s4_results.json). It records the real
metrics, candidate configuration, resource boundary, non-confirmatory status,
and the operational recommendation to use cloud or more RAM for broad
character, transformer, or tuning searches. This evidence remains
development-only and is not deployable.

## Consequences

S4 can explain aggregate failure modes without changing S3 or reopening a
sealed evaluation. A passing challenger remains a development result and does
not authorize deployment. A later stage must decide whether the diagnostic
evidence justifies hierarchical classification, taxonomy review, calibration,
or a new pre-registered experiment.
