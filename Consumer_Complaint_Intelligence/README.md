# Consumer Complaint Intelligence

**Language:** English | [Português (Brasil)](README.pt-BR.md)

This project routes free-text consumer complaints from the CFPB Consumer
Complaint Database into nine product families, and studies **what it takes to
honestly confirm that a text classifier works on data it has never seen**.

The modeling target is deliberately hostile. The critical class,
`debt_credit_management`, accounts for **0.14% of narratives** — 5,437 rows in
3.8 million. Every stage of the project is organized around one question: does
the measured gain survive an evaluation that was designed before the number
existed?

Two full cycles were run against that question. Both returned
`NOT_CONFIRMED`. The deliverable is the protocol and the evidence, not a
deployable model.

## Executive Result

Two sealed partitions were opened, each exactly once, each under a protocol
frozen and committed before the data was read.

| Cycle | Sealed window | Gates | Status |
|---|---|---|---|
| **S8** (V1, frozen S7 classifier) | `test` 2025-H1 | 2 of 3 | `NOT_CONFIRMED` |
| **V2.1-C** (hierarchical V2 over S7) | `stress` 2025-H2 | 3 of 4 | `NOT_CONFIRMED` |

The V2.1-C opening, on 269,915 clean and novel group representatives:

| Gate | Bar | Observed | Result |
|---|---:|---:|---|
| Macro-F1 | ≥ 0.6900 | 0.710748 | pass |
| Critical-class precision | ≥ 0.2000 | 0.426070 | pass |
| Paired critical-F1 gain over control | > 0 (strict) | +0.006455 | pass |
| **Critical-class F1** | **≥ 0.2715** | **0.260404** | **fail** |

`deploy=false` in both cycles. Confirmation was never defined to authorize
deployment.

The paired gate is the one the V2 cycle existed to answer, and it passed: the
hierarchical model beat its own frozen control on identical rows. Under the
pre-specified diagnostic resampling procedure, the 95% bootstrap interval was
[0.000235, 0.013661] and excluded zero. The observed gain is also **one
seventh** of the +0.047234 measured in development and too small to reach an
absolute bar fixed in August 2026, before V2 existed.

## Why the Development Gain Did Not Survive

The mechanism is in the firing counts, not in the model.

| Quantity | Development (2024-Q4) | Stress (2025-H2) |
|---|---:|---:|
| Stage-A positive decisions | 258 in 127,706 rows (0.202%) | 224 in 269,915 rows (0.083%) |
| Effective overrides | 82 | 36 |
| Critical-F1 gain over S7 | +0.047234 | +0.006455 |

The stage-A detector was fitted on 2023-08 to 2024-06 and calibrated on
2024-07 to 2024-09. The `stress` window is 12 to 18 months after the end of
fit and 9 to 15 months after the end of calibration. There, its margins reach
the frozen threshold at 41% of the development rate. Of the 36 effective
overrides, 10 were right and 26 were wrong — a 27.8% hit rate against a
42.6% critical-class precision on the same window. That is why precision fell
while recall rose, and why the net movement in F1 was small.

## The Control Arm, and What It Ruled Out

Macro-F1 is an unweighted mean over nine classes, so a shifted class mix moves
it even for an identical model. The same frozen S7 scored critical-F1 0.339665
on `validation` 2024-H2 and 0.257843 on `test` 2025-H1 — a swing of 0.081822
with no change to the model at all. Absolute metrics are not comparable across
windows.

So V2.1-C scored two arms in a **single pass** over the same sealed rows: the
primary arm (V2 hierarchical) and a control arm (frozen S7 alone). Both arms
land in one 9×9×9 joint accumulator indexed by `[truth, v2_label, s7_label]`;
marginalizing one axis recovers each arm's confusion matrix, and the bootstrap
resamples the joint cells so the correlation between arms is preserved.

The control paid for itself. Frozen S7 scored 0.253949 on `stress` 2025-H2
against 0.257843 on `test` 2025-H1. The similar critical-class F1 weakens a
simple "the stress window was harder" explanation, but it is not a formal
equivalence test and does not eliminate other window effects.

It also permits one clean comparison: V2 on `stress` (0.260404) exceeds S7 on
`test` (0.257843) by 0.002561. The entire V2 cycle moved the metric that
motivated it by roughly two and a half thousandths, against a bar that needed
thirteen and a half more.

## Business Question

The CFPB database preserves the `Product` and `Issue` options that existed
when each complaint was filed. Routing a narrative to the right product family
is the operational task; the critical class is the one where routing failures
are expensive and where every classical approach in this project failed.

Two framings are refused throughout, and the refusal is enforced in the
documentation rather than left to the reader:

- Complaint volume is **not** market prevalence and **not** a measure of
  consumer harm. The target is routing to historical form categories.
- `Issue` is not a stable global label. The field reference states that
  possible `Issue` values depend on `Product`, and 178 raw labels collide
  across families — `Incorrect information on your report` alone appears
  under six of them.

## Dataset

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
downloaded as a 821 MB Parquet snapshot.

| Quantity | Value |
|---|---:|
| Total rows | 17,094,898 |
| Rows with a narrative | 3,836,659 (22.4%) |
| Year range | 2011 to 2026 (2026 partial) |
| Raw `Product` labels | 21 |
| Modeled families | 9 |
| Distinct raw `Issue` labels | 178 |

Narrative rows by family, and the shape of the problem:

| Family | Narrative rows | Share |
|---|---:|---:|
| `credit_reporting` | 2,510,907 | 65.45% |
| `debt_collection` | 440,886 | 11.49% |
| `cards_prepaid` | 247,005 | 6.44% |
| `deposit_accounts` | 201,575 | 5.25% |
| `mortgage` | 146,267 | 3.81% |
| `money_services` | 122,121 | 3.18% |
| `consumer_lending` | 99,573 | 2.60% |
| `student_loan` | 62,596 | 1.63% |
| **`debt_credit_management`** | **5,437** | **0.14%** |
| `other_financial_services` | 292 | 0.01% |

`other_financial_services` is declared `out_of_scope_rare/abstention`: it stays
visible in every audit count but is not a supervised class.

The critical class is the second-to-last row. A 1-in-700 class with 946
positive groups in the training window is the entire difficulty of this
project, and no amount of representation engineering removed it.

The taxonomy is versioned as `cfpb-product-family-v1.0.0` and the mapper is
pure: strict mode raises on an unknown label, audit mode returns
`mapping_status=unmapped`, and no unknown label ever receives a silent
fallback family.

## Temporal Protocol and Sealed Partitions

The protocol is `post_2023_taxonomy`, selected in S2 as the only candidate
that keeps all nine families supervised. All bounds are inclusive.

| Partition | Window | Status |
|---|---|---|
| `train` | 2023-08-01 to 2024-06-30 | development |
| `validation` | 2024-07-01 to 2024-12-31 | development, fully consumed by V2.1 |
| `test` | 2025-01-01 to 2025-06-30 | **consumed** by S8, forbidden to V2 |
| `stress` | 2025-07-01 to 2025-12-31 | **consumed** by V2.1-C |
| `monitor` | 2026-01-01 to 2026-12-31 | **sealed**, never opened |

Group identity is the pair `(normalized_group_hash, normalized_length)`, where
normalization is lowercase, trim, and whitespace collapse. One normalized
fingerprint is never split across training and evaluation.

Two views are computed and reported separately:

- **Scientific (primary).** Modeled families only; groups not seen before the
  window starts; groups carrying a single label; one representative per group,
  the lowest `Complaint ID`. Groups with more than one in-scope family are
  labeled `label_ambiguous`, counted, and excluded — conflicts are never
  resolved by silently taking the smallest ID.
- **Operational (secondary).** All rows of those same clean groups, scored in
  batches. Published separately, and **structurally unable to change any
  decision**.

## Sealed Data Boundary

The boundary is enforced in code, not in prose.

- Before the unlock token is read, the runner validates only configs, hashes,
  metadata, and the frozen model manifests. A protocol violation is detected
  without touching sealed data.
- The token is read from an environment variable and pinned by SHA-256 in the
  frozen protocol. The plaintext appears in no notebook and in no repository
  file. `run_mode` defaults to `disabled`.
- Development code contains **no unlock path at all** for `test` or `monitor`.
  This is asserted by the test suite.
- After authorization, DuckDB runs with a 4 GB limit, one thread, and batches
  of 4,096. The raw Parquet is joined by `Complaint ID` only for the current
  narrative, rejecting empties.
- Nothing individual is persisted: no narratives, identifiers, per-row scores,
  per-row margins, or sealed-partition cache. Aggregates only.
- Results are written atomically. The public manifest is created only after
  completion, so an interrupted run publishes no partial metrics.

The Kaggle upload bundle is checked by dedicated tests to contain no stress
protocol, no stress results, no path containing `stress`, and no reference to
the unlock variable.

## Stage Workflow

Thirteen notebook pairs, twenty-six notebooks. PT-BR is the canonical edition;
EN-US is editorially equivalent and shares byte-identical code cells. All
logic lives in `src/`; notebooks orchestrate and report, and default to
`RUN_MODE = disabled`, reading persisted evidence when available.

| Stage | PT-BR | EN-US | Evidence role |
|---|---|---|---|
| 01 - Data inspection | [PT-BR](notebooks/pt-BR/01_Data_Inspection_PT.ipynb) | [EN-US](notebooks/en-US/01_Data_Inspection_EN.ipynb) | Sampled smoke evidence |
| 02 - S0 audit | [PT-BR](notebooks/pt-BR/02_S0_Audit_PT.ipynb) | [EN-US](notebooks/en-US/02_S0_Audit_EN.ipynb) | Full-corpus profile |
| 03 - S1 taxonomy and dedup | [PT-BR](notebooks/pt-BR/03_S1_Taxonomy_Dedup_PT.ipynb) | [EN-US](notebooks/en-US/03_S1_Taxonomy_Dedup_EN.ipynb) | Taxonomy and leakage policy |
| 04 - S2 temporal protocol | [PT-BR](notebooks/pt-BR/04_S2_Temporal_Protocol_PT.ipynb) | [EN-US](notebooks/en-US/04_S2_Temporal_Protocol_EN.ipynb) | Protocol candidates |
| 05 - S3 baseline and learning curve | [PT-BR](notebooks/pt-BR/05_S3_Baseline_Learning_Curve_PT.ipynb) | [EN-US](notebooks/en-US/05_S3_Baseline_Learning_Curve_EN.ipynb) | Frozen baseline |
| 06 - S4 error and representation | [PT-BR](notebooks/pt-BR/06_S4_Error_Representation_Challenge_PT.ipynb) | [EN-US](notebooks/en-US/06_S4_Error_Representation_Challenge_EN.ipynb) | Representation challenge |
| 07 - S5 estimator benchmark | [PT-BR](notebooks/pt-BR/07_S5_Estimator_Benchmark_PT.ipynb) | [EN-US](notebooks/en-US/07_S5_Estimator_Benchmark_EN.ipynb) | Estimator isolation |
| 08 - S6 calibrated classical | [PT-BR](notebooks/pt-BR/08_S6_Calibrated_Classical_Challenge_PT.ipynb) | [EN-US](notebooks/en-US/08_S6_Calibrated_Classical_Challenge_EN.ipynb) | Final classical round |
| 09 - S7 frozen package | [PT-BR](notebooks/pt-BR/09_S7_Frozen_Model_Package_PT.ipynb) | [EN-US](notebooks/en-US/09_S7_Frozen_Model_Package_EN.ipynb) | Freeze and package |
| 10 - S8 confirmatory | [PT-BR](notebooks/pt-BR/10_S8_Confirmatory_Evaluation_PT.ipynb) | [EN-US](notebooks/en-US/10_S8_Confirmatory_Evaluation_EN.ipynb) | **Confirmatory, `test`** |
| 11 - V2 Kaggle import | [PT-BR](notebooks/pt-BR/11_V2_Kaggle_Import_PT.ipynb) | [EN-US](notebooks/en-US/11_V2_Kaggle_Import_EN.ipynb) | D1 and D2 evidence import |
| 12 - V2 frozen package | [PT-BR](notebooks/pt-BR/12_V2_Frozen_Package_PT.ipynb) | [EN-US](notebooks/en-US/12_V2_Frozen_Package_EN.ipynb) | Selection and freeze |
| 13 - V2 stress confirmatory | [PT-BR](notebooks/pt-BR/13_V2_Stress_Confirmatory_PT.ipynb) | [EN-US](notebooks/en-US/13_V2_Stress_Confirmatory_EN.ipynb) | **Confirmatory, `stress`** |

## Cycle One: Classical Development to S8

The gates were frozen in S4, on 2026-08-16, before V2 existed, and were never
adjusted: macro-F1 ≥ 0.6900, critical-class F1 ≥ 0.2715, critical-class
precision ≥ 0.2000.

| Stage | What varied | Outcome |
|---|---|---|
| S3 | Word TF-IDF + `SGDClassifier(log_loss)` | Macro-F1 0.7004, critical F1 0.2458. Learning curve flat — more volume did not help the critical class |
| S4 | Representation and weighting (word vs `char_wb`, balanced vs sqrt-balanced) | `NO_ELIGIBLE_CHALLENGER`; no candidate passed all three gates |
| S5 | Estimator only, representation held fixed | `LinearSVC` closest; still no 3/3 |
| S6 | Calibration, with an internal fit/calibration split inside `train` | `LinearSVC` + critical-margin threshold |
| S7 | Freeze and package | Threshold 0.1135351095114484; calibration macro-F1 0.721989, critical F1 0.292994; 3/3 → `packaged_for_confirmation` |
| S8 | **Open `test` once** | `NOT_CONFIRMED`, 2/3 |

S8, on 334,230 representatives: macro-F1 0.718214 (pass), critical precision
0.404615 (pass), critical F1 0.257843 against 0.2715 (**fail**), critical
recall 0.189209. Diagnostic 95% bootstrap on critical F1: [0.233153,
0.282263].

S7's calibration used `validation` in full, so the manifest carries
`validation_independence: NOT_INDEPENDENT_EVIDENCE_AFTER_FINAL_CALIBRATION`.
The S7 numbers are calibration evidence; S8 is the independent measurement.

## Cycle Two: The V2 Hierarchical Detector

V2 isolates the intervention on the class that failed. A binary stage-A
detector overrides the frozen S7 multiclass output when its critical margin
reaches a calibrated threshold; otherwise S7 stands. Stage B is referenced by
hash, never duplicated.

### The D1 null, and the bug it exposed

The first D1 run completed 30 candidates on Kaggle and produced a
scientifically degenerate result: all 30 candidates returned **identical**
metrics, in both windows, despite 30 distinct thresholds.

Recomputing pure S7 through the runner's own code paths reproduced those
metrics cell for cell in both windows. The hierarchical system had been
identical to the fallback: stage A never produced an effective override, and
the declared winner was untouched S7.

The cause was the window, not the search. ADR-010 placed `inner_calibration`
at `train` 2024-05 to 2024-06 — **a proper subset of the frozen S7's own fit
scope**. In-sample, S7 reaches critical F1 0.895787 and recall 0.939535.
Out-of-sample it falls to 0.292994 and 0.232063. Against a fallback that
recovers 94% of the critical class from memory, no override can improve F1,
and the zero-override sentinel wins by construction. The experiment could not
falsify its own hypothesis.

ADR-011 moved exactly one variable — the temporal placement of the three
windows — and added a protocol invariant that is now validated in code: **no
calibration or evaluation window may intersect the fit scope of the frozen
fallback.** Under that invariant, the ADR-010 windows are rejected before any
execution. Two further eligibility rules were added, both restrictive: a
candidate with zero effective overrides is the fallback and is not selectable,
and a candidate that does not strictly beat the pure-fallback baseline on the
external window is not selectable either.

The cost was accepted explicitly: the external window shrank from 245,980 to
127,706 rows and from 892 to 545 critical cases. A noisier estimator on an
experiment that can fail is worth more than a precise one on an experiment
that cannot.

### D1 under V2.1, and the D2 transformer challenge

Under the corrected windows, 30 candidates ran again. All 30 passed the
development margins, 22 had effective overrides, and the same 22 beat the
fallback. The winner was
`word_char_tfidf_union_40000_60000_c_1_hard_negative`: external critical F1
0.386899 against the fallback's 0.339665, a **+0.047234** gain from 82
effective overrides in 258 positive decisions, with precision essentially
unchanged (0.437500 against 0.434286).

D2 then challenged that incumbent with a `distilbert-base-uncased` fine-tuned
as a binary detector, under a controlled design: identical fit scope, identical
hard-negative pool, identical calibration and evaluation windows, identical
threshold search, identical combination rule. Only the source of the score
changed — logit difference in place of `decision_function`.

The displacement bar was pre-registered before execution and derived, not
chosen: a parametric bootstrap over the incumbent's external confusion matrix
gives a standard deviation of 0.019347 for critical F1, so two standard
deviations is 0.0387. The transformer had to reach critical F1 ≥ 0.425599 and
critical precision ≥ 0.434286.

Three seeds were run, with the **median** pre-declared as the reported result;
reporting the best seed was prohibited. The transformer beat the fallback
comfortably and beat the incumbent by +0.015825, but landed below both bars.
Outcome: `CLASSICAL_WINNER_STANDS`. No seed would have passed, which makes the
result independent of the aggregation rule.

### Freeze, and the exact-reproduction gate

D1 persisted no fitted models, so freezing required refitting — and refitting
creates the possibility that the frozen artifact differs from the one that
produced the published numbers.

The freeze was therefore conditioned on an **exact** reproduction gate: the
threshold, both complete confusion matrices, both pairs of override counts,
and the hard-negative pool counts had to reproduce D1 with no tolerance.
Comparison by equality rather than by numerical closeness, because the question
is not whether two numbers are near each other — it is whether the frozen
object is the measured object.

The gate passed 21 of 21 checks. Outcome: `PACKAGE_FROZEN`.

The gate reproduced D1 behavior exactly, but it cannot prove row-level pool
identity retroactively because D1 did not persist a pool signature. The freeze
run matched the local D2 rehearsal and differed from the signature published
by D2 on the GPU kernel. Counts are identical (946 positives and 14,190
negatives), while row identity across all runs remains unproven. This corrects
the stronger line-by-line claim in ADR-012. It does not change D2's decision:
no seed passed, and the missed margins were materially larger than the observed
reproduction differences.

## Decision Boundary

`deployment_authorized: false`, in every protocol, at every stage. Neither
cycle authorizes serving either model.

Permitted:

- Reading the published aggregates as an independent measurement of a frozen
  package on a window it had never seen.
- Reusing the protocol design — sealed partitions, paired control arms, exact
  reproduction gates, pre-registered decision rules.
- Designing a different intervention for the critical class, on partitions not
  yet consumed.

Prohibited:

- Reopening either verdict. The V2 critical-F1 bootstrap interval [0.232937,
  0.288567] contains the 0.2715 bar, and this is recorded as **diagnostic
  only**. The pre-registered rule is on the point estimate. Using the interval
  to reopen the verdict is exactly the post-hoc move the protocol exists to
  prevent.
- A V2.2 on `stress`. ADR-014 pre-registered, before the seal was opened, that
  `NOT_CONFIRMED` closes the cycle. `stress` is now consumed.
- Opening `monitor` 2026. It stays sealed and diagnostic, reserved for later
  monitoring, and the ADR-014 amendment does not reach it.
- Treating any development number as performance. Every margin was measured on
  the same window that served as the selection surface.

## Software Architecture

| Path | Responsibility |
|---|---|
| `src/consumer_complaint_intelligence/audit.py`, `data.py`, `taxonomy.py`, `deduplication.py` | S0/S1 corpus audit, family taxonomy, three levels of duplicate detection |
| `src/consumer_complaint_intelligence/temporal_split.py` | S2 protocol candidates, group identity, partition support policy |
| `src/consumer_complaint_intelligence/s3.py` to `s8.py` (+ `_reporting`) | One module per stage, runner plus text report |
| `src/consumer_complaint_intelligence/v2_protocol.py`, `v2_detector.py`, `v2_benchmark.py` | V2 contract, stage-A detector, 30-candidate benchmark |
| `src/consumer_complaint_intelligence/v2_transformer.py` | D2 DistilBERT challenge |
| `src/consumer_complaint_intelligence/v2_package.py` | Selection, exact-reproduction gate, freeze |
| `src/consumer_complaint_intelligence/v2_stress.py` | V2.1-C two-arm confirmatory runner |
| `src/consumer_complaint_intelligence/v2_import.py` | Reproducible import and rendering of Kaggle evidence |
| `src/consumer_complaint_intelligence/kaggle_execution.py` | Bundle assembly for remote execution |
| `src/consumer_complaint_intelligence/contracts.py`, `service.py`, `tracking.py` | `Predictor` / `PredictionBatch` / `ArtifactManifest`, serving surface, optional tracker |
| `config/*.json` | Frozen protocols and public result manifests with hashes |
| `docs/ADR-001` to `ADR-014` | Every decision, with its rationale and its cost |
| `tests/` | 348 tests across 28 test modules |

Three architectural boundaries hold across the whole project:

- The scikit-learn pipeline knows nothing about Flask, HTTP, JSON, or MLflow.
  The domain exposes `Predictor`, `PredictionBatch`, and `ArtifactManifest`; a
  future adapter translates to those contracts.
- MLflow lives in orchestration only. `NullTracker` is the default and MLflow
  is imported only when explicitly selected.
- Notebooks contain no analytical logic. Everything computational is in `src/`
  and covered by tests.

The `score` field means `critical_margin` and nothing else. It is not a
probability and not a confidence. Estimator input is `en-US` only, even though
the API and documentation are bilingual.

## Execution Environments

Local execution is a 16 GB Windows machine. Some stages did not fit.

| Run | Environment | Duration |
|---|---|---:|
| S4 representation challenge | Local | 2,193 s, 7.24 GB peak RSS |
| S7 final calibration | Local | 284.3 s, 3.71 GB peak RSS |
| D1 classical benchmark | Kaggle CPU | 5,972.4 s |
| D2 transformer challenge | Kaggle GPU | ~1 h, 3 seeds |
| V2.1-P freeze | Kaggle CPU | 1,309 s |
| V2.1-C stress opening | **Local** | 2,014.4 s |

Two local D1 attempts were aborted for memory — 10.18 GiB peak RSS with
0.11 GiB left on the system — and are retained as resource evidence. Because
the runner publishes transactionally, neither produced a partial result.

V2.1-C ran locally **by choice**. It is batch inference, so it does not need
the GPU, and running it locally keeps the sealed partition and the 821 MB raw
Parquet on the machine. For a sealed opening that is the better boundary.

## Installation

```bash
mamba env create -f environment.yml
mamba activate consumer-nlp
python -m pip install -e .
python -m ipykernel install --user --name consumer-nlp --display-name "Python (consumer-nlp)"
```

The dataset is not distributed with the repository. The frozen study requires
the exact archived Parquet snapshot; a current CFPB download is not equivalent
and will be rejected by the pinned SHA-256. See
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) for the snapshot identity,
scope of reproducibility, and current source limitations.

The published `.joblib` bundles are Python pickle-based artifacts. Load them
only from a trusted checkout and verify their SHA-256 values against the frozen
protocols before deserialization.

Flask and MLflow are optional extras in `pyproject.toml`, not environment
dependencies.

## Validation

```bash
python -m unittest discover -s tests -t .
```

The suite runs 348 tests covering contracts, taxonomy, deduplication,
temporal split, every stage runner, every result manifest, notebook structure
and PT-BR/EN-US code identity, docstring and line-length conventions, Kaggle
bundle assembly, and the seal integrity assertions described above.

For review, treat `config/s8_results.json` and `config/v2_stress_results.json`
as frozen evidence. Re-running the reporting notebooks is supported. Opening a
new sealed partition would be a different study, and `monitor` is the only one
left.

## Sources

This project cites regulatory and documentary sources rather than a
methodological literature; its decisions are recorded in `docs/ADR-001`
through `ADR-014`.

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [CFPB field reference](https://cfpb.github.io/api/ccdb/fields.html)
- [August 2023 product and issue options](https://files.consumerfinance.gov/f/documents/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.pdf)
- [CFPB notice of June 24, 2026](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-is-correcting-flaws-to-restore-integrity-and-utility-to-the-consumer-complaint-system/)
- [CFPB notice of August 14, 2026](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/)

The June 2026 notice reports exceptional growth in credit-reporting complaints
and states that the portal data cannot be relied on as a reliable reflection of
market conditions without addressing the identified factors. It is treated as a
review signal, not as an automatic rule to exclude 2025 or 2026 data. Any
exclusion would be an explicit, documented decision.

## Author and Responsibility

**Author:** Enio Rubens
**Role:** Data Science and Analytics

AI coding assistants supported modularization, test scaffolding, documentation,
translation, and review. Protocol design, gate selection, acceptance criteria,
result interpretation, and every decision to open a sealed partition remained
human-led. All published claims are the author's responsibility.

## Data Attribution

The CFPB Consumer Complaint Database is published by the Consumer Financial
Protection Bureau. The raw snapshot is not redistributed here; ownership and
any original distribution terms remain with the source.

The MIT license applies to the project software and documentation, not to the
CFPB dataset or to third-party model and library licenses.
