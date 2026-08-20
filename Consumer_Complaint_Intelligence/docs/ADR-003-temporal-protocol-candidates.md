# ADR-003: Candidate temporal protocols for S2

Status: strict gate blocked; pilot sensitivity ready for review; no candidate
is sealed or approved.

## Context

S1 established a versioned product-family taxonomy and showed that normalized
narrative groups cross temporal boundaries. The project therefore needs a
reproducible protocol comparison before any estimator is trained. The CFPB
database preserves the categories available when a complaint was submitted,
and the CFPB field reference describes `Issue` as dependent on `Product`.
The project keeps the source links recorded in S1:

- [CFPB complaint database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [CFPB field reference](https://cfpb.github.io/api/ccdb/fields.html)
- [August 2023 form options](https://files.consumerfinance.gov/f/documents/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.pdf)
- [June 2026 CFPB notice](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-is-correcting-flaws-to-restore-integrity-and-utility-to-the-consumer-complaint-system/)

The 2025 window is retained as a stress period because it tests a recent
operational regime without deciding confirmatory approval. The 2026 window is
monitoring only because it is partial in the downloaded snapshot and has very
different narrative coverage.

## Decision

S2 compares three explicit candidates:

1. `historical_stress`: train 2015-2022, validation 2023, test 2024,
   stress 2025, monitor 2026.
2. `post_2023_taxonomy`: train 2023-08-01 through 2024-06-30, validation the
   remainder of 2024, test 2025-01-01 through 2025-06-30, stress the remainder
   of 2025, monitor 2026.
3. `extended_history`: train 2015-2023, validation 2024, test 2025,
   monitor 2026.

All limits are inclusive, and candidate windows may not overlap. The raw
Parquet is not modified.

## Group-aware support policy

The normalized fingerprint is versioned as lowercase, trim, whitespace
collapse, MD5, and normalized length. Validation and test require at least
500 novel unique normalized groups per modeled family. Training requires at
least 2,000 narrative rows and 1,000 unique normalized groups per family.
`other_financial_services` is explicitly `out_of_scope_rare/abstention`; it
remains visible in counts but is not a supervised class.

For every partition and family, S2 reports all-text rows, unique groups,
seen-before rows, novel rows, novel unique groups, and repeated rows within the
partition. All-text measures remain operational. Scientific metrics must also
use group-weighted aggregation or one deterministic observation per novel
group. The binomial 2.24 percentage-point approximation for 500 observations
is only a rough reference when groups are approximately independent; it is not
a substitute for bootstrap or class-specific confidence intervals.

## Cache and privacy boundary

`temp/s2/modeling_index.parquet` is built with DuckDB `COPY`, a 2 GB memory
limit, two threads, and spill in `temp/duckdb`. It contains no narrative,
normalized text, ZIP code, or other raw fields beyond the declared index
contract. Its metadata is signed by source path, size, modification time,
taxonomy version, fingerprint version, and schema version. Writes are atomic.

The JSON comparison report is incrementally cached against the index signature,
candidate definitions, and criteria. A recommendation is `READY_FOR_REVIEW`
only if at least one candidate passes; otherwise it is `BLOCKED`. Ranking uses
eligible class count, minimum novel unique-group test support, test purge rate,
and then favors `historical_stress`. Neither status is `SEALED` or `APPROVED`.

## Executed sensitivity evidence

The strict cached report is `temp/s2/s2_report.json`. It is `BLOCKED` under the
2,000 training-row and 1,000 training-group criteria. This remains the
scientific gate and was not relaxed retroactively.

The exploratory pilot cached report is `temp/s2/s2_report_pilot.json`. It uses
750 training rows and 750 training groups, while retaining 500 novel groups in
validation and test. It is `READY_FOR_REVIEW`; `post_2023_taxonomy` is `PASS`
with 9 eligible classes, a minimum of 1,398 novel test groups, and a test purge
of 18.57695%. The `historical_stress` and `extended_history` candidates remain
`FAIL` with 8 eligible classes.

For `debt_credit_management` in the pilot candidate, the observed support is
968 training rows and 959 training groups, 892 novel validation groups, and
1,398 novel test groups. The class clears the exploratory threshold of 750,
but S3 must run a learning curve before any claim about scientific sufficiency.

The pilot threshold is a data-informed sensitivity choice made after observing
the strict result. That creates data-driven-threshold risk: it may make a
candidate look viable because the threshold was selected in response to the
observed corpus. Pilot results therefore cannot be reported as confirmatory
evidence. Any approval must occur before model training, be explicitly reviewed
and documented, and then be frozen for the complete scientific evaluation.

## Consequences

This decision prevents future normalized groups from entering earlier
partitions and makes the template-repetition problem visible. It also accepts
that a candidate may fail because the taxonomy or support is insufficient. The
pilot scenario provides a transparent sensitivity analysis without changing
the strict gate. The next stage may choose a candidate explicitly and define
model training and group-aware evaluation; S2 itself does neither.
