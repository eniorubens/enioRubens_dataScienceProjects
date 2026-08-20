# ADR-002: Taxonomy and deduplication policy

## Status

Accepted for S1 audit; the temporal split gate remains **BLOCKED**.

## Context

The CFPB Consumer Complaint Database preserves the consumer's original
`Product`, `Sub-product`, `Issue`, and `Sub-issue` selections according to the
form options available when the complaint was submitted. The official page
lists product and issue changes in 2017 and August 2023:

- [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Official field reference](https://cfpb.github.io/api/ccdb/fields.html)
- [August 2023 product and issue options PDF](https://files.consumerfinance.gov/f/documents/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.pdf)

The field reference states that possible `Issue` values depend on `Product`.
Therefore, a global Issue label is not a stable semantic target across the
whole historical corpus. The August 2023 document records changes such as the
split of `Credit card or prepaid card`, the movement of `Credit repair
services` under `Debt or credit management`, and wording changes to credit
reporting issues. These are form-taxonomy changes, not evidence that the
underlying consumer experience changed in the same way.

The CFPB's [June 24, 2026 notice](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-is-correcting-flaws-to-restore-integrity-and-utility-to-the-consumer-complaint-system/)
reports exceptional growth in credit-reporting complaints and identifies
overlapping factors including credit clinics, social-media promotion, AI tools,
and possible misuse of the portal. It states that the portal data cannot be
relied on as a reliable reflection of market conditions or actual consumer
experiences without addressing those issues.

## Decision

### Product taxonomy

S1 keeps every raw `Product` value and adds a versioned `product_family`. The
conservative registry in `taxonomy.py` maps the 21 observed raw labels to
stable families under `cfpb-product-family-v1.0.0`. The mapper is pure:

- strict mode raises for an unknown label;
- audit mode returns `family=None` and `mapping_status=unmapped`;
- no unknown label receives a silent fallback family.

### Issue target

Raw `Issue` is preserved. The secondary hierarchical key is
`product_family + Issue`. S1 does not merge the 178 raw Issue labels. This
avoids treating a label with the same spelling as the same semantic category
when its parent Product, form regime, or available options differ.

The primary project interpretation is routing to historical form categories,
not estimating prevalence of harm. Complaint volume must not be described as
market prevalence or as a reliable measure of consumer harm.

### Deduplication and leakage

The raw dataset is never deleted or rewritten. S1 records three distinct
levels:

1. Unique `Complaint ID` is an integrity check, not text deduplication.
2. `exact_text` uses hash plus text length inside DuckDB.
3. `normalized_text` uses only lowercase, trim, and whitespace collapse.

The normalized fingerprint is the modeling `group_id` candidate and produces
`is_repeated`. One normalized fingerprint must never be split between training
and evaluation. Publication will report two future metrics: operational
`all-text` and leakage-controlled `novel-text/purged`. Near-duplicate templates
outside this normalization remain a later audit and are not automatically
removed in S1.

Hashes reduce memory use but do not create a mathematical guarantee against
collision. Critical groups require a collision-safe follow-up before
publication. Narratives are never materialized in Python by the full-corpus
queries, and top duplicate groups expose only hashes, lengths, dates, and
categorical counts.

## Gate and regime review

The S1 gate remains **BLOCKED** because no split cut date or minimum class
criteria were configured, and S1 does not invent either one. Before S2 can
prepare a split, it must review:

- the 2017 and August 2023 form regimes;
- yearly credit-reporting concentration and narrative coverage;
- the exceptional 2025/2026 volume pattern and its possible process or
  integrity changes;
- class support by year for the selected hierarchical target;
- normalized duplicate groups crossing dates, Products, families, or Issues.

2026 is treated as a partial period in this snapshot and may change as the
database updates. The June 2026 notice is a review signal, not an automatic
rule to exclude 2025 or 2026 data. Any exclusion, censoring, or sensitivity
analysis must be an explicit later decision with evidence.

## Consequences

The project can prepare routing labels and future scikit-learn estimators
without coupling the estimator to Flask or MLflow. Taxonomy mapping and
deduplication policies are versioned and testable before a scientific split is
sealed. The cost is that historical label changes, concentration, and
duplicate groups remain visible in the report and must be discussed in model
evaluation rather than hidden by preprocessing.
