# ADR-015: V3 sentence-embedding development cycle

## Status

Draft for user review. This ADR does not freeze a protocol, promote a model,
authorize deployment, or authorize a Kaggle run.

## Context

The V1/S8 and V2.1 cycles are closed. Their configs, artifacts, notebooks, and
results remain read-only evidence. The `test` and `stress` partitions were
consumed by prior confirmatory work. They are archived and never reusable in
V3, including for diagnostics, model selection, calibration, threshold
selection, or promotion. The `monitor` partition is separate: it remains sealed
and diagnostic-only, and it is still never selectable in V3.

V2 improved the critical-class decision slightly, but it did not close every
gate at confirmation. The next scientific question is whether semantic sentence
embeddings can recover more critical-class signal than sparse lexical features
while preserving a development-only protocol.

This is a scientific V3 because it starts a new research protocol after the
closed V1 and V2 evidence. A later public GitHub release can still be labelled
2.0 for portfolio readability; release labels are software packaging labels,
not scientific experiment versions.

## Decision

Create a bounded V3 E0 draft protocol in
`config/v3_development_protocol.json` and validate it through
`consumer_complaint_intelligence.v3_protocol`.

E0 authorizes only the E1 synthetic model-loading smoke scaffold for design
review. It does not authorize execution of that smoke. It also does not
authorize model training, threshold selection, candidate promotion, packaging,
sealed data access, project dataset loading, or a Kaggle run. Kaggle execution
can occur only after a later user review and explicit execution approval.

The input-language contract is `en-US`. Any future estimator package must state
that contract clearly at the inference boundary.

The candidate catalog is:

- incumbent TF-IDF word plus character features with `sklearn.LinearSVC`;
- primary frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings plus a
  linear probe;
- direct frozen `BAAI/bge-small-en-v1.5` embeddings plus the same linear probe;
- score-level fusion of the incumbent and the best embedding score;
- quality ceiling `sentence-transformers/all-mpnet-base-v2`;
- optional supervised fine-tuning only after frozen embeddings justify it.

The accelerated environment is Kaggle with an NVIDIA T4 GPU only. P100 is
forbidden because the project previously hit an `sm_60` compatibility failure.
The E1 smoke uses synthetic texts and must fail before data staging or model
loading when the GPU is unavailable or not T4.

## Stage Boundary

The planned stages are E0, E1, E2, E3, E4, E5, E6, and P.

E1 is a synthetic/model-loading smoke. E2 builds versioned embedding caches. E3
trains comparable linear probes. E4 evaluates long-text strategies. E5 tests
score fusion. E6 is an optional supervised fine-tune. P packages the selected
estimator behind an explicit inference contract.

Only the E1 design is authorized by the E0 draft. E1 execution is not
authorized. Every later stage is intentionally present as design context but
remains blocked until a separate review decision.

## Long-Text Risk

The protocol records the long-text risk explicitly. MiniLM truncates at 256
wordpieces, while BGE supports 512 tokens. V3 therefore cannot promote a model
before E4 compares long-text handling strategies such as head, tail, head-tail,
and chunk pooling.

The limits and embedding dimensions come from the publishers' model cards:

- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  documents 384-dimensional embeddings and default truncation above 256
  wordpieces;
- [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
  documents 384-dimensional embeddings and a 512-token sequence length;
- [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
  is retained only as the later quality-ceiling candidate.

The implementation follows the publisher-supported
[Sentence Transformers](https://sbert.net/docs/installation.html) inference
interface. E1 records the resolved model revisions and installed package
versions rather than treating a mutable model name as complete provenance.

## Consequences

The V3 work can begin without reopening V1/S8 or V2.1 evidence. The first
implementation surface is small: deterministic JSON validation plus directed
unit tests. No transformer, Torch, network, raw dataset, Kaggle token, consumed
partition, or sealed monitor partition is needed to review E0.

If the user approves E0, the next implementation is the E1 scaffold. That
scaffold must still avoid project data, print progress during long execution,
consume Kaggle credentials without logging secrets, and enforce T4-only
execution before any expensive or stateful step.
