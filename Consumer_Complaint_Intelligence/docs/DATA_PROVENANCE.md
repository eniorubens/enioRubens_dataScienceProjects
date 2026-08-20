# Data provenance and reproducibility

## Frozen snapshot

The confirmatory study used one local CFPB Consumer Complaint Database
snapshot. The raw narratives and identifiers are not distributed with this
repository.

| Property | Frozen value |
|---|---|
| Local path expected by the protocols | `dataset/processed/complaints.parquet` |
| Parquet size | 821,595,288 bytes |
| Parquet SHA-256 | `33875E0439B72E00CB6E5985238B47F76B5B0A63B0AC21DDB8C7D46B565C428F` |
| Source archive recorded during ingestion | `complaints.csv.zip` |
| Source archive size recorded during ingestion | approximately 1.414 GB |
| Expanded CSV size recorded during ingestion | approximately 9.117 GB |
| Local conversion date | 2026-08-14 |

The acquisition notebook records the archive name and conversion sizes, but no
public immutable URL for that exact archive was captured. Therefore, a current
download from the CFPB must not be represented as the same snapshot. Frozen
protocols reject a different file before reading rows.

## What can be reproduced publicly

The repository distributes aggregate JSON evidence, frozen configuration and
result manifests, tests, and the S7 and V2 inference bundles. These support:

- validation of published aggregate evidence and package hashes;
- execution of tests that do not require the undistributed Parquet index;
- inspection of the complete protocol and decision history;
- inference smoke tests on user-supplied `en-US` text.

The historical S8 result contains an internal run signature created with the
original checkout's absolute path. That field is preserved as evidence and is
not recomputed in another checkout. Portable validation relies on the public
manifest, which pins the protocol and result by relative path, byte size, and
SHA-256.

Some earlier aggregate JSON files also retain the original Windows checkout
path in provenance fields. Those strings contain no username, credential,
identifier, or complaint narrative. Rewriting them would invalidate the frozen
byte hashes, so the evidence files remain unchanged; public-facing notebooks
render project-relative paths instead.

A full historical rerun of S0 through V2.1-C requires both the exact Parquet
snapshot above and `temp/s2/modeling_index.parquet`, whose identity is pinned
as 69,858,991 bytes and SHA-256
`EDBE3C38225DA1B380E5651436FF9ABEE6591BE14A1390A988FC24B2F7D8F1A9`.
Neither file is published because it is data-derived and may expose source
records. The sealed `monitor` partition remains unopened.

## Source status

The CFPB database is a changing source rather than a versioned research
archive. On August 14, 2026, the CFPB announced that it had ceased discretionary
publication of complaint narratives and visualizations and moved previously
published narratives to its FOIA Reading Room. This further limits independent
reconstruction of the exact narrative snapshot.

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [CFPB notice of August 14, 2026](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/)

## Proveniência e reprodutibilidade

O estudo confirmatório usou um único snapshot local do CFPB Consumer Complaint
Database. As narrativas brutas e os identificadores não são distribuídos neste
repositório. O arquivo Parquet congelado tem 821.595.288 bytes e SHA-256
`33875E0439B72E00CB6E5985238B47F76B5B0A63B0AC21DDB8C7D46B565C428F`.

O notebook de aquisição registra `complaints.csv.zip`, aproximadamente 1,414
GB compactado e 9,117 GB após expansão, convertido localmente em 2026-08-14.
Ele não registra uma URL pública imutável para aquele arquivo exato. Portanto,
um download atual do CFPB não deve ser apresentado como o mesmo snapshot.

Os agregados, manifests, configurações, testes e bundles de inferência podem ser
auditados publicamente. Uma reexecução histórica completa exige o Parquet exato
e o índice derivado `temp/s2/modeling_index.parquet`; nenhum dos dois é
publicado. A partição selada `monitor` permanece fechada.
