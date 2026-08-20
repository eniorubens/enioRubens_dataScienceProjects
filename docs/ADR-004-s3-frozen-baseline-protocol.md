# ADR-004 - S3 baseline e curva de aprendizado congelados

## Status

Aceito em 2026-08-15 para desenvolvimento S3. O protocolo está congelado
antes de qualquer ajuste de modelo. A configuração canônica está em
`config/s3_protocol.json`.

## Contexto

O S2 exploratório mostrou que `post_2023_taxonomy` é o único candidato piloto
com as nove famílias supervisionadas. A aprovação explícita permite iniciar o
desenvolvimento, mas não transforma o resultado piloto em evidência
confirmatória. O índice S2 continua sendo somente hash, comprimento e metadados
de particionamento; a narrativa deve ser juntada sob uma fronteira controlada.

## Decisão

S3 usa as janelas inclusivas abaixo:

| Partição | Janela | Uso em S3 |
| --- | --- | --- |
| train | 2023-08-01 a 2024-06-30 | ajuste e curva |
| validation | 2024-07-01 a 2024-12-31 | seleção e avaliação científica |
| test | 2025-01-01 a 2025-06-30 | selada |
| stress | 2025-07-01 a 2025-12-31 | selada |
| monitor | 2026-01-01 a 2026-12-31 | selada |

As nove famílias de `MODELED_FAMILIES` são as classes supervisionadas.
`other_financial_services` permanece visível nos dados de auditoria, mas é
`out_of_scope_rare/abstention` e não entra no treino.

A identidade de grupo é sempre o par
`(normalized_group_hash, normalized_length)`. Antes de escolher qualquer
representante, S3 conta as famílias em escopo por grupo em todo o conjunto de
desenvolvimento. Se houver mais de uma família, o grupo recebe o rótulo técnico
`label_ambiguous`, é contado no relatório e é excluído do treino científico e
da avaliação científica. Não se resolve conflito escolhendo silenciosamente o
menor `Complaint ID`.

Para cada grupo com rótulo único, a materialização SQL seleciona o menor
`Complaint ID` dentro da própria partição. Grupos de treino sempre têm
precedência; somente grupos que não aparecem em treino podem fornecer uma linha
de validação científica. O resumo auditável usa exatamente essa mesma regra.

A learning curve usa somente índices de linhas da partição `train`, de forma
determinística e estratificada dentro de cada classe. Cada ponto deve conter as
nove classes. A visão all-text é operacional e permanece em
`development.parquet`; ela só é avaliada no ponto final, em batches, com
acumulador de métricas.

O baseline é `DummyClassifier` e TF-IDF com
`SGDClassifier(loss="log_loss")`. Este último é uma regressão logística linear
otimizada por SGD, escolhida para controlar memória. O limite de vocabulário,
`float32`, as frações da curva, o tamanho do batch e o orçamento de memória são
parâmetros do executor. O tracker é opcional; MLflow não é dependência
obrigatória.

Os defaults locais do caminho full refletem uma medição no computador de
desenvolvimento: `memory_limit="2GB"` e dois threads falharam por falta de
memória; `memory_limit="4GB"` com um thread concluiu em 15,8 s, com pico de
RSS de 2,65 GB. Para uma máquina com 16 GB de RAM, o batch padrão foi reduzido
de 8192 para 4096: a estimativa de 8192 era 7,03 GB, acima do orçamento de
7,0 GB, enquanto 4096 mantém a operação dentro da margem observada.

O caminho full é `run_s3_full(...)`. Ele cria ou reutiliza um cache científico
Parquet com DuckDB, carrega somente essa visão como tabela Arrow e não converte
o desenvolvimento inteiro para `list[dict]`. A guarda padrão de memória é uma
estimativa determinística de 7 GB, sem `psutil`; pode ser ajustada ou desligada
explicitamente com `memory_budget_gb=None`. O limite de 250.000 linhas e a
leitura em listas permanecem somente no caminho smoke legado.

## Fronteira de dados

O executor S3 materializa em cache somente `train` e `validation`, juntando o
índice S2 ao Parquet original por `Complaint ID`. A API rejeita explicitamente
qualquer pedido de narrativa para `test`, `stress` ou `monitor`; essas
partições não entram em treino, seleção ou métricas nesta etapa.

Os notebooks PT-BR e EN-US expõem `RUN_MODE` com os valores `disabled`, `smoke`
e `full`, com padrão `disabled`. Em `disabled`, eles apenas leem a evidência
persistida, quando disponível.

## Executed development evidence

A execução real do baseline está registrada em
[`config/s3_baseline.json`](../config/s3_baseline.json), a partir de
`temp/s3/s3_full.json`. Foram usados 345.552 grupos científicos de `train`,
245.980 de `validation` e 591.532 linhas no cache. A curva obteve macro-F1
científico de 0,7017, 0,6996 e 0,7004 nos pontos 0,25, 0,5 e 1,0; o F1 de
`debt_credit_management` caiu de 0,2715 para 0,2458. O baseline superou o
dummy (macro-F1 0,0892), enquanto a métrica operacional final foi 0,6784.
Esses números são validação de desenvolvimento: a curva é plana, a classe
crítica não melhora com mais volume, a visão operacional é separada e nenhuma
partição selada foi aberta. O resultado não é confirmatório nem deploy-ready.

## Consequências

O treino científico fica menor e menos influenciado por templates repetidos,
ao custo de não representar a frequência operacional dos textos. Essa perda é
intencional e fica visível pela comparação com a visão all-text. A aprovação
S3 congela o desenho do experimento, não um vencedor nem uma autorização de
deploy.
