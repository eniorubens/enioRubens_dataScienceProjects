# Consumer Complaint Intelligence

**Idioma:** [English](README.md) | Português (Brasil)

Este projeto roteia narrativas livres de reclamações de consumidores do CFPB
Consumer Complaint Database para nove famílias de produto, e estuda **o que é
preciso para confirmar honestamente que um classificador de texto funciona
sobre dados que ele nunca viu**.

O alvo de modelagem é deliberadamente hostil. A classe crítica,
`debt_credit_management`, corresponde a **0,14% das narrativas** — 5.437 linhas
em 3,8 milhões. Cada etapa do projeto se organiza em torno de uma pergunta: o
ganho medido sobrevive a uma avaliação desenhada antes de o número existir?

Dois ciclos completos responderam a essa pergunta. Os dois retornaram
`NOT_CONFIRMED`. A entrega é o protocolo e a evidência, não um modelo
implantável.

## Resultado executivo

Duas partições seladas foram abertas, cada uma exatamente uma vez, cada uma sob
um protocolo congelado e registrado antes de o dado ser lido.

| Ciclo | Janela selada | Gates | Status |
|---|---|---|---|
| **S8** (V1, classificador S7 congelado) | `test` 2025-H1 | 2 de 3 | `NOT_CONFIRMED` |
| **V2.1-C** (V2 hierárquico sobre o S7) | `stress` 2025-H2 | 3 de 4 | `NOT_CONFIRMED` |

A abertura do V2.1-C, sobre 269.915 representantes de grupos limpos e inéditos:

| Gate | Barra | Observado | Resultado |
|---|---:|---:|---|
| Macro-F1 | ≥ 0,6900 | 0,710748 | passa |
| Precisão da classe crítica | ≥ 0,2000 | 0,426070 | passa |
| Ganho pareado de F1 crítico sobre o controle | > 0 (estrito) | +0,006455 | passa |
| **F1 da classe crítica** | **≥ 0,2715** | **0,260404** | **falha** |

`deploy=false` nos dois ciclos. A confirmação nunca foi definida como
autorização de deployment.

O gate pareado é aquele que o ciclo V2 existia para responder, e ele passou: o
modelo hierárquico superou seu próprio controle congelado nas mesmas linhas.
Sob o procedimento diagnóstico de reamostragem pré-especificado, o intervalo
bootstrap de 95% foi [0,000235; 0,013661] e excluiu o zero. O ganho observado
também é **um sétimo** dos +0,047234 medidos em desenvolvimento e pequeno
demais para alcançar uma barra absoluta fixada em agosto de 2026, antes de o
V2 existir.

## Por que o ganho de desenvolvimento não se sustentou

O mecanismo está nas contagens de disparo, não no modelo.

| Quantidade | Desenvolvimento (2024-Q4) | Stress (2025-H2) |
|---|---:|---:|
| Decisões positivas do estágio A | 258 em 127.706 linhas (0,202%) | 224 em 269.915 linhas (0,083%) |
| Sobrescritas efetivas | 82 | 36 |
| Ganho de F1 crítico sobre o S7 | +0,047234 | +0,006455 |

O detector do estágio A foi ajustado entre 2023-08 e 2024-06 e calibrado entre
2024-07 e 2024-09. A janela de `stress` ocorre de 12 a 18 meses após o fim do
ajuste e de 9 a 15 meses após o fim da calibração. Nela, suas margens alcançam
o limiar congelado a 41% da taxa de desenvolvimento. Das 36 sobrescritas
efetivas, 10 estavam certas e 26 erradas — taxa de acerto de 27,8% contra uma
precisão de 42,6% da classe crítica na mesma janela. É por isso que a precisão
caiu enquanto o recall subiu, e por isso que o saldo em F1 foi pequeno.

## O braço de controle, e o que ele descartou

Macro-F1 é média não ponderada sobre nove classes, então uma mudança de
composição move a métrica mesmo para um modelo idêntico. O mesmo S7 congelado
marcou F1 crítico 0,339665 em `validation` 2024-H2 e 0,257843 em `test`
2025-H1 — uma diferença de 0,081822 sem alterar uma linha do modelo. Métricas
absolutas não são comparáveis entre janelas.

Por isso o V2.1-C pontuou dois braços numa **única passagem** sobre as mesmas
linhas seladas: o braço primário (V2 hierárquico) e um braço de controle (S7
congelado sozinho). Os dois braços caem num único acumulador conjunto 9×9×9
indexado por `[verdade, rótulo_v2, rótulo_s7]`; marginalizar um eixo recupera a
matriz de confusão de cada braço, e o bootstrap reamostra as células conjuntas,
de modo que a correlação entre os braços é preservada.

O controle se pagou. O S7 congelado marcou 0,253949 em `stress` 2025-H2 contra
0,257843 em `test` 2025-H1. A proximidade do F1 da classe crítica enfraquece uma
explicação simples de que "a janela de stress era mais difícil", mas não é um
teste formal de equivalência e não elimina outros efeitos de janela.

Ele também permite uma comparação limpa: o V2 em `stress` (0,260404) supera o
S7 em `test` (0,257843) por 0,002561. Todo o ciclo V2 moveu a métrica que
motivou sua existência em cerca de dois milésimos e meio, contra uma barra que
exigia mais treze e meio.

## Pergunta de negócio

A base do CFPB preserva as opções de `Product` e `Issue` que existiam quando
cada reclamação foi registrada. Rotear uma narrativa para a família correta é a
tarefa operacional; a classe crítica é aquela em que a falha de roteamento é
cara e em que toda abordagem clássica deste projeto falhou.

Duas leituras são recusadas ao longo de todo o trabalho, e a recusa é imposta
pela documentação em vez de ficar por conta do leitor:

- Volume de reclamações **não** é prevalência de mercado e **não** é medida de
  dano ao consumidor. O alvo é roteamento para categorias históricas do
  formulário.
- `Issue` não é um rótulo global estável. A referência de campos declara que os
  valores possíveis de `Issue` dependem de `Product`, e 178 rótulos brutos
  colidem entre famílias — só `Incorrect information on your report` aparece em
  seis delas.

## Dataset

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
baixado como um snapshot Parquet de 821 MB.

| Quantidade | Valor |
|---|---:|
| Linhas totais | 17.094.898 |
| Linhas com narrativa | 3.836.659 (22,4%) |
| Intervalo de anos | 2011 a 2026 (2026 parcial) |
| Rótulos brutos de `Product` | 21 |
| Famílias modeladas | 9 |
| Rótulos brutos distintos de `Issue` | 178 |

Linhas com narrativa por família, e o formato do problema:

| Família | Linhas com narrativa | Participação |
|---|---:|---:|
| `credit_reporting` | 2.510.907 | 65,45% |
| `debt_collection` | 440.886 | 11,49% |
| `cards_prepaid` | 247.005 | 6,44% |
| `deposit_accounts` | 201.575 | 5,25% |
| `mortgage` | 146.267 | 3,81% |
| `money_services` | 122.121 | 3,18% |
| `consumer_lending` | 99.573 | 2,60% |
| `student_loan` | 62.596 | 1,63% |
| **`debt_credit_management`** | **5.437** | **0,14%** |
| `other_financial_services` | 292 | 0,01% |

`other_financial_services` é declarada `out_of_scope_rare/abstention`:
permanece visível em toda contagem de auditoria, mas não é classe
supervisionada.

A classe crítica é a penúltima linha. Uma classe de 1 em 700, com 946 grupos
positivos na janela de treino, é toda a dificuldade deste projeto, e nenhuma
engenharia de representação a removeu.

A taxonomia é versionada como `cfpb-product-family-v1.0.0` e o mapeador é puro:
o modo estrito levanta erro para rótulo desconhecido, o modo de auditoria
retorna `mapping_status=unmapped`, e nenhum rótulo desconhecido recebe família
por fallback silencioso.

## Protocolo temporal e partições seladas

O protocolo é `post_2023_taxonomy`, escolhido no S2 como o único candidato que
mantém as nove famílias supervisionadas. Todos os limites são inclusivos.

| Partição | Janela | Situação |
|---|---|---|
| `train` | 2023-08-01 a 2024-06-30 | desenvolvimento |
| `validation` | 2024-07-01 a 2024-12-31 | desenvolvimento, integralmente consumida pelo V2.1 |
| `test` | 2025-01-01 a 2025-06-30 | **consumida** pelo S8, proibida ao V2 |
| `stress` | 2025-07-01 a 2025-12-31 | **consumida** pelo V2.1-C |
| `monitor` | 2026-01-01 a 2026-12-31 | **selada**, nunca aberta |

A identidade de grupo é o par `(normalized_group_hash, normalized_length)`, com
normalização por minúsculas, trim e colapso de espaços. Uma mesma impressão
digital normalizada nunca é dividida entre treino e avaliação.

Duas visões são calculadas e reportadas em separado:

- **Científica (primária).** Somente famílias modeladas; grupos não vistos
  antes do início da janela; grupos com rótulo único; um representante por
  grupo, o de menor `Complaint ID`. Grupos com mais de uma família em escopo
  recebem `label_ambiguous`, são contados e excluídos — conflito nunca é
  resolvido escolhendo silenciosamente o menor ID.
- **Operacional (secundária).** Todas as linhas desses mesmos grupos limpos,
  pontuadas em batches. Publicada separadamente e **estruturalmente incapaz de
  alterar qualquer decisão**.

## Fronteira do dado selado

A fronteira é imposta em código, não em prosa.

- Antes de o token ser lido, o runner valida apenas configurações, hashes,
  metadados e os manifestos dos modelos congelados. Uma violação de protocolo é
  detectada sem tocar no dado selado.
- O token é lido de variável de ambiente e fixado por SHA-256 no protocolo
  congelado. O plaintext não aparece em nenhum notebook nem em nenhum arquivo
  do repositório. `run_mode` tem padrão `disabled`.
- O código de desenvolvimento **não contém caminho de destravamento** para
  `test` nem para `monitor`. Isso é asseverado pela suíte de testes.
- Depois da autorização, o DuckDB roda com limite de 4 GB, uma thread e batches
  de 4.096. O Parquet bruto é unido por `Complaint ID` somente para a narrativa
  corrente, rejeitando vazios.
- Nada individual é persistido: nem narrativas, nem identificadores, nem scores
  por linha, nem margens por linha, nem cache de partição selada. Só agregados.
- Os resultados são escritos atomicamente. O manifesto público só é criado após
  a conclusão, de modo que uma execução interrompida não publica métrica
  parcial.

O bundle de upload para o Kaggle é verificado por testes dedicados: não contém
protocolo de stress, não contém resultados de stress, não contém caminho com
`stress` e não referencia a variável de destravamento.

## Fluxo de etapas

Treze pares de notebooks, vinte e seis notebooks. PT-BR é a edição canônica;
EN-US é editorialmente equivalente e compartilha células de código idênticas
byte a byte. Toda a lógica vive em `src/`; os notebooks orquestram e reportam,
e têm padrão `RUN_MODE = disabled`, lendo a evidência persistida quando
disponível.

| Etapa | PT-BR | EN-US | Papel na evidência |
|---|---|---|---|
| 01 - Inspeção de dados | [PT-BR](notebooks/pt-BR/01_Data_Inspection_PT.ipynb) | [EN-US](notebooks/en-US/01_Data_Inspection_EN.ipynb) | Evidência amostral |
| 02 - Auditoria S0 | [PT-BR](notebooks/pt-BR/02_S0_Audit_PT.ipynb) | [EN-US](notebooks/en-US/02_S0_Audit_EN.ipynb) | Perfil do corpus completo |
| 03 - S1 taxonomia e dedup | [PT-BR](notebooks/pt-BR/03_S1_Taxonomy_Dedup_PT.ipynb) | [EN-US](notebooks/en-US/03_S1_Taxonomy_Dedup_EN.ipynb) | Taxonomia e política de vazamento |
| 04 - S2 protocolo temporal | [PT-BR](notebooks/pt-BR/04_S2_Temporal_Protocol_PT.ipynb) | [EN-US](notebooks/en-US/04_S2_Temporal_Protocol_EN.ipynb) | Candidatos de protocolo |
| 05 - S3 baseline e curva | [PT-BR](notebooks/pt-BR/05_S3_Baseline_Learning_Curve_PT.ipynb) | [EN-US](notebooks/en-US/05_S3_Baseline_Learning_Curve_EN.ipynb) | Baseline congelado |
| 06 - S4 erro e representação | [PT-BR](notebooks/pt-BR/06_S4_Error_Representation_Challenge_PT.ipynb) | [EN-US](notebooks/en-US/06_S4_Error_Representation_Challenge_EN.ipynb) | Desafio de representação |
| 07 - S5 benchmark de estimadores | [PT-BR](notebooks/pt-BR/07_S5_Estimator_Benchmark_PT.ipynb) | [EN-US](notebooks/en-US/07_S5_Estimator_Benchmark_EN.ipynb) | Isolamento do estimador |
| 08 - S6 clássico calibrado | [PT-BR](notebooks/pt-BR/08_S6_Calibrated_Classical_Challenge_PT.ipynb) | [EN-US](notebooks/en-US/08_S6_Calibrated_Classical_Challenge_EN.ipynb) | Última rodada clássica |
| 09 - S7 pacote congelado | [PT-BR](notebooks/pt-BR/09_S7_Frozen_Model_Package_PT.ipynb) | [EN-US](notebooks/en-US/09_S7_Frozen_Model_Package_EN.ipynb) | Congelamento e empacotamento |
| 10 - S8 confirmatório | [PT-BR](notebooks/pt-BR/10_S8_Confirmatory_Evaluation_PT.ipynb) | [EN-US](notebooks/en-US/10_S8_Confirmatory_Evaluation_EN.ipynb) | **Confirmatório, `test`** |
| 11 - V2 import do Kaggle | [PT-BR](notebooks/pt-BR/11_V2_Kaggle_Import_PT.ipynb) | [EN-US](notebooks/en-US/11_V2_Kaggle_Import_EN.ipynb) | Import da evidência D1 e D2 |
| 12 - V2 pacote congelado | [PT-BR](notebooks/pt-BR/12_V2_Frozen_Package_PT.ipynb) | [EN-US](notebooks/en-US/12_V2_Frozen_Package_EN.ipynb) | Seleção e congelamento |
| 13 - V2 confirmatório em stress | [PT-BR](notebooks/pt-BR/13_V2_Stress_Confirmatory_PT.ipynb) | [EN-US](notebooks/en-US/13_V2_Stress_Confirmatory_EN.ipynb) | **Confirmatório, `stress`** |

## Ciclo um: desenvolvimento clássico até o S8

Os gates foram congelados no S4, em 2026-08-16, antes de o V2 existir, e nunca
foram ajustados: macro-F1 ≥ 0,6900, F1 crítico ≥ 0,2715 e precisão crítica
≥ 0,2000.

| Etapa | O que variou | Desfecho |
|---|---|---|
| S3 | TF-IDF word + `SGDClassifier(log_loss)` | Macro-F1 0,7004, F1 crítico 0,2458. Curva plana — mais volume não ajudou a classe crítica |
| S4 | Representação e ponderação (word vs `char_wb`, balanced vs sqrt-balanced) | `NO_ELIGIBLE_CHALLENGER`; nenhum candidato passou os três gates |
| S5 | Somente o estimador, representação fixa | `LinearSVC` foi o mais próximo; ainda sem 3/3 |
| S6 | Calibração, com divisão interna de ajuste e calibração dentro de `train` | `LinearSVC` + threshold de margem crítica |
| S7 | Congelamento e empacotamento | Threshold 0,1135351095114484; macro-F1 de calibração 0,721989, F1 crítico 0,292994; 3/3 → `packaged_for_confirmation` |
| S8 | **Abrir `test` uma vez** | `NOT_CONFIRMED`, 2/3 |

O S8, sobre 334.230 representantes: macro-F1 0,718214 (passa), precisão crítica
0,404615 (passa), F1 crítico 0,257843 contra 0,2715 (**falha**), recall crítico
0,189209. Bootstrap diagnóstico de 95% no F1 crítico: [0,233153; 0,282263].

A calibração do S7 usou `validation` inteira, então o manifesto carrega
`validation_independence: NOT_INDEPENDENT_EVIDENCE_AFTER_FINAL_CALIBRATION`. Os
números do S7 são evidência de calibração; o S8 é a medida independente.

## Ciclo dois: o detector hierárquico V2

O V2 isola a intervenção na classe que falhou. Um detector binário de estágio A
sobrescreve a saída multiclasse do S7 congelado quando sua margem crítica
alcança um limiar calibrado; caso contrário, o S7 permanece. O estágio B é
referenciado por hash, nunca duplicado.

### O nulo do D1, e o defeito que ele expôs

A primeira execução do D1 completou 30 candidatos no Kaggle e produziu um
resultado cientificamente degenerado: os 30 candidatos retornaram métricas
**idênticas**, nas duas janelas, apesar de 30 thresholds distintos.

Recalcular o S7 puro pelos próprios caminhos de código do runner reproduziu
essas métricas célula a célula nas duas janelas. O sistema hierárquico era
idêntico ao fallback: o estágio A nunca produziu sobrescrita efetiva, e o
vencedor declarado era o S7 intocado.

A causa era a janela, não a busca. A ADR-010 colocou `inner_calibration` em
`train` 2024-05 a 2024-06 — **um subconjunto próprio do escopo de treino do
próprio S7 congelado**. In-sample, o S7 alcança F1 crítico 0,895787 e recall
0,939535. Fora da amostra, cai para 0,292994 e 0,232063. Contra um fallback que
recupera 94% da classe crítica por memória, nenhuma sobrescrita pode aumentar o
F1, e o sentinela de zero sobrescritas vence por construção. O experimento
estava impedido de falsear a própria hipótese.

A ADR-011 mudou exatamente uma variável — o posicionamento temporal das três
janelas — e acrescentou uma invariante de protocolo agora validada em código:
**nenhuma janela de calibração ou de avaliação pode intersectar o escopo de
ajuste do fallback congelado.** Sob essa invariante, as janelas da ADR-010 são
rejeitadas antes de qualquer execução. Duas regras de elegibilidade foram
acrescentadas, ambas restritivas: um candidato com zero sobrescritas efetivas é
o fallback e não é selecionável, e um candidato que não supera estritamente o
baseline de fallback puro na janela externa também não é.

O custo foi aceito explicitamente: a janela externa encolheu de 245.980 para
127.706 linhas e de 892 para 545 casos críticos. Um estimador mais ruidoso
sobre um experimento que pode falhar vale mais que um preciso sobre um
experimento que não pode.

### O D1 sob o V2.1, e o desafio do Transformer no D2

Sob as janelas corrigidas, 30 candidatos rodaram de novo. Os 30 passaram nas
margens de desenvolvimento, 22 tiveram sobrescritas efetivas e os mesmos 22
superaram o fallback. O vencedor foi
`word_char_tfidf_union_40000_60000_c_1_hard_negative`: F1 crítico externo
0,386899 contra 0,339665 do fallback, ganho de **+0,047234** obtido por 82
sobrescritas efetivas em 258 decisões positivas, com precisão praticamente
inalterada (0,437500 contra 0,434286).

O D2 então desafiou esse titular com um `distilbert-base-uncased` ajustado como
detector binário, sob desenho controlado: mesmo escopo de ajuste, mesmo pool de
negativos difíceis, mesmas janelas de calibração e avaliação, mesma busca de
threshold, mesma regra de combinação. Variou apenas a origem do escore —
diferença de logits no lugar do `decision_function`.

A barra de deslocamento foi pré-registrada antes da execução e derivada, não
escolhida: um bootstrap paramétrico sobre a matriz de confusão externa do
titular dá desvio padrão 0,019347 para o F1 crítico, e dois desvios padrão são
0,0387. O Transformer precisava alcançar F1 crítico ≥ 0,425599 e precisão
crítica ≥ 0,434286.

Três sementes foram executadas, com a **mediana** pré-declarada como resultado
reportado; reportar a melhor semente estava proibido. O Transformer superou o
fallback com folga e superou o titular por +0,015825, mas ficou abaixo das duas
barras. Desfecho: `CLASSICAL_WINNER_STANDS`. Nenhuma das sementes passaria, o
que torna o resultado independente da regra de agregação.

### Congelamento, e o portão de reprodução exata

O D1 não persistiu nenhum modelo ajustado, então congelar exigiu reajustar — e
reajustar cria a possibilidade de o artefato congelado diferir daquele que
produziu os números publicados.

Por isso o congelamento foi condicionado a um portão de reprodução **exata**: o
limiar, as duas matrizes de confusão inteiras, os dois pares de contagem de
sobrescrita e as contagens do pool de negativos difíceis precisavam reproduzir
o D1 sem tolerância. Comparação por igualdade, e não por proximidade numérica,
porque a pergunta não é se dois números são próximos — é se o objeto congelado
é o objeto medido.

O portão passou em 21 de 21 verificações. Desfecho: `PACKAGE_FROZEN`.

O portão reproduziu exatamente o comportamento do D1, mas não pode provar
retroativamente a identidade do pool linha a linha, pois o D1 não persistiu uma
assinatura do pool. A execução de congelamento coincide com o ensaio local do
D2 e difere da assinatura publicada pelo D2 no kernel de GPU. As contagens são
idênticas (946 positivos e 14.190 negativos), enquanto a identidade das linhas
entre todas as execuções permanece não comprovada. Isso corrige a afirmação
mais forte da ADR-012. A decisão do D2 não muda: nenhuma semente passou e as
margens perdidas foram materialmente maiores que as diferenças de reprodução.

## Fronteira de decisão

`deployment_authorized: false`, em todo protocolo, em toda etapa. Nenhum dos
dois ciclos autoriza servir nenhum dos modelos.

Permitido:

- Ler os agregados publicados como medida independente de um pacote congelado
  numa janela que ele nunca tinha visto.
- Reusar o desenho de protocolo — partições seladas, braços de controle
  pareados, portões de reprodução exata, regras de decisão pré-registradas.
- Desenhar uma intervenção diferente para a classe crítica, sobre partições
  ainda não consumidas.

Proibido:

- Reabrir qualquer um dos veredictos. O intervalo bootstrap do F1 crítico do V2,
  [0,232937; 0,288567], contém a barra de 0,2715, e isso está registrado como
  **apenas diagnóstico**. A regra pré-registrada é sobre a estimativa pontual.
  Usar o intervalo para reabrir o veredicto seria exatamente a manobra post hoc
  que o protocolo existe para impedir.
- Um V2.2 sobre `stress`. A ADR-014 pré-registrou, antes de o selo ser aberto,
  que um `NOT_CONFIRMED` encerra o ciclo. `stress` está agora consumida.
- Abrir `monitor` 2026. Permanece selada e diagnóstica, reservada a
  monitoramento posterior, e a emenda da ADR-014 não a alcança.
- Tratar qualquer número de desenvolvimento como desempenho. Toda folga foi
  medida na mesma janela que serviu de superfície de seleção.

## Arquitetura de software

| Caminho | Responsabilidade |
|---|---|
| `src/consumer_complaint_intelligence/audit.py`, `data.py`, `taxonomy.py`, `deduplication.py` | Auditoria S0/S1 do corpus, taxonomia de famílias, três níveis de detecção de duplicatas |
| `src/consumer_complaint_intelligence/temporal_split.py` | Candidatos de protocolo S2, identidade de grupo, política de suporte por partição |
| `src/consumer_complaint_intelligence/s3.py` a `s8.py` (+ `_reporting`) | Um módulo por etapa, runner mais relatório em texto |
| `src/consumer_complaint_intelligence/v2_protocol.py`, `v2_detector.py`, `v2_benchmark.py` | Contrato V2, detector do estágio A, benchmark de 30 candidatos |
| `src/consumer_complaint_intelligence/v2_transformer.py` | Desafio DistilBERT do D2 |
| `src/consumer_complaint_intelligence/v2_package.py` | Seleção, portão de reprodução exata, congelamento |
| `src/consumer_complaint_intelligence/v2_stress.py` | Runner confirmatório de dois braços do V2.1-C |
| `src/consumer_complaint_intelligence/v2_import.py` | Import e renderização reprodutíveis da evidência do Kaggle |
| `src/consumer_complaint_intelligence/kaggle_execution.py` | Montagem do bundle para execução remota |
| `src/consumer_complaint_intelligence/contracts.py`, `service.py`, `tracking.py` | `Predictor` / `PredictionBatch` / `ArtifactManifest`, superfície de serving, tracker opcional |
| `config/*.json` | Protocolos congelados e manifestos públicos de resultado com hashes |
| `docs/ADR-001` a `ADR-014` | Cada decisão, com sua razão e seu custo |
| `tests/` | 348 testes em 28 módulos de teste |

Três fronteiras arquiteturais valem para todo o projeto:

- O pipeline scikit-learn não conhece Flask, HTTP, JSON nem MLflow. O domínio
  expõe `Predictor`, `PredictionBatch` e `ArtifactManifest`; um adaptador futuro
  traduz para esses contratos.
- MLflow vive apenas na orquestração. `NullTracker` é o padrão, e MLflow só é
  importado quando explicitamente selecionado.
- Notebooks não contêm lógica analítica. Tudo que é computacional está em `src/`
  e coberto por testes.

O campo `score` significa `critical_margin` e nada mais. Não é probabilidade
nem confiança. A entrada do estimador é exclusivamente `en-US`, ainda que a API
e a documentação sejam bilíngues.

## Ambientes de execução

A execução local é uma máquina Windows de 16 GB. Algumas etapas não couberam.

| Execução | Ambiente | Duração |
|---|---|---:|
| Desafio de representação S4 | Local | 2.193 s, pico de 7,24 GB de RSS |
| Calibração final S7 | Local | 284,3 s, pico de 3,71 GB de RSS |
| Benchmark clássico D1 | Kaggle CPU | 5.972,4 s |
| Desafio do Transformer D2 | Kaggle GPU | ~1 h, 3 sementes |
| Congelamento V2.1-P | Kaggle CPU | 1.309 s |
| Abertura do stress V2.1-C | **Local** | 2.014,4 s |

Duas tentativas locais do D1 foram abortadas por memória — pico de 10,18 GiB de
RSS com 0,11 GiB disponíveis no sistema — e ficam registradas como evidência de
recurso. Como o runner publica transacionalmente, nenhuma delas produziu
resultado parcial.

O V2.1-C rodou localmente **por escolha**. É inferência em lote, portanto não
exige GPU, e rodar local mantém a partição selada e o Parquet bruto de 821 MB
na máquina. Para uma abertura de selo, essa é a fronteira melhor.

## Instalação

```bash
mamba env create -f environment.yml
mamba activate consumer-nlp
python -m pip install -e .
python -m ipykernel install --user --name consumer-nlp --display-name "Python (consumer-nlp)"
```

O dataset não é distribuído com o repositório. O estudo congelado exige o
snapshot Parquet arquivado exato; um download atual do CFPB não é equivalente e
será rejeitado pelo SHA-256 fixado. Consulte
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) para a identidade do
snapshot, o escopo de reprodutibilidade e as limitações atuais da fonte.

Os bundles `.joblib` publicados são artefatos Python baseados em pickle.
Carregue-os apenas a partir de um checkout confiável e verifique seus SHA-256
contra os protocolos congelados antes da desserialização.

Flask e MLflow são extras opcionais do `pyproject.toml`, não dependências do
ambiente.

## Validação

```bash
python -m unittest discover -s tests -t .
```

A suíte roda 348 testes cobrindo contratos, taxonomia, deduplicação, split
temporal, cada runner de etapa, cada manifesto de resultado, estrutura de
notebook e identidade de código PT-BR/EN-US, convenções de docstring e largura
de linha, montagem do bundle do Kaggle e as asserções de integridade de selo
descritas acima.

Para revisão, trate `config/s8_results.json` e `config/v2_stress_results.json`
como evidência congelada. Reexecutar os notebooks de relatório é suportado.
Abrir uma nova partição selada seria outro estudo, e `monitor` é a única que
resta.

## Fontes

Este projeto cita fontes regulatórias e documentais, não uma literatura
metodológica; suas decisões estão registradas em `docs/ADR-001` a `ADR-014`.

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Referência de campos do CFPB](https://cfpb.github.io/api/ccdb/fields.html)
- [Opções de produto e issue de agosto de 2023](https://files.consumerfinance.gov/f/documents/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.pdf)
- [Comunicado do CFPB de 24 de junho de 2026](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-is-correcting-flaws-to-restore-integrity-and-utility-to-the-consumer-complaint-system/)
- [Comunicado do CFPB de 14 de agosto de 2026](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/)

O comunicado de junho de 2026 relata crescimento excepcional de reclamações de
credit reporting e afirma que os dados do portal não podem ser tratados como
reflexo confiável das condições de mercado sem que os fatores identificados
sejam endereçados. Ele é tratado como sinal de revisão, não como regra
automática de exclusão de 2025 ou 2026. Qualquer exclusão seria decisão
explícita e documentada.

## Autoria e responsabilidade

**Autor:** Enio Rubens
**Papel:** Data Science e Analytics

Assistentes de código com IA apoiaram modularização, andaimes de teste,
documentação, tradução e revisão. Desenho de protocolo, escolha de gates,
critérios de aceitação, interpretação de resultados e toda decisão de abrir uma
partição selada permaneceram conduzidos por humano. Todas as afirmações
publicadas são de responsabilidade do autor.

## Atribuição dos dados

O CFPB Consumer Complaint Database é publicado pelo Consumer Financial
Protection Bureau. O snapshot bruto não é redistribuído aqui; a propriedade e
os eventuais termos originais de distribuição permanecem com a fonte.

A licença MIT se aplica ao software e à documentação do projeto, não ao dataset
do CFPB nem às licenças de modelos e bibliotecas de terceiros.
