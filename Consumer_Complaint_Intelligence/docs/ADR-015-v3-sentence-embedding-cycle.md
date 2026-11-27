# ADR-015: protocolo temporal e de reamostragem do ciclo V3-MiniLM

## Estado

**FROZEN_FOR_V3_DEVELOPMENT**, aprovado pelo usuário em 2026-08-21.

Esta decisão substitui o rascunho Kaggle/T4 não executado preparado antes da
aprovação. O rascunho permanece arquivado apenas para rastreabilidade e não é
um contrato concorrente.

Este documento transforma a aprovação conceitual do V3-MiniLM em um contrato
exato. O congelamento autoriza E2 e E3 exclusivamente em `train` e `validation`.
Não autoriza abertura de partição selada, alegação confirmatória, deployment ou
entrada antecipada dos desafiantes condicionais.

## Contexto

Os ciclos anteriores encerraram com `NOT_CONFIRMED`. O V2.1-C superou o S7 nas
mesmas linhas de `stress`, mas o ganho pareado de F1 crítico foi 0,006455 e o F1
crítico absoluto ficou em 0,260404, abaixo de 0,2715. A taxa de disparo do
detector caiu de 0,202% no desenvolvimento para 0,083% em `stress`.

O V3 não repetirá apenas uma troca de estimador. Ele testará conjuntamente uma
representação semântica congelada, razões controladas entre positivos e
negativos, preservação da composição temporal e seleção por estabilidade entre
janelas futuras.

O candidato principal é `sentence-transformers/all-MiniLM-L6-v2` congelado com
uma cabeça linear. O controle obrigatório permanece TF-IDF word+char com a
mesma cabeça. MiniLM é uma hipótese de representação, não uma promessa de
melhora e não substitui o controle.

## Fronteira científica

Somente `train` e `validation` podem participar do desenvolvimento. `test` foi
consumido pelo S8 e `stress` pelo V2.1-C; nenhum dos dois pode ser reutilizado
para seleção V3. `monitor` permanece selado e não possui caminho de
destravamento no código de desenvolvimento.

A unidade científica continua sendo o grupo
`(normalized_group_hash, normalized_length)`, com um representante por grupo.
O protocolo usa todos os grupos críticos únicos disponíveis no fit de cada
fold. Nenhuma observação posterior ao fim do fit pode entrar na mineração de
negativos, no ajuste do vetorizador, na cabeça linear ou no ajuste do encoder.

O desenvolvimento V3 será, por construção, evidência reutilizada e otimista.
Uma futura afirmação confirmatória exigirá dados prospectivos ou externos e um
novo protocolo escrito antes de lê-los.

## Três folds rolling-origin

As contagens abaixo foram calculadas apenas como agregados do cache científico
de desenvolvimento. Os fits crescem no tempo; calibração e avaliação mantêm a
prevalência natural e não sofrem reamostragem.

| Fold | Fit | Críticos | Calibração | Críticos | Avaliação | Críticos |
|---|---|---:|---|---:|---|---:|
| F1 | 2023-08 a 2023-12 | 305 | 2024-01 a 2024-02 | 207 | 2024-03 a 2024-04 | 219 |
| F2 | 2023-08 a 2024-02 | 512 | 2024-03 a 2024-04 | 219 | 2024-05 a 2024-06 | 215 |
| F3 | 2023-08 a 2024-06 | 946 | 2024-07 a 2024-09 | 347 | 2024-10 a 2024-12 | 545 |

Uma avaliação mais antiga pode integrar o fit de um fold posterior, como em
validação rolling-origin usual. Dentro de cada fold, porém, vale a ordem estrita
`fit < calibration < evaluation`.

O S7 congelado foi ajustado até 2024-06-30. Portanto, usá-lo nos folds F1 ou F2
seria avaliação in-sample e repetiria o defeito documentado pela ADR-011. F1 e
F2 rastreiam apenas o detector binário. A comparação hierárquica com o S7 é
permitida somente em F3, inteiramente posterior ao escopo de ajuste do S7.

## Reamostragem dentro do fit

Cada candidato usa todos os grupos críticos únicos do fit, sem reposição.
Negativos são subamostrados nas razões 1:5, 1:10 e 1:15. A razão 1:15 é o
controle comparável ao V2.

| Razão | Hard negatives por positivo | Background por positivo |
|---|---:|---:|
| 1:5 | 3 | 2 |
| 1:10 | 7 | 3 |
| 1:15 | 10 | 5 |

Hard negatives vêm exclusivamente de previsões out-of-fold produzidas dentro
do fit. Negativos de background são sorteados sem reposição, estratificados por
mês e família. A identidade dos grupos sorteados deve ser idêntica entre TF-IDF
e MiniLM; assim a comparação isola a representação.

Não há oversampling antes do corte temporal nem reamostragem de calibração ou
avaliação. Se um estrato não tiver negativos suficientes, a execução falha em
vez de completar silenciosamente com outra origem.

## Representações e classificador

O braço A é o TF-IDF word+char do vencedor clássico V2, como controle. O braço B
é MiniLM congelado, inicialmente com a visão `head-256` e pooling médio padrão
do modelo. Ambos usam `LinearSVC` com os mesmos hiperparâmetros e sem
`class_weight`, pois a razão de treino já é intervenção explícita.

A revisão imutável do MiniLM foi resolvida no pré-voo e fixada como
`bc57282bc374d33e0d6c4de27f12dc1c2a87f37a`. O identificador vem da árvore do
modelo publicada em
<https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/bc57282bc374d33e0d6c4de27f12dc1c2a87f37a>.
Alterar esse commit exige emenda explícita antes de nova execução.

BGE-small, stacking e fine-tuning não pertencem ao catálogo inicial. Eles têm
condições de entrada explícitas e exigirão emenda prévia. O teste de texto longo
compara, uma única vez, `head-256` com a média de embeddings do início e do fim,
somente se o MiniLM completar o E3.

## Calibração, métricas e seleção

O classificador produz margem linear. Em cada fold, uma sigmoide ajustada apenas
na calibração permite publicar Brier e ECE na avaliação. O threshold também é
escolhido apenas na calibração: maximiza F1 crítico sujeito a precisão crítica
de pelo menos 0,40. A avaliação reutiliza esse threshold sem ajuste.

As métricas primárias são PR-AUC crítico, recall crítico no piso de precisão e
F1 crítico no threshold selecionado. Precisão, recall, Brier, ECE, taxa de
decisão positiva e estabilidade do threshold são diagnósticos obrigatórios.

Um candidato é o par `(representação, razão de negativos)` e precisa completar
os três folds, atendendo ao piso de precisão em todos. O ranking prioriza:

1. maior pior recall entre folds no piso de precisão;
2. maior mediana de PR-AUC;
3. maior mediana de F1 crítico;
4. menor amplitude de F1 entre folds;
5. menor tempo de execução.

Em F3, e somente nele, o detector selecionado pode desafiar o S7 congelado. O
sistema combinado precisa manter macro-F1 >= 0,69, F1 crítico >= 0,2715,
precisão crítica >= 0,20 e ganho pareado estritamente positivo sobre o S7.
Passar esses limites autoriza apenas empacotar um candidato de desenvolvimento.

## Execução e ambiente

O ambiente V3 será `consumer-nlp-v3`, separado do `consumer-nlp` usado para
reproduzir a publicação anterior. O modo padrão de qualquer runner é
`disabled`; smoke é diagnóstico e não seleciona modelo.

Embeddings podem ser persistidos em shards `float16` somente para `train` e
`validation`, com manifesto e hashes. Embeddings individuais não são artefatos
públicos. Uma partição confirmatória futura só poderá ser transformada durante
uma abertura autorizada por outro protocolo.

## Aprovação dos parâmetros exatos

Em 2026-08-21, o usuário aprovou explicitamente e autorizou congelar:

1. os três cortes rolling-origin;
2. o piso proposto de precisão crítica de 0,40;
3. a decomposição hard/background das três razões;
4. o uso do S7 somente em F3;
5. o catálogo inicial restrito a TF-IDF e MiniLM congelado;
6. a revisão imutável do MiniLM fixada no commit `bc57282...f37a`.

Esses seis pontos não podem ser alterados depois de observados resultados E2 ou
E3. Qualquer extensão condicionada continua exigindo emenda anterior à sua
execução.

## Pré-voo técnico V3-E1, executado em 2026-08-21

O commit fixado do MiniLM foi carregado no ambiente isolado
`consumer-nlp-v3`. Um benchmark exclusivamente sintético, sem acesso ao cache
científico, codificou 10.000 textos em 6,4654 segundos na RTX 3050 Laptop,
aproximadamente 1.546,7 textos por segundo. O lote 128 completou com pico de
484 MiB de memória CUDA reservada.

O resultado está em `temp/v3/v3_e1_gpu_smoke.json`, fixado pelo manifesto
`config/v3/smoke_results.json`. Nenhum texto, label, score individual ou
embedding foi persistido. O resultado demonstra viabilidade operacional e não
participa de seleção nem fornece evidência de qualidade preditiva.
