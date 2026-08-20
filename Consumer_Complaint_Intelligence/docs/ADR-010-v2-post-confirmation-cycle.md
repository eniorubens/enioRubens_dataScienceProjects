# ADR-010: ciclo V2 pós-confirmação

## Estado

**FROZEN_FOR_V2_DEVELOPMENT**, aprovado pelo Cientista de Dados em 2026-08-17.

## Contexto

O S8 encerrou o modelo V1 como `NOT_CONFIRMED`. O modelo passou os gates de
macro-F1 e precisão crítica, mas obteve F1 crítico 0,257843, abaixo do mínimo
0,2715. Esse resultado é imutável: o test 2025-H1 não pode ser reutilizado para
seleção, calibração, inspeção de erros ou promoção retrospectiva.

## Decisão

O V2 será uma extensão hierárquica. Um detector binário identifica
`debt_credit_management`; quando seu score supera um threshold calibrado apenas
em desenvolvimento, ele substitui a saída do classificador S7. Caso contrário,
o bundle multiclasses S7 permanece como fallback. Isso isola a intervenção na
classe que falhou e preserva uma referência conhecida para as outras famílias.

O desafio clássico compara duas representações TF-IDF, três valores de `C` e
três estratégias de balanceamento: pesos, oversampling aleatório moderado e
hard negatives derivados exclusivamente de previsões out-of-fold do inner fit.
Toda reamostragem ocorre dentro do fold de fit e mantém a identidade de grupo.
SMOTE não integra o protocolo inicial. O catálogo cartesiano é limitado a 30
candidatos clássicos completos.

Após uma primeira execução abortada por baixa memória, o oversampling preserva
os índices sorteados pelo `RandomOverSampler`, mas aplica a multiplicidade de
cada linha como `sample_weight` inteiro no LinearSVC. Para uma loss aditiva,
isso é equivalente a repetir a mesma linha esparsa e evita materializar cópias
da matriz TF-IDF. A tentativa abortada não publicou métricas nem manifesto.

Uma segunda execução monitorada, já com essa equivalência, também foi abortada
sem publicar métricas ou manifesto. Ela durou 1.543,80 segundos, atingiu pico de
10,18 GiB de RSS no processo e deixou apenas 0,11 GiB disponíveis no sistema.
Logo, a equivalência reduz a duplicação causada pelo oversampling, mas não torna
o catálogo completo operacionalmente seguro na máquina local de 16 GiB.

A mineração usa `StratifiedKFold` determinístico com três folds. Em cada fold,
o pipeline TF-IDF word e LinearSVC balanceado é reajustado sem as linhas de
avaliação daquele fold. O subset final mantém todos os positivos, dez hard
negatives e cinco negativos de background por positivo. O candidato treinado
nesse subset não reaplica `class_weight`.

## Fronteira científica

Somente `train` e `validation` podem ser lidos. O resultado agregado S8 serve
apenas como motivação e baseline histórico; narrativas, IDs, scores individuais
e erros do test não podem ser consultados. `stress` 2025-H2 e `monitor` 2026
permanecem selados, e nenhum código de desenvolvimento terá mecanismo de unlock
para essas partições.

As janelas internas permanecem 2023-08-01 a 2024-04-30 para fit e 2024-05-01 a
2024-06-30 para calibração. A avaliação externa de desenvolvimento usa
`validation`, de 2024-07-01 a 2024-12-31.

O cache científico contém 591.532 representantes: 345.552 em `train` e 245.980
em `validation`. A classe crítica tem suporte 731 no inner fit, 215 na inner
calibration e 892 na outer validation. Essas contagens são metadados de
desenvolvimento; não justificam acesso a uma partição selada.

## Ambiente de execução do D1

O benchmark clássico completo será executado no Kaggle a partir de um bundle
reproduzível e de um upload separado de `scientific.parquet`. A mudança é apenas
operacional: reutiliza o runner congelado, os hashes de baseline e protocolo,
as mesmas 30 candidaturas e as mesmas janelas. O bundle não contém dados,
resultados locais nem mecanismos de acesso a `test`, `stress` ou `monitor`.

As duas tentativas locais abortadas permanecem como evidência de recursos em
`temp/v2`, com nomes que incluem `attempt1` e `attempt2`. Como o runner publica
transacionalmente, nenhuma delas constitui resultado parcial do D1.

## Seleção e margem

Os gates científicos continuam inalterados: macro-F1 pelo menos 0,69, F1
crítico pelo menos 0,2715 e precisão crítica pelo menos 0,20. Para congelar um
candidato V2, exigimos margem de desenvolvimento: macro-F1 pelo menos 0,70, F1
crítico pelo menos 0,29 e precisão crítica pelo menos 0,22. Depois de 3/3, o
ranking prioriza F1 crítico, macro-F1, precisão crítica e menor runtime.

Um Transformer compacto poderá desafiar somente o melhor candidato clássico e
exigirá configuração de execução própria antes de usar GPU. Nenhum resultado de
desenvolvimento autoriza deploy.

## Confirmação futura

Após seleção e congelamento do pacote V2, um novo protocolo poderá autorizar uma
única abertura de `stress` 2025-H2. O S8 não será recalculado. `monitor` 2026
continuará reservado para monitoramento posterior.
