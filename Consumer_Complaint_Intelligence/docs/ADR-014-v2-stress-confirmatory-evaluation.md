# ADR-014: avaliação confirmatória do V2 em stress 2025-H2

## Estado

**NOT_CONFIRMED**, executado em 2026-08-19, com 3 de 4 gates.

As seções até "Consequências" foram escritas e congeladas em 2026-08-19 antes de
qualquer leitura do dado selado, e pré-registram a abertura única da partição
`stress`, de 2025-07-01 a 2025-12-31. Elas não foram alteradas depois da
execução. O resultado está registrado no adendo ao final.

## Contexto

O S8 encerrou o V1 como `NOT_CONFIRMED`: macro-F1 0,718214 e precisão crítica
0,404615 passaram, mas o F1 crítico foi 0,257843, abaixo do mínimo 0,2715. Esse
resultado é imutável e o `test` 2025-H1 não pode ser reutilizado pelo V2.

O ciclo V2 respondeu com uma extensão hierárquica. A ADR-013 congelou o pacote
`consumer-complaint-detector-v2` com portão de reprodução exato, 21 de 21
verificações. Sobre a janela externa de desenvolvimento, `validation` 2024-H2,
o pacote combinado obteve F1 crítico 0,386899 contra 0,339665 do S7 sozinho, um
ganho pareado de 0,047234 vindo integralmente de recall, com precisão crítica
praticamente inalterada, 0,437500 contra 0,434286.

Toda essa evidência é de desenvolvimento e o próprio pacote a marca como
`development_optimistic_not_independent`. O ciclo em `train` e `validation` está
encerrado pela ADR-011 e pela ADR-012. Falta a medida independente.

## Decisão

O V2.1-C avaliará uma única vez a partição `stress`, de 2025-07-01 a
2025-12-31, usando exclusivamente o bundle V2 congelado sobre o fallback S7
congelado, ambos fixados por hash. Não há seleção, recalibração, ajuste de
limiar, varredura ou inspeção de erros. O limiar é o congelado,
-0,13949530151425016, aplicado como `critical_margin`.

A visão científica contém somente famílias modeladas, grupos não vistos antes de
2025-07-01, grupos com um único label em `stress` e o representante de menor
`Complaint ID` por identidade `(normalized_group_hash, normalized_length)`. A
visão operacional avalia todas as linhas dos mesmos grupos limpos em batches, é
publicada em separado e não pode alterar a decisão. As duas definições são as do
S8, com as datas trocadas.

`monitor` 2026 permanece selado e continua diagnóstico.

## Promoção de stress a partição confirmatória

O relatório S2 registra a limitação de que `stress` e `monitor` são diagnósticas
e não aprovam um modelo. Esta ADR emenda essa cláusula, e apenas para `stress`.

A razão é que o `test`, a partição desenhada para confirmar, foi consumido pelo
S8 e está proibido ao V2 pela ADR-010. Sem a emenda, o V2 não teria nenhuma
janela independente e o ciclo terminaria sem medida honesta.

A emenda é sustentada por metadados que já existiam antes desta decisão, todos
lidos do relatório S2 e nenhum lido da partição selada. Submetido aos mesmos
critérios-piloto que o S2 aplicou ao `test`, o `stress` passaria: o menor número
de grupos inéditos por família modelada é 1.170, contra o mínimo de 500, e a
maior participação de família é 0,752312, contra o teto de 0,80. A partição tem
270.279 grupos inéditos, ou 80,7 por cento do `test`.

A emenda não se estende a `monitor`, que continua diagnóstico e selado.

## Os dois braços

A abertura pontua dois braços numa única passagem sobre o dado selado, o que
mantém literalmente uma abertura do selo:

O braço primário é o V2 combinado, ou seja, o estágio A sobrescrevendo o S7
quando a margem alcança o limiar congelado. O braço de controle é o S7 sozinho,
o mesmo bundle que o S8 mediu, sobre exatamente as mesmas linhas.

O controle é necessário porque `stress` 2025-H2 tem composição diferente do
`test` 2025-H1. Em grupos inéditos, `money_services` cai 62 por cento,
`student_loan` cai 44 por cento, `credit_reporting` cai 22 por cento e a classe
crítica cai 16 por cento. Como macro-F1 é média não ponderada sobre nove
classes, o número do V2 em `stress` não é comparável ao 0,718214 do S8 em
`test`, nem para o mesmo modelo.

A evidência direta disso já existe: o mesmo S7 congelado, sem alterar uma linha,
marcou F1 crítico 0,339665 em `validation` 2024-H2 e 0,257843 em `test`
2025-H1, uma diferença de 0,081822 atribuível apenas à janela. Uma barra
absoluta é sensível à janela; um contraste pareado nas mesmas linhas não é.

O braço de controle não é seleção. Os dois modelos estão congelados por hash e
nenhum será escolhido em função do resultado.

## Gates e decisão

São quatro gates simultâneos, avaliados sobre a visão científica.

Os três absolutos são os do S8, sem alteração: macro-F1 pelo menos 0,69, F1
crítico pelo menos 0,2715 e precisão crítica pelo menos 0,20. Foram congelados
em 2026-08-16, antes do V2 existir, e nunca foram ajustados. Mantê-los preserva
a comensurabilidade entre a resposta do V2 e a do S8.

O quarto é pareado e estrito: o F1 crítico do V2 combinado precisa ser
estritamente maior que o do S7 sozinho nas mesmas linhas. É o gate que responde
à pergunta que motivou o V2, e é o único imune à deriva de composição.

Com 4 de 4, o status será `CONFIRMED`; caso contrário, `NOT_CONFIRMED`. O campo
`confirmed` é separado do status. `deploy` é sempre `false` e a confirmação
nunca autoriza deployment.

O ganho pareado de desenvolvimento, 0,047234, é registrado como expectativa
pré-declarada. Ele é diagnóstico e não é gate.

Intervalos bootstrap estratificados pela matriz de confusão usam 2.000
replicações, seed 42 e nível 0,95, e incluem a diferença pareada entre os
braços. São diagnósticos e não participam dos gates.

## Fronteira de dados

Antes do token, o runner valida somente configurações, hashes, metadados e os
manifestos V2 e S7. O token é lido de `V2_STRESS_UNLOCK` apenas em modo `full` e
o plaintext não aparece no notebook nem em arquivo do repositório. Depois da
autorização, DuckDB usa limite de 4 GB, uma thread e batches de 4.096.

A execução é local. Diferente do D1 e do D2, isto é inferência em lote e não
exige o Kaggle, de modo que o parquet bruto e a partição selada não saem da
máquina. O raw é unido por `Complaint ID` somente para a narrativa corrente,
rejeitando vazios. Nenhum texto, ID, score individual, margem individual ou
cache de `stress` é persistido.

O resultado agregado é escrito atomicamente em `temp/v2/v2_stress_results.json`
e o manifesto público é criado depois da conclusão. Execuções incompletas não
persistem métricas parciais e só podem ser retomadas com o mesmo contrato,
hashes e token.

## Evidência congelada

O protocolo congela os hashes de raw, índice, relatório S2 e protocolo S3, além
dos quatro arquivos V2 e dos quatro arquivos S7. A evidência S2 registrada para
`stress` é `all_text=526872`, `novel_text=382385`, `novel_unique_groups=270279`
e `critical_novel_groups=1170`.

As contagens de escopo reconstruídas serão comparadas com essas e a diferença
preservada, como o S8 fez em `scope_counts.s2_difference`. Diferença é esperada:
o S8 reconstruiu 334.230 representantes contra 334.973 do S2, uma perda de 743
grupos ambíguos.

## Consequências

Esta é a última leitura de dado independente disponível ao V2. Depois dela não
há janela para uma terceira tentativa: `monitor` 2026 é reservado a
monitoramento posterior e continua diagnóstico por esta mesma ADR.

Um `NOT_CONFIRMED` não abre um V2.2 sobre `stress`. Encerra o ciclo com a
medida publicada, exatamente como o S8 encerrou o V1.

Um `CONFIRMED` também não autoriza deployment. Autoriza apenas registrar que o
pacote congelado passou os quatro gates pré-registrados numa janela nunca vista.


## Resultado da execução, registrado em 2026-08-19

O selo foi aberto uma única vez em 2026-08-19T19:39:06Z, localmente, em 2.014,38
segundos. A reconstrução encontrou 270.080 grupos inéditos contra 270.279
esperados pelo S2, diferença de -199. Depois da remoção de 165 grupos ambíguos,
o escopo primário ficou com 269.915 representantes limpos e inéditos e 1.168
grupos críticos contra 1.170 esperados. `monitor` não foi aberto.

A decisão é **NOT_CONFIRMED**, com 3 de 4 gates e `deploy=false`.

Macro-F1 foi 0,710748, acima de 0,69, e a precisão crítica foi 0,426070, acima
de 0,20. O gate pareado passou: o ganho de F1 crítico foi 0,006455, positivo. O
gate que falhou foi o de F1 crítico absoluto, 0,260404 contra o mínimo 0,2715.

Na visão científica, o V2 combinado obteve F1 crítico 0,260404, precisão
0,426070 e recall 0,187500. O S7 sozinho, nas mesmas linhas, obteve 0,253949,
0,437238 e 0,178938. O ganho pareado veio de recall, mais 0,008562, com custo de
precisão de menos 0,011168.

O intervalo bootstrap de 95 por cento do ganho pareado é [0,000235, 0,013661] e
exclui o zero. O intervalo do F1 crítico do V2 é [0,232937, 0,288567] e contém a
barra de 0,2715. Isso é registrado como diagnóstico. A regra de decisão
pré-registrada é sobre a estimativa pontual, como no S8, e não é reaberta pelo
intervalo. Fazer isso seria exatamente a manobra post hoc que este protocolo
existe para impedir.

A visão operacional secundária, com 379.182 linhas, obteve ganho pareado
0,006379 e não participou da decisão.

## Por que o ganho de desenvolvimento não se sustentou

O ganho pareado medido foi 0,006455 contra os 0,047234 pré-declarados, cerca de
um sétimo. O ganho observado foi positivo e seu intervalo bootstrap diagnóstico
excluiu o zero sob o procedimento de reamostragem pré-especificado; a magnitude
não satisfez o gate absoluto.

O mecanismo está nas contagens. O estágio A disparou 224 vezes em 269.915
linhas, ou 0,083 por cento, contra 258 vezes em 127.706 linhas no
desenvolvimento, ou 0,202 por cento. A taxa de disparo caiu para 41 por cento da
observada em desenvolvimento. O detector foi ajustado de 2023-08 a 2024-06 e
calibrado de 2024-07 a 2024-09. A janela de `stress` ocorre de 12 a 18 meses
após o fim do ajuste e de 9 a 15 meses após o fim da calibração; nela, as
margens já não alcançam o limiar congelado na mesma frequência.

Das 224 decisões, apenas 36 foram efetivas, isto é, incidiram sobre linhas em
que o S7 ainda não dizia crítico. Dessas 36, 10 estavam certas e 26 erradas, uma
taxa de acerto de 27,8 por cento, abaixo da precisão de 42,6 por cento da classe
crítica no conjunto. É por isso que a precisão caiu enquanto o recall subiu, e é
por isso que o saldo em F1 foi pequeno.

## Uma comparação que o braço de controle tornou possível

O S7 congelado obteve F1 crítico 0,253949 em `stress` 2025-H2 e 0,257843 em
`test` 2025-H1. A proximidade enfraquece uma explicação simples baseada apenas
em maior dificuldade da janela de `stress`, mas não constitui teste formal de
equivalência e não elimina outros efeitos de janela. O braço de controle torna
essa comparação observável, sem transformar proximidade numérica em prova.

Segue que o V2 em `stress`, 0,260404, supera o S7 em `test`, 0,257843, por
0,002561. Todo o ciclo V2 moveu a métrica que motivou sua existência em cerca de
dois milésimos e meio, contra uma barra que exigia mais treze milésimos e meio.

## Consequência

O ciclo V2 encerra aqui, como esta ADR pré-registrou. Um `NOT_CONFIRMED` não
abre um V2.2 sobre `stress`, e `stress` está agora consumido. `monitor` 2026
permanece selado e diagnóstico, reservado a monitoramento posterior, e a emenda
desta ADR não o alcança.

Nenhum deployment é autorizado, nem do V2 nem do S7.
