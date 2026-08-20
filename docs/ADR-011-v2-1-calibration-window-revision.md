# ADR-011: resultado nulo do D1 e revisão V2.1 da janela de calibração

## Estado

**FROZEN_FOR_V2_1_DEVELOPMENT**, aprovado pelo Cientista de Dados em
2026-08-17. Este ADR substitui apenas as janelas temporais definidas na
ADR-010 e acrescenta uma invariante de protocolo. Arquitetura, catálogo de
candidatos, gates, margens, fronteira de partições e ciclo de sete passos
permanecem exatamente como aprovados lá.

## Contexto: o D1 executou e produziu um nulo

O benchmark clássico D1 foi executado com sucesso operacional no Kaggle
(kernel `cci-v2-d1-classical-benchmark`, quarta versão, 5.972,4 segundos, CPU).
Os 30 candidatos completaram, os dois artefatos agregados retornaram e a
revalidação local de hashes passou. Nenhuma partição selada foi tocada.

O resultado, porém, é cientificamente degenerado. Os 30 candidatos produziram
métricas de calibração idênticas entre si, com F1 crítico 0,895787, precisão
crítica 0,855932, recall crítico 0,939535 e macro-F1 0,889007. Produziram
também métricas externas idênticas entre si, com F1 crítico 0,292994,
precisão crítica 0,397313, recall crítico 0,232063 e macro-F1 0,721989. Duas
representações, três valores de `C` e cinco estratégias de balanceamento não
diferenciaram absolutamente nada, embora tenham gerado 30 thresholds distintos.

A verificação foi feita nos dois lados. Recalculando o fallback S7 puro na
janela de calibração pelos mesmos caminhos de código do runner, obtêm-se
exatamente as métricas de calibração armazenadas, com igualdade célula a
célula da matriz de confusão. Repetindo o procedimento na janela externa,
obtêm-se exatamente as métricas externas armazenadas, também com igualdade
célula a célula. Ou seja, o sistema hierárquico foi idêntico ao fallback nas
duas janelas: o estágio A nunca produziu uma sobrescrita efetiva, e o
candidato dito vencedor é o modelo S7 intocado.

Por consequência, a aprovação 3/3 nas margens de desenvolvimento é um
artefato. A folga em F1 crítico era de 0,002994, e o que passou nas margens
foi o próprio V1, que já havia falhado a confirmação S8. Congelar aquele
candidato como pacote V2 repetiria o padrão do V1: passar em desenvolvimento
e quebrar na confirmação.

O D1 não é descartado. Fica registrado como nulo válido e sustenta duas
conclusões úteis: o runner congelado executa corretamente ponta a ponta no
Kaggle, e a degenerescência foi resposta honesta a uma pergunta mal formulada.

## Diagnóstico: a calibração era in-sample para o fallback

A causa não é defeito de implementação da busca de threshold. A busca exata
avalia como primeiro candidato o sentinela de zero sobrescritas,
`np.nextafter(max(scores), inf)`, e só troca de escolha diante de melhora
estrita. Ela concluiu corretamente que não sobrescrever era ótimo.

A falha está na janela. O `fit_scope` do pacote S7 congelado, registrado em
`config/s7_results.json`, é a partição `train` de 2023-08-01 a 2024-06-30. A
janela `inner_calibration` da ADR-010 era `train` de 2024-05-01 a 2024-06-30,
isto é, **um subconjunto próprio do escopo de treino do próprio fallback**. O
threshold do estágio A estava sendo calibrado contra um S7 que memorizava as
linhas em que era avaliado.

O efeito é quantitativamente enorme. Na janela de calibração, in-sample, o S7
alcança F1 crítico 0,895787 e recall crítico 0,939535. Na janela externa, fora
do escopo de fit, o mesmo S7 cai para F1 crítico 0,292994 e recall crítico
0,232063. Contra um fallback que recupera 94% da classe crítica por memória,
nenhuma sobrescrita pode aumentar F1, e o sentinela vence por construção. O
experimento estava impedido de falsear a hipótese.

Existe ainda deriva de prior favorável a uma janela mais tardia. A densidade
da classe crítica sobe ao longo do cache: 0,2689% na janela de fit anterior,
0,2916% na calibração anterior e 0,4268% no quarto trimestre de 2024.

## Decisão

Mudamos exatamente uma variável: o posicionamento temporal das três janelas.
Catálogo, estimador, gates, margens, ranking, arquitetura hierárquica, selos,
`batch_size` e `random_state` permanecem idênticos, de modo que V2.1 contra
V2.0 seja uma comparação controlada.

As janelas passam a ser: fit interno em `train`, de 2023-08-01 a 2024-06-30;
calibração interna em `validation`, de 2024-07-01 a 2024-09-30; avaliação
externa em `validation`, de 2024-10-01 a 2024-12-31. Calibração e avaliação
ficam integralmente fora do `fit_scope` do S7.

O censo sobre o cache científico, restrito a partições de desenvolvimento, é:

| Janela | V2.0 linhas | V2.0 críticos | V2.1 linhas | V2.1 críticos |
| --- | --- | --- | --- | --- |
| fit interno | 271.819 | 731 | 345.552 | 946 |
| calibração interna | 73.733 | 215 | 118.274 | 347 |
| avaliação externa | 245.980 | 892 | 127.706 | 545 |

## Verificação de pré-voo das novas janelas

Antes de gastar quota de execução, medimos o fallback S7 puro nas janelas
V2.1, usando apenas partições de desenvolvimento.

Na nova calibração, `validation` de 2024-07-01 a 2024-09-30, o S7 obtém F1
crítico 0,212355, precisão crítica 0,321637 e recall crítico 0,158501,
deixando 292 dos 347 casos críticos sem detecção. Contra os 0,895787 da
janela anterior, isso confirma que a hipótese voltou a ser falseável: existe
margem real para o estágio A demonstrar valor.

Na nova avaliação externa, `validation` de 2024-10-01 a 2024-12-31, o S7 puro
obtém F1 crítico 0,339665, precisão crítica 0,434286, recall crítico 0,278899
e macro-F1 0,725816, deixando 393 dos 545 críticos sem detecção.

Esse segundo número expõe um problema que teria contaminado o V2.1. Os limites
absolutos foram fixados quando o fallback marcava 0,292994 na janela externa
antiga. Na janela nova, o fallback sozinho já supera o gate científico de
0,2715, a margem de desenvolvimento de 0,29 e a margem de macro-F1 de 0,70.
Passar 3/3 deixou de ser evidência de qualquer coisa: um candidato poderia
aprovar todas as margens sendo estritamente pior do que não fazer nada.

## Regra de melhora relativa

O V2.1 acrescenta por isso uma condição de elegibilidade mais estrita, também
derivada apenas de desenvolvimento: um candidato só é selecionável se seu F1
crítico externo for estritamente maior que o F1 crítico externo do baseline de
fallback puro. Os limites absolutos congelados permanecem inalterados, e a
regra é camada adicional, nunca afrouxamento. Cada candidato publica também a
diferença contra o baseline, de modo que "pior que não fazer nada" fique
visível no próprio artefato, junto do funil de elegibilidade.

## Invariante acrescentada ao protocolo

O contrato passa a registrar o `fit_scope` do estágio B e a validar em código
que **nenhuma janela de calibração ou de avaliação intersecta o escopo de fit
do fallback congelado**. `inner_fit` é explicitamente isento, porque treinar o
estágio A sobre `train` é esperado. A ordenação temporal estrita entre as três
janelas, antes apenas convenção, também passa a ser validada.

Essa invariante é a regressão que documenta o defeito: sob ela, as janelas da
ADR-010 são rejeitadas antes de qualquer execução.

## Consequências e custos aceitos

A avaliação externa encolhe de 245.980 para 127.706 linhas e de 892 para 545
casos críticos, uma perda de 38,9%. A perda é menor que os 50% sugeridos pela
metade do intervalo porque a densidade crítica sobe no quarto trimestre. O
intervalo de confiança sobre o F1 crítico fica mais largo, e aceitamos isso
explicitamente: um estimador mais ruidoso sobre um experimento que pode
falsear é preferível a um estimador preciso sobre um experimento que não pode.

A calibração migra de `train` para `validation`. Ambas seguem permitidas, e a
ordenação fit anterior à calibração, anterior à avaliação, é preservada. O
custo é que `validation` passa a estar integralmente consumida pelo V2.1.

Registramos também que `validation` não é evidência independente para o S7. O
próprio manifesto S7 marca
`validation_independence: NOT_INDEPENDENT_EVIDENCE_AFTER_FINAL_CALIBRATION`,
porque a calibração final do S7 usou aquela partição. O viés resultante é
otimista para o fallback, o que torna a barra do estágio A mais alta, e não
mais baixa. É um viés na direção conservadora, e por isso aceitável para medir
o incremento do estágio A.

Daí decorre uma restrição vinculante: **V2.1 é a última iteração de
desenvolvimento sobre estas janelas**. Uma terceira passagem sobre a mesma
`validation` seria otimista por construção. Se o V2.1 também retornar nulo ou
não alcançar as margens, a conduta correta é encerrar o ciclo V2 sem pacote,
e não remodelar as janelas outra vez.

Como efeito colateral favorável, o fit interno passa a usar a partição `train`
inteira, subindo de 271.819 para 345.552 linhas e de 731 para 946 positivos
críticos, sem custo de fronteira.

## Observabilidade obrigatória

O nulo do D1 só foi identificado por análise forense posterior, o que é
inaceitável como regime de trabalho. O runner passa a publicar, por candidato
e para cada janela, a contagem de sobrescritas efetivas do estágio A, além de
uma linha de baseline com o fallback puro. Um resultado em que todos os
candidatos têm zero sobrescritas precisa ser visivelmente nulo no próprio
manifesto.

Essa observabilidade também vira regra de seleção. Um candidato com zero
sobrescritas efetivas na avaliação externa é o fallback, não um modelo V2, e
por isso não é selecionável, ainda que passe 3/3 nas margens. O protocolo
registra a regra em `revision.zero_override_candidate_is_not_selectable`.

Complementarmente, os resultados baixados do Kaggle passam a ser inspecionados
por um caminho reprodutível: o módulo de importação e o notebook bilíngue 11
leem o log e os JSON agregados e os apresentam como texto, incluindo a
checagem de degenerescência.

## Fronteira científica

Inalterada. Somente `train` e `validation` são legíveis em desenvolvimento. O
agregado S8 sobre `test` 2025-H1 permanece imutável e fechado, servindo apenas
como motivação e baseline histórico; narrativas, identificadores, scores e
erros daquela partição continuam vedados para seleção, calibração e ajuste.
`stress` 2025-H2 e `monitor` 2026 seguem selados, sem mecanismo de unlock em
código de desenvolvimento.

## Seleção e margem

Os gates científicos continuam macro-F1 pelo menos 0,69, F1 crítico pelo menos
0,2715 e precisão crítica pelo menos 0,20. As margens de desenvolvimento
continuam macro-F1 pelo menos 0,70, F1 crítico pelo menos 0,29 e precisão
crítica pelo menos 0,22, com 3/3 obrigatório e ranking por F1 crítico,
macro-F1, precisão crítica e menor runtime. Duas condições são acrescentadas,
ambas restritivas: o candidato com zero sobrescritas efetivas é inelegível, e
o candidato que não supera estritamente o baseline de fallback puro na janela
externa também é inelegível. Se nenhum candidato sobreviver, o resultado é um
nulo declarado, sem vencedor, e não um vencedor por omissão.

## Confirmação futura

Inalterada em relação à ADR-010. Após seleção e congelamento do pacote V2, um
novo protocolo poderá autorizar uma única abertura de `stress` 2025-H2. O S8
não será recalculado e `monitor` 2026 permanece reservado.
