# ADR-012: D2, desafio do Transformer compacto ao vencedor clássico

## Estado

**FROZEN_FOR_V2_1_D2_EXECUTION**, aprovado pelo Cientista de Dados em
2026-08-18. Este ADR executa o passo 4 do ciclo de sete passos aprovado na
ADR-010, `challenge_best_classical_with_compact_transformer`, sob as janelas
temporais fixadas pela ADR-011. Não altera janelas, gates, margens, fronteira
de partições nem o catálogo clássico. A regra de decisão abaixo é registrada
antes da execução e não pode ser revisada depois de observado o resultado.

## Contexto: existe um vencedor clássico para desafiar

O D1 sob o protocolo V2.1 produziu um resultado não degenerado. Dos 30
candidatos, 30 passaram 3/3 nas margens, 22 tiveram sobrescritas efetivas na
janela externa e os mesmos 22 superaram o baseline do fallback. O selecionado
foi `word_char_tfidf_union_40000_60000_c_1_hard_negative`, com threshold de
calibração -0,139495 e, na janela externa, F1 crítico 0,386899, precisão
crítica 0,437500, recall crítico 0,346789 e macro-F1 0,731214, contra o
fallback S7 puro em 0,339665, 0,434286, 0,278899 e 0,725816. O ganho é
+0,047234 de F1 crítico, obtido por 82 sobrescritas efetivas em 127.706
linhas, com precisão praticamente inalterada.

O `transformer_challenge` do protocolo estava em
`DEFERRED_UNTIL_CLASSICAL_WINNER`. A condição está satisfeita, e o desafio
passa a ser executável com `maximum_models: 1` e
`requires_separate_frozen_execution_config: true`. A configuração congelada
correspondente é `config/v2_d2_execution.json`.

## O que o D2 pergunta, e o que ele não pergunta

O D2 pergunta uma única coisa: mantido tudo o mais constante, uma
representação contextual densa produz um detector de estágio A melhor do que
a união TF-IDF word+char com LinearSVC?

O D2 não reabre a seleção entre os 30 candidatos clássicos, não altera as
janelas, não ajusta gates nem margens e não toca partição selada. Se o
Transformer não vencer, o vencedor clássico do D1 permanece, e o D2 fica
registrado como um desafio que não deslocou o titular. Esse é um desfecho
legítimo e esperado a priori, não uma falha operacional.

## Desenho controlado

Para que a comparação isole a família de modelo, tudo o que não é a família
de modelo é mantido idêntico ao do vencedor clássico:

- mesmo escopo de ajuste, `inner_fit`, `train` de 2023-08-01 a 2024-06-30;
- pool de treino de negativos difíceis gerado pelo mesmo procedimento,
  `generate_hard_negative_indices` com os mesmos parâmetros do protocolo,
  10 negativos difíceis e 5 negativos de fundo por positivo, OOF em 3 folds
  com `random_state` 42, o que reproduz as contagens de 946 grupos positivos e
  14.190 negativos do D1, num pool de 15.136 linhas. Como o D1 não persistiu
  assinatura, identidade linha a linha não pode ser provada retroativamente;
- mesma janela de calibração, `validation` de 2024-07-01 a 2024-09-30, e
  mesma busca exata de threshold, `search_detector_threshold_exact`;
- mesma janela externa, `validation` de 2024-10-01 a 2024-12-31;
- mesma arquitetura hierárquica, com o estágio A sobrescrevendo o fallback S7
  congelado acima do threshold calibrado;
- mesmos gates científicos, mesmas margens de desenvolvimento e mesmo
  baseline de fallback publicado.

Varia apenas o estágio A. No lugar do vetorizador TF-IDF e do LinearSVC entra
um `distilbert-base-uncased` ajustado como classificador binário, e o escore
usado como margem é a diferença de logits, `logit_crítico` menos
`logit_não_crítico`. Essa definição é deliberada: ela torna o escore do
Transformer um substituto direto do `decision_function` do LinearSVC, de modo
que a busca exata de threshold e toda a mecânica de sobrescrita seguem sem
qualquer alteração. O que muda é a origem do número, não o que se faz com ele.

Os hiperparâmetros são pré-registrados e não formam superfície de busca:
comprimento máximo 256 tokens, 3 épocas, taxa de aprendizado 2e-5, decaimento
de peso 0,01, aquecimento de 10%, lote de treino 32, lote de avaliação 128,
AdamW com agendamento linear, precisão mista quando houver CUDA. Não há
early stopping e não há seleção de checkpoint: vale o último passo da última
época. Qualquer varredura de hiperparâmetros está proibida por esta ADR.

## Sementes: réplicas, não modelos

O protocolo autoriza `maximum_models: 1`. Interpretamos "um modelo" como uma
arquitetura com uma configuração de hiperparâmetros. Sementes são réplicas do
mesmo modelo, e não modelos distintos, de modo que executar mais de uma
semente não amplia o espaço de busca.

Executamos três sementes, 42, 43 e 44. A razão é honestidade estatística. Um
ajuste fino sobre 946 positivos tem variância de semente não desprezível, e
uma única extração deixaria indistinguíveis "o Transformer é melhor" e "esta
semente foi sortuda". Para que a réplica não vire seleção disfarçada, a
métrica reportada é fixada de antemão: **a mediana das três**, com o vetor
completo de métricas da semente que atingir a mediana. Reportar a melhor
semente está proibido. A amplitude entre a menor e a maior é publicada junto,
como medida da instabilidade do procedimento.

## Regra de decisão pré-registrada

Avaliada sobre a semente reportada, na janela externa. Para deslocar o
vencedor clássico, o Transformer precisa satisfazer as cinco condições:

1. passar 3/3 nas margens de desenvolvimento, macro-F1 pelo menos 0,70, F1
   crítico pelo menos 0,29 e precisão crítica pelo menos 0,22;
2. ter mais de zero sobrescritas efetivas na janela externa, pela mesma regra
   da ADR-011 que impede um candidato idêntico ao fallback de ser selecionado;
3. superar o baseline do fallback puro em F1 crítico;
4. ter precisão crítica de no mínimo 0,434286, que é a precisão do próprio
   fallback na janela externa. Essa é exatamente a barra que o titular
   cumpriu, e existe para impedir que um ganho de recall seja comprado com
   perda de precisão;
5. ter F1 crítico de no mínimo **0,425599**, isto é, superar o titular por ao
   menos **+0,0387**.

O incremento de 0,0387 não é arbitrário. Um bootstrap paramétrico com 200.000
extrações sobre a matriz de confusão externa do titular, reamostrando os
verdadeiros positivos como binomial em 545 positivos e os falsos positivos
como binomial no restante das 127.706 linhas, dá desvio padrão 0,019347 para
o F1 crítico. Dois desvios padrão são 0,0387, e esse valor coincide com uma
melhora relativa de 10% sobre 0,386899. Uma diferença menor do que isso não é
distinguível do ruído amostral desta janela, e não justificaria substituir um
modelo que roda em CPU por um que exige GPU em inferência.

Se qualquer condição falhar, o desfecho é `CLASSICAL_WINNER_STANDS`, o
resultado do Transformer é publicado como evidência e o pacote V2 é congelado
sobre o vencedor clássico. Não haverá segunda configuração, segunda janela,
nem reabertura da seleção clássica.

## Multiplicidade, e por que ela permanece controlada

A janela externa já foi usada como superfície de seleção uma vez, entre 22
candidatos elegíveis. O D2 acrescenta uma segunda olhada sobre a mesma
janela, o que é um custo estatístico real e precisa ser declarado.

Três coisas o mantêm limitado. O acréscimo é de exatamente um modelo, não de
outro catálogo, porque o protocolo fixa `maximum_models: 1` e esta ADR proíbe
varredura. A barra de deslocamento é de dois desvios padrão, e não de zero,
o que torna a segunda olhada muito menos permissiva do que a primeira. E a
regra é registrada antes da execução, neste arquivo e em
`config/v2_d2_execution.json`, de modo que não pode ser afrouxada depois de
visto o número.

Ainda assim, registramos a consequência sem suavizá-la: o número do D2, tanto
faz quem vença, é evidência de desenvolvimento com viés otimista, e a
confirmação continua dependendo da abertura única de `stress` 2025-H2 sob
protocolo próprio.

## Fronteira científica

Vale integralmente a fronteira da ADR-010 e da ADR-011. Apenas `train` e
`validation` são legíveis. `test`, `stress` e `monitor` permanecem selados, e
o código de desenvolvimento não contém caminho de destravamento. O agregado
S8 de 2025-H1 continua servindo apenas como motivação e linha de base
imutável, e permanece proibido para seleção de candidato, seleção de
threshold, seleção de hiperparâmetro e inspeção de narrativas ou predições de
`test`.

O artefato do D2 publica apenas agregados. Não persiste narrativas,
identificadores, margens individuais nem pesos ajustados. O modelo ajustado
vive apenas na sessão de execução e é descartado com ela.

## Observabilidade obrigatória

Por semente e por janela, o runner publica contagem de decisões positivas do
estágio A, contagem de sobrescritas efetivas, threshold calibrado e o vetor
de métricas agregadas. Publica também o baseline do fallback puro recalculado
nas duas janelas, o eco da configuração congelada com seu hash, o hash do
artefato do D1 usado como titular e a assinatura do pool de negativos
difíceis. A assinatura torna execuções D2 comparáveis entre si, mas não prova
identidade com o D1, que não persistiu assinatura equivalente.

A revisão resolvida do modelo baixado é registrada no artefato. O
identificador é fixado em `distilbert-base-uncased`, mas a revisão exata só é
conhecida em tempo de execução, e registrá-la é preferível a fixar de antemão
um hash que não pode ser verificado localmente.

## Consequências e custos aceitos

O D2 exige GPU no Kaggle, o que é uma configuração de execução nova e
separada do D1. O custo estimado é de aproximadamente uma hora de GPU, três
ajustes finos sobre 15.136 linhas e três passagens de inferência sobre as
245.980 linhas somadas das janelas de calibração e externa.

Aceitamos que o D2 pode não deslocar o titular, e que esse é o desfecho mais
provável a priori, dado que o pool de treino tem apenas 946 positivos e que
o TF-IDF word+char é forte precisamente em sinais lexicais e de subpalavra,
que é onde reclamações de cobrança tendem a se concentrar. O valor do D2 não
depende de ele vencer: um Transformer compacto que não supera a união TF-IDF
neste regime é um resultado publicável e fecha o passo 4 do ciclo com
evidência em vez de omissão.

## Passo seguinte

Concluído o D2, o ciclo avança para o passo 5, aplicação da margem de
segurança e seleção do candidato V2, e para o passo 6, congelamento do pacote
V2. O passo 7, abertura única de `stress` 2025-H2, exigirá protocolo
confirmatório próprio e ADR própria, e não é autorizado por este documento.
