# ADR-013: seleção do candidato V2 sob margem e congelamento do pacote

## Estado

**FROZEN_FOR_V2_PACKAGE_EXECUTION**, aprovado pelo Cientista de Dados em
2026-08-19. Este ADR executa os passos 5 e 6 do ciclo de sete passos da
ADR-010, `apply_safety_margin_and_select_v2_candidate` e
`freeze_v2_package`, sob as janelas fixadas pela ADR-011 e depois de
encerrado o desafio da ADR-012. Não altera janelas, gates, margens,
fronteira de partições nem o catálogo clássico, e não autoriza a abertura
de nenhuma partição selada.

## Contexto: o que já está decidido antes deste documento

O D1 sob o protocolo V2.1 avaliou 30 candidatos. Os 30 passaram 3/3 nas
margens de desenvolvimento, 22 tiveram sobrescritas efetivas na janela
externa e os mesmos 22 superaram o baseline do fallback. Pela ordenação
pré-registrada, `critical_f1` decrescente à frente de macro-F1, precisão
crítica e tempo de execução, o primeiro colocado foi
`word_char_tfidf_union_40000_60000_c_1_hard_negative`.

O D2 desafiou esse titular com um `distilbert-base-uncased` sob desenho
controlado e regra de decisão pré-registrada, e o desfecho publicado foi
`CLASSICAL_WINNER_STANDS`. O Transformer superou o fallback com folga e
superou o titular por +0,015825 de F1 crítico, mas ficou abaixo do piso de
precisão de 0,434286 e abaixo da barra de deslocamento de 0,425599. Nenhuma
das três sementes passaria, o que torna o desfecho independente da regra de
agregação.

Portanto o passo 5 não abre nenhuma escolha nova. Ele ratifica um resultado
que já existe e que foi produzido por regras registradas antes de qualquer
número. Reabrir a seleção clássica está proibido pela ADR-012, e este ADR
não a reabre.

## Passo 5: o candidato V2 sob a margem de segurança

O candidato V2 é `word_char_tfidf_union_40000_60000_c_1_hard_negative`,
avaliado na janela externa contra as margens de desenvolvimento, que são
mais exigentes do que os gates científicos:

- macro-F1 0,731214 contra mínimo de 0,70, folga +0,031214;
- F1 crítico 0,386899 contra mínimo de 0,29, folga +0,096899;
- precisão crítica 0,437500 contra mínimo de 0,22, folga +0,217500.

Passa 3/3. Tem 82 sobrescritas efetivas em 258 decisões positivas do
estágio A sobre 127.706 linhas, ou seja, não é um candidato degenerado
idêntico ao fallback. E supera o fallback puro em +0,047234 de F1 crítico,
com precisão praticamente inalterada, 0,437500 contra 0,434286.

Registramos também o que a margem não diz. Essas folgas foram medidas na
mesma janela que serviu de superfície de seleção, primeiro entre 22
elegíveis e depois no desafio do D2. São folgas de desenvolvimento com
viés otimista, não estimativas de desempenho futuro. A única evidência
independente virá do passo 7.

## Passo 6: o que exatamente é congelado

O pacote V2 é o sistema hierárquico completo, não apenas o detector:

- estágio A, o detector binário crítico selecionado, com o vetorizador
  união word mais char ajustado e o LinearSVC binário ajustado;
- o limiar calibrado, -0,13949530151425016;
- estágio B, o pacote S7 congelado, referenciado por hash e não duplicado;
- a regra de combinação, sobrescrita do estágio B pelo estágio A somente
  quando a margem do estágio A atinge o limiar.

O artefato ajustado é persistido em
`artifacts/v2/consumer_complaint_detector_v2.joblib`. A configuração
congelada é `config/v2_frozen_package.json`, o resultado completo é
`temp/v2/v2_package.json` e o manifesto com hashes é
`config/v2_results.json`.

## Escopo de ajuste e origem do limiar: por que não repetimos o S7

O S7 foi congelado ajustando em `train` e recalibrando o limiar sobre a
`validation` inteira, os dois trimestres. É natural perguntar se o V2 deve
fazer o mesmo, já que mais dados de calibração costumam ser melhores.

Não deve, e a razão não é preferência nossa. O protocolo congelado fixa
`selection.threshold_source: "inner_calibration_only"`. O limiar do estágio
A só pode vir da janela de calibração interna, `validation` de 2024-07-01 a
2024-09-30. Recalibrar sobre a `validation` inteira consumiria a janela
externa, que é exatamente a superfície onde o candidato foi selecionado, e
violaria uma cláusula do protocolo em vigor desde antes do D1.

O escopo de ajuste segue igualmente inalterado, `inner_fit`, `train` de
2023-08-01 a 2024-06-30, com o mesmo pool de negativos difíceis de 15.136
linhas, 946 positivos e 14.190 negativos difíceis mais de fundo. Estender o
ajuste para incluir a `validation` exigiria regenerar o pool OOF sobre um
escopo maior, o que mudaria o pool sobre o qual os 30 candidatos foram
comparados. Isso seria uma nova iteração de desenvolvimento sobre estas
janelas, proibida pela ADR-011.

Há um custo real nessa escolha e o declaramos: o pacote V2 não aproveita
dois trimestres de dados de `validation` que existem e estão liberados. Em
troca, o artefato congelado é numericamente o mesmo objeto sobre o qual
todas as afirmações do D1 e do D2 foram feitas. Um passo confirmatório só
confirma alguma coisa se o que ele testa for o que foi medido.

## O portão de reprodução

Nenhum modelo ajustado foi persistido no D1. O benchmark publicou apenas
agregados e descartou os 30 estimadores com a sessão. Congelar o pacote,
portanto, exige reajustar o candidato selecionado, e reajustar cria a
possibilidade de o artefato congelado diferir daquele que produziu os
números publicados.

Por isso o congelamento é condicionado a um portão de reprodução exata. A
execução do passo 6 reajusta o candidato sob as mesmas regras e só publica
o pacote se reproduzir, sem tolerância, os valores publicados pelo D1:

- o limiar calibrado, -0,13949530151425016;
- a matriz de confusão inteira da janela de calibração e a da janela
  externa;
- as contagens de decisões positivas e de sobrescritas efetivas nas duas
  janelas, 57 e 16 na calibração, 258 e 82 na externa;
- as contagens do pool de negativos difíceis, 946 positivos e 14.190
  negativos.

Comparação exata, e não por tolerância numérica, porque a pergunta aqui não
é se dois números são próximos. É se o objeto congelado é o mesmo objeto
medido. Se qualquer item divergir, o desfecho é `REPRODUCTION_MISMATCH`,
nenhum pacote é publicado, a divergência é publicada como evidência e a
decisão sobre como proceder volta ao Cientista de Dados. Nada é congelado
em silêncio sobre números que não batem.

Esse portão também trata uma ressalva registrada no relatório do D2. O D1 não
publicou assinatura do pool, campo que só foi introduzido no D2, de modo que a
identidade entre os pools do D1 e do D2 não pode ser provada por hash
retroativamente. Uma reprodução exata da matriz de confusão externa do D1
prova reprodução comportamental nas verificações medidas, mas não identidade
linha a linha do pool: pools distintos podem, em princípio, produzir a mesma
matriz agregada.

Se a execução local não reproduzir, a hipótese principal é diferença de
BLAS ou de desempate numérico entre a máquina local e a imagem do Kaggle,
onde o D1 rodou. Nesse caso a execução é repetida no Kaggle, no mesmo
ambiente do D1, antes de qualquer outra conclusão. Uma divergência que
persista no ambiente original seria um achado científico, não um
contratempo operacional, e seria tratada como tal.

## Fronteira científica e privacidade

Vale integralmente a fronteira da ADR-010, da ADR-011 e da ADR-012. Apenas
`train` e `validation` são legíveis. `test`, `stress` e `monitor` permanecem
selados e o código de desenvolvimento não contém caminho de destravamento.
O agregado S8 de 2025-H1 continua servindo apenas como motivação e linha de
base imutável.

Uma cláusula muda em relação ao D2, e a mudança é deliberada. O D2 declarava
`persists_fitted_weights: false`, porque um desafio precisa publicar
evidência e não artefato. O passo 6 declara `persists_fitted_weights: true`,
porque congelar um pacote é precisamente persistir pesos ajustados. O que
não muda é `persists_narratives_or_identifiers: false`. O pacote guarda
vocabulário, pesos e o limiar. Não guarda narrativas, identificadores,
índices de linha, margens individuais nem o pool de negativos difíceis, que
segue sendo um objeto de memória regenerável de forma determinística.

## Estado de implantação

O protocolo declara `deployment_authorized: false` e este ADR não altera
isso. O pacote é congelado com status `FROZEN_FOR_CONFIRMATION`. Ele é
carregável e aplicável em código, o que é necessário para o passo 7, mas não
é um pacote de produção e não há autorização para servi-lo.

## Consequências

O ciclo de desenvolvimento do V2 se encerra aqui. Depois deste ADR não há
mais seleção, ajuste de limiar, varredura ou revisão de janelas sobre
`train` e `validation`. Qualquer número novo produzido sobre essas janelas
passa a ser diagnóstico, nunca decisório.

O passo 7, abertura única de `stress` 2025-H2, exigirá protocolo
confirmatório próprio e ADR própria, e não é autorizado por este documento.
O que este documento entrega é o objeto que aquele passo deverá testar, com
identidade provada por hash e por reprodução.

## Resultado da execução, registrado em 2026-08-19

Esta seção registra o que aconteceu. Ela não altera nenhuma regra
pré-registrada acima, que permaneceu congelada durante toda a execução.

A execução rodou no Kaggle, kernel `eniorubens/cci-v2-1-p-frozen-package`,
em 1.309 segundos de CPU. O portão de reprodução passou em 21 de 21
verificações, incluindo as 8 canônicas. O limiar, as duas matrizes de
confusão inteiras, os dois pares de contagem de sobrescrita e as contagens
do pool reproduziram o D1 exatamente. O desfecho é `PACKAGE_FROZEN` e o
pacote foi persistido em
`artifacts/v2/consumer_complaint_detector_v2.joblib`, 1.774.532 bytes,
sha256 `CD9193D3B3B1BD94931F4FE38393245D498DCF379AFBE0AF1039C79AB66649E0`.

O portão esclareceu parte da questão do pool, sem provar identidade retroativa
com o D1. A assinatura do pool desta execução é
`D173E5CD476B2F10404E4F694DAE1ED4BC457EB34D687FC719339DF07146C9D6`. Como a
reprodução da matriz externa foi exata, o comportamento medido do D1 foi
reproduzido. Isso não torna o pool necessariamente idêntico ao do D1. A
assinatura coincide com a obtida no ensaio local do D2. A assinatura publicada
pelo D2, porém, é
`A9DED33BC707823A01ACF511FCBD22487B3D13808B306608DA058E03CAFA0288`, e
portanto difere.

A divergência observável está na execução D2 publicada: ela rodou no kernel de
GPU, com uma imagem diferente, e uma diferença de desempate numérico na
ordenação OOF é uma hipótese plausível para a troca de negativos na fronteira
do corte. As contagens são idênticas, 946 positivos e 14.190 negativos
difíceis, mas a identidade linha a linha entre todas as execuções não é
comprovada. Quantas linhas diferem não é recuperável, porque os índices do pool
nunca são persistidos, o que é uma decisão de privacidade que mantemos.

O que isso faz com o D2: retira a afirmação da ADR-012 de que o pool seria
idêntico linha a linha; a identidade do pool do D1 permanece não comprovada. O
que isso não faz: mudar o desfecho do D2. O Transformer ficou 0,0229 abaixo da
barra de deslocamento e 0,0057 abaixo do piso de precisão, e nenhuma das
três sementes passaria. Uma troca de negativos quase empatados na
fronteira do ranking é uma perturbação de ordem muito menor do que essas
distâncias. Registramos a imprecisão sem convertê-la em dúvida sobre uma
conclusão que ela não alcança.
