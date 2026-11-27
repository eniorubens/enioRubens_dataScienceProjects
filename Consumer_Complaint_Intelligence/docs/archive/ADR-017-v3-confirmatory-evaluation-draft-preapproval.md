# ADR-017: rascunho da confirmação temporal V3

## Estado

**DRAFT_AWAITING_USER_APPROVAL_AND_COMPLETE_SOURCE**, proposto em 2026-08-24.

Este documento ainda não autoriza execução. O pacote V3 está congelado, mas
`monitor` não será aberto, nenhum runner confirmatório será implementado e
nenhuma alegação confirmatória será feita enquanto o protocolo não receber
aprovação explícita e a fonte de 2026 não estiver completa.

## Por que não executar agora

A janela registrada como `monitor` vai de 2026-01-01 a 2026-12-31. Hoje é
2026-08-24. O snapshot local foi convertido em 2026-08-14 e o relatório S1
mostra 2026-07-23 como a data recebida mais recente. Logo, os dados atuais são
parciais.

Avaliar agosto como se fosse dezembro mudaria silenciosamente a pergunta e
favoreceria uma decisão baseada numa janela escolhida por conveniência. A
conduta proposta é esperar o encerramento do ano e o atraso de publicação do
CFPB, adquirir uma nova fotografia a partir de 2027-02-01 e congelar hash,
tamanho e manifesto antes de qualquer predição.

O monitor foi perfilado por metadados agregados no S2, mas nunca recebeu
predições V3. Sua descrição honesta é, portanto, *performance-sealed e
metadata-profiled*, não “jamais visto sob qualquer forma”.

## Objeto congelado

O V3-P reproduziu exatamente os seis blocos registrados em E3 e publicou o
detector `tfidf_word_char_ratio_1_10` com threshold
`0.10162120744876312`. O bundle V3 tem SHA-256
`FDA94C2C...DD99250A`; o S7 permanece o controle e fallback congelado, com
SHA-256 `5B302E1D...B4931049`.

Não haverá refit, recalibração, alteração de threshold, nova seleção de
candidato ou inspeção de erros antes da decisão. O braço primário será o
override V3 sobre o S7; o controle será o S7 sozinho. Ambos serão pontuados nas
mesmas linhas e numa única abertura.

## População científica

A visão primária conterá as nove famílias modeladas, somente reclamações
recebidas em 2026 e grupos não presentes no índice histórico congelado anterior
a 2026. A identidade continuará sendo
`(normalized_group_hash, normalized_length)`.

Grupos com mais de um label em 2026 serão contados e excluídos. Para cada grupo
limpo, o representante será o menor `Complaint ID`. A visão operacional com
todas as linhas dos mesmos grupos limpos será secundária e incapaz de alterar a
decisão.

Antes das predições, a janela precisará conter as nove famílias, pelo menos 500
grupos críticos únicos e participação máxima de família de 0,80. Falha de
escopo produzirá `INCONCLUSIVE_INSUFFICIENT_SCOPE`; não autorizará escolher
outra fração da mesma janela.

## Gates congelados para comparabilidade

Quatro gates preservam as barras usadas no S8 e no V2.1-C:

1. macro-F1 pelo menos 0,69;
2. F1 crítico pelo menos 0,2715;
3. precisão crítica pelo menos 0,20;
4. ganho pareado de F1 crítico sobre o S7 estritamente positivo.

Esses quatro números serão sempre publicados, ainda que a regra V3 seja mais
forte, para manter comparação direta com os ciclos anteriores.

## Decisão pendente: quinto gate V3

O protocolo V3-E0 usou precisão crítica de 0,40 para escolher thresholds e
exigiu que cada candidato cumprisse esse piso nos três folds. O requisito do
sistema combinado em F3, porém, permaneceu no legado 0,20.

Propõe-se carregar o compromisso mais forte para a confirmação: **precisão
crítica do sistema V3 pelo menos 0,40**, tornando a decisão 5 de 5. Isso é
prospectivo — ainda não vimos desempenho V3 em `monitor` —, mas precisa de
aprovação explícita porque altera o número de gates confirmatórios em relação
aos ciclos anteriores.

Se aprovado, `CONFIRMED` exigirá escopo adequado e cinco gates. Se rejeitado
antes da execução, exigirá os quatro gates históricos. A escolha será congelada
no protocolo final e não poderá ser revista depois do resultado.

## Incerteza e limites

Serão publicados intervalos bootstrap pareados com 2.000 réplicas, seed 42 e
95% de confiança, além da visão operacional secundária. São diagnósticos e não
substituem a regra pontual dos gates.

Confirmação nunca autoriza deployment. `deploy=false` continuará obrigatório.
Falhar um gate com escopo adequado produzirá `NOT_CONFIRMED`; não haverá ajuste
e segunda tentativa na mesma janela.

## Próximo portão de aprovação

Antes de escrever ou executar o runner, o Cientista de Dados deverá aprovar:

1. a promoção excepcional de `monitor` completo a janela confirmatória V3;
2. a espera por nova fotografia adquirida a partir de 2027-02-01;
3. os quatro gates históricos;
4. aceitar ou rejeitar o quinto gate de precisão 0,40;
5. a regra de escopo insuficiente;
6. a abertura única, sem refit e com `deploy=false`.

Depois disso, mas ainda antes de inferência, um protocolo final substituirá
todos os campos `TBD` pelos hashes da nova fonte e receberá novo manifesto.
