# ADR-017: confirmação temporal V3

## Estado

**APPROVED_AWAITING_COMPLETE_SOURCE_FREEZE**, aprovado pelo Cientista de Dados
em 2026-08-24.

Os seis pontos do portão metodológico e os cinco gates foram aprovados antes de
qualquer predição V3 em `monitor`. Esta ADR congela o método, mas não autoriza a
execução: a fotografia completa de 2026 ainda não existe no projeto e sua
identidade deverá ser fixada e aprovada antes da abertura.

## Por que não executar agora

A janela confirmatória vai de 2026-01-01 a 2026-12-31. Em 2026-08-24, o
snapshot local contém reclamações recebidas somente até 2026-07-23. Usá-lo
responderia a uma pergunta parcial, diferente da pergunta registrada.

O protocolo exige uma nova fotografia oficial do CFPB adquirida a partir de
2027-02-01, após o fechamento da janela e uma margem para publicação. Antes de
qualquer inferência, serão congelados o hash, o tamanho, a data máxima e o
manifesto dessa fonte. A execução dependerá de aprovação explícita da
identidade do snapshot.

O monitor já foi perfilado por metadados agregados no S2, mas nunca recebeu
predições V3. A descrição correta é *performance-sealed e metadata-profiled*;
não se afirma que nenhum metadado jamais foi observado.

## Aprovação dos seis pontos

Em 2026-08-24, o usuário aprovou explicitamente:

1. promover o `monitor` completo de 2026 a janela confirmatória V3;
2. esperar uma nova fotografia adquirida a partir de 2027-02-01;
3. preservar os quatro gates históricos;
4. adicionar precisão crítica do sistema de pelo menos 0,40 como quinto gate;
5. classificar falha de escopo como inconclusiva, sem reutilizar outra fração da
   mesma janela;
6. fazer uma única abertura, sem refit, mantendo `deploy=false`.

A expectativa pessoal de que o sistema talvez não passe todos os gates não
altera o protocolo. Esse é precisamente o papel do pré-registro: manter a regra
quando o resultado esperado parece desfavorável.

## Objeto congelado

O braço primário é o detector V3 `tfidf_word_char_ratio_1_10`, com threshold
`0.10162120744876312`, sobrescrevendo o fallback S7 somente quando decide pela
classe `debt_credit_management`. O controle é o mesmo S7 sozinho, pontuado nas
mesmas linhas e na mesma passagem.

O bundle V3 tem SHA-256
`FDA94C2CFEF2DF3389F9E74089BA64540BBA130F0E55B373A892A472DD99250A`.
O bundle S7 tem SHA-256
`5B302E1D958EC99A2ECB3C8B2DA5E5F52F613C5112153A76086FC5ACB4931049`.

Refit, recalibração, mudança de threshold, nova seleção de candidato e inspeção
de erros antes da decisão estão proibidos.

## População científica

A visão primária conterá somente as nove famílias modeladas e reclamações
recebidas em 2026. Grupos já presentes no índice histórico anterior a 2026
serão removidos pela identidade
`(normalized_group_hash, normalized_length)`.

Grupos com mais de um label serão contados e excluídos. O representante de cada
grupo limpo será o menor `Complaint ID`. A visão operacional com todas as linhas
dos mesmos grupos limpos será secundária e não poderá alterar a decisão.

Antes de produzir predições, a janela deverá conter:

- as nove famílias modeladas;
- pelo menos 500 grupos críticos únicos;
- participação máxima de qualquer família de 0,80.

Falha de escopo produzirá `INCONCLUSIVE_INSUFFICIENT_SCOPE`, não
`NOT_CONFIRMED`. Outra fração da mesma janela não poderá ser escolhida como
substituta.

## Os cinco gates confirmatórios

Todos são simultâneos e avaliados sobre a visão científica:

1. macro-F1 pelo menos 0,69;
2. F1 crítico pelo menos 0,2715;
3. precisão crítica pelo menos 0,20;
4. ganho pareado de F1 crítico sobre o S7 estritamente positivo;
5. **precisão crítica do sistema V3 pelo menos 0,40**.

Os quatro primeiros serão reportados separadamente para comparação direta com
S8 e V2.1-C. O quinto é a barra de segurança própria do V3. Embora os gates 3 e
5 usem a mesma métrica, ambos permanecem registrados: o primeiro preserva a
série histórica e o segundo determina a exigência mais forte do V3.

`CONFIRMED` exigirá escopo adequado e 5 de 5. Com escopo adequado, qualquer
falha produzirá `NOT_CONFIRMED`. O número de gates não poderá ser reduzido após
acesso à nova fonte.

## Diagnósticos e limites

Intervalos bootstrap pareados usarão 2.000 réplicas, seed 42 e confiança de
95%. A visão operacional e os intervalos serão diagnósticos; não substituirão
a regra pontual dos gates.

Confirmação nunca autoriza implantação. `deploy=false` é invariável, inclusive
se os cinco gates passarem. Não haverá segunda tentativa sobre a mesma janela.

## Próximo portão

O próximo passo só poderá ocorrer a partir de 2027-02-01:

1. adquirir a fotografia completa do CFPB;
2. verificar que a data recebida máxima alcança 2026-12-31;
3. congelar hashes, tamanhos e manifesto da fonte;
4. substituir os campos `TBD` sem modificar método, população ou gates;
5. solicitar aprovação da identidade da fonte e da execução única.

Até lá, o runner permanece inexistente, o token de abertura permanece
indefinido e `monitor` continua selado para desempenho V3.
