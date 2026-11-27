# ADR-016: congelamento do pacote de desenvolvimento V3

## Estado

**FROZEN_FOR_V3_PACKAGE_EXECUTION**, aprovado pelo Cientista de Dados em
2026-08-24. Este ADR autoriza somente o estágio `V3-P` previsto no protocolo
V3-E0. Não autoriza abrir `test`, `stress` ou `monitor` e não autoriza alegação
confirmatória ou implantação.

## Contexto

O V3-E3 comparou seis candidatos sob três folds rolling-origin. Os três
candidatos TF-IDF cumpriram o piso de precisão crítica de 0,40 em todas as
janelas. Os três candidatos MiniLM falharam esse requisito em F2.

Pela ordenação congelada — pior recall crítico no piso de precisão, PR-AUC
mediana, F1 mediano, amplitude de F1 e runtime — o candidato selecionado foi
`tfidf_word_char_ratio_1_10`.

O sistema avaliado em F3 combina esse detector com o S7 congelado. O detector
sobrescreve a previsão para `debt_credit_management` quando sua probabilidade
calibrada atinge o threshold; caso contrário, a previsão S7 é preservada. Em F3
o sistema passou os quatro requisitos de desenvolvimento e superou o S7 em
`0.0272706794` de F1 crítico.

## Identidade do objeto a congelar

O pacote deve reproduzir exatamente o objeto medido em F3:

- ajuste: `train`, de 2023-08-01 a 2024-06-30;
- 946 positivos críticos únicos;
- razão negativa 1:10, com 6.622 hard negatives e 2.838 negativos de fundo;
- assinatura do pool:
  `C20F9823FB5C55D6CAC4AC8243CD5FC3AE7B6DD45F77048F60A7770F768D6138`;
- TF-IDF word/char `40000/60000` e LinearSVC `C=1`;
- calibrador sigmoide ajustado somente em `validation` de 2024-07-01 a
  2024-09-30;
- threshold `0.10162120744876312`;
- avaliação de reprodução em `validation` de 2024-10-01 a 2024-12-31;
- S7 referenciado por hash, sem duplicação no pacote V3.

Não haverá refit incluindo outubro-dezembro. Fazer isso criaria um modelo novo,
diferente daquele que produziu a evidência selecionada.

## Portão de reprodução

O V3-E3 descartou os estimadores e não persistiu IDs do pool. V3-P precisa
regenerar deterministicamente a mineração OOF e reajustar o candidato. O pacote
só será publicado se reproduzir, sem tolerância numérica:

- assinatura e contagens do pool;
- bloco completo de calibração publicado;
- bloco completo de avaliação binária publicado;
- threshold selecionado;
- bloco completo da comparação híbrida com S7 em F3;
- identidade e ordem do candidato selecionado;
- hashes das entradas congeladas.

Se qualquer verificação divergir, o resultado será
`REPRODUCTION_MISMATCH`, o desvio será documentado e nenhum bundle será
publicado.

## Privacidade e fronteira científica

O bundle pode persistir vocabulário, pesos ajustados, calibrador e threshold.
Ele não pode persistir narrativas, IDs, índices de linha, scores individuais,
probabilidades individuais ou posições do pool. `test`, `stress` e `monitor`
permanecem selados durante toda a execução.

## Consequência

Se o portão passar, o bundle recebe status `FROZEN_FOR_CONFIRMATION` e fica
carregável apenas para uma futura avaliação confirmatória pré-registrada. Um
protocolo confirmatório separado deverá ser aprovado antes de qualquer abertura
de `monitor` ou aquisição de nova janela futura.

## Resultado da execução, registrado em 2026-08-24

O V3-P terminou como `PACKAGE_FROZEN`, com status
`FROZEN_FOR_CONFIRMATION`. Os seis checks reproduziram exatamente o V3-E3:
identidade do candidato, contagens do pool, assinatura do pool, calibração,
avaliação binária e comparação do sistema híbrido em F3. A execução levou
3.531,14 segundos e não abriu `test`, `stress` ou `monitor`.

O primeiro arquivo joblib foi criado pelo entrypoint `python -m`, que atribuiu
à classe serializada o nome não importável `__main__.FrozenV3Detector`. Os
conteúdos científicos e os hashes estavam corretos, mas um novo processo não
conseguia carregar o objeto. O incidente foi detectado pelo teste pós-pacote.

A correção reserializou o mesmo objeto já ajustado sob o módulo canônico
`consumer_complaint_intelligence.v3.package`; não houve refit, recálculo de
métricas ou acesso a dados. Um smoke test confirmou carregamento normal e
predições invariantes ao tamanho do batch. Os hashes finais são:

- bundle: `FDA94C2C...DD99250A`;
- resultado agregado: `FAB58BFA...C5C09DF`;
- manifesto: `A0E10ABD...D266923`.

O protocolo confirmatório permanece um estágio separado. O congelamento do
pacote não autoriza executá-lo.
