# ADR-009: S8 avaliação confirmatória

## Estado

**NOT_CONFIRMED**, executado em 2026-08-16.
O teste confirmatório foi aberto uma única vez com o protocolo congelado. O
resultado agregado foi publicado sem ajuste, recalibração ou novo treinamento.

## Decisão

O S8 avaliará uma única vez a partição `test`, de 2025-01-01 a 2025-06-30,
usando exclusivamente o bundle S7 congelado. O estimador recebe narrativas
`en-US`, usa a ordem fixa de `MODELED_FAMILIES` e aplica o threshold
`0.1135351095114484` como `critical_margin` para
`debt_credit_management`.

A visão científica contém somente famílias modeladas, grupos não vistos antes
do test, grupos com um único label no test e o representante de menor
`Complaint ID` por identidade `(normalized_group_hash, normalized_length)`.
A visão operacional avalia todas as linhas dos mesmos grupos limpos em batches.
Ela é publicada separadamente e não pode alterar a decisão.

## Gates e decisão

Os três gates simultâneos são macro-F1 maior ou igual a 0.69, F1 da classe
crítica maior ou igual a 0.2715 e precisão da classe crítica maior ou igual a
0.2. Com 3/3, o status será `CONFIRMED_FOR_STRESS_EVALUATION`; caso contrário,
`NOT_CONFIRMED`. O campo `confirmed` é separado do status. `deploy` é sempre
`false`; a confirmação nunca autoriza deployment.

Intervalos bootstrap estratificados pela matriz de confusão usam 2.000
replicações, seed 42 e nível 0.95. São diagnósticos e não participam dos gates.

## Fronteira de dados

Antes do token, o runner valida somente configurações, hashes, metadados e o
manifesto S7. O token é lido de `S8_CONFIRMATORY_UNLOCK` apenas em `full` e o
plaintext não aparece no notebook. Depois da autorização, DuckDB usa limite de
4 GB, uma thread e batches de 4.096. O raw é unido por `Complaint ID` somente
para a narrativa corrente, rejeitando vazios; nenhum texto, ID, score individual
ou cache de test é persistido.

O resultado agregado é escrito atomicamente em `temp/s8/s8_results.json` e o
manifesto público é criado depois da conclusão. Cache completo compatível não
relê raw ou índice e não exige token. Execuções incompletas não persistem
métricas parciais e só podem ser retomadas com o mesmo contrato, hashes e token.

## Evidência congelada

O protocolo congela os hashes atuais do config, manifesto, resultado e bundle
S7, além do raw e do índice hash-only. A evidência S2 registrada é
`post_2023_taxonomy: PASS`, com `test all_text=695184`, `novel_text=566040`,
`novel_unique_groups=334973` e `critical_novel_groups=1398`. `stress` de
2025-H2 e `monitor` de 2026 permanecem selados.

## Resultado confirmatório

A visão científica avaliou 334.230 representantes de grupos limpos e inéditos.
O macro-F1 foi 0.718214, a precisão da classe crítica foi 0.404615 e o F1 da
classe crítica foi 0.257843. Assim, os gates de macro-F1 e precisão crítica
passaram, mas o gate de F1 crítico (`0.257843 < 0.2715`) falhou. A decisão
confirmatória é `NOT_CONFIRMED`, com 2/3 gates, e `deploy=false`.

O recall da classe crítica foi 0.189209. Os intervalos bootstrap diagnósticos de
95% foram [0.714793, 0.721536] para macro-F1, [0.370913, 0.441502] para precisão
crítica e [0.233153, 0.282263] para F1 crítico. A visão operacional secundária,
com 520.292 linhas, obteve macro-F1 0.709749 e F1 crítico 0.246283; ela não
participou da decisão.

As diferenças em relação às contagens exploratórias S2 foram auditadas: -315
linhas inéditas, -427 grupos inéditos e -8 grupos críticos inéditos. Essas
diferenças decorrem da reconstrução confirmatória com o contrato congelado e
estão preservadas em `scope_counts.s2_difference`. O resultado completo está em
`temp/s8/s8_results.json`, e `config/s8_results.json` valida hashes, protocolo,
bundle e fronteiras seladas. `stress` e `monitor` não foram abertos.
