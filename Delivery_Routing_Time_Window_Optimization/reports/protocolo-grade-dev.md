# Protocolo pré-registrado — grade de desenvolvimento

Registrado em 2026-11-17, **antes** de qualquer execução da grade. Só usa as 14 instâncias com
`set = dev` em `data/reference/split.csv`. As instâncias de avaliação não são lidas.

## Configurações

| Fator | Níveis |
|---|---|
| Estratégia de frota | `fixed_cost` (um `solve` com custo fixo por veículo) · `reduce_fleet` (laço que re-resolve com um veículo a menos) |
| Solução inicial | `PARALLEL_CHEAPEST_INSERTION` · `PATH_CHEAPEST_ARC` · `SAVINGS` |
| Metaheurística | `GUIDED_LOCAL_SEARCH` · `TABU_SEARCH` · `SIMULATED_ANNEALING` |

- 2 × 3 × 3 = **18 configurações × 14 instâncias = 252 execuções**.
- Orçamento **total** de 30 s de relógio por execução, igual para as duas estratégias de frota
  (`reduce_fleet`: primeiro solve com 10 s; reduções com o tempo restante, em passos de 5 s, sem
  ultrapassar 30 s no total).
- Referência adicional (fora do torneio): as 3 construções sem metaheurística e as duas baselines.
- 1 processo OR-Tools por núcleo, até 8 em paralelo; uma réplica por célula (a busca é determinística
  dado o limite; a variação de relógio será medida depois, na avaliação, com réplicas).
- Toda solução é re-verificada em double; métricas vêm do verificador, nunca do objetivo do solver.

## Regra de seleção (fixada antes de ver resultados)

1. Para cada instância, ordenar as 18 configurações lexicograficamente por
   (factível, menos veículos, menor distância). Infactível = pior posição.
2. Calcular a **posição média** de cada configuração nas 14 instâncias.
3. Vencedora = menor posição média. Desempates, nesta ordem: menor média de Δ veículos vs SINTEF;
   menor mediana de gap de distância % entre as instâncias com frota igual à referência.
4. A configuração vencedora é **congelada** (`outputs/frozen/selected_config.json`) e é a única
   levada à avaliação nas 42 instâncias.

O orçamento de tempo da avaliação final e o número de réplicas serão decididos com base no tempo
medido nesta grade, antes de abrir a avaliação.

---

## Resultados (executado em 2026-11-17, após o registro acima)

Execução: `scripts/run_dev_grid.py --budget 30 --workers 8` → 322 células (252 do torneio + 70 de
referência), **322/322 factíveis** no verificador em double. Seleção: `scripts/select_config.py`,
regra aplicada sem alteração. Artefatos: `outputs/dev_grid/results.csv`, `outputs/dev_grid/ranking.csv`,
`outputs/frozen/selected_config.json`.

### Configuração congelada

**`fixed_cost` + `SAVINGS` + `GUIDED_LOCAL_SEARCH`, 30 s** — posição média 5,61 de 18
(2ª: `fixed_cost`/`SAVINGS`/`TABU_SEARCH`, 5,89; 3ª: `fixed_cost`/`PARALLEL_CHEAPEST_INSERTION`/GLS, 6,39).

| Instância | Família | SINTEF (veíc. / dist.) | Congelada (veíc. / dist.) | Melhor das 18 (veíc. / dist.) |
|---|---|---|---|---|
| C103 | C1 | 10 / 828,06 | 10 / 859,00 | 10 / 828,06 |
| C106 | C1 | 10 / 828,94 | 10 / 828,94 | 10 / 828,94 |
| C203 | C2 | 3 / 591,17 | 3 / 600,21 | 3 / 591,17 |
| C206 | C2 | 3 / 588,49 | 3 / 588,49 | 3 / 588,49 |
| R108 | R1 | 9 / 960,88 | 10 / 970,31 | 10 / 967,80 |
| R110 | R1 | 10 / 1118,84 | 12 / 1102,83 | 12 / 1102,83 |
| R111 | R1 | 10 / 1096,73 | 11 / 1107,99 | 11 / 1107,99 |
| R201 | R2 | 4 / 1252,37 | 4 / 1301,51 | 4 / 1283,08 |
| R202 | R2 | 3 / 1191,70 | 4 / 1114,65 | 4 / 1114,65 |
| R207 | R2 | 2 / 890,61 | 3 / 851,58 | 3 / 851,58 |
| RC106 | RC1 | 11 / 1424,73 | 14 / 1493,40 | 13 / 1444,30 |
| RC108 | RC1 | 10 / 1139,82 | 11 / 1261,69 | 11 / 1208,02 |
| RC203 | RC2 | 3 / 1049,62 | 4 / 1023,75 | 3 / 1198,09 |
| RC206 | RC2 | 3 / 1146,32 | 4 / 1192,64 | 4 / 1136,33 |

### Leitura

- A configuração congelada iguala a frota do SINTEF em **5/14** (todas as C e R201); C106 e C206
  reproduzem a distância de referência. Nas famílias R e RC fica em geral **+1 veículo** (+2 em R110, +3 em RC106).
  Distâncias menores que a referência (R110, R202, R207, RC203) vêm sempre com **mais veículos** — não
  são ganhos na convenção hierárquica.
- A busca local é decisiva: sem metaheurística, a melhor construção (SAVINGS) fica +2,8 veículos em
  média; com GLS, +0,86. As baselines ficam em +1,5 (NN-TW) e +4,9 (gulosa).

### Limitações encontradas (registradas, sem alterar a seleção)

1. **`reduce_fleet` não foi testada de forma efetiva.** Em 126/126 células a primeira tentativa com um
   veículo a menos não encontrou solução em 5 s e o laço parou (tempo médio usado ≈ 10 s de 30 s).
   Falhar por tempo não prova infactibilidade: o passo de 5 s foi curto demais. A comparação
   `fixed_cost` × `reduce_fleet` deste torneio é, portanto, inconclusiva sobre o mérito da estratégia.
2. `TABU_SEARCH` terminou antes do limite em parte das células (tempo médio 22–27 s); todas com status
   `ROUTING_SUCCESS`.
3. Uma réplica por célula; a variabilidade de relógio ainda não foi medida.

## Correção (2026-11-18)

O item 1 das limitações acima atribui a falha de `reduce_fleet` a um passo de 5 s curto demais. **Essa
explicação estava errada.** Os tempos totais das 126 células (`outputs/frozen/dev_grid/results.csv`) são
10,07–10,28 s com `PARALLEL_CHEAPEST_INSERTION` (mediana 10,14 s), mediana 10,47 s com `SAVINGS` (1,08–12,10 s)
e cerca de 15 s só com `PATH_CHEAPEST_ARC` (mediana 15,01 s; 40 de 42 células usaram o passo inteiro de 5 s).
Ou seja, com as duas primeiras construções a segunda tentativa (um veículo a menos) falhou quase de imediato,
compatível com falha na construção (o modelo não permite descartar clientes e a busca parte do zero), e o
laço descartou o restante do orçamento: `reduce_fleet` competiu com cerca de 10 s contra os 30 s de
`fixed_cost`. Com `PATH_CHEAPEST_ARC` o limite de tempo não pode ser descartado como causa. A célula de
1,08 s (C106, `SAVINGS`, `TABU_SEARCH`) tem a primeira resolução encerrada cedo (a mesma configuração com
`fixed_cost` levou 0,95 s) e a segunda tentativa falhou em seguida. A comparação `fixed_cost` ×
`reduce_fleet` deste torneio continua **inconclusiva e enviesada contra `reduce_fleet`**; corrigi-la exige
partida a quente e descarte penalizado (experimento B3 de `reports/plano-sanidade-auditoria.md`). Tabela
reproduzível em `analysis.reduce_fleet_timing` (notebook 04, seção 8). A seleção da configuração congelada
não foi alterada; o texto original foi mantido acima por rastreabilidade.
