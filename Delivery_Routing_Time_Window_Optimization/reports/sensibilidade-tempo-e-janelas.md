# Sensibilidade ao tempo de busca e cenário de janelas alargadas

Data: 2026-11-17 · Somente as 14 instâncias de desenvolvimento · Configuração congelada
(`fixed_cost` + `SAVINGS` + `GUIDED_LOCAL_SEARCH`) · Parâmetros em `config/experiment.toml` ·
Resultados em `outputs/frozen/time_sensitivity/` e `outputs/frozen/window_scenarios/`.

## 1. Camada A — mais tempo de busca reduz a frota?

70/70 soluções factíveis no verificador em double.

| Orçamento | Δ veículos médio vs SINTEF | Instâncias com frota igual ao SINTEF |
|---|---|---|
| 10 s | 1,00 | 5/14 |
| 30 s | 0,86 | 5/14 |
| 60 s | 0,79 | 6/14 |
| 120 s | 0,64 | 7/14 |
| 300 s | 0,64 | 7/14 |

Veículos por instância (SINTEF entre parênteses):

| Família | Instância | 10 s | 30 s | 60 s | 120 s | 300 s |
|---|---|---|---|---|---|---|
| C1 / C2 | C103 (10), C106 (10), C203 (3), C206 (3) | = | = | = | = | = |
| R1 | R108 (9) | 10 | 10 | 10 | 10 | 10 |
| R1 | R110 (10) | 12 | 12 | 12 | 12 | 12 |
| R1 | R111 (10) | 12 | 11 | 11 | 11 | 11 |
| R2 | R201 (4) | 4 | 4 | 4 | 4 | 4 |
| R2 | R202 (3) | 4 | 4 | 4 | 4 | 4 |
| R2 | R207 (2) | 3 | 3 | 3 | 3 | 3 |
| RC1 | RC106 (11) | 14 | 14 | 14 | 13 | 13 |
| RC1 | RC108 (10) | 12 | 11 | 11 | 11 | 11 |
| RC2 | RC203 (3) | 4 | 4 | **3** | **3** | **3** |
| RC2 | RC206 (3) | 4 | 4 | 4 | **3** | **3** |

Gap de distância onde a frota é igual: R201 cai de 4,71% (10 s) para 1,16% (300 s); C203 chega a
0,00% em 300 s; RC203 e RC206 igualam a frota com gap de 7,2% e 10,6%.

**Leitura.** Mais tempo ajuda até ~120 s e depois estabiliza (120 s e 300 s idênticos em frota).
O ganho vem das famílias RC2 e RC1; **R1 e R2 não mudam de frota nem com 300 s** — o veículo extra
nessas famílias não é fechado por tempo de busca com esta configuração. Custo: 4× o tempo (30 → 120 s)
reduz o Δ médio de 0,86 para 0,64.

## 2. Camada B — janelas de atendimento mais largas (sem comparação com o SINTEF)

Rodada limpa, 56/56 soluções factíveis no verificador **contra a instância alargada**. Comparação com
o próprio plano no fator 1,0.

| Fator de largura | Veículos economizados (média) | Variação de distância (média) |
|---|---|---|
| 1,25 | 0,29 | −4,5% |
| 1,50 | 0,57 | −5,8% |
| 2,00 | 0,79 | −7,9% |

- **Famílias C: nenhum veículo economizado em nenhum fator** — coerente com o notebook 02, em que a
  frota das famílias C coincide com o limite por capacidade.
- R e RC: com janelas 2× mais largas economiza-se 1 veículo em quase todas (R110 e RC106: 2–3).
- Interpretação de negócio: flexibilizar janelas só reduz frota onde as janelas, e não a capacidade,
  são a restrição ativa.

## 3. Achado metodológico — variabilidade por relógio

Execuções **idênticas** (mesma instância, configuração e 30 s) não produzem sempre a mesma solução:

- grade de desenvolvimento × curva de tempo (30 s): idênticas nas 14 instâncias;
- rodada limpa do cenário (fator 1,0) × curva de tempo (30 s): RC203 com 3 veículos contra 4;
  distâncias diferentes em 3/14 instâncias;
- primeira rodada do cenário (com notebooks e testes executando em paralelo): 4 instâncias com ±1
  veículo em relação à rodada limpa.

Causa: o OR-Tools para pelo relógio; o ponto de parada (e portanto a melhor solução encontrada) depende
da carga da máquina. Nas famílias C o resultado é estável; nas R e RC, ±1 veículo é possível.

**Consequências:**

1. Diferenças de ±1 veículo em uma única execução por instância não são evidência; as conclusões acima
   valem como tendência agregada.
2. A avaliação final precisa de **réplicas** (proposta: 5 por instância) e de reportar mediana e
   amplitude, ou de um critério de parada determinístico (`solution_limit`, contado em soluções
   encontradas) com o tempo de relógio apenas registrado.
3. Experimentos cronometrados não devem coincidir com outras cargas na máquina.

## 4. Implicações para a avaliação (decisão pendente)

| Opção | Orçamento | Execuções (42 inst.) | Tempo de relógio estimado (8 processos) |
|---|---|---|---|
| A | 30 s × 5 réplicas | 210 | ~15 min |
| B | 120 s × 5 réplicas | 210 | ~55 min |
| C | 120 s × 3 réplicas | 126 | ~35 min |

O orçamento congelado no torneio foi 30 s; usar 120 s na avaliação é legítimo desde que decidido
**antes** de abrir as instâncias de avaliação e reportado como tal.

## Correção (2026-11-18)

A tabela de janelas alargadas acima traz uma média única de distância por fator, que mistura instâncias que
reduzem a frota com instâncias que a mantêm. No notebook 06 a distância é reportada por classe de frota: com
fator 2,0, −2,0% nas 6 instâncias de frota igual e −12,3% nas 8 de frota reduzida (veículos economizados:
0,79 em média). Em ordem lexicográfica (veículos, depois distância), as 6 células não monotônicas das
janelas alargadas são todas pioras de distância com a mesma frota (C103 a ×1,5, C106 a ×1,25 e ×2, C203 a ×2,
R108 a ×1,5, RC108 a ×2). O texto original foi mantido por rastreabilidade.
