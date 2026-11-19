# Cenário de entrega antecipada — ganho de tempo e realocação de entregas

Data: 2026-11-17 · Camada B (cenário de negócio; **sem comparação com o SINTEF**) · 14 instâncias de
desenvolvimento · configuração congelada, 30 s · 3 réplicas por célula · máquina sem outra carga ·
parâmetros em `config/experiment.toml` (`[early_delivery]`) · resultados em
`outputs/frozen/early_delivery/`.

## Definição

No benchmark, chegar antes da abertura da janela obriga a esperar. No cenário, a abertura de cada janela
é antecipada em uma fração da sua largura (25%, 50%, 100%) ou removida (“full”: pode entregar a qualquer
hora até o fim da janela). O fim da janela não muda. Solver e verificador não foram alterados: apenas os
dados de entrada são transformados, e cada plano é verificado contra a instância transformada
(210/210 factíveis).

**Duração justa.** As durações são calculadas com a **saída do depósito ajustada** (atraso máximo útil
da partida pela folga de tempo de Savelsbergh). Medir com todos os veículos saindo no instante 0 infla a
espera do plano original e superestima o ganho (ex.: no smoke, RC106 “full”: −16,2% com saída às 0
contra −7,4% com saída ajustada).

O OR-Tools continua minimizando frota e depois distância; **o ganho de tempo é efeito colateral do novo
plano, não um objetivo otimizado**.

## Resultados (médias das 14 instâncias, medianas das réplicas)

| Antecipação | Veículos economizados | Distância | Duração das rotas | Último retorno (makespan) |
|---|---|---|---|---|
| 25% | 0,29 | −1,9% | −3,7% | +0,1% |
| 50% | 0,36 | −5,3% | −6,4% | −1,4% |
| 100% | 0,50 | −7,1% | −9,4% | −4,1% |
| full | 0,50 | −13,2% | **−15,4%** | −12,9% |

Por família (duração das rotas, saída ajustada):

| Família | 25% | 50% | 100% | full | Veículos economizados (full) |
|---|---|---|---|---|---|
| C1 / C2 | ≈ 0% | ≈ 0% | ≈ 0% | ≈ 0% | −0,5 / 0 |
| R1 | −0,6% | −2,2% | −3,5% | −3,8% | 1,3 |
| R2 | −3,4% | −12,6% | −17,0% | **−36,7%** | 0 |
| RC1 | −6,5% | −7,7% | −6,8% | −8,7% | 1,5 |
| RC2 | −13,1% | −14,9% | −28,1% | **−37,9%** | 0,5 |

**Leitura.**

- O ganho de tempo concentra-se nas **séries 2 (R2, RC2)**, de horizonte longo: rotas longas com muita
  espera; liberar a antecipação elimina a espera e encurta as rotas em mais de um terço.
- Nas **séries 1 (R1, RC1)** o ganho aparece mais como **frota**: 1 a 1,5 veículo a menos.
- Nas **famílias C** não há ganho: a capacidade é a restrição ativa (coerente com o notebook 02 e com o
  cenário de janelas alargadas).
- Variações percentuais de *espera* são instáveis onde a espera original é pequena (ex.: R1 a 25%:
  +120% sobre uma base quase nula); por isso a tabela usa a duração total, não a espera.

## Realocação de entregas

Comparação de cada plano do cenário com o plano original da mesma réplica. O patamar de ruído compara
réplicas do próprio plano original entre si.

| Antecipação | Clientes que mudaram de veículo | Pares que continuam juntos | Trechos mantidos |
|---|---|---|---|
| ruído (original × original) | 0% | 100% | 100% |
| 25% | 43% | 54% | 40% |
| 50% | 39% | 49% | 39% |
| 100% | 41% | 53% | 36% |
| full | 34% | 53% | 26% |

**Leitura.** Mesmo uma antecipação pequena (25%) **reorganiza o plano**: cerca de 40% dos clientes mudam
de veículo e só 26–40% dos trechos permanecem. O ganho de tempo e de frota não vem de ajustes locais,
mas de uma redistribuição das entregas entre veículos. Implicação operacional: a política de aceitar
entrega antecipada tem um **custo de mudança** (rotas e motoristas diferentes) que deve ser pesado
contra o ganho.

## Limitações

1. **Patamar de ruído zero nesta rodada**: as 3 réplicas do plano original foram idênticas (máquina
   ociosa). Em rodadas anteriores, execuções idênticas variaram ±1 veículo em R/RC
   (`reports/sensibilidade-tempo-e-janelas.md`); o patamar real pode não ser zero.
2. **Não monotonicidade heurística**: relaxar janelas nunca piora o ótimo, mas a heurística às vezes
   piora — C106 “full” com 11 veículos contra 10; R202 “full” com 4 contra 3 a 50%. Reflete a busca com
   tempo limitado, não o problema.
3. A realocação usa casamento guloso de rotas por maior sobreposição; outras métricas de casamento podem
   dar valores um pouco diferentes.
4. 14 instâncias de desenvolvimento; resultados de cenário devem ser lidos como tendência, não como
   estimativa precisa para uma operação real.

## Correção (2026-11-18)

Os números da tabela de resultados acima são médias entre instâncias de **medianas coluna a coluna** das
réplicas (`outputs/frozen/early_delivery/summary.csv`), misturando instâncias que reduzem a frota com
instâncias que não a alteram. Foram substituídos no notebook 06 por (i) um plano real por célula (réplica de
posição ⌈R/2⌉ na ordem lexicográfica por infactibilidade, veículos e distância; arquivo derivado
`outputs/frozen/early_delivery/summary_lex.csv`) e (ii) resultados separados por classe de frota. Com
tolerância livre: 0,50 veículo economizado em média; frota igual (7 instâncias): distância −18,1% e duração
(saída ótima) −20,8%; frota reduzida (6): −11,0% e −11,7%; frota maior (1, C106): +8,1% e +0,7%. A
limitação 2 também mudou: em ordem lexicográfica há 14 células não monotônicas na entrega antecipada (2 de
frota, 12 de distância com a mesma frota) e 6 nas janelas alargadas. Texto original mantido por
rastreabilidade.
