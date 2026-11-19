# Fase C — robustez e cenários operacionais (resultados)

Data: 2026-11-18 · Protocolo: `reports/protocolo-fase-c.md` (registrado e esclarecido antes da execução) ·
Camada B: **sem comparação com o SINTEF** · Solver v2 (escala 10 000), 30 s, 3 réplicas · dados em
`outputs/frozen/phase_c/`. Todos os blocos rodaram em sequência, com a máquina sem outra carga.

## C1 — Robustez do plano fixo sob atrasos (sem replanejar)

Planos da avaliação v1 (42 instâncias; OR-Tools na réplica mediana e as duas baselines), 1 000 sorteios por
plano e cenário. "Atrasado" = início do serviço depois do fim da janela.

| Cenário | Clientes atrasados: OR-Tools / NN / gulosa | Retorno tardio ao depósito: OR-Tools / NN / gulosa | Rotas com ≥ 1 atraso: OR-Tools / NN / gulosa |
|---|---|---|---|
| viagem +0–5% | **4,6%** / 5,6% / 13,4% | **9,6%** / 19,5% / 32,2% | 55,6% / 44,2% / 56,6% |
| viagem +0–10% | **6,3%** / 9,3% / 21,1% | **13,4%** / 32,9% / 43,3% | 63,8% / 53,5% / 66,9% |
| viagem +0–20% | **9,3%** / 16,0% / 31,7% | **18,4%** / 45,7% / 55,7% | 70,2% / 63,8% / 77,3% |
| serviço +0–20% | **16,6%** / 21,4% / 29,8% | **22,9%** / 51,3% / 53,3% | 84,2% / 71,1% / 72,5% |
| viagem 10% + serviço | **20,3%** / 28,3% / 40,8% | **27,7%** / 60,5% / 67,4% | 89,9% / 79,7% / 84,3% |

**Leitura.**

- O plano otimizado **não é mais frágil** que as baselines por cliente: com 10% de atraso de viagem,
  6,3% dos clientes do OR-Tools atrasam, contra 9,3% (vizinho mais próximo) e 21,1% (gulosa); o retorno
  tardio ao depósito é 2 a 3× menor. A hipótese de que rotas otimizadas "no limite do `due`" seriam
  especialmente frágeis não se confirmou em termos de clientes.
- Por rota, porém, o OR-Tools tem **mais rotas com algum atraso** sob choque de serviço (84% contra 71–73%):
  usa menos veículos, com rotas mais longas e mais clientes, e um único atraso acumulado basta para marcar a
  rota.
- O atraso nunca é desprezível: mesmo +0–5% na viagem atrasa algum cliente em mais da metade das rotas.
- **A folga explica a fragilidade:** correlação de Spearman entre folga nominal e taxa de atraso de −0,37 a
  −0,78 (todas negativas). Por família (OR-Tools, viagem +10%): R1 13,1% e RC1 11,1% de clientes atrasados
  (folga média 38–41) contra C1 1,0% (folga 168).

## C2 — Cenários operacionais (14 instâncias dev)

### C2.1 Capacidade reduzida

| Capacidade | Veículos adicionais (média) | Instâncias com frota maior | Distância nessas instâncias |
|---|---|---|---|
| 90% | 0,21 | 3/14 | +13,9% |
| 80% | 0,57 | 6/14 (+1,33 veículo cada) | +13,6% |

Com frota igual, a distância praticamente não muda (−0,5% a +0,2%).

### C2.2 Frota limitada com descarte penalizado e prioridade

K = frota do plano de referência − 1 e − 2; 20% dos clientes prioritários (penalidade 10×).

| Frota | Variante | Clientes descartados (total, 14 inst.) | Prioritários descartados | Demanda descartada (média/inst.) |
|---|---|---|---|---|
| K − 1 | uniforme | 111 | 23 (8% dos prioritários) | 150,6 |
| K − 1 | **com prioridade** | 122 | **0** | 162,1 |
| K − 2 | uniforme | 344 | 69 (25%) | 437,7 |
| K − 2 | **com prioridade** | 363 | **3 (1%)** | 454,3 |

**Leitura.** Com um veículo a menos, cerca de 8 clientes por instância deixam de ser atendidos; com dois,
cerca de 25. A prioridade **protege quase todos os prioritários** (de 23 para 0 e de 69 para 3) ao custo de
descartar um pouco mais de clientes comuns (+0,8 e +1,4 cliente por instância). "Clientes não atendidos"
deixa de ser zero por construção e passa a ser uma decisão explícita de negócio.

### C2.3 Janelas flexíveis com atraso penalizado — **pouco informativo**

K = referência − 1. Resultado em só **3 de 14** instâncias (R110, R202, RC106): 4 são infactíveis por
capacidade (famílias C, previsto no esclarecimento 2) e **7 falharam na construção inicial em < 2 s**
(mesmo modo de falha do achado 4: o `SAVINGS` não monta um plano com um veículo a menos e a busca não
começa). Nas 3 instâncias resolvidas, custo de atraso 1× → 8,7 clientes atrasados em média (atraso total
111); 10× → 3,7 (atraso total 65). O bloco **não sustenta conclusão geral**; corrigi-lo exige partida a
quente a partir do plano de referência com uma rota esvaziada (como em `minimize_fleet_v2`) — mudança não
pré-registrada, deixada como trabalho futuro.

## C3 — Estabilidade da reotimização (14 instâncias dev, 3 réplicas)

| Evento | Variante | Clientes que mudam de veículo | Pares mantidos | Trechos mantidos | Δ veículos | Δ distância (mediana) |
|---|---|---|---|---|---|---|
| remover 10% | **a quente** | **26,7%** | 64,9% | 61,5% | −0,57 | −3,4% |
| remover 10% | a frio | 40,0% | 52,7% | 41,6% | −0,52 | −2,6% |
| inserir 10% | **a quente** | **30,0%** | 55,2% | 60,2% | +0,64 | +6,1% |
| inserir 10% | a frio | 36,7% | 46,9% | 42,5% | +0,57 | +7,0% |
| ruído (duas réplicas a frio) | — | 0% | 100% | 100% | — | — |

**Leitura.** Partir do plano anterior **preserva mais o plano** (menos clientes realocados, mais trechos
mantidos) e chega a distância igual ou melhor. Mesmo assim, um evento de 10% dos clientes realoca de 27% a
30% dos demais — o replanejamento tem custo operacional relevante. O patamar de ruído foi zero nesta
rodada (máquina ociosa).

## Limitações

1. C2.3 falhou em 7 de 10 instâncias possíveis por limitação da construção inicial (não por tempo); ver acima.
2. C1 usa planos de avaliação de forma descritiva; choques independentes por trecho, sem correlação
   espacial ou temporal (trânsito real é correlacionado).
3. C2 e C3 usam apenas 14 instâncias dev; grupos por classe de frota podem ter 3 a 8 instâncias.
4. O ganho de tempo e a robustez não são objetivos do solver (que minimiza frota e distância).

---

## Correção (2026-11-19)

O texto acima **não foi reescrito**. Uma revisão externa, verificada, encontrou dois problemas na seção C3 e um
na C2.2.

**C3 — agregação.** A tabela C3 acima agregou as 42 réplicas por variante (3 réplicas × 14 instâncias, todas
juntas) em vez da **réplica representativa por instância** do protocolo (posição ⌈R/2⌉ = 2ª de 3, ordenando por
infactível, veículos e distância; empate pelo id da réplica). Recalculado com a regra
(`phase_c.reopt_aggregate_representative`; arquivos novos `outputs/frozen/phase_c/C3/<bloco>/aggregate_representative.csv`),
as 14 instâncias pesam uma vez por variante:

| Evento | Variante | Clientes que mudam de veículo (mediana) | Pares mantidos | Trechos mantidos | Δ veículos | Δ distância (mediana) |
|---|---|---|---|---|---|---|
| remover 10% | a quente | 26,7% (antes 26,7%) | 64,9% | 61,5% | −0,57 | −3,4% |
| remover 10% | a frio | **40,6%** (antes 40,0%) | 51,0% (antes 52,7%) | 41,4% (antes 41,6%) | −0,50 (antes −0,52) | −2,6% |
| inserir 10% | a quente | 30,0% | 55,2% | 60,2% | +0,64 | +6,1% |
| inserir 10% | a frio | 36,7% | 46,9% | 42,5% | +0,57 | +7,0% |

A conclusão não muda: partir do plano anterior preserva mais o plano (26,7% contra 40,6% dos clientes realocados
na remoção; 30,0% contra 36,7% na inserção) e chega a distância igual ou melhor.

**C3 — patamar de ruído.** A frase "O patamar de ruído foi zero nesta rodada" vale só para a **mediana**. Das 42
comparações entre réplicas a frio, na remoção 4 realocaram clientes (R111: 23,3% em 2 comparações; RC203: 3,3%
em 2 comparações; máximo 23,3%); na inserção, 0 de 42. O patamar de ruído é uma distribuição (mediana 0%, 4/42
acima de zero na remoção), e diferenças entre quente e frio do tamanho de 23% em uma única instância não são
evidência.

**C2.2 — dispersão entre réplicas.** A penalidade de descarte incentiva o atendimento, mas a heurística não
garante o descarte mínimo: em RC108 com K − 2 e penalidade uniforme, as 3 réplicas descartaram 9, 7 e 7 clientes
(prioritários: 2, 1 e 1). Dos 56 grupos (instância × limite de frota × variante), 1 tem número diferente de
clientes descartados entre réplicas e 5 têm conjuntos diferentes (RC108, C106, R110 ×2 e RC106;
`outputs/frozen/phase_c/C2/fleet_priority/dropped_spread.csv`). As diferenças médias de +0,8 e +1,4 cliente
comum por instância atribuídas à prioridade são menores que a maior dispersão entre réplicas numa instância (2
clientes) e devem ser lidas como pequenas.

**C2.3 — ressalva mantida.** O bloco de janelas flexíveis é **exploratório** (3 de 14 instâncias resolvidas) e
não deve ser citado sem essa ressalva.
