# Pré-análise da proposta — Otimização de entregas com capacidade e janelas de tempo

Data: 2026-11-17 · Base: auditoria em [pre-analise-viabilidade.md](pre-analise-viabilidade.md).
Objetivo deste documento: decidir **o que da proposta é viável como está, o que precisa ser
reformulado e o que deve ficar fora**, antes de qualquer notebook ou solver.

## 1. Veredito resumido

| Elemento da proposta | Veredito | Ajuste necessário |
|---|---|---|
| Pergunta de negócio (frota → distância, sem violar restrições) | ✅ viável | coincide com a convenção SINTEF; manter a ordem lexicográfica explícita |
| "Continuação conceitual do Olist" | ⚠️ reformular | ligação **temática**, não empírica: Olist não tem frota, capacidade, janelas nem rotas |
| 56 instâncias Solomon, 100 clientes | ✅ viável | dados auditados, duas fontes idênticas, hash registrado |
| BKS do SINTEF para "medir objetivamente" | ✅ com ressalva | BKS são **melhores conhecidas por heurísticas**, não ótimos; comparar só sob a mesma convenção (hierárquica, double, 2 casas) |
| OR-Tools | ✅ viável | instalado e testado; exige custos inteiros e um mecanismo explícito para minimizar frota |
| Heurística gulosa e nearest neighbor | ✅ com ressalva | precisam ser **cientes de janela** (senão geram soluções infactíveis, comparação injusta) |
| OR-Tools com diferentes estratégias de busca | ✅ viável | grade pequena e congelada; seleção só em instâncias de desenvolvimento |
| Solução sob limite fixo de tempo | ✅ viável | o tempo de CPU é o eixo principal; reportar curva qualidade × tempo |
| Sensibilidade a capacidade e frota | ✅ viável | perturbações diretas do contrato de dados |
| Sensibilidade a atrasos | ⚠️ reformular | "atraso" muda o problema (janela flexível com penalidade) → cenário separado, nunca comparado ao BKS |
| Clientes prioritários | ⚠️ reformular | Solomon não tem prioridade; é uma **extensão sintética** (penalidades de descarte diferenciadas) |
| Indicadores | ⚠️ definir | vários são triviais (=0) no problema rígido; ver seção 5 |
| Estabilidade sob perturbações | ⚠️ definir protocolo | precisa de perturbação, métrica de mudança de rota e sementes definidas a priori |
| "Prescriptive analytics" | ✅ legítimo | a saída é um plano de rotas verificável; não alegar otimalidade |

**Conclusão:** a proposta é viável no ambiente local. O núcleo (benchmark + OR-Tools + baselines + BKS)
pode seguir; os cenários de negócio (atrasos, prioridade, perturbações) devem ser uma **segunda
camada**, separada da comparação com o benchmark.

## 2. Duas camadas que nunca se misturam

| | Camada A — Benchmark | Camada B — Cenários de negócio |
|---|---|---|
| Problema | VRPTW rígido de Solomon | variações: janelas flexíveis, descarte com penalidade, prioridade, perturbações |
| Objetivo | lexicográfico: veículos, depois distância | custo ponderado declarado (ex.: custo fixo por veículo + km + penalidade de atraso/descarte) |
| Referência | BKS SINTEF | o próprio plano da Camada A (antes × depois) |
| Pode reportar "gap para BKS"? | sim | **não** |
| Violações e não atendidos | devem ser 0 (factibilidade) | são resultados medidos |

Esta separação responde à decisão aberta 2 do briefing.

## 3. Riscos técnicos do OR-Tools e como tratá-los

1. **Frota primeiro.** O OR-Tools minimiza uma soma; a hierarquia é obtida com custo fixo por
   veículo (`SetFixedCostOfAllVehicles`) maior que qualquer distância possível (ex.: 10⁵ com
   distâncias em escala ×100). Complemento: busca iterativa reduzindo o número de veículos
   disponíveis até a infactibilidade. Ambos devem ser testados em desenvolvimento.
2. **Custos inteiros.** Distâncias em double são escaladas (×100 ou ×1000) e arredondadas. Isso pode
   tornar uma rota "factível no solver" infactível em double (o mesmo fenômeno observado nas 8
   soluções do CVRPLIB). Regra: **toda solução é re-verificada com o verificador em double**; a
   distância reportada é a recalculada, nunca o objetivo interno do solver. Arredondar o tempo de
   viagem **para cima** no solver evita esse risco.
3. **Janelas e espera.** Dimensão de tempo com espera permitida (`slack`), horizonte = `DUE DATE`
   do depósito, serviço somado ao trânsito do nó de origem.
4. **Não determinismo.** Busca com limite de tempo em relógio depende da carga da máquina. Fixar
   threads (1 worker), registrar CPU, e repetir cada configuração com algumas réplicas.
5. **Descarte de clientes.** `AddDisjunction` só na Camada B. Na Camada A o descarte é proibido;
   um cliente não atendido torna a solução inválida para o benchmark.

## 4. Baselines — definição mínima para comparação justa

| Baseline | Definição | Papel |
|---|---|---|
| Gulosa sequencial | abre uma rota, adiciona o cliente factível de menor custo de inserção no fim; quando nenhum cabe, abre outra rota | piso de qualidade |
| Nearest neighbor com janelas (Solomon, 1987) | próximo cliente = menor combinação de distância, espera e urgência da janela, respeitando capacidade e janela | heurística clássica da literatura |
| OR-Tools — solução inicial apenas | `PATH_CHEAPEST_ARC`, `PARALLEL_CHEAPEST_INSERTION`, `SAVINGS`, sem metaheurística | isola o efeito da construção |
| OR-Tools — com metaheurística | `GUIDED_LOCAL_SEARCH`, `TABU_SEARCH`, `SIMULATED_ANNEALING` sob orçamento fixo | candidato principal |

Todas as baselines passam pelo mesmo verificador. A proposta lista "gulosa" e "nearest neighbor"
como itens separados; ambas são viáveis, mas sem consciência de janela não são comparáveis.

## 5. Indicadores — o que cada um significa aqui

| Indicador | Camada A | Camada B | Observação |
|---|---|---|---|
| Número de veículos | principal | sim | — |
| Distância total (double, 2 casas) | principal | sim | sempre recalculada pelo verificador |
| Tempo total | secundário | sim | definir: **duração das rotas** (viagem + espera + serviço); diferente de distância porque inclui espera |
| Violações de janela | deve ser 0 | medido | na Camada A é teste de factibilidade, não métrica |
| Clientes não atendidos | deve ser 0 | medido | idem |
| Gap para BKS | sim, separado por veículos e distância | **não** | gap de distância só quando o nº de veículos for igual ao BKS |
| Tempo de solver | sim | sim | CPU, limite fixo |
| Estabilidade | — | sim | ver seção 6 |

Como o objetivo é lexicográfico, o gap deve ser reportado em duas colunas (Δ veículos, Δ distância %)
e nunca agregado num número único.

## 6. Estabilidade sob perturbações — protocolo proposto

- **Perturbações** (definidas antes de rodar): demanda ±10–20%; janelas deslocadas em ±5–10% do horizonte (Solomon não tem unidade de tempo real);
  tempo de viagem × (1 + ε), ε ~ U(0; 0,2) para simular atraso; remoção/inserção de 5–10% dos clientes.
- **Duas perguntas distintas:**
  1. *Robustez do plano*: o plano original ainda é factível sob a perturbação? (taxa de violações)
  2. *Estabilidade da reotimização*: quanto o novo plano difere do original? (fração de arcos
     preservados, clientes que trocam de veículo, Δ veículos, Δ distância)
- Sementes registradas; nº de réplicas fixado antes da execução.

## 7. Orçamento computacional (estimativa a confirmar)

Máquina: i9-11900H, 16 threads; execução com 1 worker por processo, até ~8 processos paralelos.

| Bloco | Execuções | Tempo por execução | Total sequencial | ~8 em paralelo |
|---|---|---|---|---|
| Baselines construtivas | 56 × 2 | < 1 s | < 2 min | — |
| Grade OR-Tools (3 construções × 3 metaheurísticas) em dev | ~15 inst. × 9 × 3 réplicas | 30 s | ~3,4 h | ~25 min |
| Configuração congelada na avaliação | ~40 inst. × 3 réplicas | 60 s | 2 h | ~15 min |
| Curva qualidade × tempo | 6 inst. × 5 orçamentos (5–300 s) × 3 | ~30–300 s | ~2,5 h | ~20 min |
| Sensibilidade e perturbações (Camada B) | ~6 inst. × ~20 cenários × 3 | 30 s | ~3 h | ~25 min |

Ordem de grandeza: **poucas horas de CPU no total, sem GPU**. A memória é irrelevante (matriz 101×101).
Estes números são estimativas e devem ser medidos no caso canônico C101 antes de congelar a grade.

## 8. Divisão desenvolvimento/avaliação (proposta)

Estratificada por família, ~25% para desenvolvimento: C1 2/9, C2 2/8, R1 3/12, R2 3/11, RC1 2/8,
RC2 2/8 → **14 instâncias de desenvolvimento e 42 de avaliação**. Sorteio com semente fixa, registrado em arquivo **antes** de
olhar qualquer resultado do solver. As BKS da avaliação só são lidas no torneio final.

## 9. Riscos de narrativa

- **Dados sintéticos:** declarar que Solomon é benchmark acadêmico (coordenadas no plano, velocidade 1),
  não operação real.
- **Olist:** a ligação é de tema (entrega no e-commerce), não de dados.
- **Otimalidade:** OR-Tools com limite de tempo não prova ótimo; as BKS também não são ótimos provados
  na convenção hierárquica. Usar "factível", "melhor encontrada", "gap para a melhor conhecida".
- **Expectativa realista:** atingir o BKS de frota em R1/RC1 com OR-Tools e orçamento curto é
  improvável; o resultado honesto provavelmente terá Δ veículos > 0 nessas famílias. Isso é um achado,
  não um fracasso.

## 10. Próximos passos sugeridos (ainda dentro do gate)

1. Rodar C101 no OR-Tools com limite curto, validar com o verificador e comparar com 10 / 828,94.
2. Medir o tempo por família em 1 instância de cada (6 execuções) para confirmar a seção 7.
3. Sortear e registrar o split dev/avaliação.
4. Só então: módulo `src/`, testes do parser e do verificador, e notebooks.
