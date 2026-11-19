# Plano de sanidade — auditoria externa (Antigravity)

Data: 2026-11-17 · Status: **plano, nada executado além da triagem só de leitura abaixo**.

## 1. Triagem já feita (só leitura, sem solver)

| Achado | Afirmação | Verificação | Status |
|---|---|---|---|
| 1 | Camada B não tem descarte penalizado, prioridade, capacidade/frota, atrasos | código de `solver.py` / `scenarios.py` | **Confirmado** (escopo não executado) |
| 2 | Robustez sob perturbações não foi medida | código e relatórios | **Confirmado** |
| 3 | Avaliação não grava duração/espera | `RESULT_COLS` em `run_evaluation.py` | **Confirmado** |
| 4 | `reduce_fleet` falha na construção, não por tempo | runtime das 126 células: PCI 10,07–10,28 s; SAVINGS mediana 10,47 s; só PATH_CHEAPEST_ARC chega a ~15 s | **Confirmado — nosso diagnóstico ("passo de 5 s curto") está errado** |
| 5 | `scale=100` + `ceil` poda rotas válidas | rotas CVRPLIB factíveis em double tornam-se infactíveis na aritmética do solver: **R106 e R110** com escala 100; nenhuma com escala 10.000 | **Confirmado** |
| 6 | Relaxamentos sem warm-start geram não monotonicidade subnotificada | ordem lexicográfica (veículos, distância): **6** células em janelas, **14** em entrega antecipada | **Confirmado** |
| 7 | Médias de distância misturam frotas; mediana marginal por coluna | código de `early_delivery_summary` e tabelas de `business.py` | **Confirmado** |
| 8 | `vehicle_fixed_cost=10^7` distorce o GLS; *throttling* com 8 processos | — | **Hipótese — exige experimento** |
| 9 | Barra de cores do notebook 02 usa a escala da última instância | `viz_eda.plot_family_layouts` | **Confirmado** |
| 10 | Contradições documentais; sem git | cabeçalho de `outputs/frozen/README.md`; README diz "sem dados" nos testes; notebooks 01/02 leem BKS de avaliação (declarado como limitação no NB01, mas viola a regra textual da pré-análise) | **Confirmado** |

Observação extra: uma célula `reduce_fleet`/`SAVINGS` levou só 1,08 s no total — a primeira resolução terminou muito antes do limite. Investigar junto com o achado 4.

## 2. Princípio de governança para as correções

A avaliação final (v1) **já foi aberta**. Qualquer mudança no solver (escala, custo fixo, redução de
frota) define uma **versão v2** e:

1. a configuração v2 é escolhida **só nas 14 instâncias dev**, com a mesma regra de seleção, e
   pré-registrada em `reports/protocolo-v2.md` antes de rodar;
2. a v1 permanece congelada e é reportada como o resultado original;
3. uma reavaliação nas 42 instâncias é rotulada **"v2, pós-hoc"**: as instâncias de avaliação já foram
   vistas, então ela não é mais um conjunto intocado. Ambas aparecem lado a lado, sem escolher a melhor.

Antes de qualquer alteração: **inicializar git e commitar o estado atual como `v1`** (ancora no tempo
tudo o que existe; não prova retroativamente a ordem de 17/11, o que fica declarado).

## 3. Fases

### Fase A — Correções sem reexecutar solver (baixo risco, ~1 h de agente)

1. **Achado 4 (texto):** corrigir o diagnóstico em `README*`, `reports/protocolo-grade-dev.md`,
   `reports/sensibilidade-tempo-e-janelas.md` e NB04 (Insight de limitações): a tentativa com k−1
   veículos falha na construção inicial sem descarte permitido; o tempo restante foi descartado; o
   torneio comparou `reduce_fleet` com ~10 s contra 30 s de `fixed_cost`.
2. **Achado 3:** recalcular duração, espera e duração com saída ajustada **a partir das rotas congeladas**
   da avaliação (estão salvas em `results.csv`) — sem reexecutar o solver; acrescentar ao NB05.
3. **Achado 7:** substituir a mediana marginal por **seleção lexicográfica de réplica** (como na
   avaliação); separar, nas tabelas da Camada B, "frota igual" × "frota reduzida" ao reportar distância;
   reexecutar NB06 (só leitura de CSVs).
4. **Achado 6 (texto):** `nonmonotonic_cases` passa a usar ordem lexicográfica; NB06 relata 6 + 14.
5. **Achado 9:** normalizar `ready / horizon` em [0, 1] na barra de cores; reexecutar NB02.
6. **Achado 10:** corrigir cabeçalho de `outputs/frozen/README.md`; trocar "somente fixtures sintéticas"
   por "testes unitários com fixtures sintéticas; testes de contrato leem notebooks e `split.csv`";
   registrar no README/NB01 o desvio da regra "BKS de avaliação só no torneio final" (uso descritivo,
   não usado para seleção).

### Fase B — Diagnósticos do solver, só dev (~1,5 h de CPU, máquina ociosa)

Pré-registrar em `reports/protocolo-v2.md` antes de rodar.

| Experimento | Desenho | Pergunta |
|---|---|---|
| B1 escala (achado 5) | `scale` ∈ {100, 10.000} × config congelada, 14 dev, 120 s, 3 réplicas | R110 (e outras) reduzem frota? |
| B2 custo fixo (achado 8) | `vehicle_fixed_cost` ∈ {10^5, 5×10^5, 10^7} com a melhor escala de B1 | o GLS melhora frota/distância? |
| B3 redução de frota (achado 4) | `minimize_fleet` corrigido: *warm-start* a partir da solução anterior com uma rota esvaziada e `AddDisjunction` com penalidade alta só durante a redução; orçamento total igual | a estratégia fecha o veículo extra em R1/RC1? |
| B4 *throttling* (achado 8) | mesma célula com `workers` = 4 e 8, 3 réplicas | a carga muda o resultado com 120 s? |
| B5 warm-start Camada B (achado 6) | cada nível de `widen_windows`/`allow_early` inicia da solução do nível anterior (`ReadAssignmentFromRoutes`) | as 20 células não monótonas desaparecem? |

Resultado: configuração v2 escolhida pela mesma regra lexicográfica nas 14 dev.

### Fase C — Escopo da Camada B não executado (achados 1 e 2; ~2–3 h)

1. **Robustez do plano fixo (barato, sem solver):** aplicar choques de tempo de viagem `×(1+ε)`,
   `ε ~ U(0; 0,2)`, e de tempo de serviço aos **planos já congelados**; medir pelo verificador a taxa de
   janelas violadas e de retornos tardios (Monte Carlo, semente fixa). Responde "quão frágeis são as
   rotas no limite do `due`".
2. **Solver da Camada B:** `AddDisjunction` com penalidade por prioridade (clientes VIP × comuns),
   janelas flexíveis com `SetCumulVarSoftUpperBound` (atraso penalizado), sensibilidade a capacidade
   `Q` (−10%, −20%) e frota `K` (limitada abaixo do necessário) — aqui "clientes não atendidos" e
   "violações de janela" deixam de ser zero por construção.
3. **Estabilidade da reotimização** sob remoção/inserção de 5–10% dos clientes (métrica `reallocation`
   já existente, com patamar de ruído).

### Fase D — Reavaliação v2 (decisão do usuário; ~55 min)

Rodar a configuração v2 nas 42 instâncias, rotulada "v2 pós-hoc", e apresentar v1 e v2 lado a lado no
NB05, sem substituir a v1.

## 4. Ordem recomendada e decisões pendentes

1. **git init + commit `v1`** (decisão do usuário).
2. Fase A (corrige o que está errado hoje, inclusive afirmação nossa refutada).
3. Fase B com protocolo v2 pré-registrado.
4. Fase C1 (robustez, barata) → C2/C3.
5. Fase D somente se a v2 diferir da v1 de forma material nas instâncias dev.

Ponto em aberto: o achado 8 (custo fixo e *throttling*) é hipótese plausível, mas não demonstrada; a
estabilidade de frota entre réplicas com 120 s sugere que o efeito de *throttling* no resultado
principal é pequeno — B4 decide.

## 5. Andamento

- **2026-11-18 — Fase A concluída** (sem executar solver; derivada dos resultados congelados).
  A1: diagnóstico de `reduce_fleet` corrigido — falha quase imediata na construção com
  `PARALLEL_CHEAPEST_INSERTION` (42/42 células) e `SAVINGS` (28/42); com `PATH_CHEAPEST_ARC` o passo de 5 s
  foi usado inteiro (40/42), então ali o limite de tempo não pode ser descartado. A2: duração e espera da
  avaliação derivadas das rotas (`outputs/frozen/evaluation/time_metrics*.csv`, NB05 §10). A3: réplica
  representativa lexicográfica e distância separada por classe de frota na Camada B
  (`outputs/frozen/early_delivery/summary_lex.csv`). A4: não monotonicidade lexicográfica — 6 (janelas,
  todas de distância com frota igual) e 14 (entrega antecipada: 2 de frota, 12 de distância). A5: barra de
  cores normalizada no NB02. A6: documentação corrigida. 290 testes, ruff limpo.
- Próximo: Fase B (protocolo v2 pré-registrado, somente instâncias dev).
- **2026-11-18 — Fase B concluída** (`reports/protocolo-v2.md`): v2 = v1 + escala 10 000 (R110 12→11);
  custo fixo e estratégia de frota mantidos; partida a quente elimina a não monotonicidade da Camada B.
- **2026-11-18 — Fase C concluída** (`reports/fase-c-robustez.md`): robustez dos planos fixos (OR-Tools
  menos frágil por cliente que as baselines), capacidade, frota limitada com prioridade, reotimização;
  janelas flexíveis (C2.3) pouco informativas por falha de construção em 7/10 instâncias.
- Pendentes: notebooks com as Fases B e C; Fase D (reavaliação v2 pós-hoc) por decisão do usuário.
- **2026-11-18 — Fase D concluída** (`reports/protocolo-fase-d.md`): v2 pós-hoc ≈ v1 (frota igual 24 → 25;
  só RC207 4 → 3); a v1 segue como resultado principal.
- **2026-11-18 — Notebooks atualizados:** NB04 §9 (Fase B), NB05 §11 (v2 pós-hoc), NB06 §7 (B5) e NB07 novo
  (Fase C), em PT-BR e EN-US; READMEs com a seção "Auditoria e v2". 427 testes, ruff limpo.
  **Plano de sanidade encerrado.**

- **2026-11-19 — Correção de leitura (revisão externa verificada):** a linha de 2026-11-18 "v2 = v1 + escala
  10 000" está incompleta. A v2 é a v1 com escala 10 000 **e** custo fixo relativo k = 10, cuja magnitude é de
  49,9% a 66,2% do custo fixo absoluto da v1 (10^7 com escala 100) nas 42 instâncias de avaliação; a comparação
  v1 × v2 é, portanto, entre duas configurações, e RC207 (4 → 3) é atribuível à configuração v2, não só à escala
  (ver as notas "Correção (2026-11-19)" em `reports/protocolo-v2.md` e `reports/protocolo-fase-d.md`). Também
  corrigidos: o *resume* dos scripts de execução agora é protegido por impressão digital da configuração
  (`delivery_routing_vrptw.runs`; `--allow-legacy-resume` para pastas antigas); a agregação do C3 e o patamar de
  ruído (`reports/fase-c-robustez.md`); e a ressalva de dispersão entre réplicas no C2.2.
- **2026-11-19 — Correção da retomada (segunda verificação externa):** o número de processos (`workers`) estava
  excluído da impressão digital, de modo que uma execução do B4 com 4 processos podia ser retomada com 8 e ter seus
  metadados reescritos. Agora ele faz parte da impressão digital de todos os scripts que retomam pastas
  (`run_dev_grid`, `run_time_sensitivity`, `run_window_scenarios`, `run_early_delivery`, `run_evaluation`,
  `run_v2_solver_grid`, `run_v2_layerb_warm`, `run_phase_c_solver`, `run_reoptimisation`, `run_robustness`),
  porque limites de tempo de relógio fazem o resultado depender da contenção de CPU. Resultados congelados não
  são afetados: as linhas do B4 congelado foram geradas em pasta nova e cada linha registra o próprio `workers`
  (verificado em `outputs/frozen/v2/B4/results.csv`: 42 linhas, todas com `workers` = 4). Também: `run_reoptimisation`
  passa a gravar `aggregate_representative.csv` (agregado do protocolo) além do `aggregate.csv` agrupado.
