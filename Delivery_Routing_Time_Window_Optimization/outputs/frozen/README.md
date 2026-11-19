# Resultados congelados

Esta pasta é a única parte de `outputs/` versionada. Os arquivos abaixo são **cópias congeladas**
(não são regenerados nem editados): os notebooks 03 a 06 leem apenas daqui, nunca das pastas de
trabalho (`outputs/dev_grid/`, `outputs/gate/`).

- **Somente instâncias de desenvolvimento** (`set = dev` em `data/reference/split.csv`): `dev_grid/`, `gate/`,
  `time_sensitivity/`, `window_scenarios/`, `window_scenarios_run1_contended/` e `early_delivery/`, além de
  `selected_config.json`.
- **Somente instâncias de avaliação** (`set = eval`, as 42 instâncias): `evaluation/`, a avaliação final
  pré-registrada, que só é lida pelo notebook 05 por meio de carregadores que recusam instâncias de
  desenvolvimento.

A tabela a seguir cobre `selected_config.json`, `dev_grid/` e `gate/`; as demais pastas estão descritas nas
seções seguintes.

| Arquivo | Conteúdo | Comando que o produziu |
|---|---|---|
| `selected_config.json` | Configuração vencedora do torneio (`fixed_cost` + `SAVINGS` + `GUIDED_LOCAL_SEARCH`), estatísticas de seleção e protocolo | `python scripts/select_config.py` |
| `dev_grid/results.csv` | 322 células da grade de desenvolvimento (252 do torneio: 18 configurações × 14 instâncias; 70 de referência: 3 construções do OR-Tools sem busca local e as 2 heurísticas construtivas), com métricas do verificador e rotas em JSON | `python scripts/run_dev_grid.py --budget 30 --workers 8` |
| `dev_grid/ranking.csv` | Ranking das 18 configurações pela regra pré-registrada (posição média, Δ veículos, gap mediano) | `python scripts/select_config.py` |
| `dev_grid/run_metadata.json` | Metadados da execução da grade (orçamento, processos, versões, argumentos) | `python scripts/run_dev_grid.py --budget 30 --workers 8` |
| `gate/gate_results.csv` | Teste de fumaça (gate) em 6 instâncias de desenvolvimento: gulosa, vizinho mais próximo e OR-Tools | `python scripts/run_gate.py` |

Protocolo pré-registrado: `reports/protocolo-grade-dev.md`.

## Sensibilidade ao tempo e cenários de janelas (2026-11-17)

- `time_sensitivity/` — `scripts/run_time_sensitivity.py` (configuração congelada; orçamentos 10/30/60/120/300 s; 14 instâncias dev; 8 processos). Camada A: comparável ao SINTEF.
- `window_scenarios/` — `scripts/run_window_scenarios.py --out outputs/window_scenarios_rerun` (fatores 1,0/1,25/1,5/2,0; modo simétrico; 30 s; 14 instâncias dev), rodado sem outra carga na máquina. Camada B: sem comparação com o SINTEF. **Versão oficial.**
- `window_scenarios_run1_contended/` — primeira rodada do mesmo experimento, que coincidiu parcialmente com a execução de notebooks e testes. Mantida apenas como evidência da variabilidade por relógio; não usar como resultado.
- `early_delivery/` — `scripts/run_early_delivery.py --out outputs/early_delivery` (antecipação de 0/25/50/100% da largura da janela + "full"; 3 réplicas; 30 s; 14 instâncias dev; máquina sem outra carga). Camada B: sem comparação com o SINTEF. Inclui `summary.csv`, `reallocation.csv` e `reallocation_summary.csv`.
- `evaluation/` — **avaliação final pré-registrada** (`reports/protocolo-avaliacao.md`): `scripts/run_evaluation.py --confirm-eval --workers 8 --out outputs/evaluation`; 42 instâncias de avaliação; configuração congelada; 120 s × 5 réplicas + baselines; 294/294 factíveis; 54 min. Resultado principal = réplica mediana (`instance_table.csv`, `aggregate_table.csv`).

## Arquivos derivados (2026-11-18, sem executar solver)

Os arquivos congelados acima **nunca são editados**. As correções da auditoria (`reports/plano-sanidade-auditoria.md`,
Fase A) geram arquivos **novos**, ao lado dos originais, apenas a partir de arquivos congelados; os originais
ficam preservados para rastreabilidade.

| Arquivo novo | Derivado de | Script | Conteúdo |
|---|---|---|---|
| `evaluation/time_metrics.csv` | `evaluation/results.csv` (rotas) e instâncias SINTEF | `python scripts/derive_eval_time_metrics.py` | uma linha por célula (294): duração e espera com saída às 0, duração e espera com saída ótima, último retorno, recalculados com `route_compare.schedule_metrics` na instância original (diferença de distância contra o arquivo original: 0) |
| `evaluation/time_metrics_aggregate.csv` | `evaluation/time_metrics.csv` | mesmo script | réplica mediana por instância (regra lexicográfica de `evaluation`), depois mediana por escopo (geral e família) e método (OR-Tools e baselines) |
| `early_delivery/summary_lex.csv` | `early_delivery/results.csv` | `python scripts/derive_layerb_summaries.py` | resumo por (instância, tolerância) com **um plano real por célula** (réplica de posição ⌈R/2⌉ na ordem lexicográfica infactível, veículos, distância; colunas `*_rep`, `rep_replicate`, `*_min`, `*_max`), no lugar da mediana coluna a coluna de `early_delivery/summary.csv` |

`early_delivery/summary.csv` (mediana coluna a coluna) é mantido apenas por rastreabilidade; o notebook 06 usa o
resumo lexicográfico, recalculado dos resultados congelados por `business.early_results_summary`. O resumo das
janelas alargadas (`window_scenarios/summary.csv`) tem uma réplica por célula e não muda: o script verifica que
ele continua igual ao recalculado. As tabelas do notebook 06 separam frota igual, reduzida e maior, e a contagem
de células não monotônicas é lexicográfica (6 em janelas, 14 em entrega antecipada).


## Fase B — protocolo v2 (2026-11-18)

- `v2/` — experimentos B1–B5 de `reports/protocolo-v2.md`, **somente instâncias dev**: `B1` (escala), `B2`
  (custo fixo), `B3` (redução de frota v2), `B4` (4 processos), `B5_windows` e `B5_early` (partida a quente
  entre níveis). Comandos: `scripts/run_v2_solver_grid.py`, `scripts/select_v2.py`,
  `scripts/run_v2_layerb_warm.py`. Cada bloco com `decision.json` (quando aplicável) e `run_metadata.json`.

## Fase C — robustez e cenários operacionais (2026-11-18)

- `phase_c/` — `reports/protocolo-fase-c.md`. `C1/` (robustez dos planos da avaliação v1 sob choques,
  `scripts/run_robustness.py`; usa as 42 instâncias de avaliação de forma descritiva); `C2/` (capacidade,
  frota limitada com prioridade, janelas flexíveis; `scripts/run_phase_c_solver.py`; somente dev); `C3/`
  (reotimização após remoção/inserção de 10% dos clientes; `scripts/run_reoptimisation.py`; somente dev).

## Fase D — reavaliação v2 pós-hoc (2026-11-18)

- `evaluation_v2/` — `scripts/run_evaluation.py --version v2 --confirm-eval --post-hoc` (42 instâncias de
  avaliação, já vistas na v1; configuração v2, escala 10 000; 120 s × 5 réplicas; baselines reutilizadas da v1),
  `scripts/compare_eval_versions.py` e `scripts/derive_eval_time_metrics.py`. **Rótulo: "v2, pós-hoc"** — não
  substitui `evaluation/`.

## Correções de 2026-11-19 (sem executar solver; arquivos novos ao lado dos originais)

Os arquivos congelados continuam sem edição. Arquivos derivados novos, gerados por
`python scripts/derive_phase_c_corrections.py` somente a partir de CSVs congelados:

| Arquivo novo | Derivado de | Conteúdo |
|---|---|---|
| `phase_c/C3/removal/aggregate_representative.csv`, `phase_c/C3/insertion/aggregate_representative.csv` | `results.csv` do bloco | agregado C3 pela **réplica representativa de cada instância** (2ª de 3; infactível, veículos, distância), no lugar do agregado com as réplicas juntas de `aggregate.csv` (mantido por rastreabilidade); a linha `noise` traz o patamar de ruído como distribuição (`noise_n_positive`, `noise_max_share`, `noise_instances`) |
| `phase_c/C2/fleet_priority/dropped_spread.csv` | `results.csv` do bloco | dispersão entre réplicas dos clientes descartados por (instância, nível, variante): valores por réplica, amplitude, número de conjuntos distintos |

**Proteção de *resume* dos scripts de execução.** Cada `scripts/run_*.py` que retoma uma pasta calcula uma
impressão digital (SHA-256 de um JSON canônico) da configuração efetiva que determina os resultados (versão,
escala, regra e valor do custo fixo, limite de tempo, solução inicial, metaheurística, estratégia de frota, réplicas,
parâmetros do bloco ou do cenário e o número de processos; não datas nem caminhos), grava-a em `run_metadata.json` **depois** de
verificar a pasta e em uma coluna `config_fingerprint` das linhas novas. Retomar uma pasta com impressão digital
diferente é recusado com erro (use outro `--out`); retomar uma pasta **legada** (resultados sem impressão digital,
como todas as congeladas acima) exige `--allow-legacy-resume`, que preserva os metadados antigos em
`run_metadata.legacy.json`. As pastas congeladas nunca são modificadas por esta correção. Scripts protegidos:
`run_evaluation`, `run_time_sensitivity`, `run_window_scenarios`, `run_early_delivery`, `run_dev_grid`,
`run_v2_solver_grid`, `run_v2_layerb_warm`, `run_phase_c_solver`, `run_reoptimisation` e `run_robustness`
(`run_gate.py` não retoma: sobrescreve).
